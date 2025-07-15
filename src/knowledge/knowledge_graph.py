"""
Knowledge Graph System for Physics Concepts
Provides visual concept mapping, semantic connections, and relationship tracking.
"""

import json
import sqlite3
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle
import hashlib

from ..config.settings import get_settings


class ConceptType(Enum):
    """Types of physics concepts in the knowledge graph."""
    FUNDAMENTAL_PRINCIPLE = "fundamental_principle"
    PHYSICAL_LAW = "physical_law"
    MATHEMATICAL_FORMULA = "mathematical_formula"
    PHYSICAL_QUANTITY = "physical_quantity"
    EXPERIMENTAL_TECHNIQUE = "experimental_technique"
    THEORETICAL_FRAMEWORK = "theoretical_framework"
    PHENOMENON = "phenomenon"
    PARTICLE = "particle"
    FORCE = "force"
    FIELD = "field"
    CONSERVATION_LAW = "conservation_law"
    SYMMETRY = "symmetry"


class RelationshipType(Enum):
    """Types of relationships between physics concepts."""
    CAUSES = "causes"
    REQUIRES = "requires"
    ENABLES = "enables"
    CONTRADICTS = "contradicts"
    GENERALIZES = "generalizes"
    SPECIALIZES = "specializes"
    APPLIES_TO = "applies_to"
    DERIVED_FROM = "derived_from"
    EQUIVALENT_TO = "equivalent_to"
    ANALOGOUS_TO = "analogous_to"
    MEASURED_BY = "measured_by"
    CONSERVES = "conserves"
    VIOLATES = "violates"
    DESCRIBES = "describes"
    PART_OF = "part_of"


@dataclass
class PhysicsConcept:
    """Represents a physics concept in the knowledge graph."""
    id: str
    name: str
    concept_type: ConceptType
    domain: str  # e.g., "quantum_mechanics", "thermodynamics"
    description: str
    mathematical_representation: Optional[str] = None
    units: Optional[str] = None
    typical_values: Optional[Dict[str, Any]] = None
    historical_context: Optional[str] = None
    applications: Optional[List[str]] = None
    related_experiments: Optional[List[str]] = None
    difficulty_level: str = "undergraduate"
    confidence_score: float = 1.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.applications is None:
            self.applications = []
        if self.related_experiments is None:
            self.related_experiments = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ConceptRelationship:
    """Represents a relationship between two physics concepts."""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    description: Optional[str] = None
    evidence: Optional[List[str]] = None
    context: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.evidence is None:
            self.evidence = []
        if self.metadata is None:
            self.metadata = {}


class PhysicsKnowledgeGraph:
    """
    Comprehensive knowledge graph for physics concepts and relationships.
    
    Features:
    - Concept nodes with rich metadata
    - Typed relationships with weights
    - Semantic similarity calculations
    - Graph traversal and analysis
    - Persistent storage with SQLite
    - NetworkX integration for graph algorithms
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the physics knowledge graph.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.settings = get_settings()
        # Create data directory if it doesn't exist
        data_dir = Path("./data")
        data_dir.mkdir(exist_ok=True)
        self.db_path = db_path or str(data_dir / "knowledge_graph.db")
        
        # NetworkX graph for algorithms
        self.graph = nx.DiGraph()
        
        # In-memory storage for fast access
        self.concepts: Dict[str, PhysicsConcept] = {}
        self.relationships: Dict[str, ConceptRelationship] = {}
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_from_database()
        
        # Initialize with core physics concepts
        self._initialize_core_concepts()
    
    def _init_database(self):
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Concepts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS concepts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    concept_type TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    description TEXT,
                    mathematical_representation TEXT,
                    units TEXT,
                    typical_values TEXT,  -- JSON
                    historical_context TEXT,
                    applications TEXT,  -- JSON
                    related_experiments TEXT,  -- JSON
                    difficulty_level TEXT DEFAULT 'undergraduate',
                    confidence_score REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT  -- JSON
                )
            """)
            
            # Relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    confidence REAL NOT NULL,
                    description TEXT,
                    evidence TEXT,  -- JSON
                    context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,  -- JSON
                    FOREIGN KEY (source_id) REFERENCES concepts (id),
                    FOREIGN KEY (target_id) REFERENCES concepts (id)
                )
            """)
            
            # Graph snapshots table (for versioning)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_hash TEXT UNIQUE,
                    graph_data BLOB,  -- Pickled NetworkX graph
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_concepts_domain ON concepts(domain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_concepts_type ON concepts(concept_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type)")
            
            conn.commit()
    
    def _load_from_database(self):
        """Load concepts and relationships from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='concepts'")
                if not cursor.fetchone():
                    return  # Tables don't exist yet, skip loading
                
                # Load concepts
                cursor.execute("SELECT * FROM concepts")
            for row in cursor.fetchall():
                concept_data = {
                    'id': row[0],
                    'name': row[1],
                    'concept_type': ConceptType(row[2]),
                    'domain': row[3],
                    'description': row[4],
                    'mathematical_representation': row[5],
                    'units': row[6],
                    'typical_values': json.loads(row[7]) if row[7] else None,
                    'historical_context': row[8],
                    'applications': json.loads(row[9]) if row[9] else [],
                    'related_experiments': json.loads(row[10]) if row[10] else [],
                    'difficulty_level': row[11],
                    'confidence_score': row[12],
                    'created_at': datetime.fromisoformat(row[13]) if row[13] else None,
                    'updated_at': datetime.fromisoformat(row[14]) if row[14] else None,
                    'metadata': json.loads(row[15]) if row[15] else {}
                }
                concept = PhysicsConcept(**concept_data)
                self.concepts[concept.id] = concept
                self.graph.add_node(concept.id, **asdict(concept))
            
            # Load relationships
            cursor.execute("SELECT * FROM relationships")
            for row in cursor.fetchall():
                relationship_data = {
                    'source_id': row[1],
                    'target_id': row[2],
                    'relationship_type': RelationshipType(row[3]),
                    'strength': row[4],
                    'confidence': row[5],
                    'description': row[6],
                    'evidence': json.loads(row[7]) if row[7] else [],
                    'context': row[8],
                    'created_at': datetime.fromisoformat(row[9]) if row[9] else None,
                    'metadata': json.loads(row[10]) if row[10] else {}
                }
                relationship = ConceptRelationship(**relationship_data)
                rel_id = f"{relationship.source_id}_{relationship.target_id}_{relationship.relationship_type.value}"
                self.relationships[rel_id] = relationship
                
                # Add to NetworkX graph
                self.graph.add_edge(
                    relationship.source_id,
                    relationship.target_id,
                    relationship_type=relationship.relationship_type.value,
                    strength=relationship.strength,
                    confidence=relationship.confidence,
                    description=relationship.description
                )
        except Exception as e:
            # If loading fails, just continue with empty graph
            print(f"Warning: Could not load from database: {e}")
            return
    
    def _initialize_core_concepts(self):
        """Initialize core physics concepts if the graph is empty."""
        if len(self.concepts) > 0:
            return  # Already initialized
        
        # Core physics concepts
        core_concepts = [
            # Fundamental principles
            PhysicsConcept(
                id="energy_conservation",
                name="Conservation of Energy",
                concept_type=ConceptType.CONSERVATION_LAW,
                domain="general_physics",
                description="Energy cannot be created or destroyed, only transformed from one form to another",
                mathematical_representation="E_total = constant",
                difficulty_level="high_school"
            ),
            PhysicsConcept(
                id="momentum_conservation",
                name="Conservation of Momentum",
                concept_type=ConceptType.CONSERVATION_LAW,
                domain="mechanics",
                description="The total momentum of a closed system remains constant",
                mathematical_representation="Σp_i = constant",
                difficulty_level="high_school"
            ),
            PhysicsConcept(
                id="newtons_second_law",
                name="Newton's Second Law",
                concept_type=ConceptType.PHYSICAL_LAW,
                domain="mechanics",
                description="The acceleration of an object is directly proportional to the net force acting on it",
                mathematical_representation="F = ma",
                difficulty_level="high_school"
            ),
            
            # Quantum mechanics
            PhysicsConcept(
                id="wave_function",
                name="Wave Function",
                concept_type=ConceptType.THEORETICAL_FRAMEWORK,
                domain="quantum_mechanics",
                description="Mathematical description of the quantum state of a system",
                mathematical_representation="ψ(x,t)",
                difficulty_level="undergraduate"
            ),
            PhysicsConcept(
                id="uncertainty_principle",
                name="Heisenberg Uncertainty Principle",
                concept_type=ConceptType.FUNDAMENTAL_PRINCIPLE,
                domain="quantum_mechanics",
                description="Fundamental limit to the precision with which certain pairs of properties can be known",
                mathematical_representation="Δx·Δp ≥ ℏ/2",
                difficulty_level="undergraduate"
            ),
            
            # Thermodynamics
            PhysicsConcept(
                id="entropy",
                name="Entropy",
                concept_type=ConceptType.PHYSICAL_QUANTITY,
                domain="thermodynamics",
                description="Measure of disorder or randomness in a system",
                mathematical_representation="S = k_B ln(Ω)",
                units="J/K",
                difficulty_level="undergraduate"
            ),
            PhysicsConcept(
                id="second_law_thermodynamics",
                name="Second Law of Thermodynamics",
                concept_type=ConceptType.PHYSICAL_LAW,
                domain="thermodynamics",
                description="The entropy of an isolated system never decreases",
                mathematical_representation="ΔS ≥ 0",
                difficulty_level="undergraduate"
            ),
            
            # Electromagnetism
            PhysicsConcept(
                id="electric_field",
                name="Electric Field",
                concept_type=ConceptType.FIELD,
                domain="electromagnetism",
                description="Region around a charged particle where other charges experience a force",
                mathematical_representation="E = F/q",
                units="N/C or V/m",
                difficulty_level="undergraduate"
            ),
            PhysicsConcept(
                id="magnetic_field",
                name="Magnetic Field",
                concept_type=ConceptType.FIELD,
                domain="electromagnetism",
                description="Region around a magnet where magnetic forces can be detected",
                mathematical_representation="B",
                units="T (Tesla)",
                difficulty_level="undergraduate"
            ),
            
            # Relativity
            PhysicsConcept(
                id="spacetime",
                name="Spacetime",
                concept_type=ConceptType.THEORETICAL_FRAMEWORK,
                domain="relativity",
                description="Four-dimensional continuum combining space and time",
                mathematical_representation="ds² = -c²dt² + dx² + dy² + dz²",
                difficulty_level="graduate"
            )
        ]
        
        # Add core concepts
        for concept in core_concepts:
            self.add_concept(concept)
        
        # Add core relationships
        core_relationships = [
            # Energy and momentum conservation
            ConceptRelationship(
                source_id="energy_conservation",
                target_id="momentum_conservation",
                relationship_type=RelationshipType.ANALOGOUS_TO,
                strength=0.8,
                confidence=0.9,
                description="Both are fundamental conservation laws"
            ),
            
            # Newton's second law relationships
            ConceptRelationship(
                source_id="newtons_second_law",
                target_id="momentum_conservation",
                relationship_type=RelationshipType.ENABLES,
                strength=0.9,
                confidence=0.95,
                description="Newton's second law leads to momentum conservation"
            ),
            
            # Quantum mechanics relationships
            ConceptRelationship(
                source_id="wave_function",
                target_id="uncertainty_principle",
                relationship_type=RelationshipType.REQUIRES,
                strength=0.9,
                confidence=0.9,
                description="Wave function formalism leads to uncertainty principle"
            ),
            
            # Thermodynamics relationships
            ConceptRelationship(
                source_id="entropy",
                target_id="second_law_thermodynamics",
                relationship_type=RelationshipType.PART_OF,
                strength=1.0,
                confidence=0.95,
                description="Entropy is central to the second law"
            ),
            
            # Electromagnetic relationships
            ConceptRelationship(
                source_id="electric_field",
                target_id="magnetic_field",
                relationship_type=RelationshipType.ANALOGOUS_TO,
                strength=0.8,
                confidence=0.9,
                description="Electric and magnetic fields are related phenomena"
            )
        ]
        
        # Add core relationships
        for relationship in core_relationships:
            self.add_relationship(relationship)
    
    def add_concept(self, concept: PhysicsConcept) -> bool:
        """Add a new concept to the knowledge graph.
        
        Args:
            concept: PhysicsConcept to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add to in-memory storage
            self.concepts[concept.id] = concept
            
            # Add to NetworkX graph
            self.graph.add_node(concept.id, **asdict(concept))
            
            # Save to database
            self._save_concept_to_db(concept)
            
            return True
            
        except Exception as e:
            print(f"Error adding concept {concept.id}: {e}")
            return False
    
    def add_relationship(self, relationship: ConceptRelationship) -> bool:
        """Add a new relationship to the knowledge graph.
        
        Args:
            relationship: ConceptRelationship to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate that both concepts exist
            if relationship.source_id not in self.concepts:
                print(f"Source concept {relationship.source_id} not found")
                return False
            if relationship.target_id not in self.concepts:
                print(f"Target concept {relationship.target_id} not found")
                return False
            
            # Create relationship ID
            rel_id = f"{relationship.source_id}_{relationship.target_id}_{relationship.relationship_type.value}"
            
            # Add to in-memory storage
            self.relationships[rel_id] = relationship
            
            # Add to NetworkX graph
            self.graph.add_edge(
                relationship.source_id,
                relationship.target_id,
                relationship_type=relationship.relationship_type.value,
                strength=relationship.strength,
                confidence=relationship.confidence,
                description=relationship.description
            )
            
            # Save to database
            self._save_relationship_to_db(relationship)
            
            return True
            
        except Exception as e:
            print(f"Error adding relationship: {e}")
            return False
    
    def _save_concept_to_db(self, concept: PhysicsConcept):
        """Save concept to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO concepts (
                    id, name, concept_type, domain, description,
                    mathematical_representation, units, typical_values,
                    historical_context, applications, related_experiments,
                    difficulty_level, confidence_score, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                concept.id,
                concept.name,
                concept.concept_type.value,
                concept.domain,
                concept.description,
                concept.mathematical_representation,
                concept.units,
                json.dumps(concept.typical_values) if concept.typical_values else None,
                concept.historical_context,
                json.dumps(concept.applications),
                json.dumps(concept.related_experiments),
                concept.difficulty_level,
                concept.confidence_score,
                concept.created_at.isoformat() if concept.created_at else None,
                concept.updated_at.isoformat() if concept.updated_at else None,
                json.dumps(concept.metadata)
            ))
            conn.commit()
    
    def _save_relationship_to_db(self, relationship: ConceptRelationship):
        """Save relationship to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            rel_id = f"{relationship.source_id}_{relationship.target_id}_{relationship.relationship_type.value}"
            cursor.execute("""
                INSERT OR REPLACE INTO relationships (
                    id, source_id, target_id, relationship_type, strength, confidence,
                    description, evidence, context, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rel_id,
                relationship.source_id,
                relationship.target_id,
                relationship.relationship_type.value,
                relationship.strength,
                relationship.confidence,
                relationship.description,
                json.dumps(relationship.evidence),
                relationship.context,
                relationship.created_at.isoformat() if relationship.created_at else None,
                json.dumps(relationship.metadata)
            ))
            conn.commit()
    
    def get_concept(self, concept_id: str) -> Optional[PhysicsConcept]:
        """Get a concept by ID."""
        return self.concepts.get(concept_id)
    
    def get_concept_by_name(self, name: str) -> Optional[PhysicsConcept]:
        """Get a concept by its name."""
        for concept in self.concepts.values():
            if concept.name.lower() == name.lower():
                return concept
        return None
    
    def get_concept_relationships(self, concept_id: str) -> List[ConceptRelationship]:
        """Get all relationships for a concept."""
        relationships = []
        for relationship in self.relationships.values():
            if relationship.source_id == concept_id or relationship.target_id == concept_id:
                relationships.append(relationship)
        return relationships
    
    def get_concepts_by_domain(self, domain: str) -> List[PhysicsConcept]:
        """Get all concepts in a specific domain."""
        return [concept for concept in self.concepts.values() if concept.domain == domain]
    
    def get_concepts_by_type(self, concept_type: ConceptType) -> List[PhysicsConcept]:
        """Get all concepts of a specific type."""
        return [concept for concept in self.concepts.values() if concept.concept_type == concept_type]
    
    def get_related_concepts(self, concept_id: str, relationship_types: Optional[List[RelationshipType]] = None) -> List[Tuple[PhysicsConcept, ConceptRelationship]]:
        """Get concepts related to a given concept.
        
        Args:
            concept_id: ID of the source concept
            relationship_types: Optional list of relationship types to filter by
            
        Returns:
            List of (concept, relationship) tuples
        """
        related = []
        
        for rel_id, relationship in self.relationships.items():
            if relationship.source_id == concept_id:
                if relationship_types is None or relationship.relationship_type in relationship_types:
                    target_concept = self.concepts.get(relationship.target_id)
                    if target_concept:
                        related.append((target_concept, relationship))
        
        return related
    
    def find_path(self, source_id: str, target_id: str, max_length: int = 5) -> Optional[List[str]]:
        """Find shortest path between two concepts.
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            max_length: Maximum path length
            
        Returns:
            List of concept IDs forming the path, or None if no path exists
        """
        try:
            if source_id not in self.graph or target_id not in self.graph:
                return None
            
            path = nx.shortest_path(self.graph, source_id, target_id)
            if len(path) <= max_length + 1:  # +1 because path includes both endpoints
                return path
            else:
                return None
                
        except nx.NetworkXNoPath:
            return None
    
    def get_concept_centrality(self) -> Dict[str, float]:
        """Calculate centrality measures for all concepts."""
        try:
            centrality = nx.betweenness_centrality(self.graph)
            return centrality
        except:
            return {}
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            "num_concepts": len(self.concepts),
            "num_relationships": len(self.relationships),
            "num_domains": len(set(concept.domain for concept in self.concepts.values())),
            "concept_types": {ct.value: len(self.get_concepts_by_type(ct)) for ct in ConceptType},
            "relationship_types": {rt.value: len([r for r in self.relationships.values() if r.relationship_type == rt]) for rt in RelationshipType},
            "graph_density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph) if len(self.graph) > 0 else False,
            "num_components": nx.number_weakly_connected_components(self.graph) if len(self.graph) > 0 else 0
        }
    
    def search_concepts(self, query: str, domain: Optional[str] = None, limit: int = 10) -> List[PhysicsConcept]:
        """Search for concepts by name or description.
        
        Args:
            query: Search query
            domain: Optional domain filter
            limit: Maximum number of results
            
        Returns:
            List of matching concepts
        """
        query_lower = query.lower()
        matches = []
        
        for concept in self.concepts.values():
            if domain and concept.domain != domain:
                continue
            
            score = 0
            
            # Name match (highest priority)
            if query_lower in concept.name.lower():
                score += 10
            
            # Description match
            if concept.description and query_lower in concept.description.lower():
                score += 5
            
            # Mathematical representation match
            if concept.mathematical_representation and query_lower in concept.mathematical_representation.lower():
                score += 3
            
            if score > 0:
                matches.append((concept, score))
        
        # Sort by score and return top results
        matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in matches[:limit]]
    
    def create_subgraph(self, concept_ids: List[str], include_relationships: bool = True) -> 'PhysicsKnowledgeGraph':
        """Create a subgraph containing only specified concepts.
        
        Args:
            concept_ids: List of concept IDs to include
            include_relationships: Whether to include relationships between concepts
            
        Returns:
            New PhysicsKnowledgeGraph instance with the subgraph
        """
        subgraph = PhysicsKnowledgeGraph(db_path=":memory:")  # In-memory database
        
        # Add concepts
        for concept_id in concept_ids:
            if concept_id in self.concepts:
                subgraph.add_concept(self.concepts[concept_id])
        
        # Add relationships if requested
        if include_relationships:
            for relationship in self.relationships.values():
                if (relationship.source_id in concept_ids and 
                    relationship.target_id in concept_ids):
                    subgraph.add_relationship(relationship)
        
        return subgraph
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export the knowledge graph to a dictionary."""
        return {
            "concepts": {cid: asdict(concept) for cid, concept in self.concepts.items()},
            "relationships": {rid: asdict(relationship) for rid, relationship in self.relationships.items()},
            "statistics": self.get_graph_statistics()
        }
    
    def save_snapshot(self, description: str = "") -> str:
        """Save a snapshot of the current graph state.
        
        Args:
            description: Optional description of the snapshot
            
        Returns:
            Snapshot hash
        """
        # Create hash of current state
        graph_data = pickle.dumps(self.graph)
        snapshot_hash = hashlib.md5(graph_data).hexdigest()
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO graph_snapshots (snapshot_hash, graph_data, description)
                VALUES (?, ?, ?)
            """, (snapshot_hash, graph_data, description))
            conn.commit()
        
        return snapshot_hash 