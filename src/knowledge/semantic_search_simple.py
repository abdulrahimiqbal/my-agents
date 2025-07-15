"""
Simplified Semantic Search System for Phase 3 - Physics Knowledge Discovery

This module provides semantic search capabilities for physics concepts and knowledge.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .knowledge_graph import PhysicsKnowledgeGraph, PhysicsConcept, ConceptRelationship
from ..database.knowledge_api import KnowledgeAPI


class SearchMode(Enum):
    """Different modes of semantic search."""
    SEMANTIC = "semantic"
    CONCEPTUAL = "conceptual"
    CONTEXTUAL = "contextual"
    HYBRID = "hybrid"
    MULTIMODAL = "multimodal"


class QueryIntent(Enum):
    """Different types of query intentions."""
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    PROBLEM_SOLVING = "problem_solving"
    COMPARISON = "comparison"
    APPLICATION = "application"
    DERIVATION = "derivation"
    EXAMPLE = "example"
    RELATIONSHIP = "relationship"


class ExpertiseLevel(Enum):
    """User expertise levels for personalized results."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class SearchResult:
    """Represents a search result with relevance scoring."""
    content_id: str
    content_type: str
    title: str
    description: str
    relevance_score: float
    confidence_score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_concepts: List[str] = field(default_factory=list)
    explanation: Optional[str] = None


@dataclass
class SearchContext:
    """Context information for search queries."""
    user_expertise: ExpertiseLevel
    physics_domain: Optional[str] = None
    previous_queries: List[str] = field(default_factory=list)
    session_history: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)


class SimpleSemanticSearch:
    """
    Simplified semantic search system for physics knowledge.
    
    Features:
    - Basic semantic similarity using text matching
    - Concept-based search through knowledge graph
    - Contextual understanding based on physics domains
    - Personalized results based on expertise level
    """
    
    def __init__(self, 
                 knowledge_graph: Optional[PhysicsKnowledgeGraph] = None,
                 knowledge_api: Optional[KnowledgeAPI] = None,
                 enable_embeddings: bool = False):
        """Initialize the semantic search system."""
        self.knowledge_graph = knowledge_graph or PhysicsKnowledgeGraph()
        self.knowledge_api = knowledge_api or KnowledgeAPI()
        self.enable_embeddings = enable_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        
        # Initialize embeddings if available
        if self.enable_embeddings:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.concept_embeddings: Dict[str, np.ndarray] = {}
                self._initialize_concept_embeddings()
            except Exception as e:
                logging.warning(f"Failed to initialize embeddings: {e}")
                self.enable_embeddings = False
        
        # Initialize search index
        self.concept_index: Dict[str, PhysicsConcept] = {}
        self.domain_index: Dict[str, List[str]] = {}
        self.keyword_index: Dict[str, Set[str]] = {}
        
        self._build_search_index()
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_concept_embeddings(self):
        """Initialize concept embeddings if sentence transformers is available."""
        if not self.enable_embeddings:
            return
            
        try:
            concepts = self.knowledge_graph.get_all_concepts()
            for concept in concepts:
                concept_text = f"{concept.name} {concept.description}"
                if concept.mathematical_representation:
                    concept_text += f" {concept.mathematical_representation}"
                
                embedding = self.embedding_model.encode(concept_text)
                self.concept_embeddings[concept.id] = embedding
        except Exception as e:
            self.logger.warning(f"Failed to initialize concept embeddings: {e}")
    
    def _build_search_index(self):
        """Build search indices for fast lookup."""
        try:
            concepts = self.knowledge_graph.get_all_concepts()
            
            for concept in concepts:
                # Concept index
                self.concept_index[concept.id] = concept
                
                # Domain index
                if concept.domain not in self.domain_index:
                    self.domain_index[concept.domain] = []
                self.domain_index[concept.domain].append(concept.id)
                
                # Keyword index
                keywords = self._extract_keywords(concept)
                for keyword in keywords:
                    if keyword not in self.keyword_index:
                        self.keyword_index[keyword] = set()
                    self.keyword_index[keyword].add(concept.id)
                    
        except Exception as e:
            self.logger.warning(f"Failed to build search index: {e}")
    
    def _extract_keywords(self, concept: PhysicsConcept) -> List[str]:
        """Extract keywords from a concept for indexing."""
        keywords = []
        
        # Add name words
        keywords.extend(concept.name.lower().split())
        
        # Add description words
        if concept.description:
            keywords.extend(concept.description.lower().split())
        
        # Add domain
        keywords.append(concept.domain.lower())
        
        # Add concept type
        keywords.append(concept.concept_type.value.lower())
        
        return [kw for kw in keywords if len(kw) > 2]  # Filter short words
    
    def search(self, 
               query: str,
               search_mode: SearchMode = SearchMode.HYBRID,
               context: Optional[SearchContext] = None,
               max_results: int = 10) -> List[SearchResult]:
        """
        Perform semantic search for physics concepts and knowledge.
        
        Args:
            query: Search query
            search_mode: Mode of search to perform
            context: Search context for personalization
            max_results: Maximum number of results to return
            
        Returns:
            List of search results ranked by relevance
        """
        try:
            # Classify query intent
            intent = self._classify_query_intent(query)
            
            # Perform search based on mode
            if search_mode == SearchMode.SEMANTIC and self.enable_embeddings:
                results = self._semantic_search(query, context, max_results)
            elif search_mode == SearchMode.CONCEPTUAL:
                results = self._conceptual_search(query, context, max_results)
            elif search_mode == SearchMode.CONTEXTUAL:
                results = self._contextual_search(query, context, max_results)
            elif search_mode == SearchMode.HYBRID:
                results = self._hybrid_search(query, context, max_results)
            else:
                results = self._keyword_search(query, context, max_results)
            
            # Personalize results based on expertise level
            if context and context.user_expertise:
                results = self._personalize_results(results, context)
            
            return results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def _classify_query_intent(self, query: str) -> QueryIntent:
        """Classify the intent of a search query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'define', 'definition']):
            return QueryIntent.DEFINITION
        elif any(word in query_lower for word in ['how', 'why', 'explain']):
            return QueryIntent.EXPLANATION
        elif any(word in query_lower for word in ['solve', 'calculate', 'find']):
            return QueryIntent.PROBLEM_SOLVING
        elif any(word in query_lower for word in ['compare', 'difference', 'vs']):
            return QueryIntent.COMPARISON
        elif any(word in query_lower for word in ['apply', 'use', 'application']):
            return QueryIntent.APPLICATION
        elif any(word in query_lower for word in ['derive', 'proof', 'derivation']):
            return QueryIntent.DERIVATION
        elif any(word in query_lower for word in ['example', 'instance', 'case']):
            return QueryIntent.EXAMPLE
        elif any(word in query_lower for word in ['relation', 'connect', 'related']):
            return QueryIntent.RELATIONSHIP
        else:
            return QueryIntent.EXPLANATION
    
    def _semantic_search(self, query: str, context: Optional[SearchContext], max_results: int) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        if not self.enable_embeddings:
            return self._keyword_search(query, context, max_results)
        
        try:
            query_embedding = self.embedding_model.encode(query)
            similarities = []
            
            for concept_id, concept_embedding in self.concept_embeddings.items():
                similarity = np.dot(query_embedding, concept_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(concept_embedding)
                )
                similarities.append((concept_id, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for concept_id, similarity in similarities[:max_results]:
                concept = self.concept_index.get(concept_id)
                if concept:
                    result = SearchResult(
                        content_id=concept.id,
                        content_type="concept",
                        title=concept.name,
                        description=concept.description,
                        relevance_score=similarity,
                        confidence_score=similarity,
                        source="knowledge_graph",
                        metadata={"domain": concept.domain, "type": concept.concept_type.value}
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def _conceptual_search(self, query: str, context: Optional[SearchContext], max_results: int) -> List[SearchResult]:
        """Search based on concept relationships."""
        results = []
        query_words = set(query.lower().split())
        
        for concept_id, concept in self.concept_index.items():
            concept_words = set(self._extract_keywords(concept))
            
            # Calculate word overlap
            overlap = len(query_words & concept_words)
            if overlap > 0:
                relevance = overlap / len(query_words)
                
                result = SearchResult(
                    content_id=concept.id,
                    content_type="concept",
                    title=concept.name,
                    description=concept.description,
                    relevance_score=relevance,
                    confidence_score=relevance,
                    source="knowledge_graph",
                    metadata={"domain": concept.domain, "type": concept.concept_type.value}
                )
                results.append(result)
        
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)[:max_results]
    
    def _contextual_search(self, query: str, context: Optional[SearchContext], max_results: int) -> List[SearchResult]:
        """Search with contextual understanding."""
        results = self._keyword_search(query, context, max_results)
        
        # Enhance with context
        if context and context.physics_domain:
            domain_results = []
            for result in results:
                if result.metadata.get("domain") == context.physics_domain:
                    result.relevance_score *= 1.5  # Boost domain-relevant results
                domain_results.append(result)
            results = domain_results
        
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)[:max_results]
    
    def _hybrid_search(self, query: str, context: Optional[SearchContext], max_results: int) -> List[SearchResult]:
        """Combine multiple search approaches."""
        # Get results from different approaches
        semantic_results = self._semantic_search(query, context, max_results) if self.enable_embeddings else []
        conceptual_results = self._conceptual_search(query, context, max_results)
        keyword_results = self._keyword_search(query, context, max_results)
        
        # Combine and deduplicate
        combined_results = {}
        
        # Add semantic results with high weight
        for result in semantic_results:
            combined_results[result.content_id] = result
            result.relevance_score *= 0.6  # Semantic weight
        
        # Add conceptual results
        for result in conceptual_results:
            if result.content_id in combined_results:
                combined_results[result.content_id].relevance_score += result.relevance_score * 0.3
            else:
                result.relevance_score *= 0.3
                combined_results[result.content_id] = result
        
        # Add keyword results
        for result in keyword_results:
            if result.content_id in combined_results:
                combined_results[result.content_id].relevance_score += result.relevance_score * 0.1
            else:
                result.relevance_score *= 0.1
                combined_results[result.content_id] = result
        
        # Sort by combined relevance
        results = list(combined_results.values())
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)[:max_results]
    
    def _keyword_search(self, query: str, context: Optional[SearchContext], max_results: int) -> List[SearchResult]:
        """Basic keyword-based search."""
        results = []
        query_words = set(query.lower().split())
        
        # Search through keyword index
        matching_concepts = set()
        for word in query_words:
            if word in self.keyword_index:
                matching_concepts.update(self.keyword_index[word])
        
        # Score the matches
        for concept_id in matching_concepts:
            concept = self.concept_index.get(concept_id)
            if concept:
                concept_words = set(self._extract_keywords(concept))
                overlap = len(query_words & concept_words)
                relevance = overlap / len(query_words) if query_words else 0
                
                result = SearchResult(
                    content_id=concept.id,
                    content_type="concept",
                    title=concept.name,
                    description=concept.description,
                    relevance_score=relevance,
                    confidence_score=relevance,
                    source="knowledge_graph",
                    metadata={"domain": concept.domain, "type": concept.concept_type.value}
                )
                results.append(result)
        
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)[:max_results]
    
    def _personalize_results(self, results: List[SearchResult], context: SearchContext) -> List[SearchResult]:
        """Personalize results based on user expertise and context."""
        expertise_weights = {
            ExpertiseLevel.BEGINNER: {"basic": 1.5, "intermediate": 1.0, "advanced": 0.5, "expert": 0.2},
            ExpertiseLevel.INTERMEDIATE: {"basic": 1.2, "intermediate": 1.5, "advanced": 1.0, "expert": 0.7},
            ExpertiseLevel.ADVANCED: {"basic": 0.8, "intermediate": 1.2, "advanced": 1.5, "expert": 1.0},
            ExpertiseLevel.EXPERT: {"basic": 0.5, "intermediate": 0.8, "advanced": 1.2, "expert": 1.5}
        }
        
        weights = expertise_weights.get(context.user_expertise, {})
        
        for result in results:
            # Adjust based on difficulty level (if available)
            concept = self.concept_index.get(result.content_id)
            if concept and hasattr(concept, 'difficulty_level'):
                difficulty = concept.difficulty_level
                weight = weights.get(difficulty, 1.0)
                result.relevance_score *= weight
        
        return results
    
    def get_related_concepts(self, concept_id: str, max_related: int = 5) -> List[SearchResult]:
        """Get concepts related to a given concept."""
        try:
            concept = self.concept_index.get(concept_id)
            if not concept:
                return []
            
            related_results = []
            
            # Get relationships from knowledge graph
            relationships = self.knowledge_graph.get_concept_relationships(concept_id)
            
            for rel in relationships:
                target_concept = self.concept_index.get(rel.target_id)
                if target_concept:
                    result = SearchResult(
                        content_id=target_concept.id,
                        content_type="concept",
                        title=target_concept.name,
                        description=target_concept.description,
                        relevance_score=rel.strength,
                        confidence_score=rel.confidence,
                        source="knowledge_graph",
                        metadata={"relationship": rel.relationship_type.value}
                    )
                    related_results.append(result)
            
            return sorted(related_results, key=lambda x: x.relevance_score, reverse=True)[:max_related]
            
        except Exception as e:
            self.logger.error(f"Failed to get related concepts: {e}")
            return []
    
    def get_search_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """Get search suggestions based on partial query."""
        suggestions = []
        partial_lower = partial_query.lower()
        
        # Search through concept names
        for concept in self.concept_index.values():
            if partial_lower in concept.name.lower():
                suggestions.append(concept.name)
        
        # Search through keywords
        for keyword in self.keyword_index.keys():
            if partial_lower in keyword.lower():
                suggestions.append(keyword)
        
        return list(set(suggestions))[:max_suggestions]
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search system."""
        return {
            "total_concepts": len(self.concept_index),
            "total_domains": len(self.domain_index),
            "total_keywords": len(self.keyword_index),
            "embeddings_enabled": self.enable_embeddings,
            "concept_embeddings": len(self.concept_embeddings) if self.enable_embeddings else 0
        }


# Backward compatibility
SemanticSearchEngine = SimpleSemanticSearch 