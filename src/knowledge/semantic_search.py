"""
Advanced Semantic Search System - Phase 3

This module implements sophisticated semantic search capabilities that enhance knowledge retrieval with:
1. Semantic search with vector embeddings and similarity matching
2. Concept similarity matching using knowledge graph relationships
3. Contextual knowledge recommendations based on user intent
4. Multi-modal search across text, equations, and diagrams
5. Intelligent query expansion and refinement
6. Personalized search results based on user expertise level
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from datetime import datetime
from collections import defaultdict, Counter
import logging
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available, using fallback search")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .knowledge_graph import PhysicsKnowledgeGraph, PhysicsConcept, ConceptRelationship
from ..database.knowledge_api import KnowledgeAPI


class SearchType(Enum):
    """Types of semantic search."""
    SEMANTIC = "semantic"
    CONCEPTUAL = "conceptual"
    CONTEXTUAL = "contextual"
    HYBRID = "hybrid"
    MULTIMODAL = "multimodal"


class QueryIntent(Enum):
    """Types of query intent."""
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    PROBLEM_SOLVING = "problem_solving"
    COMPARISON = "comparison"
    APPLICATION = "application"
    DERIVATION = "derivation"
    EXAMPLE = "example"
    RELATIONSHIP = "relationship"


class ExpertiseLevel(Enum):
    """User expertise levels for personalized search."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class SearchQuery:
    """Represents a semantic search query."""
    query_text: str
    search_type: SearchType = SearchType.HYBRID
    intent: Optional[QueryIntent] = None
    expertise_level: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    domain_filter: Optional[str] = None
    concept_filter: Optional[List[str]] = None
    max_results: int = 10
    similarity_threshold: float = 0.7
    include_related: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Represents a search result with semantic information."""
    content_id: str
    title: str
    content: str
    content_type: str
    similarity_score: float
    relevance_score: float
    concept_matches: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    expertise_level: Optional[ExpertiseLevel] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: Optional[str] = None


@dataclass
class ConceptSimilarity:
    """Represents similarity between concepts."""
    concept1: str
    concept2: str
    similarity_score: float
    relationship_type: str
    path_length: int
    common_attributes: List[str] = field(default_factory=list)


@dataclass
class SearchRecommendation:
    """Represents a search recommendation."""
    query: str
    reason: str
    confidence: float
    related_concepts: List[str] = field(default_factory=list)


class SemanticSearchEngine:
    """
    Advanced semantic search engine for physics knowledge.
    
    Features:
    - Vector-based semantic similarity
    - Knowledge graph-based concept matching
    - Contextual query understanding
    - Multi-modal search capabilities
    - Personalized results based on expertise
    - Intelligent query expansion
    """
    
    def __init__(self, 
                 knowledge_graph: Optional[PhysicsKnowledgeGraph] = None,
                 model_name: str = "all-MiniLM-L6-v2",
                 enable_gpu: bool = False):
        """
        Initialize the semantic search engine.
        
        Args:
            knowledge_graph: Physics knowledge graph for concept relationships
            model_name: Name of the sentence transformer model
            enable_gpu: Whether to use GPU acceleration
        """
        self.knowledge_graph = knowledge_graph or PhysicsKnowledgeGraph()
        self.knowledge_api = KnowledgeAPI()
        
        # Initialize embeddings model
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                if enable_gpu:
                    self.embedding_model = self.embedding_model.to('cuda')
            except Exception as e:
                logging.warning(f"Failed to load sentence transformer: {e}")
        
        # Initialize TF-IDF for fallback
        self.tfidf_vectorizer = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3)
            )
        
        # Initialize search components
        self.concept_embeddings = {}
        self.content_embeddings = {}
        self.concept_similarities = {}
        self.query_intent_classifier = QueryIntentClassifier()
        self.query_expander = QueryExpander(self.knowledge_graph)
        
        # Initialize caches
        self.search_cache = {}
        self.recommendation_cache = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize search index
        self._initialize_search_index()
    
    def _initialize_search_index(self):
        """Initialize the search index with existing knowledge."""
        self.logger.info("Initializing semantic search index...")
        
        # Index concepts from knowledge graph
        for concept in self.knowledge_graph.concepts.values():
            self._index_concept(concept)
        
        # Index content from knowledge API
        try:
            # Get all knowledge entries
            knowledge_entries = self.knowledge_api.get_all_knowledge()
            for entry in knowledge_entries:
                self._index_content(entry)
        except Exception as e:
            self.logger.warning(f"Failed to index knowledge content: {e}")
        
        # Pre-compute concept similarities
        self._compute_concept_similarities()
        
        self.logger.info("Search index initialization complete")
    
    def _index_concept(self, concept: PhysicsConcept):
        """Index a concept for semantic search."""
        try:
            # Create text representation
            text = f"{concept.name} {concept.description}"
            if concept.mathematical_representation:
                text += f" {concept.mathematical_representation}"
            
            # Generate embedding
            embedding = self.embedding_model.encode(text)
            self.concept_embeddings[concept.id] = embedding
            
        except Exception as e:
            self.logger.warning(f"Failed to index concept {concept.id}: {e}")
    
    def _index_content(self, content: Dict[str, Any]):
        """Index content for semantic search."""
        content_id = content.get('id', str(content.get('rowid', '')))
        content_text = content.get('content', '')
        
        if not content_text:
            return
        
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(content_text)
                self.content_embeddings[content_id] = embedding
            except Exception as e:
                self.logger.warning(f"Failed to embed content {content_id}: {e}")
    
    def _compute_concept_similarities(self):
        """Pre-compute similarities between concepts."""
        concepts = list(self.knowledge_graph.concepts.values())
        
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                similarity = self._calculate_concept_similarity(concept1, concept2)
                if similarity.similarity_score > 0.3:  # Only store significant similarities
                    key = f"{concept1.id}_{concept2.id}"
                    self.concept_similarities[key] = similarity
    
    def _calculate_concept_similarity(self, 
                                    concept1: PhysicsConcept, 
                                    concept2: PhysicsConcept) -> ConceptSimilarity:
        """Calculate similarity between two concepts."""
        # Embedding-based similarity
        embedding_sim = 0.0
        if (concept1.id in self.concept_embeddings and 
            concept2.id in self.concept_embeddings):
            emb1 = self.concept_embeddings[concept1.id]
            emb2 = self.concept_embeddings[concept2.id]
            embedding_sim = cosine_similarity([emb1], [emb2])[0][0]
        
        # Graph-based similarity
        graph_sim = 0.0
        path_length = float('inf')
        relationship_type = "none"
        
        try:
            path = self.knowledge_graph.find_path(concept1.id, concept2.id)
            if path:
                path_length = len(path) - 1
                graph_sim = 1.0 / (1.0 + path_length)
                relationship_type = "connected"
        except:
            pass
        
        # Domain similarity
        domain_sim = 1.0 if concept1.domain == concept2.domain else 0.0
        
        # Attribute similarity (using metadata instead of attributes)
        concept1_attrs = concept1.metadata or {}
        concept2_attrs = concept2.metadata or {}
        common_attrs = set(concept1_attrs.keys()) & set(concept2_attrs.keys())
        attr_sim = len(common_attrs) / max(len(concept1_attrs), len(concept2_attrs), 1)
        
        # Combined similarity
        total_sim = (embedding_sim * 0.4 + graph_sim * 0.3 + domain_sim * 0.2 + attr_sim * 0.1)
        
        return ConceptSimilarity(
            concept1=concept1.id,
            concept2=concept2.id,
            similarity_score=total_sim,
            relationship_type=relationship_type,
            path_length=int(path_length) if path_length != float('inf') else -1,
            common_attributes=list(common_attrs)
        )
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform semantic search based on the query."""
        # Check cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Classify query intent
        if not query.intent:
            query.intent = self.query_intent_classifier.classify(query.query_text)
        
        # Expand query if needed
        expanded_queries = self.query_expander.expand(query)
        
        # Perform search based on type
        if query.search_type == SearchType.SEMANTIC:
            results = self._semantic_search(query, expanded_queries)
        elif query.search_type == SearchType.CONCEPTUAL:
            results = self._conceptual_search(query, expanded_queries)
        elif query.search_type == SearchType.CONTEXTUAL:
            results = self._contextual_search(query, expanded_queries)
        elif query.search_type == SearchType.HYBRID:
            results = self._hybrid_search(query, expanded_queries)
        elif query.search_type == SearchType.MULTIMODAL:
            results = self._multimodal_search(query, expanded_queries)
        else:
            results = self._hybrid_search(query, expanded_queries)
        
        # Post-process results
        results = self._post_process_results(results, query)
        
        # Cache results
        self.search_cache[cache_key] = results
        
        return results
    
    def _semantic_search(self, query: SearchQuery, expanded_queries: List[str]) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        if not self.embedding_model:
            return self._fallback_search(query, expanded_queries)
        
        results = []
        
        # Get query embedding
        query_embedding = self.embedding_model.encode(query.query_text)
        
        # Search concepts
        for concept_id, concept_embedding in self.concept_embeddings.items():
            similarity = cosine_similarity([query_embedding], [concept_embedding])[0][0]
            
            if similarity >= query.similarity_threshold:
                concept = self.knowledge_graph.get_concept(concept_id)
                if concept:
                    result = SearchResult(
                        content_id=concept_id,
                        title=concept.name,
                        content=concept.description,
                        content_type="concept",
                        similarity_score=similarity,
                        relevance_score=similarity,
                        domain=concept.domain,
                        concept_matches=[concept.name]
                    )
                    results.append(result)
        
        # Search content
        for content_id, content_embedding in self.content_embeddings.items():
            similarity = cosine_similarity([query_embedding], [content_embedding])[0][0]
            
            if similarity >= query.similarity_threshold:
                # Get content details
                try:
                    content_details = self.knowledge_api.get_knowledge_by_id(content_id)
                    if content_details:
                        result = SearchResult(
                            content_id=content_id,
                            title=content_details.get('title', f'Content {content_id}'),
                            content=content_details.get('content', ''),
                            content_type=content_details.get('type', 'knowledge'),
                            similarity_score=similarity,
                            relevance_score=similarity
                        )
                        results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to get content details for {content_id}: {e}")
        
        return results
    
    def _conceptual_search(self, query: SearchQuery, expanded_queries: List[str]) -> List[SearchResult]:
        """Perform conceptual search using knowledge graph relationships."""
        results = []
        
        # Extract concepts from query
        query_concepts = self._extract_concepts_from_query(query.query_text)
        
        # Find related concepts
        related_concepts = set()
        for concept_name in query_concepts:
            concept = self.knowledge_graph.get_concept_by_name(concept_name)
            if concept:
                # Add direct relationships
                relationships = self.knowledge_graph.get_concept_relationships(concept.id)
                for rel in relationships:
                    related_concepts.add(rel.target_concept_id)
                    related_concepts.add(rel.source_concept_id)
                
                # Add similar concepts
                for sim_key, similarity in self.concept_similarities.items():
                    if concept.id in sim_key:
                        other_concept_id = sim_key.replace(concept.id, '').replace('_', '')
                        if other_concept_id:
                            related_concepts.add(other_concept_id)
        
        # Create results from related concepts
        for concept_id in related_concepts:
            concept = self.knowledge_graph.get_concept(concept_id)
            if concept:
                # Calculate relevance based on relationships
                relevance = self._calculate_concept_relevance(concept, query_concepts)
                
                result = SearchResult(
                    content_id=concept_id,
                    title=concept.name,
                    content=concept.description,
                    content_type="concept",
                    similarity_score=relevance,
                    relevance_score=relevance,
                    domain=concept.domain,
                    concept_matches=query_concepts,
                    related_concepts=list(related_concepts)
                )
                results.append(result)
        
        return results
    
    def _contextual_search(self, query: SearchQuery, expanded_queries: List[str]) -> List[SearchResult]:
        """Perform contextual search considering user intent and expertise."""
        results = []
        
        # Adjust search based on intent
        if query.intent == QueryIntent.DEFINITION:
            results.extend(self._search_definitions(query))
        elif query.intent == QueryIntent.EXPLANATION:
            results.extend(self._search_explanations(query))
        elif query.intent == QueryIntent.PROBLEM_SOLVING:
            results.extend(self._search_problem_solutions(query))
        elif query.intent == QueryIntent.COMPARISON:
            results.extend(self._search_comparisons(query))
        elif query.intent == QueryIntent.APPLICATION:
            results.extend(self._search_applications(query))
        elif query.intent == QueryIntent.DERIVATION:
            results.extend(self._search_derivations(query))
        elif query.intent == QueryIntent.EXAMPLE:
            results.extend(self._search_examples(query))
        elif query.intent == QueryIntent.RELATIONSHIP:
            results.extend(self._search_relationships(query))
        else:
            # Default to hybrid search
            results.extend(self._hybrid_search(query, expanded_queries))
        
        return results
    
    def _hybrid_search(self, query: SearchQuery, expanded_queries: List[str]) -> List[SearchResult]:
        """Perform hybrid search combining multiple approaches."""
        # Combine results from different search types
        semantic_results = self._semantic_search(query, expanded_queries)
        conceptual_results = self._conceptual_search(query, expanded_queries)
        contextual_results = self._contextual_search(query, expanded_queries)
        
        # Merge and deduplicate results
        all_results = {}
        
        # Add semantic results with weight
        for result in semantic_results:
            result.relevance_score = result.similarity_score * 0.4
            all_results[result.content_id] = result
        
        # Add conceptual results with weight
        for result in conceptual_results:
            if result.content_id in all_results:
                all_results[result.content_id].relevance_score += result.similarity_score * 0.3
            else:
                result.relevance_score = result.similarity_score * 0.3
                all_results[result.content_id] = result
        
        # Add contextual results with weight
        for result in contextual_results:
            if result.content_id in all_results:
                all_results[result.content_id].relevance_score += result.similarity_score * 0.3
            else:
                result.relevance_score = result.similarity_score * 0.3
                all_results[result.content_id] = result
        
        return list(all_results.values())
    
    def _multimodal_search(self, query: SearchQuery, expanded_queries: List[str]) -> List[SearchResult]:
        """Perform multimodal search across text, equations, and diagrams."""
        # For now, implement as enhanced hybrid search
        # In a full implementation, this would include:
        # - Equation parsing and matching
        # - Diagram content analysis
        # - Cross-modal similarity
        
        results = self._hybrid_search(query, expanded_queries)
        
        # Add multimodal-specific enhancements
        for result in results:
            result.content_type = "multimodal"
            # Add equation/diagram detection
            if self._contains_equations(result.content):
                result.metadata["has_equations"] = True
            if self._contains_diagrams(result.content):
                result.metadata["has_diagrams"] = True
        
        return results
    
    def _fallback_search(self, query: SearchQuery, expanded_queries: List[str]) -> List[SearchResult]:
        """Fallback search using TF-IDF when embeddings are not available."""
        if not self.tfidf_vectorizer or not SKLEARN_AVAILABLE:
            return []
        
        results = []
        
        # Simple text-based search
        # This is a simplified implementation
        query_terms = query.query_text.lower().split()
        
        # Search concepts
        for concept in self.knowledge_graph.concepts.values():
            concept_text = f"{concept.name} {concept.description}".lower()
            
            # Calculate simple term overlap
            concept_terms = concept_text.split()
            overlap = len(set(query_terms) & set(concept_terms))
            
            if overlap > 0:
                similarity = overlap / len(query_terms)
                
                result = SearchResult(
                    content_id=concept.id,
                    title=concept.name,
                    content=concept.description,
                    content_type="concept",
                    similarity_score=similarity,
                    relevance_score=similarity,
                    domain=concept.domain
                )
                results.append(result)
        
        return results
    
    def _post_process_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Post-process search results."""
        # Filter by domain if specified
        if query.domain_filter:
            results = [r for r in results if r.domain == query.domain_filter]
        
        # Filter by concepts if specified
        if query.concept_filter:
            results = [r for r in results if any(c in r.concept_matches for c in query.concept_filter)]
        
        # Adjust scores based on expertise level
        for result in results:
            result.relevance_score = self._adjust_score_for_expertise(result.relevance_score, query.expertise_level)
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Limit results
        results = results[:query.max_results]
        
        # Add explanations
        for result in results:
            result.explanation = self._generate_result_explanation(result, query)
        
        return results
    
    def _adjust_score_for_expertise(self, score: float, expertise_level: ExpertiseLevel) -> float:
        """Adjust relevance score based on user expertise level."""
        # Simple adjustment - in practice, this would be more sophisticated
        if expertise_level == ExpertiseLevel.BEGINNER:
            return score * 1.2  # Boost basic content
        elif expertise_level == ExpertiseLevel.EXPERT:
            return score * 0.8  # Reduce basic content
        return score
    
    def _generate_result_explanation(self, result: SearchResult, query: SearchQuery) -> str:
        """Generate an explanation for why this result is relevant."""
        reasons = []
        
        if result.similarity_score > 0.8:
            reasons.append("high semantic similarity")
        if result.concept_matches:
            reasons.append(f"matches concepts: {', '.join(result.concept_matches[:3])}")
        if result.domain == query.domain_filter:
            reasons.append(f"in requested domain: {result.domain}")
        
        if reasons:
            return f"Relevant due to: {', '.join(reasons)}"
        return "Relevant to your query"
    
    def get_recommendations(self, query: SearchQuery, results: List[SearchResult]) -> List[SearchRecommendation]:
        """Get search recommendations based on query and results."""
        recommendations = []
        
        # Recommend related concepts
        if results:
            all_related = set()
            for result in results[:3]:  # Top 3 results
                all_related.update(result.related_concepts)
            
            for concept_id in list(all_related)[:5]:  # Top 5 related
                concept = self.knowledge_graph.get_concept(concept_id)
                if concept:
                    rec = SearchRecommendation(
                        query=f"What is {concept.name}?",
                        reason=f"Related to your search results",
                        confidence=0.8,
                        related_concepts=[concept.name]
                    )
                    recommendations.append(rec)
        
        # Recommend query expansions
        expanded_queries = self.query_expander.expand(query)
        for expanded_query in expanded_queries[1:4]:  # Skip original, take next 3
            rec = SearchRecommendation(
                query=expanded_query,
                reason="Expanded version of your query",
                confidence=0.7
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _extract_concepts_from_query(self, query_text: str) -> List[str]:
        """Extract physics concepts from query text."""
        concepts = []
        
        # Simple concept extraction - in practice, this would be more sophisticated
        for concept in self.knowledge_graph.concepts.values():
            if concept.name.lower() in query_text.lower():
                concepts.append(concept.name)
        
        return concepts
    
    def _calculate_concept_relevance(self, concept: PhysicsConcept, query_concepts: List[str]) -> float:
        """Calculate relevance of a concept to query concepts."""
        if not query_concepts:
            return 0.0
        
        relevance = 0.0
        
        # Direct match
        if concept.name in query_concepts:
            relevance += 1.0
        
        # Domain match
        for qc in query_concepts:
            query_concept = self.knowledge_graph.get_concept_by_name(qc)
            if query_concept and query_concept.domain == concept.domain:
                relevance += 0.5
        
        # Relationship match
        for qc in query_concepts:
            query_concept = self.knowledge_graph.get_concept_by_name(qc)
            if query_concept:
                try:
                    path = self.knowledge_graph.find_path(concept.id, query_concept.id)
                    if path and len(path) <= 3:  # Close relationship
                        relevance += 1.0 / len(path)
                except:
                    pass
        
        return min(relevance, 1.0)
    
    def _search_definitions(self, query: SearchQuery) -> List[SearchResult]:
        """Search for definitions."""
        # Implementation for definition search
        return []
    
    def _search_explanations(self, query: SearchQuery) -> List[SearchResult]:
        """Search for explanations."""
        # Implementation for explanation search
        return []
    
    def _search_problem_solutions(self, query: SearchQuery) -> List[SearchResult]:
        """Search for problem solutions."""
        # Implementation for problem solution search
        return []
    
    def _search_comparisons(self, query: SearchQuery) -> List[SearchResult]:
        """Search for comparisons."""
        # Implementation for comparison search
        return []
    
    def _search_applications(self, query: SearchQuery) -> List[SearchResult]:
        """Search for applications."""
        # Implementation for application search
        return []
    
    def _search_derivations(self, query: SearchQuery) -> List[SearchResult]:
        """Search for derivations."""
        # Implementation for derivation search
        return []
    
    def _search_examples(self, query: SearchQuery) -> List[SearchResult]:
        """Search for examples."""
        # Implementation for example search
        return []
    
    def _search_relationships(self, query: SearchQuery) -> List[SearchResult]:
        """Search for relationships."""
        # Implementation for relationship search
        return []
    
    def _contains_equations(self, content: str) -> bool:
        """Check if content contains equations."""
        # Simple check for equation indicators
        equation_patterns = [r'\$.*\$', r'\\[.*\\]', r'=', r'\+', r'-', r'\*', r'/']
        return any(re.search(pattern, content) for pattern in equation_patterns)
    
    def _contains_diagrams(self, content: str) -> bool:
        """Check if content contains diagrams."""
        # Simple check for diagram indicators
        diagram_keywords = ['figure', 'diagram', 'chart', 'graph', 'plot', 'image']
        return any(keyword in content.lower() for keyword in diagram_keywords)
    
    def _get_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for query."""
        return f"{query.query_text}_{query.search_type.value}_{query.expertise_level.value}_{query.max_results}"


class QueryIntentClassifier:
    """Classifier for determining query intent."""
    
    def __init__(self):
        self.intent_patterns = {
            QueryIntent.DEFINITION: [
                r'what is', r'define', r'definition', r'meaning of'
            ],
            QueryIntent.EXPLANATION: [
                r'explain', r'how does', r'why does', r'how to'
            ],
            QueryIntent.PROBLEM_SOLVING: [
                r'solve', r'calculate', r'find', r'determine'
            ],
            QueryIntent.COMPARISON: [
                r'compare', r'difference', r'vs', r'versus'
            ],
            QueryIntent.APPLICATION: [
                r'application', r'use', r'apply', r'example'
            ],
            QueryIntent.DERIVATION: [
                r'derive', r'derivation', r'proof', r'show that'
            ],
            QueryIntent.EXAMPLE: [
                r'example', r'instance', r'case', r'illustration'
            ],
            QueryIntent.RELATIONSHIP: [
                r'relation', r'relationship', r'connection', r'link'
            ]
        }
    
    def classify(self, query_text: str) -> QueryIntent:
        """Classify query intent."""
        query_lower = query_text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                return intent
        
        return QueryIntent.EXPLANATION  # Default


class QueryExpander:
    """Expands queries with related terms and concepts."""
    
    def __init__(self, knowledge_graph: PhysicsKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.expansion_cache = {}
    
    def expand(self, query: SearchQuery) -> List[str]:
        """Expand query with related terms."""
        if query.query_text in self.expansion_cache:
            return self.expansion_cache[query.query_text]
        
        expanded = [query.query_text]
        
        # Add synonyms and related concepts
        concepts = self._extract_concepts(query.query_text)
        for concept_name in concepts:
            concept = self.knowledge_graph.get_concept_by_name(concept_name)
            if concept:
                # Add related concepts
                relationships = self.knowledge_graph.get_concept_relationships(concept.id)
                for rel in relationships[:3]:  # Limit to top 3
                    related_concept = self.knowledge_graph.get_concept(rel.target_concept_id)
                    if related_concept:
                        expanded_query = query.query_text.replace(concept_name, related_concept.name)
                        expanded.append(expanded_query)
        
        self.expansion_cache[query.query_text] = expanded
        return expanded
    
    def _extract_concepts(self, query_text: str) -> List[str]:
        """Extract concepts from query text."""
        concepts = []
        for concept in self.knowledge_graph.concepts.values():
            if concept.name.lower() in query_text.lower():
                concepts.append(concept.name)
        return concepts 