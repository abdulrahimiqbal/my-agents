"""
Shared types and classes for the agent system to avoid circular imports.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class TaskType(Enum):
    """Types of tasks that can be routed to different agents"""
    PHYSICS_ANALYSIS = "physics_analysis"
    MATHEMATICAL_COMPUTATION = "mathematical_computation"
    PATTERN_RECOGNITION = "pattern_recognition"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    EXPERIMENTAL_DESIGN = "experimental_design"
    LITERATURE_REVIEW = "literature_review"
    CONCEPT_EXPLANATION = "concept_explanation"


class AgentCapability(Enum):
    """Agent capabilities for routing decisions"""
    QUANTUM_MECHANICS = "quantum_mechanics"
    THERMODYNAMICS = "thermodynamics"
    ELECTROMAGNETISM = "electromagnetism"
    MECHANICS = "mechanics"
    RELATIVITY = "relativity"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    CURVE_FITTING = "curve_fitting"
    PATTERN_MATCHING = "pattern_matching"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    PATTERN_RECOGNITION = "pattern_recognition"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    CREATIVE_THINKING = "creative_thinking"
    INTERDISCIPLINARY_ANALYSIS = "interdisciplinary_analysis"
    DATA_ANALYSIS = "data_analysis"
    RELATIONSHIP_DISCOVERY = "relationship_discovery"


class ConsensusMethod(Enum):
    """Methods for achieving consensus among agents"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    EXPERT_OVERRIDE = "expert_override"
    ITERATIVE_REFINEMENT = "iterative_refinement"


@dataclass
class TaskRequest:
    """Request for task execution"""
    task_id: str
    task_type: TaskType
    content: str
    priority: int = 1  # 1=low, 5=high
    required_capabilities: Set[AgentCapability] = field(default_factory=set)
    physics_domain: Optional[str] = None
    deadline: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result from task execution"""
    task_id: str
    agent_name: str
    result: Any
    confidence: float
    execution_time: float
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = field(default_factory=datetime.now)


@dataclass
class ConsensusResult:
    """Result from consensus detection"""
    final_result: Any
    confidence: float
    agreement_level: float
    participating_agents: List[str]
    method_used: ConsensusMethod
    individual_results: List[TaskResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = field(default_factory=datetime.now)