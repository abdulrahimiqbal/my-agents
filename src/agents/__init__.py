"""
Agent implementations for the agents project.

This package contains base agent classes and specialized implementations
following LangChain Academy patterns.
"""

from .base import BaseAgent
from .chat import ChatAgent
from .physics_expert import PhysicsExpertAgent
from .hypothesis_generator import (
    HypothesisGeneratorAgent,
    EnhancedHypothesisGeneratorAgent,
    CreativeReasoningPattern,
    ResearchGapType,
    ResearchGap,
    CreativeHypothesis,
    InterdisciplinaryConnection
)
from .supervisor import SupervisorAgent
from .collaborative_system import CollaborativePhysicsSystem

# Phase 3 imports
try:
    from .enhanced_physics_expert import EnhancedPhysicsExpertAgent, PhysicsDomain, DifficultyLevel
    from .mathematical_analysis import MathematicalAnalysisAgent
    from .pattern_recognition import PatternRecognitionAgent
    from .advanced_supervisor import (
        AdvancedSupervisorAgent,
        TaskType,
        AgentCapability,
        TaskRequest,
        TaskResult,
        ConsensusResult,
        ConsensusMethod,
        AgentProfile
    )
    from .parallel_orchestrator import (
        ParallelAgentOrchestrator,
        OrchestrationStrategy,
        SynchronizationMode,
        TaskPriority,
        OrchestrationSession
    )
    
    # Mark Phase 3 as available
    PHASE_3_AVAILABLE = True
    
except ImportError as e:
    print(f"Warning: Some Phase 3 agents not available: {e}")
    PHASE_3_AVAILABLE = False

__all__ = [
    # Base classes
    "BaseAgent", 
    "ChatAgent",
    "PhysicsExpertAgent",
    "HypothesisGeneratorAgent",
    "SupervisorAgent",
    "CollaborativePhysicsSystem",
    
    # Enhanced Hypothesis Generator (Phase 3)
    "EnhancedHypothesisGeneratorAgent",
    "CreativeReasoningPattern",
    "ResearchGapType", 
    "ResearchGap",
    "CreativeHypothesis",
    "InterdisciplinaryConnection",
]

# Add Phase 3 exports if available
if PHASE_3_AVAILABLE:
    __all__.extend([
        # Enhanced Physics Expert
        "EnhancedPhysicsExpertAgent",
        "PhysicsDomain",
        "DifficultyLevel",
        
        # Phase 2 agents
        "MathematicalAnalysisAgent",
        "PatternRecognitionAgent", 
        
        # Advanced Supervisor
        "AdvancedSupervisorAgent",
        "TaskType",
        "AgentCapability",
        "TaskRequest",
        "TaskResult",
        "ConsensusResult",
        "ConsensusMethod",
        "AgentProfile",
        
        # Parallel Orchestrator
        "ParallelAgentOrchestrator",
        "OrchestrationStrategy",
        "SynchronizationMode",
        "TaskPriority",
        "OrchestrationSession"
    ]) 