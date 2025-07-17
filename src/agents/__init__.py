"""
DataAgent package initialization
"""

# Try to import existing agents if available
try:
    from .base import BaseAgent
    from .chat import ChatAgent
    from .data_agent import DataAgent
    from .data_tools import DATA_AGENT_TOOLS
    from .physics_expert import PhysicsExpertAgent
    from .hypothesis_generator import HypothesisGeneratorAgent
    from .collaborative_system import CollaborativePhysicsSystem
    
    __all__ = [
        "BaseAgent", 
        "ChatAgent", 
        "DataAgent", 
        "DATA_AGENT_TOOLS",
        "PhysicsExpertAgent",
        "HypothesisGeneratorAgent", 
        "CollaborativePhysicsSystem"
    ]
    
except ImportError as e:
    # If imports fail, define minimal structure
    print(f"Warning: Some agent imports failed: {e}")
    __all__ = [] 