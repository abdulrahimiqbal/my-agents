"""
Agent implementations for the agents project.

This package contains base agent classes and specialized implementations
following LangChain Academy patterns.
"""

from .base import BaseAgent
from .chat import ChatAgent
from .physics_expert import PhysicsExpertAgent
from .hypothesis_generator import HypothesisGeneratorAgent
from .supervisor import SupervisorAgent
from .collaborative_system import CollaborativePhysicsSystem

__all__ = [
    "BaseAgent", 
    "ChatAgent",
    "PhysicsExpertAgent",
    "HypothesisGeneratorAgent",
    "SupervisorAgent",
    "CollaborativePhysicsSystem"
] 