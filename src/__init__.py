"""
My Agents Project - A production-ready framework for building AI agents.

Built upon LangChain Academy principles with modular, extensible architecture.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .agents import BaseAgent, ChatAgent, PhysicsExpertAgent, HypothesisGeneratorAgent, CollaborativePhysicsSystem
from .memory import MemoryStore
from .config import Settings
from .database import KnowledgeAPI, DatabaseMigrator

__all__ = [
    "BaseAgent",
    "ChatAgent", 
    "PhysicsExpertAgent",
    "HypothesisGeneratorAgent",
    "CollaborativePhysicsSystem",
    "MemoryStore",
    "Settings",
    "KnowledgeAPI",
    "DatabaseMigrator"
] 