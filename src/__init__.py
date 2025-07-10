"""
My Agents Project - A production-ready framework for building AI agents.

Built upon LangChain Academy principles with modular, extensible architecture.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .agents import BaseAgent, ChatAgent
from .memory import MemoryStore
from .config import Settings

__all__ = [
    "BaseAgent",
    "ChatAgent", 
    "MemoryStore",
    "Settings"
] 