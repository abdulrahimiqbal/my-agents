"""
Agent implementations for the agents project.

This package contains base agent classes and specialized implementations
following LangChain Academy patterns.
"""

from .base import BaseAgent
from .chat import ChatAgent

__all__ = ["BaseAgent", "ChatAgent"] 