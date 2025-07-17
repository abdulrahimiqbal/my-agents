"""
DataAgent package initialization
"""

# Try to import existing agents if available
try:
    from .base import BaseAgent
    from .data_agent import DataAgent
    from .data_tools import DATA_AGENT_TOOLS
    
    __all__ = ["BaseAgent", "DataAgent", "DATA_AGENT_TOOLS"]
    
except ImportError:
    # If imports fail, define minimal structure
    __all__ = [] 