"""
Tools package for agent capabilities.

This package contains reusable tools following LangChain Academy patterns
for tool creation and integration.
"""

from typing import List
from langchain_core.tools import BaseTool

from .calculator import get_calculator_tools
from .web_search import get_web_search_tools
from .physics_calculator import get_physics_calculator_tools
from .physics_constants import get_physics_constants_tools
from .physics_research import get_physics_research_tools
from .unit_converter import get_unit_converter_tools
from .hypothesis_tools import get_hypothesis_tools


def get_basic_tools() -> List[BaseTool]:
    """
    Get a collection of basic tools for general-purpose agents.
    
    Returns:
        List of basic tools including calculator and web search
    """
    tools = []
    tools.extend(get_calculator_tools())
    tools.extend(get_web_search_tools())
    return tools


__all__ = [
    "get_basic_tools",
    "get_calculator_tools", 
    "get_web_search_tools",
    "get_physics_calculator_tools",
    "get_physics_constants_tools",
    "get_physics_research_tools",
    "get_unit_converter_tools",
    "get_hypothesis_tools"
] 