"""
LangGraph Studio compatible chat agent.

This module provides a graph definition that can be loaded by LangGraph Studio
for visual debugging and development.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.agents import ChatAgent
from src.config import get_settings


def create_chat_graph():
    """
    Create a chat agent graph for LangGraph Studio.
    
    Returns:
        Compiled LangGraph for the chat agent
    """
    # Ensure settings are loaded
    settings = get_settings()
    
    # Create chat agent with default configuration
    agent = ChatAgent(
        personality="helpful",
        enable_basic_tools=True
    )
    
    return agent.graph


# For direct execution
if __name__ == "__main__":
    graph = create_chat_graph()
    print("Chat agent graph created successfully!")
    print(f"Graph nodes: {list(graph.nodes.keys())}")
    print(f"Graph edges: {list(graph.edges)}") 