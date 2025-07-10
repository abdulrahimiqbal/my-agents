"""
LangGraph Studio compatible research agent.

This module demonstrates multi-agent patterns from LangChain Academy Module 4,
showing how to create specialized agents that work together.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.agents import BaseAgent
from src.tools import get_web_search_tools, get_calculator_tools
from src.config import get_settings


class ResearchAgent(BaseAgent):
    """
    Specialized research agent for information gathering and analysis.
    
    Demonstrates Module 4 patterns for specialized agent roles.
    """
    
    def __init__(self, **kwargs):
        # Combine research-specific tools
        research_tools = []
        research_tools.extend(get_web_search_tools())
        research_tools.extend(get_calculator_tools())
        
        # Research-focused system message
        system_message = """You are a research specialist AI agent. Your role is to:

1. **Information Gathering**: Search for accurate, up-to-date information on any topic
2. **Source Analysis**: Evaluate the credibility and relevance of information sources
3. **Data Synthesis**: Combine information from multiple sources into coherent insights
4. **Fact Verification**: Cross-reference claims and verify accuracy when possible

You have access to web search tools and calculation tools. Always:
- Cite your sources when providing information
- Indicate when information might be outdated or uncertain
- Provide multiple perspectives on controversial topics
- Use calculations when dealing with numerical data

Be thorough, accurate, and objective in your research approach."""

        super().__init__(
            tools=research_tools,
            system_message=system_message,
            **kwargs
        )
    
    def get_agent_info(self):
        """Get information about the research agent."""
        return {
            "type": "ResearchAgent",
            "specialization": "Information gathering and analysis",
            "tools_count": len(self.tools),
            "tool_names": [tool.name for tool in self.tools],
            "model": self.llm.model_name,
            "capabilities": [
                "Web search and information retrieval",
                "Source verification and analysis", 
                "Data synthesis and summarization",
                "Numerical analysis and calculations",
                "Multi-source research coordination"
            ]
        }


def create_research_graph():
    """
    Create a research agent graph for LangGraph Studio.
    
    Returns:
        Compiled LangGraph for the research agent
    """
    # Ensure settings are loaded
    settings = get_settings()
    
    # Create research agent with specialized configuration
    agent = ResearchAgent()
    
    return agent.graph


# For direct execution
if __name__ == "__main__":
    graph = create_research_graph()
    print("Research agent graph created successfully!")
    print(f"Graph nodes: {list(graph.nodes.keys())}")
    print(f"Graph edges: {list(graph.edges)}")
    
    # Test the research agent
    agent = ResearchAgent()
    info = agent.get_agent_info()
    print(f"\nAgent Info: {info}") 