"""Main entry point for LangGraph Studio."""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.chat import ChatAgent
from src.agents.base import BaseAgent
from src.tools.calculator import get_calculator_tools
from src.tools.web_search import get_web_search_tools
from src.utils.helpers import load_env

# Load environment variables
load_env()

# Create chat agent
chat_agent = ChatAgent(memory_enabled=True)
chat_graph = chat_agent.graph

# Create research agent with web search capabilities
class ResearchAgent(BaseAgent):
    """Research agent with web search and calculation capabilities."""
    
    def __init__(self, **kwargs):
        """Initialize the research agent."""
        self.system_message = (
            "You are a research assistant AI. You can search the web for information, "
            "perform calculations, and provide comprehensive answers to questions. "
            "Always cite your sources when providing information from web searches. "
            "Be thorough, accurate, and helpful in your responses."
        )
        super().__init__(**kwargs)
    
    def _build_graph(self):
        """Build the research agent graph."""
        from langchain_core.messages import SystemMessage
        from langgraph.graph import StateGraph, MessagesState, START
        from langgraph.prebuilt import ToolNode, tools_condition
        
        # Get all tools
        tools = get_calculator_tools() + get_web_search_tools()
        
        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(tools)
        
        # Create the graph
        builder = StateGraph(MessagesState)
        
        # Define the assistant node
        def assistant(state: MessagesState):
            """Assistant node that processes messages."""
            system_msg = SystemMessage(content=self.system_message)
            messages = [system_msg] + state["messages"]
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        
        # Define tool node
        tool_node = ToolNode(tools)
        
        # Add nodes
        builder.add_node("assistant", assistant)
        builder.add_node("tools", tool_node)
        
        # Add edges
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")
        
        # Compile with memory if enabled
        if self.memory_enabled and self.memory:
            return builder.compile(checkpointer=self.memory)
        else:
            return builder.compile()

# Create research agent
research_agent = ResearchAgent(memory_enabled=True)
research_graph = research_agent.graph 