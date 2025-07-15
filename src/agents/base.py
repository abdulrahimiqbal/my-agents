"""
Base agent implementation following LangChain Academy patterns.

This module provides the foundational BaseAgent class that incorporates:
- Module 1: Basic agent and tool patterns
- Module 2: State management and memory
- Module 3: Human-in-the-loop capabilities
- Module 5: Advanced memory patterns
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from ..config import get_settings


class BaseAgent(ABC):
    """
    Base agent class with memory, tools, and LangGraph integration.
    
    Incorporates patterns from LangChain Academy:
    - Stateful conversation management (Module 2)
    - Tool integration and routing (Module 1)
    - Memory persistence (Module 5)
    - Async support for production use
    """
    
    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[Any] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the base agent.
        
        Args:
            tools: List of tools available to the agent
            memory: Memory store for conversation persistence
            model_name: LLM model to use
            temperature: Temperature for LLM responses
            max_tokens: Maximum tokens for responses
            system_message: System message for the agent
            **kwargs: Additional arguments
        """
        self.settings = get_settings()
        
        # Initialize LLM
        self.llm = self._init_llm(model_name, temperature, max_tokens)
        
        # Initialize tools
        self.tools = tools or []
        if self.tools:
            if not all(hasattr(t, 'name') for t in self.tools):
                print('Warning: Tools do not have name attribute - initializing as BaseTool')
                self.tools = [BaseTool(name=str(t), func=lambda x: x) for t in self.tools]  # Fallback
            self.tool_node = ToolNode(self.tools)
        else:
            self.tool_node = None
        
        # Initialize memory
        if memory is None:
            from ..memory import get_memory_store
            self.memory = get_memory_store()
        else:
            self.memory = memory
        
        # System message
        self.system_message = system_message or self._get_default_system_message()
        
        # Initialize graph
        self.graph = self._build_graph()
        
    def _init_llm(
        self, 
        model_name: Optional[str] = None, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> ChatOpenAI:
        """Initialize the LLM with configuration."""
        config = self.settings.get_llm_config()
        
        return ChatOpenAI(
            model=model_name or config["model"],
            temperature=temperature or config["temperature"],
            max_tokens=max_tokens or config["max_tokens"],
            api_key=self.settings.openai_api_key,
        )
    
    def _get_default_system_message(self) -> str:
        """Get the default system message for the agent."""
        return """You are a helpful AI assistant. You have access to tools and can maintain 
        conversation context. Always be helpful, accurate, and professional."""
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph for the agent.
        
        Based on Module 1 patterns for agent routing and tool usage.
        """
        graph = StateGraph(MessagesState)
        
        # Add nodes
        graph.add_node("agent", self._agent_node)
        if self.tools:
            graph.add_node("tools", self.tool_node)
        
        # Add edges
        graph.set_entry_point("agent")
        
        if self.tools:
            # Conditional edge based on whether tools are called
            graph.add_conditional_edges(
                "agent",
                tools_condition,
                {
                    "tools": "tools",
                    "__end__": "__end__",
                }
            )
            graph.add_edge("tools", "agent")
        else:
            graph.add_edge("agent", "__end__")
        
        # Compile with checkpointer for memory
        try:
            checkpointer = SqliteSaver.from_conn_string(self.settings.memory_db_path)
            return graph.compile(checkpointer=checkpointer)
        except Exception as e:
            # If checkpointer fails, compile without it
            print(f"Warning: Could not initialize checkpointer: {e}")
            return graph.compile()
    
    async def _agent_node(self, state: MessagesState) -> Dict[str, Any]:
        """
        Agent node that processes messages and generates responses.
        
        Incorporates system message and tool binding.
        """
        messages = state["messages"]
        
        # Add system message if not present
        if not messages or not isinstance(messages[0], type(messages[0])) or \
           "system" not in str(messages[0]).lower():
            system_msg = HumanMessage(content=self.system_message)
            messages = [system_msg] + messages
        
        # Bind tools to LLM if available
        llm = self.llm
        if self.tools:
            llm = self.llm.bind_tools(self.tools)
        
        # Generate response
        response = await llm.ainvoke(messages)
        
        # Store in memory
        if len(messages) > 1:  # Don't store system message
            await self.memory.store_conversation(
                messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1]),
                response.content if hasattr(response, 'content') else str(response)
            )
        
        return {"messages": [response]}
    
    def chat(self, message: str, thread_id: str = "default") -> str:
        """
        Synchronous chat interface.
        
        Args:
            message: User message
            thread_id: Thread ID for conversation continuity
            
        Returns:
            Agent response
        """
        return asyncio.run(self.achat(message, thread_id))
    
    async def achat(self, message: str, thread_id: str = "default") -> str:
        """
        Asynchronous chat interface.
        
        Args:
            message: User message
            thread_id: Thread ID for conversation continuity
            
        Returns:
            Agent response
        """
        # Create input state
        input_state = {
            "messages": [HumanMessage(content=message)]
        }
        
        # Configure with thread ID for memory
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke graph
        result = await self.graph.ainvoke(input_state, config=config)
        
        # Extract response
        if result and "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            return str(last_message)
        
        return "I'm sorry, I couldn't generate a response."
    
    async def stream_chat(self, message: str, thread_id: str = "default") -> AsyncGenerator[str, None]:
        """
        Streaming chat interface for real-time responses.
        
        Args:
            message: User message
            thread_id: Thread ID for conversation continuity
            
        Yields:
            Chunks of the agent response
        """
        input_state = {
            "messages": [HumanMessage(content=message)]
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        async for chunk in self.graph.astream(input_state, config=config):
            if "messages" in chunk and chunk["messages"]:
                last_message = chunk["messages"][-1]
                if hasattr(last_message, 'content'):
                    yield last_message.content
                else:
                    yield str(last_message)
    
    def get_conversation_history(self, thread_id: str = "default", limit: int = 10) -> List[BaseMessage]:
        """
        Get conversation history for a thread.
        
        Args:
            thread_id: Thread ID
            limit: Maximum number of messages to return
            
        Returns:
            List of messages in the conversation
        """
        # This would integrate with the checkpointer to get history
        # For now, return empty list - implement based on specific needs
        return []
    
    def clear_conversation(self, thread_id: str = "default") -> None:
        """
        Clear conversation history for a thread.
        
        Args:
            thread_id: Thread ID to clear
        """
        # Implementation would clear the specific thread from checkpointer
        pass
    
    @abstractmethod
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the agent.
        
        Returns:
            Dictionary with agent information
        """