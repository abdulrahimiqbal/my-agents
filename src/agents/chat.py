"""
Chat agent implementation for general conversational AI.

This agent demonstrates the basic patterns from LangChain Academy Module 1
for creating conversational agents with tool support.
"""

from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool

from .base import BaseAgent
from ..tools import get_basic_tools


class ChatAgent(BaseAgent):
    """
    General-purpose chat agent for conversations.
    
    Features:
    - Natural conversation flow
    - Basic tool integration
    - Memory persistence
    - Configurable personality
    """
    
    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        personality: str = "helpful",
        enable_basic_tools: bool = True,
        **kwargs
    ):
        """
        Initialize the chat agent.
        
        Args:
            tools: Custom tools for the agent
            personality: Agent personality (helpful, creative, analytical, etc.)
            enable_basic_tools: Whether to include basic tools (calculator, search)
            **kwargs: Additional arguments passed to BaseAgent
        """
        self.personality = personality
        
        # Combine custom tools with basic tools if enabled
        agent_tools = tools or []
        if enable_basic_tools:
            agent_tools.extend(get_basic_tools())
        
        # Set system message based on personality
        system_message = kwargs.get('system_message') or self._get_personality_system_message()
        
        super().__init__(
            tools=agent_tools,
            system_message=system_message,
            **kwargs
        )
    
    def _get_personality_system_message(self) -> str:
        """Get system message based on personality."""
        personality_messages = {
            "helpful": """You are a helpful and friendly AI assistant. You aim to be useful, 
            accurate, and supportive in all your interactions. You have access to tools to 
            help answer questions and solve problems. Always be polite and professional.""",
            
            "creative": """You are a creative and imaginative AI assistant. You love to think 
            outside the box and come up with innovative solutions. You have access to tools 
            to help with research and calculations. Feel free to be expressive and suggest 
            creative approaches to problems.""",
            
            "analytical": """You are an analytical and logical AI assistant. You approach 
            problems systematically and provide detailed, well-reasoned responses. You have 
            access to tools for calculations and research. Always break down complex problems 
            into manageable parts.""",
            
            "casual": """You are a casual and friendly AI assistant. You communicate in a 
            relaxed, conversational style while still being helpful and accurate. You have 
            access to tools to help with various tasks. Feel free to use a more informal tone.""",
        }
        
        return personality_messages.get(self.personality, personality_messages["helpful"])
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the chat agent."""
        return {
            "type": "ChatAgent",
            "personality": self.personality,
            "tools_count": len(self.tools),
            "tool_names": [tool.name for tool in self.tools],
            "model": self.llm.model_name,
            "capabilities": [
                "Natural conversation",
                "Tool usage",
                "Memory persistence",
                "Multi-turn dialogue"
            ]
        }
    
    def set_personality(self, personality: str) -> None:
        """
        Change the agent's personality.
        
        Args:
            personality: New personality type
        """
        self.personality = personality
        self.system_message = self._get_personality_system_message()
        # Note: In a full implementation, you'd want to rebuild the graph
        # with the new system message, but for simplicity we'll just update it
    
    def get_available_personalities(self) -> List[str]:
        """Get list of available personality types."""
        return ["helpful", "creative", "analytical", "casual"]
    
    async def explain_capabilities(self) -> str:
        """Explain what the agent can do."""
        info = self.get_agent_info()
        
        explanation = f"""I'm a {info['personality']} chat agent with the following capabilities:

🤖 **Core Features:**
- Natural conversation with memory of our chat history
- Access to {info['tools_count']} tools for various tasks
- Powered by {info['model']} for high-quality responses

🛠️ **Available Tools:**
"""
        
        for tool_name in info['tool_names']:
            explanation += f"- {tool_name}\n"
        
        explanation += """
💬 **How to interact with me:**
- Ask questions or request help with tasks
- I can use tools when needed to provide accurate information
- I remember our conversation context
- Feel free to ask me to explain my reasoning or approach

What would you like to help you with today?"""
        
        return explanation 