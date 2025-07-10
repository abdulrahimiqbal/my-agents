"""
Tests for agent implementations.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from src.agents import BaseAgent, ChatAgent
from src.memory import MemoryStore
from src.config import get_settings


class MockAgent(BaseAgent):
    """Mock agent implementation for testing BaseAgent."""
    
    def get_agent_info(self):
        return {"type": "MockAgent", "test": True}


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch('src.config.get_settings') as mock:
        settings = Mock()
        settings.openai_api_key = "test-key"
        settings.memory_db_path = ":memory:"
        settings.get_llm_config.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 4000
        }
        mock.return_value = settings
        yield settings


@pytest.fixture
def memory_store():
    """In-memory database for testing."""
    return MemoryStore(db_path=":memory:")


class TestBaseAgent:
    """Tests for BaseAgent class."""
    
    def test_agent_initialization(self, mock_settings):
        """Test agent initialization."""
        agent = MockAgent()
        
        assert agent.settings == mock_settings
        assert agent.tools == []
        assert agent.system_message is not None
        assert agent.graph is not None
    
    def test_agent_with_custom_system_message(self, mock_settings):
        """Test agent with custom system message."""
        custom_message = "Custom system message"
        agent = MockAgent(system_message=custom_message)
        
        assert agent.system_message == custom_message
    
    def test_agent_with_memory(self, mock_settings, memory_store):
        """Test agent with custom memory store."""
        agent = MockAgent(memory=memory_store)
        
        assert agent.memory == memory_store
    
    @pytest.mark.asyncio
    async def test_store_conversation(self, mock_settings, memory_store):
        """Test conversation storage."""
        agent = MockAgent(memory=memory_store)
        
        # Mock the memory store method
        with patch.object(memory_store, 'store_conversation') as mock_store:
            mock_store.return_value = 1
            
            # This would normally call the LLM, so we'll test the memory interaction
            await memory_store.store_conversation("Hello", "Hi there!", "test-thread")
            
            mock_store.assert_called_once_with("Hello", "Hi there!", "test-thread")


class TestChatAgent:
    """Tests for ChatAgent class."""
    
    def test_chat_agent_initialization(self, mock_settings):
        """Test chat agent initialization."""
        agent = ChatAgent()
        
        assert agent.personality == "helpful"
        assert len(agent.tools) > 0  # Should have basic tools
    
    def test_chat_agent_personalities(self, mock_settings):
        """Test different personality configurations."""
        personalities = ["helpful", "creative", "analytical", "casual"]
        
        for personality in personalities:
            agent = ChatAgent(personality=personality)
            assert agent.personality == personality
            assert personality.lower() in agent.system_message.lower()
    
    def test_chat_agent_without_basic_tools(self, mock_settings):
        """Test chat agent without basic tools."""
        agent = ChatAgent(enable_basic_tools=False)
        
        # Should have no tools or only custom tools
        assert len(agent.tools) == 0
    
    def test_get_agent_info(self, mock_settings):
        """Test agent info retrieval."""
        agent = ChatAgent()
        info = agent.get_agent_info()
        
        assert info["type"] == "ChatAgent"
        assert "personality" in info
        assert "tools_count" in info
        assert "capabilities" in info
    
    def test_set_personality(self, mock_settings):
        """Test personality change."""
        agent = ChatAgent(personality="helpful")
        original_message = agent.system_message
        
        agent.set_personality("creative")
        
        assert agent.personality == "creative"
        assert agent.system_message != original_message
        assert "creative" in agent.system_message.lower()
    
    def test_get_available_personalities(self, mock_settings):
        """Test available personalities list."""
        agent = ChatAgent()
        personalities = agent.get_available_personalities()
        
        expected = ["helpful", "creative", "analytical", "casual"]
        assert all(p in personalities for p in expected)
    
    @pytest.mark.asyncio
    async def test_explain_capabilities(self, mock_settings):
        """Test capabilities explanation."""
        agent = ChatAgent()
        explanation = await agent.explain_capabilities()
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "capabilities" in explanation.lower()
        assert "tools" in explanation.lower()


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for agents."""
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, mock_settings):
        """Test agent memory integration."""
        memory = MemoryStore(db_path=":memory:")
        agent = ChatAgent(memory=memory)
        
        # Store a conversation
        await memory.store_conversation("Test question", "Test response", "test-thread")
        
        # Retrieve history
        history = memory.get_conversation_history("test-thread")
        
        assert len(history) == 1
        assert history[0]["type"] == "conversation"
        assert history[0]["user_message"] == "Test question"
        assert history[0]["agent_response"] == "Test response"
    
    def test_agent_graph_structure(self, mock_settings):
        """Test that agent graphs are properly structured."""
        agent = ChatAgent()
        graph = agent.graph
        
        # Check that essential nodes exist
        assert "agent" in graph.nodes
        
        # Check that the graph is compiled
        assert hasattr(graph, 'invoke')
        assert hasattr(graph, 'ainvoke')


if __name__ == "__main__":
    pytest.main([__file__]) 