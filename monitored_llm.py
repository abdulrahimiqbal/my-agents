"""
Monitored LLM wrapper that captures agent conversations for visualization.
"""

from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from agent_monitor import agent_monitor
import time
import re

class MonitoredChatOpenAI(ChatOpenAI):
    """ChatOpenAI wrapper that monitors and captures agent conversations."""
    
    def __init__(self, agent_name: str = "unknown", **kwargs):
        # Remove agent_name from kwargs before passing to parent
        self._agent_name = agent_name
        self._conversation_started = False
        
        # Initialize parent class without the agent_name
        super().__init__(**kwargs)
    
    @property
    def agent_name(self):
        """Get the agent name."""
        return self._agent_name
    
    @property
    def conversation_started(self):
        """Get conversation started status."""
        return self._conversation_started
    
    @conversation_started.setter
    def conversation_started(self, value):
        """Set conversation started status."""
        self._conversation_started = value
        
    def _generate(self, messages: List[BaseMessage], **kwargs) -> Any:
        """Override to capture agent thinking process."""
        
        # Debug logging
        print(f"📞 MonitoredChatOpenAI._generate called for {self.agent_name}")
        print(f"📊 Monitoring active: {agent_monitor.monitoring}")
        
        # Start monitoring if not already started
        if not self.conversation_started and agent_monitor.monitoring:
            # Extract task from the last human message
            task_description = "Physics analysis task"
            if messages:
                last_human_msg = None
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        last_human_msg = msg.content
                        break
                if last_human_msg:
                    # Extract first 100 chars as task description
                    task_description = last_human_msg[:100] + "..." if len(last_human_msg) > 100 else last_human_msg
            
            print(f"🚀 Starting conversation for {self.agent_name}: {task_description}")
            agent_monitor.start_agent_conversation(self.agent_name, task_description)
            self.conversation_started = True
            
        # Simulate thinking process
        if agent_monitor.monitoring:
            agent_monitor.update_agent_progress(self.agent_name, 10, "Processing input...")
            
        # Call the original method
        start_time = time.time()
        try:
            result = super()._generate(messages, **kwargs)
            print(f"✅ LLM generation completed for {self.agent_name}")
        except Exception as e:
            print(f"❌ LLM generation failed for {self.agent_name}: {e}")
            raise
            
        duration = time.time() - start_time
        
        # Extract and parse the response
        if result and hasattr(result, 'generations') and result.generations:
            response_text = result.generations[0].text if hasattr(result.generations[0], 'text') else str(result.generations[0])
            
            # Parse the response for thoughts, decisions, etc.
            if agent_monitor.monitoring:
                print(f"🔍 Parsing response for {self.agent_name}")
                self._parse_and_monitor_response(response_text)
                
                # Update progress
                agent_monitor.update_agent_progress(self.agent_name, 90, "Finalizing analysis...")
                time.sleep(0.5)  # Small delay for UI
                
                # Complete the conversation
                print(f"✅ Completing conversation for {self.agent_name}")
                agent_monitor.complete_agent_conversation(self.agent_name, response_text)
        
        return result
    
    def invoke(self, input, config=None, **kwargs):
        """Override invoke method as well, in case CrewAI uses this."""
        print(f"📞 MonitoredChatOpenAI.invoke called for {self.agent_name}")
        return super().invoke(input, config, **kwargs)
    
    def stream(self, input, config=None, **kwargs):
        """Override stream method as well."""
        print(f"📞 MonitoredChatOpenAI.stream called for {self.agent_name}")
        return super().stream(input, config, **kwargs)
    
    def _parse_and_monitor_response(self, response_text: str):
        """Parse agent response and extract thoughts, decisions, questions."""
        
        # Split response into sentences for analysis
        sentences = re.split(r'[.!?]+', response_text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Update progress based on sentence processing
            progress = min(90, 20 + (i * 60 // len(sentences)))
            agent_monitor.update_agent_progress(self.agent_name, progress, f"Analyzing aspect {i+1}...")
            
            # Detect different types of content
            sentence_lower = sentence.lower()
            
            # Thoughts and analysis
            if any(phrase in sentence_lower for phrase in [
                'i think', 'i believe', 'it appears', 'this suggests', 'we can see',
                'the analysis shows', 'based on', 'considering', 'examining'
            ]):
                agent_monitor.add_agent_thought(self.agent_name, sentence)
                
            # Decisions and conclusions
            elif any(phrase in sentence_lower for phrase in [
                'therefore', 'thus', 'consequently', 'i conclude', 'the result is',
                'we can determine', 'this means', 'it follows that'
            ]):
                agent_monitor.add_agent_decision(self.agent_name, sentence)
                
            # Questions and considerations
            elif '?' in sentence or any(phrase in sentence_lower for phrase in [
                'what if', 'how might', 'could it be', 'is it possible', 'we should consider'
            ]):
                agent_monitor.add_agent_thought(self.agent_name, f"Question: {sentence}")
                
            # Mathematical expressions
            elif any(symbol in sentence for symbol in ['=', '∫', '∂', '∇', '±', '≈', '∝']):
                agent_monitor.add_agent_thought(self.agent_name, f"Mathematical insight: {sentence}")
                
            # Key physics concepts
            elif any(concept in sentence_lower for concept in [
                'quantum', 'relativity', 'entropy', 'energy', 'momentum', 'field',
                'particle', 'wave', 'force', 'mass', 'charge', 'spin'
            ]):
                agent_monitor.add_agent_thought(self.agent_name, f"Physics concept: {sentence}")
            
            # Small delay to make the conversation feel more natural
            time.sleep(0.1)

def create_monitored_llm(agent_name: str, temperature: float = 0.1, model: str = "gpt-4o-mini") -> MonitoredChatOpenAI:
    """Create a monitored LLM for an agent."""
    try:
        print(f"🔧 Creating monitored LLM for {agent_name}")
        llm = MonitoredChatOpenAI(
            agent_name=agent_name,
            model=model,
            temperature=temperature
        )
        print(f"✅ Successfully created monitored LLM for {agent_name}")
        return llm
    except Exception as e:
        print(f"❌ Failed to create monitored LLM for {agent_name}: {e}")
        print(f"🔄 Falling back to standard ChatOpenAI for {agent_name}")
        # Fallback to standard ChatOpenAI if monitoring fails
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature
        )