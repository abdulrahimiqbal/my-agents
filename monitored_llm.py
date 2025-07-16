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
        super().__init__(**kwargs)
        self.agent_name = agent_name
        self.conversation_started = False
        
    def _generate(self, messages: List[BaseMessage], **kwargs) -> Any:
        """Override to capture agent thinking process."""
        
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
            
            agent_monitor.start_agent_conversation(self.agent_name, task_description)
            self.conversation_started = True
            
        # Simulate thinking process
        if agent_monitor.monitoring:
            agent_monitor.update_agent_progress(self.agent_name, 10, "Processing input...")
            
        # Call the original method
        start_time = time.time()
        result = super()._generate(messages, **kwargs)
        duration = time.time() - start_time
        
        # Extract and parse the response
        if result and hasattr(result, 'generations') and result.generations:
            response_text = result.generations[0].text if hasattr(result.generations[0], 'text') else str(result.generations[0])
            
            # Parse the response for thoughts, decisions, etc.
            if agent_monitor.monitoring:
                self._parse_and_monitor_response(response_text)
                
                # Update progress
                agent_monitor.update_agent_progress(self.agent_name, 90, "Finalizing analysis...")
                time.sleep(0.5)  # Small delay for UI
                
                # Complete the conversation
                agent_monitor.complete_agent_conversation(self.agent_name, response_text)
        
        return result
    
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
    return MonitoredChatOpenAI(
        agent_name=agent_name,
        model=model,
        temperature=temperature
    )