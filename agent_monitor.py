"""
Real-time agent conversation monitoring for PhysicsGPT.
Captures actual agent thoughts, decisions, and interactions.
"""

import time
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re

class AgentConversation:
    """Captures a single agent's conversation and thought process."""
    
    def __init__(self, agent_name: str, task_description: str):
        self.agent_name = agent_name
        self.task_description = task_description
        self.start_time = time.time()
        self.end_time = None
        self.status = "thinking"
        
        # Conversation elements
        self.thoughts = []
        self.decisions = []
        self.questions = []
        self.conclusions = []
        self.final_output = ""
        
        # Progress tracking
        self.progress_percentage = 0
        self.current_step = "Starting analysis..."
        
    def add_thought(self, thought: str, timestamp: float = None):
        """Add a thought or reasoning step."""
        self.thoughts.append({
            "content": thought,
            "timestamp": timestamp or time.time(),
            "type": "thought"
        })
        
    def add_decision(self, decision: str, reasoning: str = "", timestamp: float = None):
        """Add a decision point with reasoning."""
        self.decisions.append({
            "decision": decision,
            "reasoning": reasoning,
            "timestamp": timestamp or time.time(),
            "type": "decision"
        })
        
    def add_question(self, question: str, context: str = "", timestamp: float = None):
        """Add a question the agent is considering."""
        self.questions.append({
            "question": question,
            "context": context,
            "timestamp": timestamp or time.time(),
            "type": "question"
        })
        
    def update_progress(self, percentage: int, step: str):
        """Update progress information."""
        self.progress_percentage = min(100, max(0, percentage))
        self.current_step = step
        
    def complete(self, final_output: str):
        """Mark the conversation as complete."""
        self.end_time = time.time()
        self.status = "completed"
        self.final_output = final_output
        self.progress_percentage = 100
        self.current_step = "Analysis complete"
        
    def get_duration(self) -> float:
        """Get the duration of the conversation."""
        end = self.end_time or time.time()
        return end - self.start_time
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for UI display."""
        return {
            "agent_name": self.agent_name,
            "task_description": self.task_description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.get_duration(),
            "status": self.status,
            "progress_percentage": self.progress_percentage,
            "current_step": self.current_step,
            "thoughts": self.thoughts,
            "decisions": self.decisions,
            "questions": self.questions,
            "conclusions": self.conclusions,
            "final_output": self.final_output,
            "formatted_start": datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S"),
            "total_interactions": len(self.thoughts) + len(self.decisions) + len(self.questions)
        }

class AgentMonitor:
    """Monitors and captures agent conversations in real-time."""
    
    def __init__(self):
        self.conversations: Dict[str, AgentConversation] = {}
        self.active_agents: List[str] = []
        self.monitoring = False
        self.update_callbacks = []
        
    def start_monitoring(self):
        """Start monitoring agent conversations."""
        self.monitoring = True
        self.conversations.clear()
        self.active_agents.clear()
        
    def stop_monitoring(self):
        """Stop monitoring agent conversations."""
        self.monitoring = False
        
    def add_update_callback(self, callback):
        """Add a callback function to be called when conversations update."""
        self.update_callbacks.append(callback)
        
    def _notify_updates(self):
        """Notify all callbacks of updates."""
        for callback in self.update_callbacks:
            try:
                callback(self.conversations)
            except Exception as e:
                print(f"Error in update callback: {e}")
                
    def start_agent_conversation(self, agent_name: str, task_description: str):
        """Start tracking a new agent conversation."""
        if not self.monitoring:
            return
            
        conversation = AgentConversation(agent_name, task_description)
        self.conversations[agent_name] = conversation
        
        if agent_name not in self.active_agents:
            self.active_agents.append(agent_name)
            
        self._notify_updates()
        
    def add_agent_thought(self, agent_name: str, thought: str):
        """Add a thought to an agent's conversation."""
        if agent_name in self.conversations:
            self.conversations[agent_name].add_thought(thought)
            self._notify_updates()
            
    def add_agent_decision(self, agent_name: str, decision: str, reasoning: str = ""):
        """Add a decision to an agent's conversation."""
        if agent_name in self.conversations:
            self.conversations[agent_name].add_decision(decision, reasoning)
            self._notify_updates()
            
    def update_agent_progress(self, agent_name: str, percentage: int, step: str):
        """Update an agent's progress."""
        if agent_name in self.conversations:
            self.conversations[agent_name].update_progress(percentage, step)
            self._notify_updates()
            
    def complete_agent_conversation(self, agent_name: str, final_output: str):
        """Mark an agent conversation as complete."""
        if agent_name in self.conversations:
            self.conversations[agent_name].complete(final_output)
            if agent_name in self.active_agents:
                self.active_agents.remove(agent_name)
            self._notify_updates()
            
    def get_active_conversations(self) -> List[Dict[str, Any]]:
        """Get all active conversations as dictionaries."""
        return [conv.to_dict() for conv in self.conversations.values()]
        
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of all conversations."""
        total_agents = len(self.conversations)
        active_agents = len(self.active_agents)
        completed_agents = total_agents - active_agents
        
        total_interactions = sum(
            len(conv.thoughts) + len(conv.decisions) + len(conv.questions)
            for conv in self.conversations.values()
        )
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "completed_agents": completed_agents,
            "total_interactions": total_interactions,
            "monitoring": self.monitoring
        }

# Global monitor instance
agent_monitor = AgentMonitor()

def parse_agent_output(agent_name: str, output: str):
    """Parse agent output to extract thoughts, decisions, and questions."""
    if not agent_monitor.monitoring:
        return
        
    # Simple patterns to detect different types of content
    thought_patterns = [
        r"I think that?[:\s](.+)",
        r"My analysis suggests[:\s](.+)",
        r"Based on[:\s](.+)",
        r"It appears that[:\s](.+)",
        r"The key insight is[:\s](.+)"
    ]
    
    decision_patterns = [
        r"I conclude that?[:\s](.+)",
        r"Therefore[:\s](.+)",
        r"My recommendation is[:\s](.+)",
        r"The best approach is[:\s](.+)"
    ]
    
    question_patterns = [
        r"What if[:\s](.+\?)",
        r"How might[:\s](.+\?)",
        r"Could it be that[:\s](.+\?)",
        r"Is it possible[:\s](.+\?)"
    ]
    
    # Extract thoughts
    for pattern in thought_patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        for match in matches:
            agent_monitor.add_agent_thought(agent_name, match.strip())
            
    # Extract decisions
    for pattern in decision_patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        for match in matches:
            agent_monitor.add_agent_decision(agent_name, match.strip())
            
    # Extract questions
    for pattern in question_patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        for match in matches:
            agent_monitor.add_agent_thought(agent_name, f"Question: {match.strip()}")

def simulate_agent_progress(agent_name: str, task_description: str, duration: float = 30):
    """Simulate agent progress for demonstration purposes."""
    if not agent_monitor.monitoring:
        return
        
    # Start the conversation
    agent_monitor.start_agent_conversation(agent_name, task_description)
    
    # Simulate thinking process
    steps = [
        (10, "Analyzing the problem..."),
        (25, "Reviewing relevant physics principles..."),
        (40, "Formulating hypotheses..."),
        (60, "Checking mathematical consistency..."),
        (75, "Considering experimental evidence..."),
        (90, "Drawing conclusions..."),
        (100, "Finalizing analysis...")
    ]
    
    step_duration = duration / len(steps)
    
    for progress, step_desc in steps:
        if not agent_monitor.monitoring:
            break
            
        agent_monitor.update_agent_progress(agent_name, progress, step_desc)
        
        # Add some simulated thoughts
        if progress == 25:
            agent_monitor.add_agent_thought(agent_name, "The fundamental principles suggest multiple pathways for analysis")
        elif progress == 40:
            agent_monitor.add_agent_thought(agent_name, "I need to consider both classical and quantum mechanical aspects")
        elif progress == 60:
            agent_monitor.add_agent_decision(agent_name, "Focus on the most promising theoretical framework")
        elif progress == 75:
            agent_monitor.add_agent_thought(agent_name, "The experimental data supports this interpretation")
            
        time.sleep(step_duration)
    
    # Complete the conversation
    final_output = f"Completed analysis for {task_description}"
    agent_monitor.complete_agent_conversation(agent_name, final_output)