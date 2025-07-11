"""Collaborative Physics Research System - Unified interface for multi-agent physics research."""

from typing import Optional, Dict, Any, List
from langchain_core.messages import HumanMessage

from .physics_expert import PhysicsExpertAgent
from .hypothesis_generator import HypothesisGeneratorAgent
from .supervisor import SupervisorAgent
from ..memory.stores import get_memory_store
from ..config.settings import get_settings


class CollaborativePhysicsSystem:
    """Unified system for collaborative physics research using multiple specialized agents."""
    
    def __init__(self, 
                 difficulty_level: str = "undergraduate",
                 creativity_level: str = "high",
                 collaboration_style: str = "balanced",
                 memory_enabled: bool = True):
        """Initialize the collaborative physics system.
        
        Args:
            difficulty_level: Physics difficulty level (high_school, undergraduate, graduate, research)
            creativity_level: Creativity level for hypothesis generation (conservative, moderate, high, bold)
            collaboration_style: Collaboration style (balanced, expert_led, creative_led)
            memory_enabled: Whether to enable memory across sessions
        """
        self.difficulty_level = difficulty_level
        self.creativity_level = creativity_level
        self.collaboration_style = collaboration_style
        self.memory_enabled = memory_enabled
        
        # Initialize memory store if enabled
        self.memory_store = get_memory_store() if memory_enabled else None
        
        # Initialize agents
        self._initialize_agents()
        
        # Track active sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def _initialize_agents(self):
        """Initialize all specialized agents."""
        settings = get_settings()
        
        # Initialize Physics Expert Agent
        self.physics_expert = PhysicsExpertAgent(
            difficulty_level=self.difficulty_level,
            memory_enabled=self.memory_enabled,
            memory=self.memory_store,
            model_name=settings.default_model,
            temperature=settings.default_temperature
        )
        
        # Initialize Hypothesis Generator Agent
        self.hypothesis_generator = HypothesisGeneratorAgent(
            creativity_level=self.creativity_level,
            exploration_scope="broad",
            risk_tolerance="medium",
            memory_enabled=self.memory_enabled,
            memory=self.memory_store,
            model_name=settings.default_model,
            temperature=settings.default_temperature + 0.1  # Slightly higher temperature for creativity
        )
        
        # Initialize Supervisor Agent
        self.supervisor = SupervisorAgent(
            physics_expert=self.physics_expert,
            hypothesis_generator=self.hypothesis_generator,
            collaboration_style=self.collaboration_style,
            memory_enabled=self.memory_enabled,
            memory=self.memory_store,
            model_name=settings.default_model,
            temperature=settings.default_temperature
        )
    
    def start_collaborative_session(self, 
                                  topic: str, 
                                  mode: str = "research",
                                  context: str = "",
                                  session_id: Optional[str] = None) -> Dict[str, Any]:
        """Start a new collaborative research session.
        
        Args:
            topic: Physics topic or problem to investigate
            mode: Collaboration mode (research, debate, brainstorm, teaching)
            context: Additional context or constraints
            session_id: Optional session ID for tracking
            
        Returns:
            Session information and initial response
        """
        # Generate session ID if not provided
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
        
        # Initialize session
        session_info = {
            "session_id": session_id,
            "topic": topic,
            "mode": mode,
            "context": context,
            "created_at": None,  # Would use datetime in real implementation
            "status": "active",
            "iteration_count": 0,
            "agents_involved": ["supervisor", "physics_expert", "hypothesis_generator"]
        }
        
        self.active_sessions[session_id] = session_info
        
        # Start collaboration
        response = self.supervisor.collaborate_on_topic(
            topic=topic,
            mode=mode,
            context=context,
            thread_id=session_id
        )
        
        return {
            "session_info": session_info,
            "response": response,
            "next_actions": self._suggest_next_actions(mode, topic)
        }
    
    def continue_collaboration(self, 
                             session_id: str, 
                             user_input: str,
                             mode: Optional[str] = None) -> Dict[str, Any]:
        """Continue an existing collaborative session.
        
        Args:
            session_id: Session ID to continue
            user_input: User's input or question
            mode: Optional mode change
            
        Returns:
            Updated session info and response
        """
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        # Update mode if provided
        if mode:
            session["mode"] = mode
        
        # Continue collaboration
        response = self.supervisor.chat(user_input, thread_id=session_id)
        
        # Update session
        session["iteration_count"] += 1
        
        return {
            "session_info": session,
            "response": response,
            "next_actions": self._suggest_next_actions(session["mode"], session["topic"])
        }
    
    def get_expert_analysis(self, 
                          problem: str, 
                          session_id: Optional[str] = None) -> str:
        """Get expert physics analysis for a specific problem.
        
        Args:
            problem: Physics problem or question
            session_id: Optional session ID for context
            
        Returns:
            Expert analysis
        """
        return self.physics_expert.solve_physics_problem(
            problem=problem,
            thread_id=session_id
        )
    
    def generate_hypotheses(self, 
                          topic: str, 
                          num_hypotheses: int = 3,
                          session_id: Optional[str] = None) -> str:
        """Generate creative hypotheses for a physics topic.
        
        Args:
            topic: Physics topic
            num_hypotheses: Number of hypotheses to generate
            session_id: Optional session ID for context
            
        Returns:
            Generated hypotheses
        """
        return self.hypothesis_generator.generate_hypotheses(
            topic=topic,
            num_hypotheses=num_hypotheses,
            thread_id=session_id
        )
    
    def facilitate_debate(self, 
                         hypothesis: str, 
                         topic: str,
                         session_id: Optional[str] = None) -> str:
        """Facilitate a structured debate about a hypothesis.
        
        Args:
            hypothesis: The hypothesis to debate
            topic: Physics topic context
            session_id: Optional session ID for context
            
        Returns:
            Debate results and synthesis
        """
        return self.supervisor.facilitate_debate(
            hypothesis=hypothesis,
            topic=topic,
            thread_id=session_id
        )
    
    def collaborative_problem_solving(self, 
                                    problem: str, 
                                    constraints: str = "",
                                    session_id: Optional[str] = None) -> str:
        """Solve a physics problem collaboratively.
        
        Args:
            problem: Physics problem to solve
            constraints: Any constraints or requirements
            session_id: Optional session ID for context
            
        Returns:
            Collaborative solution
        """
        return self.supervisor.collaborative_problem_solving(
            problem=problem,
            constraints=constraints,
            thread_id=session_id
        )
    
    def research_topic_collaboratively(self, 
                                     topic: str, 
                                     include_gaps: bool = True,
                                     session_id: Optional[str] = None) -> Dict[str, str]:
        """Research a physics topic collaboratively.
        
        Args:
            topic: Physics topic to research
            include_gaps: Whether to identify research gaps
            session_id: Optional session ID for context
            
        Returns:
            Collaborative research results
        """
        # Get expert analysis
        expert_analysis = self.physics_expert.research_topic(
            topic=topic,
            thread_id=session_id
        )
        
        # Get creative perspective and gaps
        creative_analysis = self.hypothesis_generator.identify_research_gaps(
            field=topic,
            thread_id=session_id
        ) if include_gaps else self.hypothesis_generator.generate_hypotheses(
            topic=topic,
            thread_id=session_id
        )
        
        # Get supervisor synthesis
        synthesis_prompt = f"""
        Synthesize the following research perspectives on {topic}:
        
        Expert Analysis:
        {expert_analysis}
        
        Creative/Gap Analysis:
        {creative_analysis}
        
        Provide a comprehensive synthesis that combines both perspectives.
        """
        
        synthesis = self.supervisor.chat(synthesis_prompt, thread_id=session_id)
        
        return {
            "expert_analysis": expert_analysis,
            "creative_analysis": creative_analysis,
            "synthesis": synthesis,
            "topic": topic
        }
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of a collaborative session.
        
        Args:
            session_id: Session ID to summarize
            
        Returns:
            Session summary
        """
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        # Generate summary using supervisor
        summary_prompt = f"""
        Provide a summary of this collaborative physics research session:
        
        Topic: {session['topic']}
        Mode: {session['mode']}
        Iterations: {session['iteration_count']}
        
        Summarize:
        1. Key insights discovered
        2. Hypotheses generated
        3. Expert validations
        4. Areas of agreement/disagreement
        5. Suggested next steps
        """
        
        summary = self.supervisor.chat(summary_prompt, thread_id=session_id)
        
        return {
            "session_info": session,
            "summary": summary,
            "recommendations": self._suggest_next_actions(session["mode"], session["topic"])
        }
    
    def _suggest_next_actions(self, mode: str, topic: str) -> List[str]:
        """Suggest next actions based on collaboration mode and topic.
        
        Args:
            mode: Current collaboration mode
            topic: Physics topic
            
        Returns:
            List of suggested next actions
        """
        base_actions = [
            "Continue the collaborative discussion",
            "Switch to a different collaboration mode",
            "Focus on a specific aspect of the topic",
            "Generate additional hypotheses",
            "Seek expert validation of ideas"
        ]
        
        mode_specific_actions = {
            "research": [
                "Conduct deeper literature review",
                "Design experiments to test hypotheses",
                "Identify required resources and collaborators"
            ],
            "debate": [
                "Explore alternative viewpoints",
                "Seek additional evidence",
                "Refine the hypothesis based on debate"
            ],
            "brainstorm": [
                "Evaluate feasibility of generated ideas",
                "Prioritize the most promising concepts",
                "Develop implementation plans"
            ],
            "teaching": [
                "Create educational materials",
                "Develop practice problems",
                "Explain concepts at different levels"
            ]
        }
        
        return base_actions + mode_specific_actions.get(mode, [])
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status information for all agents.
        
        Returns:
            Status information for each agent
        """
        return {
            "physics_expert": {
                "difficulty_level": self.physics_expert.difficulty_level,
                "specialty": getattr(self.physics_expert, 'specialty', None),
                "memory_enabled": self.physics_expert.memory_enabled
            },
            "hypothesis_generator": {
                "creativity_level": self.hypothesis_generator.creativity_level,
                "exploration_scope": self.hypothesis_generator.exploration_scope,
                "risk_tolerance": self.hypothesis_generator.risk_tolerance,
                "memory_enabled": self.hypothesis_generator.memory_enabled
            },
            "supervisor": {
                "collaboration_style": self.supervisor.collaboration_style,
                "max_iterations": self.supervisor.max_iterations,
                "memory_enabled": self.supervisor.memory_enabled
            },
            "system": {
                "active_sessions": len(self.active_sessions),
                "memory_store": "enabled" if self.memory_store else "disabled"
            }
        }
    
    def update_agent_settings(self, 
                            agent: str, 
                            settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update settings for a specific agent.
        
        Args:
            agent: Agent name (physics_expert, hypothesis_generator, supervisor)
            settings: New settings to apply
            
        Returns:
            Updated agent status
        """
        if agent == "physics_expert":
            if "difficulty_level" in settings:
                self.physics_expert.difficulty_level = settings["difficulty_level"]
                self.physics_expert.system_message = self.physics_expert._create_physics_system_message()
            if "specialty" in settings:
                self.physics_expert.specialty = settings["specialty"]
                self.physics_expert.system_message = self.physics_expert._create_physics_system_message()
                
        elif agent == "hypothesis_generator":
            if "creativity_level" in settings:
                self.hypothesis_generator.creativity_level = settings["creativity_level"]
                self.hypothesis_generator.system_message = self.hypothesis_generator._create_hypothesis_system_message()
            if "exploration_scope" in settings:
                self.hypothesis_generator.exploration_scope = settings["exploration_scope"]
                self.hypothesis_generator.system_message = self.hypothesis_generator._create_hypothesis_system_message()
            if "risk_tolerance" in settings:
                self.hypothesis_generator.risk_tolerance = settings["risk_tolerance"]
                self.hypothesis_generator.system_message = self.hypothesis_generator._create_hypothesis_system_message()
                
        elif agent == "supervisor":
            if "collaboration_style" in settings:
                self.supervisor.collaboration_style = settings["collaboration_style"]
                self.supervisor.system_message = self.supervisor._create_supervisor_system_message()
            if "max_iterations" in settings:
                self.supervisor.max_iterations = settings["max_iterations"]
        
        return self.get_agent_status()
    
    def close_session(self, session_id: str) -> Dict[str, Any]:
        """Close a collaborative session.
        
        Args:
            session_id: Session ID to close
            
        Returns:
            Session closure information
        """
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        session["status"] = "closed"
        
        # Generate final summary
        final_summary = self.get_session_summary(session_id)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        return {
            "message": "Session closed successfully",
            "final_summary": final_summary,
            "session_id": session_id
        } 