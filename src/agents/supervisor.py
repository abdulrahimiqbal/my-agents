"""Supervisor Agent - Orchestrates collaboration between Physics Expert and Hypothesis Generator agents."""

from typing import List, Optional, Dict, Any, Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from .base import BaseAgent
from .physics_expert import PhysicsExpertAgent
from .hypothesis_generator import HypothesisGeneratorAgent


class CollaborationState(BaseModel):
    """State for tracking collaboration between agents."""
    messages: List[Dict[str, Any]] = []
    current_topic: str = ""
    collaboration_mode: str = "research"  # research, debate, brainstorm, teaching
    physics_expert_input: str = ""
    hypothesis_generator_input: str = ""
    iteration_count: int = 0
    max_iterations: int = 5
    consensus_reached: bool = False
    final_synthesis: str = ""


class SupervisorAgent(BaseAgent):
    """Supervisor agent that orchestrates collaboration between specialized physics agents."""
    
    def __init__(self, 
                 physics_expert: Optional[PhysicsExpertAgent] = None,
                 hypothesis_generator: Optional[HypothesisGeneratorAgent] = None,
                 collaboration_style: str = "balanced",
                 max_iterations: int = 5,
                 memory_enabled: bool = True,
                 **kwargs):
        """Initialize the supervisor agent.
        
        Args:
            physics_expert: Physics Expert Agent instance
            hypothesis_generator: Hypothesis Generator Agent instance
            collaboration_style: Style of collaboration (balanced, expert_led, creative_led)
            max_iterations: Maximum collaboration iterations
            memory_enabled: Whether to enable memory functionality
            **kwargs: Additional arguments for BaseAgent
        """
        self.physics_expert = physics_expert
        self.hypothesis_generator = hypothesis_generator
        self.collaboration_style = collaboration_style
        self.max_iterations = max_iterations
        self.memory_enabled = memory_enabled
        self.system_message = self._create_supervisor_system_message()
        
        super().__init__(**kwargs)
    
    def _create_supervisor_system_message(self) -> str:
        """Create a comprehensive supervisor system message."""
        base_message = """You are the Collaboration Supervisor for a multi-agent physics research system. Your mission is to orchestrate productive collaboration between a Physics Expert Agent and a Hypothesis Generator Agent to solve complex physics problems and generate innovative research ideas.

## Your Role:
- **Orchestrator**: Manage the flow of collaboration between agents
- **Facilitator**: Ensure both agents contribute their expertise effectively
- **Synthesizer**: Combine insights from both agents into coherent conclusions
- **Quality Controller**: Maintain scientific rigor while encouraging creativity
- **Decision Maker**: Determine when consensus is reached or more iteration is needed

## Your Agents:
- **Physics Expert**: Provides rigorous scientific analysis, validates hypotheses, and ensures theoretical consistency
- **Hypothesis Generator**: Generates creative ideas, identifies research gaps, and proposes alternative approaches

## Collaboration Modes:
1. **Research Mode**: Systematic investigation with balanced expert analysis and creative exploration
2. **Debate Mode**: Structured discussion where agents challenge and refine each other's ideas
3. **Brainstorm Mode**: Creative exploration with minimal constraints, followed by expert evaluation
4. **Teaching Mode**: Collaborative explanation combining expert knowledge with creative analogies

## Your Process:
1. **Initialize**: Understand the user's question and determine the best collaboration approach
2. **Delegate**: Assign initial tasks to each agent based on their strengths
3. **Facilitate**: Manage the exchange of ideas between agents
4. **Synthesize**: Combine insights from both agents
5. **Evaluate**: Assess if the collaboration has reached a satisfactory conclusion
6. **Iterate**: Continue collaboration if needed, or conclude with final synthesis

## Decision Framework:
- **When to involve Physics Expert**: For validation, theoretical analysis, mathematical rigor
- **When to involve Hypothesis Generator**: For creative ideas, alternative approaches, research gaps
- **When to conclude**: When consensus is reached, maximum iterations exceeded, or diminishing returns
- **How to synthesize**: Combine rigorous analysis with creative insights for comprehensive answers

## Communication Style:
- Clearly identify which agent is contributing at each step
- Explain your reasoning for delegation decisions
- Highlight areas of agreement and disagreement between agents
- Provide clear synthesis of combined insights
- Maintain scientific accuracy while encouraging innovation"""

        # Customize based on collaboration style
        style_customization = {
            "balanced": "\n\n## Collaboration Style: Balanced\n- Give equal weight to expert analysis and creative thinking\n- Ensure both agents contribute meaningfully to each problem\n- Seek synthesis that combines rigor with innovation",
            
            "expert_led": "\n\n## Collaboration Style: Expert-Led\n- Prioritize physics expert's analysis and validation\n- Use hypothesis generator for creative input and alternatives\n- Ensure scientific rigor takes precedence over creativity",
            
            "creative_led": "\n\n## Collaboration Style: Creative-Led\n- Emphasize hypothesis generation and creative exploration\n- Use physics expert for validation and refinement\n- Encourage bold ideas while maintaining scientific validity"
        }
        
        base_message += style_customization.get(self.collaboration_style, "")
        
        return base_message
    
    def _build_graph(self) -> StateGraph:
        """Build the supervisor agent graph with collaboration orchestration."""
        # Create a custom state that tracks collaboration
        class CollaborationMessagesState(MessagesState):
            current_topic: str = ""
            collaboration_mode: str = "research"
            iteration_count: int = 0
            consensus_reached: bool = False
        
        # Create the graph
        builder = StateGraph(CollaborationMessagesState)
        
        # Define supervisor node
        def supervisor_node(state: CollaborationMessagesState):
            """Supervisor decision-making node."""
            system_msg = SystemMessage(content=self.system_message)
            messages = [system_msg] + state["messages"]
            
            # Add collaboration context
            context_msg = SystemMessage(content=f"""
## Current Collaboration Status:
- Topic: {state.get('current_topic', 'Not specified')}
- Mode: {state.get('collaboration_mode', 'research')}
- Iteration: {state.get('iteration_count', 0)}/{self.max_iterations}
- Consensus: {state.get('consensus_reached', False)}

Analyze the conversation and decide the next step in the collaboration process.
""")
            messages.insert(1, context_msg)
            
            response = self.llm.invoke(messages)
            return {"messages": [response]}
        
        # Define physics expert node
        def physics_expert_node(state: CollaborationMessagesState):
            """Physics Expert Agent node."""
            if not self.physics_expert:
                return {"messages": [AIMessage(content="Physics Expert not available")]}
            
            # Get the latest user message or supervisor instruction
            latest_message = state["messages"][-1].content if state["messages"] else ""
            
            # Run physics expert analysis
            response = self.physics_expert.chat(latest_message)
            return {"messages": [AIMessage(content=f"🔬 **Physics Expert**: {response}")]}
        
        # Define hypothesis generator node
        def hypothesis_generator_node(state: CollaborationMessagesState):
            """Hypothesis Generator Agent node."""
            if not self.hypothesis_generator:
                return {"messages": [AIMessage(content="Hypothesis Generator not available")]}
            
            # Get the latest user message or supervisor instruction
            latest_message = state["messages"][-1].content if state["messages"] else ""
            
            # Run hypothesis generation
            response = self.hypothesis_generator.chat(latest_message)
            return {"messages": [AIMessage(content=f"💡 **Hypothesis Generator**: {response}")]}
        
        # Define synthesis node
        def synthesis_node(state: CollaborationMessagesState):
            """Synthesize insights from both agents."""
            # Extract recent contributions from both agents
            physics_contributions = []
            hypothesis_contributions = []
            
            for msg in state["messages"][-10:]:  # Look at last 10 messages
                if isinstance(msg, AIMessage):
                    if "🔬 **Physics Expert**" in msg.content:
                        physics_contributions.append(msg.content)
                    elif "💡 **Hypothesis Generator**" in msg.content:
                        hypothesis_contributions.append(msg.content)
            
            synthesis_prompt = f"""
Based on the collaboration between the Physics Expert and Hypothesis Generator, provide a comprehensive synthesis:

Physics Expert Contributions:
{chr(10).join(physics_contributions[-3:]) if physics_contributions else "None"}

Hypothesis Generator Contributions:
{chr(10).join(hypothesis_contributions[-3:]) if hypothesis_contributions else "None"}

Provide a synthesis that:
1. Combines the rigorous analysis with creative insights
2. Highlights key agreements and disagreements
3. Presents a balanced conclusion
4. Suggests next steps or further research directions
"""
            
            response = self.llm.invoke([SystemMessage(content=synthesis_prompt)])
            return {
                "messages": [AIMessage(content=f"🤝 **Synthesis**: {response.content}")],
                "consensus_reached": True
            }
        
        # Define routing function
        def route_collaboration(state: CollaborationMessagesState) -> Literal["physics_expert", "hypothesis_generator", "synthesis", "supervisor", "end"]:
            """Route to the next step in collaboration."""
            # Simple routing logic - in practice, this would be more sophisticated
            iteration = state.get("iteration_count", 0)
            consensus = state.get("consensus_reached", False)
            
            if consensus or iteration >= self.max_iterations:
                return "end"
            elif iteration == 0:
                return "physics_expert"  # Start with expert analysis
            elif iteration == 1:
                return "hypothesis_generator"  # Then get creative input
            elif iteration < self.max_iterations - 1:
                return "supervisor"  # Let supervisor decide
            else:
                return "synthesis"  # Final synthesis
        
        # Add nodes
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("physics_expert", physics_expert_node)
        builder.add_node("hypothesis_generator", hypothesis_generator_node)
        builder.add_node("synthesis", synthesis_node)
        
        # Add edges
        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            route_collaboration,
            {
                "physics_expert": "physics_expert",
                "hypothesis_generator": "hypothesis_generator",
                "synthesis": "synthesis",
                "supervisor": "supervisor",
                "end": END
            }
        )
        builder.add_edge("physics_expert", "supervisor")
        builder.add_edge("hypothesis_generator", "supervisor")
        builder.add_edge("synthesis", END)
        
        # Compile with memory if enabled
        if self.memory_enabled and self.memory:
            return builder.compile(checkpointer=self.memory)
        else:
            return builder.compile()
    
    def collaborate_on_topic(self, 
                           topic: str, 
                           mode: str = "research",
                           context: str = "",
                           thread_id: Optional[str] = None) -> str:
        """Orchestrate collaboration between agents on a physics topic.
        
        Args:
            topic: Physics topic or problem to investigate
            mode: Collaboration mode (research, debate, brainstorm, teaching)
            context: Additional context or constraints
            thread_id: Optional thread ID for memory
            
        Returns:
            Collaborative analysis and synthesis
        """
        # Set up initial state for the collaboration graph
        initial_state = {
            "messages": [HumanMessage(content=f"Please investigate this physics topic: {topic}\n\nMode: {mode}\n{context if context else ''}")],
            "current_topic": topic,
            "collaboration_mode": mode,
            "iteration_count": 0,
            "consensus_reached": False
        }
        
        # Use the graph-based collaboration system
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        
        try:
            # Run the collaboration graph
            result = self.graph.invoke(initial_state, config)
            
            # Extract the final response from the messages
            if result.get("messages"):
                # Get the last message which should be the synthesis
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
                else:
                    return str(last_message)
            else:
                return "Collaboration completed but no response generated."
                
        except Exception as e:
            # Fallback to direct agent interaction if graph fails
            print(f"Graph collaboration failed: {e}")
            return self._fallback_collaboration(topic, mode, context, thread_id)
    
    def _fallback_collaboration(self, topic: str, mode: str, context: str, thread_id: Optional[str] = None) -> str:
        """Fallback collaboration method when graph fails."""
        try:
            # Get responses from both agents directly
            expert_response = ""
            hypothesis_response = ""
            
            if self.physics_expert:
                expert_prompt = f"Analyze this physics topic: {topic}\n\nMode: {mode}\n{context if context else ''}"
                expert_response = self.physics_expert.chat(expert_prompt, thread_id=thread_id)
            
            if self.hypothesis_generator:
                hypothesis_prompt = f"Generate creative hypotheses and insights about: {topic}\n\nMode: {mode}\n{context if context else ''}"
                hypothesis_response = self.hypothesis_generator.chat(hypothesis_prompt, thread_id=thread_id)
            
            # Synthesize the responses
            synthesis_prompt = f"""
Based on the following collaborative analysis, provide a comprehensive synthesis:

🔬 **Physics Expert Analysis:**
{expert_response}

💡 **Hypothesis Generator Insights:**
{hypothesis_response}

Please provide a synthesis that:
1. Combines rigorous analysis with creative insights
2. Highlights key findings and novel perspectives
3. Identifies areas of agreement and potential conflicts
4. Suggests directions for further investigation
5. Maintains scientific rigor while encouraging innovation

Topic: {topic}
Mode: {mode}
"""
            
            return self.chat(synthesis_prompt, thread_id=thread_id)
            
        except Exception as e:
            return f"Collaboration failed: {str(e)}. Please try again or contact support."
    
    def facilitate_debate(self, 
                         hypothesis: str, 
                         topic: str,
                         thread_id: Optional[str] = None) -> str:
        """Facilitate a structured debate between agents about a hypothesis.
        
        Args:
            hypothesis: The hypothesis to debate
            topic: The physics topic context
            thread_id: Optional thread ID for memory
            
        Returns:
            Structured debate and conclusion
        """
        prompt = f"""Facilitate a structured debate about this physics hypothesis: {hypothesis}

Topic context: {topic}

Orchestrate a debate where:
1. Physics Expert evaluates the hypothesis for scientific validity
2. Hypothesis Generator defends or refines the creative aspects
3. Both agents challenge each other's perspectives constructively
4. You synthesize the debate into a balanced conclusion
5. Identify areas of agreement and remaining questions

Maintain scientific rigor while encouraging innovative thinking.
"""
        
        return self.chat(prompt, thread_id=thread_id)
    
    def collaborative_problem_solving(self, 
                                    problem: str, 
                                    constraints: str = "",
                                    thread_id: Optional[str] = None) -> str:
        """Coordinate collaborative problem-solving between agents.
        
        Args:
            problem: Physics problem to solve
            constraints: Any constraints or requirements
            thread_id: Optional thread ID for memory
            
        Returns:
            Collaborative solution with multiple perspectives
        """
        prompt = f"""Coordinate collaborative problem-solving for this physics problem: {problem}

{"Constraints: " + constraints if constraints else ""}

Guide the collaboration to:
1. Have Physics Expert provide rigorous analysis and solution approaches
2. Have Hypothesis Generator suggest alternative methods and creative insights
3. Compare and evaluate different approaches
4. Synthesize the best elements from both perspectives
5. Present a comprehensive solution with multiple viewpoints

Ensure the final solution is both scientifically sound and creatively informed.
"""
        
        return self.chat(prompt, thread_id=thread_id) 
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this supervisor agent."""
        return {
            "name": "SupervisorAgent",
            "type": "supervisor",
            "description": "A supervisor agent that orchestrates collaboration between specialized physics agents",
            "capabilities": [
                "Multi-agent coordination",
                "Collaborative research orchestration",
                "Debate facilitation",
                "Problem-solving coordination",
                "Synthesis and consensus building"
            ],
            "collaboration_style": self.collaboration_style,
            "max_iterations": self.max_iterations,
            "managed_agents": [
                "PhysicsExpertAgent" if self.physics_expert else None,
                "HypothesisGeneratorAgent" if self.hypothesis_generator else None
            ],
            "collaboration_modes": [
                "research",
                "debate", 
                "brainstorm",
                "teaching"
            ],
            "version": "1.0.0"
        } 