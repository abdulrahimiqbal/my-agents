"""Physics Expert Agent - Specialized agent for physics problems and explanations."""

from typing import List, Optional, Dict, Any
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from .base import BaseAgent
from ..tools.physics_calculator import get_physics_calculator_tools
from ..tools.physics_research import get_physics_research_tools
from ..tools.physics_constants import get_physics_constants_tools
from ..tools.unit_converter import get_unit_converter_tools
from ..database.knowledge_api import KnowledgeAPI


class PhysicsExpertAgent(BaseAgent):
    """A specialized physics expert agent with comprehensive physics knowledge."""
    
    def __init__(self, 
                 difficulty_level: str = "undergraduate",
                 specialty: Optional[str] = None,
                 memory_enabled: bool = True,
                 **kwargs):
        """Initialize the physics expert agent.
        
        Args:
            difficulty_level: Target difficulty (high_school, undergraduate, graduate, research)
            specialty: Physics specialty (mechanics, electromagnetism, quantum, etc.)
            memory_enabled: Whether to enable memory functionality
            **kwargs: Additional arguments for BaseAgent
        """
        self.difficulty_level = difficulty_level
        self.specialty = specialty
        self.memory_enabled = memory_enabled
        self.system_message = self._create_physics_system_message()
        
        # Initialize KnowledgeAPI for event logging and knowledge management
        self.knowledge_api = KnowledgeAPI()
        
        super().__init__(**kwargs)
    
    def _create_physics_system_message(self) -> str:
        """Create a comprehensive physics expert system message."""
        base_message = """You are PhysicsGPT, a world-class physics expert and educator with deep knowledge across all areas of physics. Your mission is to help users understand physics concepts, solve problems, and explore the fascinating world of physics.

## Your Expertise:
- Classical Mechanics (Newtonian, Lagrangian, Hamiltonian)
- Electromagnetism (Maxwell's equations, circuits, fields)
- Thermodynamics and Statistical Mechanics
- Quantum Mechanics and Quantum Field Theory
- Relativity (Special and General)
- Optics and Wave Physics
- Particle Physics and Cosmology
- Condensed Matter Physics
- Computational Physics

## Your Approach:
1. **Problem-Solving**: Break down complex problems into manageable steps
2. **Conceptual Understanding**: Explain the physics intuition behind equations
3. **Mathematical Rigor**: Show detailed mathematical derivations when needed
4. **Visual Thinking**: Use analogies and suggest visualizations
5. **Unit Awareness**: Always check units and dimensional analysis
6. **Real-World Connections**: Connect physics to everyday phenomena

## Your Tools:
- Scientific calculator with advanced functions
- Unit conversion and dimensional analysis
- Physics constants database
- Research paper search (ArXiv)
- Equation solving and symbolic math
- Graph plotting and visualization

## Communication Style:
- Start with the key physics concept or principle
- Show step-by-step solutions with clear reasoning
- Use proper physics notation and terminology
- Provide multiple approaches when possible
- Suggest follow-up questions or related concepts
- Always verify answers using dimensional analysis

## Collaboration Guidelines:
- **With Hypothesis Generators**: Provide rigorous analysis while encouraging creativity
- **In Peer Review**: Give constructive feedback that maintains scientific standards
- **In Debates**: Present evidence-based arguments respectfully
- **In Research**: Focus on experimental validation and theoretical consistency
- **In Teaching**: Demonstrate scientific method and critical thinking"""

        # Customize based on difficulty level
        level_customization = {
            "high_school": "\n\n## Current Focus: High School Physics\n- Emphasize conceptual understanding\n- Use simpler mathematical approaches\n- Connect to everyday examples\n- Avoid advanced mathematical formalism",
            
            "undergraduate": "\n\n## Current Focus: Undergraduate Physics\n- Balance conceptual and mathematical rigor\n- Introduce advanced concepts gradually\n- Show multiple solution methods\n- Connect different areas of physics",
            
            "graduate": "\n\n## Current Focus: Graduate Physics\n- Use advanced mathematical formalism\n- Discuss cutting-edge research connections\n- Explore theoretical implications\n- Reference primary literature",
            
            "research": "\n\n## Current Focus: Research-Level Physics\n- Discuss latest developments and open questions\n- Reference recent papers and preprints\n- Explore theoretical and experimental frontiers\n- Consider interdisciplinary connections"
        }
        
        base_message += level_customization.get(self.difficulty_level, "")
        
        # Add specialty focus if specified
        if self.specialty:
            base_message += f"\n\n## Specialty Focus: {self.specialty.title()}\n- Prioritize problems and concepts in this area\n- Provide deeper insights into this field\n- Connect to current research in this specialty"
        
        return base_message
    
    def _build_graph(self) -> StateGraph:
        """Build the physics expert agent graph with specialized tools."""
        # Get all physics-specific tools
        tools = (
            get_physics_calculator_tools() +
            get_physics_research_tools() +
            get_physics_constants_tools() +
            get_unit_converter_tools()
        )
        
        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(tools)
        
        # Create the graph
        builder = StateGraph(MessagesState)
        
        # Define the physics expert assistant node
        def physics_assistant(state: MessagesState):
            """Physics expert assistant node with specialized reasoning."""
            system_msg = SystemMessage(content=self.system_message)
            messages = [system_msg] + state["messages"]
            
            # Add context about current conversation
            if len(state["messages"]) > 1:
                context_msg = SystemMessage(content=f"""
## Current Context:
- Difficulty Level: {self.difficulty_level}
- Specialty Focus: {self.specialty or 'General Physics'}
- Conversation Length: {len(state["messages"])} messages

Remember to maintain consistency with the established difficulty level and build upon previous concepts discussed.
""")
                messages.insert(1, context_msg)
            
            # Get the user's latest message for logging
            user_message = state["messages"][-1] if state["messages"] else None
            
            response = llm_with_tools.invoke(messages)
            
            # Log the physics expert activity
            if user_message:
                import asyncio
                try:
                    # Determine event type based on message content
                    message_content = user_message.content.lower()
                    if any(word in message_content for word in ['solve', 'calculate', 'problem']):
                        event_type = "problem_solving"
                    elif any(word in message_content for word in ['explain', 'what is', 'how does']):
                        event_type = "concept_explanation"
                    elif any(word in message_content for word in ['evaluate', 'hypothesis', 'theory']):
                        event_type = "hypothesis_evaluation"
                    else:
                        event_type = "physics_consultation"
                    
                    # Log the event
                    asyncio.create_task(self.knowledge_api.log_event(
                        source="physics_expert",
                        event_type=event_type,
                        payload={
                            "user_query": user_message.content,
                            "response_length": len(response.content),
                            "difficulty_level": self.difficulty_level,
                            "specialty": self.specialty,
                            "tools_used": bool(response.tool_calls) if hasattr(response, 'tool_calls') else False
                        }
                    ))
                except Exception as e:
                    print(f"⚠️ Event logging failed: {e}")
            
            return {"messages": [response]}
        
        # Define tool node
        tool_node = ToolNode(tools)
        
        # Add nodes
        builder.add_node("physics_assistant", physics_assistant)
        builder.add_node("tools", tool_node)
        
        # Add edges
        builder.add_edge(START, "physics_assistant")
        builder.add_conditional_edges(
            "physics_assistant",
            tools_condition,
        )
        builder.add_edge("tools", "physics_assistant")
        
        # Compile with memory if enabled
        if hasattr(self, 'memory_enabled') and self.memory_enabled and self.memory:
            return builder.compile(checkpointer=self.memory)
        else:
            return builder.compile()
    
    def solve_physics_problem(self, 
                            problem: str, 
                            show_steps: bool = True,
                            thread_id: Optional[str] = None) -> str:
        """Solve a physics problem with detailed steps.
        
        Args:
            problem: Physics problem description
            show_steps: Whether to show detailed solution steps
            thread_id: Optional thread ID for memory
            
        Returns:
            Detailed solution with steps
        """
        prompt = f"""Please solve this physics problem: {problem}

{"Please show detailed step-by-step solution including:" if show_steps else "Please provide a concise solution including:"}
1. Given information and what we need to find
2. Relevant physics principles and equations
3. {"Detailed mathematical steps" if show_steps else "Key calculations"}
4. Final answer with proper units
5. Physical interpretation of the result
{"6. Alternative solution methods if applicable" if show_steps else ""}
"""
        
        return self.chat(prompt, thread_id=thread_id)
    
    def explain_concept(self, 
                       concept: str, 
                       level: Optional[str] = None,
                       thread_id: Optional[str] = None) -> str:
        """Explain a physics concept at the appropriate level.
        
        Args:
            concept: Physics concept to explain
            level: Override difficulty level for this explanation
            thread_id: Optional thread ID for memory
            
        Returns:
            Detailed concept explanation
        """
        target_level = level or self.difficulty_level
        
        prompt = f"""Please explain the physics concept: {concept}

Target this explanation for {target_level} level understanding. Include:
1. Core definition and key principles
2. Mathematical formulation (appropriate for the level)
3. Physical intuition and analogies
4. Real-world examples and applications
5. Common misconceptions to avoid
6. Connection to other physics concepts
7. Suggested follow-up topics to explore
"""
        
        return self.chat(prompt, thread_id=thread_id)
    
    def research_topic(self, 
                      topic: str, 
                      include_recent: bool = True,
                      thread_id: Optional[str] = None) -> str:
        """Research a physics topic using available tools.
        
        Args:
            topic: Physics topic to research
            include_recent: Whether to include recent research papers
            thread_id: Optional thread ID for memory
            
        Returns:
            Research summary with sources
        """
        prompt = f"""Please research the physics topic: {topic}

{"Search for recent research papers and developments in this area." if include_recent else "Focus on established knowledge and foundational concepts."}

Provide:
1. Overview of the current understanding
2. Key theoretical frameworks and equations
3. Experimental evidence and observations
4. {"Recent developments and open questions" if include_recent else "Historical development"}
5. Applications and technological relevance
6. References to key papers or sources
"""
        
        return self.chat(prompt, thread_id=thread_id)
    
    def evaluate_hypothesis(self, 
                          hypothesis: str, 
                          context: str = "",
                          thread_id: Optional[str] = None,
                          session_id: Optional[str] = None) -> str:
        """Evaluate a physics hypothesis for scientific validity and feasibility.
        
        Args:
            hypothesis: The hypothesis to evaluate
            context: Additional context about the hypothesis
            thread_id: Optional thread ID for memory
            session_id: Optional session ID for collaboration tracking
            
        Returns:
            Detailed evaluation with scientific assessment
        """
        prompt = f"""Please evaluate this physics hypothesis: {hypothesis}

{"Additional context: " + context if context else ""}

As a physics expert, provide a comprehensive evaluation including:
1. **Scientific Validity**: Is this hypothesis consistent with established physics?
2. **Theoretical Foundation**: What theoretical framework supports or challenges this?
3. **Experimental Testability**: How could this hypothesis be tested?
4. **Potential Challenges**: What are the main scientific or technical obstacles?
5. **Literature Review**: What existing research relates to this hypothesis?
6. **Mathematical Framework**: What mathematical tools would be needed?
7. **Feasibility Assessment**: Is this practically achievable with current technology?
8. **Potential Impact**: What would be the significance if proven correct?

Be rigorous but constructive in your analysis, and suggest improvements or refinements if needed.
"""
        
        result = self.chat(prompt, thread_id=thread_id)
        
        # Log the hypothesis evaluation event
        import asyncio
        try:
            asyncio.create_task(self.knowledge_api.log_event(
                source="physics_expert",
                event_type="hypothesis_evaluation",
                payload={
                    "hypothesis": hypothesis,
                    "context": context,
                    "evaluation_length": len(result),
                    "difficulty_level": self.difficulty_level
                },
                thread_id=thread_id,
                session_id=session_id
            ))
        except Exception as e:
            print(f"⚠️ Hypothesis evaluation logging failed: {e}")
        
        return result
    
    def peer_review_analysis(self, 
                           analysis: str, 
                           topic: str,
                           thread_id: Optional[str] = None) -> str:
        """Provide peer review feedback on a physics analysis or hypothesis.
        
        Args:
            analysis: The analysis to review
            topic: The physics topic being analyzed
            thread_id: Optional thread ID for memory
            
        Returns:
            Peer review feedback with suggestions
        """
        prompt = f"""Please provide peer review feedback on this physics analysis about {topic}:

{analysis}

As a physics expert, evaluate:
1. **Accuracy**: Are the physics principles correctly applied?
2. **Completeness**: What important aspects might be missing?
3. **Clarity**: Is the analysis clearly presented and well-reasoned?
4. **Evidence**: Is the analysis supported by appropriate evidence?
5. **Methodology**: Are the proposed methods sound and appropriate?
6. **Assumptions**: Are any assumptions questionable or need clarification?
7. **Alternative Perspectives**: Are there other valid viewpoints to consider?
8. **Improvements**: What specific improvements would strengthen this analysis?

Provide constructive feedback that maintains scientific rigor while encouraging innovation.
"""
        
        return self.chat(prompt, thread_id=thread_id)
    
    def collaborate_with_hypothesis_generator(self, 
                                            hypothesis_input: str, 
                                            topic: str,
                                            thread_id: Optional[str] = None) -> str:
        """Collaborate with a hypothesis generator by providing expert analysis.
        
        Args:
            hypothesis_input: Input from hypothesis generator
            topic: The topic being discussed
            thread_id: Optional thread ID for memory
            
        Returns:
            Expert response with analysis and suggestions
        """
        prompt = f"""A hypothesis generator has provided this creative input on {topic}:

{hypothesis_input}

As a physics expert, please:
1. **Evaluate the Scientific Merit**: Which ideas have strong scientific foundation?
2. **Identify Promising Directions**: Which hypotheses deserve further investigation?
3. **Provide Technical Analysis**: What are the physics principles involved?
4. **Suggest Refinements**: How could promising ideas be improved or made more precise?
5. **Propose Experiments**: What specific experiments could test these hypotheses?
6. **Connect to Literature**: How do these ideas relate to existing research?
7. **Assess Feasibility**: Which ideas are most feasible with current technology?
8. **Build on Ideas**: How can we develop the most promising concepts further?

Your goal is to provide rigorous scientific analysis while encouraging creative thinking and innovation.
"""
        
        return self.chat(prompt, thread_id=thread_id)
    
    def validate_experimental_design(self, 
                                   experiment_design: str, 
                                   hypothesis: str,
                                   thread_id: Optional[str] = None) -> str:
        """Validate an experimental design for testing a physics hypothesis.
        
        Args:
            experiment_design: The proposed experimental design
            hypothesis: The hypothesis being tested
            thread_id: Optional thread ID for memory
            
        Returns:
            Validation feedback with improvements
        """
        prompt = f"""Please validate this experimental design for testing the hypothesis: {hypothesis}

Proposed experimental design:
{experiment_design}

As a physics expert, evaluate:
1. **Experimental Validity**: Will this experiment actually test the hypothesis?
2. **Controls and Variables**: Are appropriate controls and variables identified?
3. **Measurement Methods**: Are the proposed measurements accurate and precise enough?
4. **Systematic Errors**: What sources of error should be considered?
5. **Statistical Analysis**: Is the statistical approach appropriate?
6. **Equipment and Resources**: Are the required resources realistic and available?
7. **Safety Considerations**: Are there any safety concerns to address?
8. **Alternative Approaches**: Are there better or complementary experimental methods?
9. **Expected Results**: What results would support or refute the hypothesis?
10. **Reproducibility**: Can this experiment be reproduced by other researchers?

Provide specific suggestions for improving the experimental design while maintaining scientific rigor.
"""
        
        return self.chat(prompt, thread_id=thread_id)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this physics expert agent."""
        return {
            "name": "PhysicsExpertAgent",
            "type": "physics_expert",
            "description": "A specialized physics expert agent with comprehensive physics knowledge",
            "capabilities": [
                "Physics problem solving",
                "Concept explanation",
                "Hypothesis evaluation",
                "Peer review analysis",
                "Experimental design validation",
                "Research topic analysis"
            ],
            "difficulty_level": self.difficulty_level,
            "specialty": self.specialty,
            "tools": [
                "Physics calculator",
                "Research paper search",
                "Physics constants database",
                "Unit converter"
            ],
            "version": "1.0.0"
        } 