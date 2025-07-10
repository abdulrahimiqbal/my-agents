"""Physics Expert Agent - Specialized agent for physics problems and explanations."""

from typing import List, Optional
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from .base import BaseAgent
from ..tools.physics_calculator import get_physics_calculator_tools
from ..tools.physics_research import get_physics_research_tools
from ..tools.physics_constants import get_physics_constants_tools
from ..tools.unit_converter import get_unit_converter_tools


class PhysicsExpertAgent(BaseAgent):
    """A specialized physics expert agent with comprehensive physics knowledge."""
    
    def __init__(self, 
                 difficulty_level: str = "undergraduate",
                 specialty: Optional[str] = None,
                 **kwargs):
        """Initialize the physics expert agent.
        
        Args:
            difficulty_level: Target difficulty (high_school, undergraduate, graduate, research)
            specialty: Physics specialty (mechanics, electromagnetism, quantum, etc.)
            **kwargs: Additional arguments for BaseAgent
        """
        self.difficulty_level = difficulty_level
        self.specialty = specialty
        self.system_message = self._create_physics_system_message()
        
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
- Always verify answers using dimensional analysis"""

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
            
            response = llm_with_tools.invoke(messages)
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
        if self.memory_enabled and self.memory:
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
        
        return self.run(prompt, thread_id=thread_id)
    
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
        
        return self.run(prompt, thread_id=thread_id)
    
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
        
        return self.run(prompt, thread_id=thread_id) 