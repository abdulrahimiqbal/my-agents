"""Hypothesis Generator Agent - Specialized agent for creative hypothesis generation and research gap identification."""

from typing import List, Optional, Dict, Any
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from .base import BaseAgent
from ..tools.hypothesis_tools import get_hypothesis_tools
from ..tools.physics_research import get_physics_research_tools


class HypothesisGeneratorAgent(BaseAgent):
    """Hypothesis generator agent specialized in creative physics research."""
    
    def __init__(self, 
                 creativity_level: str = "high",
                 exploration_scope: str = "broad",
                 risk_tolerance: str = "medium",
                 memory_enabled: bool = True,
                 **kwargs):
        """Initialize the hypothesis generator agent.
        
        Args:
            creativity_level: Level of creativity (conservative, moderate, high, bold)
            exploration_scope: Scope of exploration (focused, broad, interdisciplinary)
            risk_tolerance: Tolerance for speculative ideas (low, medium, high)
            memory_enabled: Whether to enable memory for the agent
            **kwargs: Additional arguments for BaseAgent
        """
        self.creativity_level = creativity_level
        self.exploration_scope = exploration_scope
        self.risk_tolerance = risk_tolerance
        self.memory_enabled = memory_enabled
        self.system_message = self._create_hypothesis_system_message()
        
        super().__init__(**kwargs)
    
    def _create_hypothesis_system_message(self) -> str:
        """Create a comprehensive hypothesis generator system message."""
        base_message = """You are HypothesisGPT, a creative physics research assistant specialized in generating novel hypotheses, identifying research gaps, and proposing alternative approaches to physics problems. Your mission is to think outside conventional boundaries while maintaining scientific rigor.

## Your Expertise:
- **Creative Hypothesis Generation**: Proposing novel, testable ideas
- **Research Gap Identification**: Finding unexplored areas in physics
- **Alternative Approaches**: Suggesting unconventional problem-solving methods
- **Experimental Design**: Proposing ways to test new ideas
- **Interdisciplinary Thinking**: Connecting physics with other fields
- **Pattern Recognition**: Identifying hidden connections and analogies

## Your Approach:
1. **Creative Thinking**: Use analogies, inversions, and lateral thinking
2. **Scientific Rigor**: Ensure hypotheses are testable and falsifiable
3. **Risk Assessment**: Evaluate feasibility while encouraging bold ideas
4. **Systematic Exploration**: Consider multiple perspectives and scales
5. **Collaborative Spirit**: Work well with physics experts and other researchers
6. **Evidence-Based**: Ground speculation in existing scientific knowledge

## Your Tools:
- Creative hypothesis generation frameworks
- Research gap analysis methods
- Experimental design templates
- Feasibility evaluation tools
- Alternative approach brainstorming
- Scientific literature search

## Communication Style:
- Start with creative, "what if" questions
- Propose multiple hypotheses for each problem
- Explain the reasoning behind each idea
- Suggest experimental tests for hypotheses
- Acknowledge uncertainty and risk levels
- Encourage exploration of unconventional ideas
- Connect ideas across different physics domains

## Collaboration Guidelines:
- **With Physics Experts**: Provide creative input while respecting established knowledge
- **In Debates**: Present alternative viewpoints constructively
- **In Research**: Focus on novelty and unexplored directions
- **In Teaching**: Demonstrate creative scientific thinking processes"""

        # Customize based on creativity level
        creativity_customization = {
            "conservative": "\n\n## Current Mode: Conservative Creativity\n- Focus on incremental advances and safe hypotheses\n- Build closely on established theories\n- Prioritize high-probability, low-risk ideas\n- Emphasize feasibility over novelty",
            
            "moderate": "\n\n## Current Mode: Moderate Creativity\n- Balance innovation with feasibility\n- Propose testable but non-obvious hypotheses\n- Consider both incremental and breakthrough possibilities\n- Mix safe and speculative approaches",
            
            "high": "\n\n## Current Mode: High Creativity\n- Encourage bold, innovative hypotheses\n- Challenge conventional assumptions\n- Explore interdisciplinary connections\n- Prioritize novelty while maintaining scientific validity",
            
            "bold": "\n\n## Current Mode: Bold Creativity\n- Propose paradigm-shifting ideas\n- Question fundamental assumptions\n- Explore highly speculative but scientifically grounded concepts\n- Push the boundaries of current understanding"
        }
        
        base_message += creativity_customization.get(self.creativity_level, "")
        
        # Add exploration scope customization
        scope_customization = {
            "focused": "\n\n## Exploration Scope: Focused\n- Concentrate on specific subfields or problems\n- Deep dive into particular aspects\n- Maintain tight connection to the main topic",
            
            "broad": "\n\n## Exploration Scope: Broad\n- Consider connections across physics subfields\n- Explore multiple scales and contexts\n- Look for patterns and analogies across domains",
            
            "interdisciplinary": "\n\n## Exploration Scope: Interdisciplinary\n- Draw connections to other sciences and fields\n- Consider biological, chemical, and mathematical analogies\n- Explore applications beyond traditional physics"
        }
        
        base_message += scope_customization.get(self.exploration_scope, "")
        
        # Add risk tolerance customization
        risk_customization = {
            "low": "\n\n## Risk Tolerance: Conservative\n- Focus on hypotheses with high probability of success\n- Prefer well-established experimental methods\n- Minimize speculative elements",
            
            "medium": "\n\n## Risk Tolerance: Balanced\n- Balance speculative ideas with practical considerations\n- Consider both safe and risky approaches\n- Evaluate trade-offs between novelty and feasibility",
            
            "high": "\n\n## Risk Tolerance: Adventurous\n- Embrace highly speculative but scientifically sound ideas\n- Propose cutting-edge experimental approaches\n- Accept high uncertainty for potential breakthrough discoveries"
        }
        
        base_message += risk_customization.get(self.risk_tolerance, "")
        
        return base_message
    
    def _build_graph(self) -> StateGraph:
        """Build the hypothesis generator agent graph with specialized tools."""
        # Get hypothesis generation and research tools
        tools = (
            get_hypothesis_tools() +
            get_physics_research_tools()
        )
        
        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(tools)
        
        # Create the graph
        builder = StateGraph(MessagesState)
        
        # Define the hypothesis generator assistant node
        def hypothesis_assistant(state: MessagesState):
            """Hypothesis generator assistant node with creative reasoning."""
            system_msg = SystemMessage(content=self.system_message)
            messages = [system_msg] + state["messages"]
            
            # Add context about current conversation
            if len(state["messages"]) > 1:
                context_msg = SystemMessage(content=f"""
## Current Context:
- Creativity Level: {self.creativity_level}
- Exploration Scope: {self.exploration_scope}
- Risk Tolerance: {self.risk_tolerance}
- Conversation Length: {len(state["messages"])} messages

Remember to maintain your creative approach while building on the conversation context. Look for opportunities to propose novel hypotheses and alternative perspectives.
""")
                messages.insert(1, context_msg)
            
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        
        # Define tool node
        tool_node = ToolNode(tools)
        
        # Add nodes
        builder.add_node("hypothesis_assistant", hypothesis_assistant)
        builder.add_node("tools", tool_node)
        
        # Add edges
        builder.add_edge(START, "hypothesis_assistant")
        builder.add_conditional_edges(
            "hypothesis_assistant",
            tools_condition,
        )
        builder.add_edge("tools", "hypothesis_assistant")
        
        # Compile with memory if enabled
        if self.memory_enabled and self.memory:
            return builder.compile(checkpointer=self.memory)
        else:
            return builder.compile()
    
    def generate_hypotheses(self, 
                          topic: str, 
                          context: str = "",
                          num_hypotheses: int = 3,
                          thread_id: Optional[str] = None) -> str:
        """Generate creative hypotheses for a physics topic.
        
        Args:
            topic: Physics topic or problem
            context: Additional context or constraints
            num_hypotheses: Number of hypotheses to generate
            thread_id: Optional thread ID for memory
            
        Returns:
            Generated hypotheses with reasoning
        """
        prompt = f"""Please generate {num_hypotheses} creative hypotheses for the topic: {topic}

{"Additional context: " + context if context else ""}

Use your creative thinking tools to:
1. Generate novel, testable hypotheses
2. Explain the reasoning behind each hypothesis
3. Suggest how each could be tested experimentally
4. Evaluate the potential impact of each hypothesis
5. Identify any risks or challenges in testing them

Focus on ideas that might not be immediately obvious but are scientifically sound and potentially groundbreaking.
"""
        
        return self.chat(prompt, thread_id=thread_id)
    
    def identify_research_gaps(self, 
                             field: str, 
                             current_knowledge: str = "",
                             thread_id: Optional[str] = None) -> str:
        """Identify research gaps in a physics field.
        
        Args:
            field: Physics field to analyze
            current_knowledge: Summary of current understanding
            thread_id: Optional thread ID for memory
            
        Returns:
            Research gap analysis with opportunities
        """
        prompt = f"""Please analyze research gaps in the field: {field}

{"Current knowledge summary: " + current_knowledge if current_knowledge else ""}

Use your research gap analysis tools to:
1. Identify unexplored areas and missing connections
2. Suggest novel research directions
3. Evaluate the potential impact of addressing each gap
4. Propose experimental or theoretical approaches
5. Consider interdisciplinary opportunities

Focus on gaps that could lead to significant advances or new understanding in physics.
"""
        
        return self.chat(prompt, thread_id=thread_id)
    
    def propose_alternative_approaches(self, 
                                     problem: str, 
                                     current_approach: str = "",
                                     thread_id: Optional[str] = None) -> str:
        """Propose alternative approaches to a physics problem.
        
        Args:
            problem: The physics problem to address
            current_approach: Current or conventional approach
            thread_id: Optional thread ID for memory
            
        Returns:
            Alternative approaches with creative suggestions
        """
        prompt = f"""Please propose alternative approaches to this physics problem: {problem}

{"Current/conventional approach: " + current_approach if current_approach else ""}

Use your creative thinking tools to:
1. Brainstorm unconventional approaches
2. Consider analogies from other fields
3. Explore different scales or perspectives
4. Suggest interdisciplinary methods
5. Evaluate the potential of each approach

Think beyond traditional methods and consider approaches that might reveal new insights or lead to breakthrough discoveries.
"""
        
        return self.chat(prompt, thread_id=thread_id)
    
    def design_experiment(self, 
                         hypothesis: str, 
                         constraints: str = "",
                         thread_id: Optional[str] = None) -> str:
        """Design an experimental framework to test a hypothesis.
        
        Args:
            hypothesis: The hypothesis to test
            constraints: Experimental constraints
            thread_id: Optional thread ID for memory
            
        Returns:
            Experimental design framework
        """
        prompt = f"""Please design an experimental framework to test this hypothesis: {hypothesis}

{"Constraints to consider: " + constraints if constraints else ""}

Use your experimental design tools to:
1. Outline a comprehensive experimental approach
2. Identify key variables and controls
3. Suggest measurement methods and techniques
4. Evaluate feasibility and resource requirements
5. Consider alternative experimental designs
6. Assess potential risks and mitigation strategies

Focus on creative but practical experimental approaches that could provide clear evidence for or against the hypothesis.
"""
        
        return self.chat(prompt, thread_id=thread_id)
    
    def collaborate_with_expert(self, 
                              expert_input: str, 
                              topic: str,
                              thread_id: Optional[str] = None) -> str:
        """Collaborate with a physics expert by providing creative input.
        
        Args:
            expert_input: Input or analysis from physics expert
            topic: The topic being discussed
            thread_id: Optional thread ID for memory
            
        Returns:
            Creative response with hypotheses and alternatives
        """
        prompt = f"""A physics expert has provided this analysis on {topic}:

{expert_input}

As a creative hypothesis generator, please:
1. Identify aspects that could be explored further
2. Propose alternative perspectives or interpretations
3. Generate novel hypotheses based on the expert's input
4. Suggest creative experimental approaches
5. Ask thought-provoking questions that might lead to new insights
6. Consider connections to other areas of physics or science

Your goal is to complement the expert's knowledge with creative thinking and novel ideas while maintaining scientific rigor.
"""
        
        return self.chat(prompt, thread_id=thread_id) 
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this hypothesis generator agent."""
        return {
            "name": "HypothesisGeneratorAgent",
            "type": "hypothesis_generator",
            "description": "A specialized agent for generating creative hypotheses and alternative approaches in physics",
            "capabilities": [
                "Creative hypothesis generation",
                "Research gap identification",
                "Alternative approach proposal",
                "Experimental design",
                "Creative collaboration"
            ],
            "creativity_level": self.creativity_level,
            "exploration_scope": self.exploration_scope,
            "risk_tolerance": self.risk_tolerance,
            "tools": [
                "Hypothesis generation frameworks",
                "Research gap analysis",
                "Alternative approach brainstorming",
                "Experimental design templates"
            ],
            "version": "1.0.0"
        } 