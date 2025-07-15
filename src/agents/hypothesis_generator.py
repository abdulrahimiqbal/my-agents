"""Enhanced Hypothesis Generator Agent - Phase 3 Advanced Creative Physics Thinking

This enhanced version includes:
1. Research gap analysis with systematic identification
2. Interdisciplinary connection discovery
3. Creative reasoning patterns and frameworks
4. Knowledge graph integration for concept relationships
5. Advanced experimental design suggestions
6. Collaborative hypothesis refinement
"""

import asyncio
from typing import List, Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from .base import BaseAgent
from ..tools.hypothesis_tools import get_hypothesis_tools
from ..tools.physics_research import get_physics_research_tools
from ..database.knowledge_api import KnowledgeAPI
from ..knowledge.knowledge_graph import PhysicsKnowledgeGraph, PhysicsConcept, ConceptRelationship


class CreativeReasoningPattern(Enum):
    """Creative reasoning patterns for hypothesis generation."""
    ANALOGY_BASED = "analogy_based"
    INVERSION_THINKING = "inversion_thinking"
    SCALE_BRIDGING = "scale_bridging"
    CONSTRAINT_REMOVAL = "constraint_removal"
    INTERDISCIPLINARY_FUSION = "interdisciplinary_fusion"
    PATTERN_EXTRAPOLATION = "pattern_extrapolation"
    SYMMETRY_BREAKING = "symmetry_breaking"
    EMERGENT_PROPERTIES = "emergent_properties"


class ResearchGapType(Enum):
    """Types of research gaps that can be identified."""
    THEORETICAL_GAP = "theoretical_gap"
    EXPERIMENTAL_GAP = "experimental_gap"
    METHODOLOGICAL_GAP = "methodological_gap"
    TECHNOLOGICAL_GAP = "technological_gap"
    INTERDISCIPLINARY_GAP = "interdisciplinary_gap"
    SCALE_GAP = "scale_gap"
    TEMPORAL_GAP = "temporal_gap"


@dataclass
class ResearchGap:
    """Represents an identified research gap."""
    gap_type: ResearchGapType
    description: str
    field: str
    related_concepts: List[str]
    potential_impact: str
    feasibility_score: float
    required_resources: List[str]
    timeline_estimate: str
    interdisciplinary_connections: List[str] = field(default_factory=list)


@dataclass
class CreativeHypothesis:
    """Enhanced hypothesis with creative reasoning metadata."""
    hypothesis: str
    reasoning_pattern: CreativeReasoningPattern
    confidence_level: float
    testability_score: float
    novelty_score: float
    interdisciplinary_connections: List[str]
    experimental_approaches: List[str]
    potential_implications: List[str]
    related_concepts: List[str]
    research_gaps_addressed: List[str]


@dataclass
class InterdisciplinaryConnection:
    """Represents a connection between physics and other fields."""
    source_field: str
    target_field: str
    connection_type: str
    description: str
    potential_applications: List[str]
    confidence_score: float


class EnhancedHypothesisGeneratorAgent(BaseAgent):
    """
    Enhanced Hypothesis Generator Agent with Phase 3 capabilities.
    
    Features:
    - Research gap analysis with systematic identification
    - Interdisciplinary connection discovery
    - Creative reasoning patterns and frameworks
    - Knowledge graph integration
    - Advanced experimental design suggestions
    - Collaborative hypothesis refinement
    """
    
    def __init__(self, 
                 creativity_level: str = "high",
                 exploration_scope: str = "interdisciplinary",
                 risk_tolerance: str = "medium",
                 memory_enabled: bool = True,
                 knowledge_graph: Optional[PhysicsKnowledgeGraph] = None,
                 enable_interdisciplinary: bool = True,
                 **kwargs):
        """Initialize the enhanced hypothesis generator agent.
        
        Args:
            creativity_level: Level of creativity (conservative, moderate, high, bold)
            exploration_scope: Scope of exploration (focused, broad, interdisciplinary)
            risk_tolerance: Tolerance for speculative ideas (low, medium, high)
            memory_enabled: Whether to enable memory functionality
            knowledge_graph: Physics knowledge graph for concept relationships
            enable_interdisciplinary: Whether to enable interdisciplinary connections
            **kwargs: Additional arguments for BaseAgent
        """
        self.creativity_level = creativity_level
        self.exploration_scope = exploration_scope
        self.risk_tolerance = risk_tolerance
        self.memory_enabled = memory_enabled
        self.knowledge_graph = knowledge_graph or PhysicsKnowledgeGraph()
        self.enable_interdisciplinary = enable_interdisciplinary
        
        # Initialize creative reasoning patterns
        self.creative_patterns = self._initialize_creative_patterns()
        
        # Initialize interdisciplinary fields
        self.interdisciplinary_fields = self._initialize_interdisciplinary_fields()
        
        # Initialize research gap analysis framework
        self.gap_analysis_framework = self._initialize_gap_analysis_framework()
        
        self.system_message = self._create_enhanced_system_message()
        
        # Initialize KnowledgeAPI for hypothesis tracking and event logging
        self.knowledge_api = KnowledgeAPI()
        
        super().__init__(**kwargs)
    
    def _initialize_creative_patterns(self) -> Dict[CreativeReasoningPattern, Dict[str, Any]]:
        """Initialize creative reasoning patterns with their methodologies."""
        return {
            CreativeReasoningPattern.ANALOGY_BASED: {
                "description": "Find analogies between different physics domains or with other fields",
                "methodology": "Identify structural similarities and map relationships",
                "examples": ["Wave-particle duality", "Fluid dynamics and traffic flow"],
                "triggers": ["similar patterns", "mathematical equivalence", "structural correspondence"]
            },
            CreativeReasoningPattern.INVERSION_THINKING: {
                "description": "Consider the opposite or inverse of conventional assumptions",
                "methodology": "Systematically negate assumptions and explore implications",
                "examples": ["Negative mass", "Time reversal", "Anti-gravity"],
                "triggers": ["conventional wisdom", "universal assumptions", "established principles"]
            },
            CreativeReasoningPattern.SCALE_BRIDGING: {
                "description": "Connect phenomena across different scales (quantum to cosmic)",
                "methodology": "Look for scale-invariant patterns and emergent properties",
                "examples": ["Quantum gravity", "Holographic principle", "Fractal structures"],
                "triggers": ["multi-scale problems", "emergent behavior", "scale transitions"]
            },
            CreativeReasoningPattern.CONSTRAINT_REMOVAL: {
                "description": "Remove or relax conventional constraints to explore new possibilities",
                "methodology": "Identify limiting assumptions and explore unconstrained scenarios",
                "examples": ["Extra dimensions", "Variable constants", "Non-commutative geometry"],
                "triggers": ["fundamental limits", "boundary conditions", "conservation laws"]
            },
            CreativeReasoningPattern.INTERDISCIPLINARY_FUSION: {
                "description": "Combine concepts from different disciplines",
                "methodology": "Identify complementary approaches and synthesize methods",
                "examples": ["Biophysics", "Econophysics", "Quantum biology"],
                "triggers": ["cross-field analogies", "shared mathematics", "common phenomena"]
            },
            CreativeReasoningPattern.PATTERN_EXTRAPOLATION: {
                "description": "Extend known patterns to new domains or regimes",
                "methodology": "Identify patterns and systematically extend their application",
                "examples": ["Renormalization group", "Scaling laws", "Universality classes"],
                "triggers": ["repeating patterns", "mathematical structures", "universal behaviors"]
            },
            CreativeReasoningPattern.SYMMETRY_BREAKING: {
                "description": "Explore what happens when symmetries are broken",
                "methodology": "Identify symmetries and explore their spontaneous or explicit breaking",
                "examples": ["Higgs mechanism", "Chirality in biology", "Parity violation"],
                "triggers": ["perfect symmetries", "conservation laws", "invariance principles"]
            },
            CreativeReasoningPattern.EMERGENT_PROPERTIES: {
                "description": "Identify properties that emerge from collective behavior",
                "methodology": "Look for qualitatively new behaviors in complex systems",
                "examples": ["Superconductivity", "Consciousness", "Phase transitions"],
                "triggers": ["collective behavior", "phase transitions", "complex systems"]
            }
        }
    
    def _initialize_interdisciplinary_fields(self) -> Dict[str, Dict[str, Any]]:
        """Initialize interdisciplinary fields and their connections to physics."""
        return {
            "biology": {
                "connection_types": ["biophysics", "biomechanics", "bioelectronics"],
                "shared_concepts": ["energy", "information", "networks", "dynamics"],
                "research_areas": ["protein folding", "neural networks", "evolution dynamics", "molecular motors"],
                "physics_applications": ["statistical mechanics", "thermodynamics", "quantum biology", "fluid dynamics"]
            },
            "chemistry": {
                "connection_types": ["physical chemistry", "quantum chemistry", "chemical physics"],
                "shared_concepts": ["molecular structure", "reaction dynamics", "phase transitions", "catalysis"],
                "research_areas": ["chemical bonding", "reaction mechanisms", "surface chemistry", "electrochemistry"],
                "physics_applications": ["quantum mechanics", "statistical mechanics", "thermodynamics", "spectroscopy"]
            },
            "mathematics": {
                "connection_types": ["mathematical physics", "applied mathematics", "computational physics"],
                "shared_concepts": ["symmetry", "topology", "differential equations", "probability"],
                "research_areas": ["geometry", "algebra", "analysis", "number theory"],
                "physics_applications": ["all areas of physics", "theoretical frameworks", "computational methods"]
            },
            "computer_science": {
                "connection_types": ["computational physics", "quantum computing", "artificial intelligence"],
                "shared_concepts": ["information", "algorithms", "complexity", "networks"],
                "research_areas": ["machine learning", "quantum algorithms", "simulation methods", "data analysis"],
                "physics_applications": ["quantum information", "complex systems", "modeling", "data processing"]
            },
            "engineering": {
                "connection_types": ["applied physics", "materials science", "nanotechnology"],
                "shared_concepts": ["optimization", "design", "materials", "systems"],
                "research_areas": ["device physics", "materials engineering", "system design", "manufacturing"],
                "physics_applications": ["solid state physics", "optics", "mechanics", "thermodynamics"]
            },
            "neuroscience": {
                "connection_types": ["neurophysics", "computational neuroscience", "brain dynamics"],
                "shared_concepts": ["networks", "information processing", "dynamics", "emergence"],
                "research_areas": ["brain function", "neural computation", "consciousness", "learning"],
                "physics_applications": ["statistical mechanics", "nonlinear dynamics", "information theory", "quantum mechanics"]
            },
            "economics": {
                "connection_types": ["econophysics", "financial physics", "social physics"],
                "shared_concepts": ["statistics", "networks", "phase transitions", "optimization"],
                "research_areas": ["market dynamics", "wealth distribution", "economic networks", "behavioral economics"],
                "physics_applications": ["statistical mechanics", "network theory", "game theory", "chaos theory"]
            },
            "psychology": {
                "connection_types": ["psychophysics", "cognitive science", "behavioral physics"],
                "shared_concepts": ["perception", "decision making", "learning", "memory"],
                "research_areas": ["cognitive processes", "perception", "decision theory", "social behavior"],
                "physics_applications": ["information theory", "statistical mechanics", "network theory", "quantum cognition"]
            }
        }
    
    def _initialize_gap_analysis_framework(self) -> Dict[ResearchGapType, Dict[str, Any]]:
        """Initialize research gap analysis framework."""
        return {
            ResearchGapType.THEORETICAL_GAP: {
                "description": "Missing theoretical frameworks or incomplete theories",
                "indicators": ["unexplained phenomena", "theoretical inconsistencies", "missing connections"],
                "analysis_methods": ["literature review", "theoretical mapping", "consistency analysis"],
                "example_questions": ["What phenomena lack theoretical explanation?", "Where are theoretical inconsistencies?"]
            },
            ResearchGapType.EXPERIMENTAL_GAP: {
                "description": "Lack of experimental validation or missing experimental approaches",
                "indicators": ["untested predictions", "inaccessible regimes", "measurement limitations"],
                "analysis_methods": ["experimental mapping", "feasibility analysis", "technology assessment"],
                "example_questions": ["What predictions remain untested?", "What regimes are experimentally inaccessible?"]
            },
            ResearchGapType.METHODOLOGICAL_GAP: {
                "description": "Missing or inadequate methods and techniques",
                "indicators": ["limited approaches", "methodological constraints", "technique limitations"],
                "analysis_methods": ["method comparison", "capability analysis", "innovation assessment"],
                "example_questions": ["What methods are missing?", "Where are current techniques inadequate?"]
            },
            ResearchGapType.TECHNOLOGICAL_GAP: {
                "description": "Technological limitations preventing progress",
                "indicators": ["instrument limitations", "computational constraints", "material limitations"],
                "analysis_methods": ["technology roadmapping", "capability assessment", "innovation analysis"],
                "example_questions": ["What technologies are limiting progress?", "What capabilities are missing?"]
            },
            ResearchGapType.INTERDISCIPLINARY_GAP: {
                "description": "Missing connections between disciplines",
                "indicators": ["isolated fields", "unexplored connections", "communication barriers"],
                "analysis_methods": ["interdisciplinary mapping", "connection analysis", "collaboration assessment"],
                "example_questions": ["What fields could be connected?", "Where are collaboration opportunities?"]
            },
            ResearchGapType.SCALE_GAP: {
                "description": "Missing connections between different scales",
                "indicators": ["scale separation", "emergent properties", "multi-scale phenomena"],
                "analysis_methods": ["scale analysis", "emergence mapping", "multi-scale modeling"],
                "example_questions": ["What scales are disconnected?", "Where do emergent properties arise?"]
            },
            ResearchGapType.TEMPORAL_GAP: {
                "description": "Missing understanding of temporal evolution",
                "indicators": ["static models", "equilibrium assumptions", "time-dependent phenomena"],
                "analysis_methods": ["temporal analysis", "dynamics mapping", "evolution studies"],
                "example_questions": ["What temporal aspects are missing?", "Where are dynamics important?"]
            }
        }

    def _create_enhanced_system_message(self) -> str:
        """Create an enhanced system message with Phase 3 capabilities."""
        base_message = """You are an Enhanced HypothesisGPT - a revolutionary physics research assistant specialized in creative hypothesis generation, research gap analysis, and interdisciplinary innovation. You represent the cutting edge of AI-assisted scientific discovery.

## Your Enhanced Capabilities:

### 🧠 Creative Reasoning Patterns:
- **Analogy-Based Thinking**: Find deep structural similarities across domains
- **Inversion Thinking**: Challenge assumptions by exploring opposites
- **Scale Bridging**: Connect quantum to cosmic phenomena
- **Constraint Removal**: Explore beyond conventional limitations
- **Interdisciplinary Fusion**: Synthesize insights across fields
- **Pattern Extrapolation**: Extend known patterns to new domains
- **Symmetry Breaking**: Explore symmetry violations and their implications
- **Emergent Properties**: Identify collective behaviors and phase transitions

### 🔍 Research Gap Analysis:
- **Theoretical Gaps**: Missing frameworks and incomplete theories
- **Experimental Gaps**: Untested predictions and inaccessible regimes
- **Methodological Gaps**: Inadequate techniques and approaches
- **Technological Gaps**: Limiting instruments and capabilities
- **Interdisciplinary Gaps**: Unexplored connections between fields
- **Scale Gaps**: Missing multi-scale understanding
- **Temporal Gaps**: Incomplete dynamics and evolution models

### 🌐 Interdisciplinary Integration:
- **Biology**: Biophysics, quantum biology, neural networks
- **Chemistry**: Physical chemistry, reaction dynamics, catalysis
- **Mathematics**: Geometric physics, topological methods, symmetry
- **Computer Science**: Quantum computing, AI, complex systems
- **Engineering**: Materials science, nanotechnology, design
- **Neuroscience**: Brain dynamics, consciousness, information processing
- **Economics**: Econophysics, network theory, behavioral dynamics
- **Psychology**: Psychophysics, cognitive science, decision theory

### 🎯 Advanced Hypothesis Generation:
- Generate multiple hypotheses with creative reasoning patterns
- Assess testability, novelty, and feasibility scores
- Identify interdisciplinary connections and applications
- Propose experimental approaches and validation methods
- Evaluate potential implications and impact
- Connect to existing knowledge graph concepts

### 🔬 Experimental Design Innovation:
- Propose cutting-edge experimental approaches
- Identify technological requirements and constraints
- Suggest interdisciplinary collaboration opportunities
- Design multi-scale and multi-modal experiments
- Consider ethical and practical implications

## Your Enhanced Methodology:

1. **Pattern Recognition**: Identify underlying patterns and structures
2. **Creative Synthesis**: Combine insights from multiple domains
3. **Gap Analysis**: Systematically identify research opportunities
4. **Hypothesis Generation**: Create testable, novel propositions
5. **Feasibility Assessment**: Evaluate practical constraints and requirements
6. **Impact Evaluation**: Assess potential scientific and societal implications
7. **Collaboration Planning**: Identify interdisciplinary partnerships

## Your Communication Style:
- Start with provocative "what if" questions
- Present multiple creative hypotheses with reasoning patterns
- Explain interdisciplinary connections and analogies
- Suggest specific experimental tests and validation approaches
- Acknowledge uncertainty while maintaining scientific rigor
- Encourage exploration of unconventional ideas
- Connect insights to broader scientific and societal contexts

## Collaboration Guidelines:
- **With Physics Experts**: Provide creative alternatives and novel perspectives
- **In Research Teams**: Focus on innovation and breakthrough potential
- **With Interdisciplinary Partners**: Bridge concepts and methodologies
- **In Debates**: Present constructive alternatives and creative solutions
- **In Teaching**: Demonstrate advanced creative scientific thinking"""

        # Add creativity and exploration customizations
        if self.creativity_level == "bold":
            base_message += "\n\n## Current Mode: BOLD CREATIVITY - PARADIGM SHIFTING\n- Challenge fundamental assumptions and established paradigms\n- Propose revolutionary concepts that could transform physics\n- Explore highly speculative but scientifically grounded ideas\n- Push the absolute boundaries of current understanding\n- Consider paradigm-shifting implications and applications"
        
        if self.exploration_scope == "interdisciplinary":
            base_message += "\n\n## Enhanced Interdisciplinary Focus:\n- Actively seek connections across all scientific disciplines\n- Identify unexpected analogies and structural similarities\n- Propose novel interdisciplinary research collaborations\n- Synthesize insights from biology, chemistry, mathematics, computer science, and beyond\n- Consider applications and implications across multiple fields"
        
        if self.enable_interdisciplinary:
            base_message += "\n\n## Interdisciplinary Integration Enabled:\n- Access to comprehensive interdisciplinary knowledge base\n- Ability to identify and explore cross-field connections\n- Enhanced pattern recognition across disciplines\n- Creative synthesis of multi-domain insights"
        
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
            
            # Get the user's latest message for logging
            user_message = state["messages"][-1] if state["messages"] else None
            
            response = llm_with_tools.invoke(messages)
            
            # Log the hypothesis generator activity
            if user_message:
                import asyncio
                try:
                    # Determine event type based on message content
                    message_content = user_message.content.lower()
                    if any(word in message_content for word in ['generate', 'hypothesis', 'hypotheses']):
                        event_type = "hypothesis_generation"
                    elif any(word in message_content for word in ['gap', 'research gap', 'missing']):
                        event_type = "research_gap_identification"
                    elif any(word in message_content for word in ['alternative', 'different approach', 'other way']):
                        event_type = "alternative_approach_proposal"
                    elif any(word in message_content for word in ['experiment', 'test', 'design']):
                        event_type = "experimental_design"
                    else:
                        event_type = "creative_thinking"
                    
                    # Log the event
                    asyncio.create_task(self.knowledge_api.log_event(
                        source="hypothesis_generator",
                        event_type=event_type,
                        payload={
                            "user_query": user_message.content,
                            "response_length": len(response.content),
                            "creativity_level": self.creativity_level,
                            "exploration_scope": self.exploration_scope,
                            "risk_tolerance": self.risk_tolerance,
                            "tools_used": bool(response.tool_calls) if hasattr(response, 'tool_calls') else False
                        }
                    ))
                except Exception as e:
                    print(f"⚠️ Event logging failed: {e}")
            
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
        if hasattr(self, 'memory_enabled') and self.memory_enabled and self.memory:
            return builder.compile(checkpointer=self.memory)
        else:
            return builder.compile()
    
    def generate_hypotheses(self, 
                          topic: str, 
                          context: str = "",
                          num_hypotheses: int = 3,
                          thread_id: Optional[str] = None,
                          session_id: Optional[str] = None) -> str:
        """Generate creative hypotheses for a physics topic.
        
        Args:
            topic: Physics topic or problem
            context: Additional context or constraints
            num_hypotheses: Number of hypotheses to generate
            thread_id: Optional thread ID for memory
            session_id: Optional session ID for collaboration tracking
            
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

For each hypothesis, please format your response as:
**Hypothesis N:** [Clear statement of the hypothesis]
**Reasoning:** [Explanation of the reasoning]
**Testing:** [How it could be tested]
**Impact:** [Potential impact if proven correct]
"""
        
        result = self.chat(prompt, thread_id=thread_id)
        
        # Parse and create hypotheses in the database
        import asyncio
        import re
        try:
            # Extract individual hypotheses from the response
            hypothesis_pattern = r'\*\*Hypothesis \d+:\*\*\s*(.+?)(?=\*\*Reasoning:|$)'
            hypotheses = re.findall(hypothesis_pattern, result, re.DOTALL)
            
            # Create each hypothesis in the database
            for i, hypothesis_text in enumerate(hypotheses):
                if hypothesis_text.strip():
                    # Determine confidence based on creativity level
                    confidence_map = {
                        "conservative": 0.7,
                        "moderate": 0.6,
                        "high": 0.5,
                        "bold": 0.4
                    }
                    initial_confidence = confidence_map.get(self.creativity_level, 0.5)
                    
                    # Create hypothesis in database
                    asyncio.create_task(self.knowledge_api.propose_hypothesis(
                        statement=hypothesis_text.strip(),
                        created_by="hypothesis_generator",
                        thread_id=thread_id,
                        session_id=session_id,
                        initial_confidence=initial_confidence,
                        domain=self._extract_domain_from_topic(topic)
                    ))
                    
        except Exception as e:
            print(f"⚠️ Hypothesis database creation failed: {e}")
        
        return result
    
    def _extract_domain_from_topic(self, topic: str) -> str:
        """Extract physics domain from topic string."""
        topic_lower = topic.lower()
        
        # Domain keywords mapping
        domain_keywords = {
            "quantum": ["quantum", "qubit", "entanglement", "superposition", "wave function"],
            "classical": ["classical", "newton", "mechanics", "motion", "force"],
            "thermodynamics": ["thermodynamics", "heat", "temperature", "entropy", "energy"],
            "electromagnetism": ["electromagnetic", "electric", "magnetic", "field", "maxwell"],
            "relativity": ["relativity", "spacetime", "einstein", "lorentz", "minkowski"],
            "particle": ["particle", "quark", "lepton", "boson", "standard model"],
            "astrophysics": ["astrophysics", "cosmology", "star", "galaxy", "universe"],
            "condensed_matter": ["condensed matter", "solid state", "crystal", "material"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                return domain
        
        return "general"
    
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

    def analyze_research_gaps(self, 
                            field: str, 
                            current_knowledge: str = "",
                            gap_types: Optional[List[ResearchGapType]] = None,
                            thread_id: Optional[str] = None) -> List[ResearchGap]:
        """
        Perform comprehensive research gap analysis using systematic framework.
        
        Args:
            field: The field to analyze (e.g., "quantum_mechanics", "thermodynamics")
            current_knowledge: Current state of knowledge in the field
            gap_types: Specific types of gaps to focus on (optional)
            thread_id: Thread ID for conversation context
            
        Returns:
            List of identified research gaps with detailed analysis
        """
        if gap_types is None:
            gap_types = list(ResearchGapType)
        
        identified_gaps = []
        
        # Get related concepts from knowledge graph
        related_concepts = []
        if self.knowledge_graph:
            domain_concepts = self.knowledge_graph.get_concepts_by_domain(field)
            related_concepts = [concept.name for concept in domain_concepts]
        
        # Analyze each gap type
        for gap_type in gap_types:
            framework = self.gap_analysis_framework[gap_type]
            
            # Use creative reasoning patterns to identify gaps
            for pattern in self.creative_patterns:
                pattern_info = self.creative_patterns[pattern]
                
                # Apply pattern-specific analysis
                if self._pattern_applicable_to_gap(pattern, gap_type):
                    gap_description = self._generate_gap_description(
                        field, gap_type, pattern, current_knowledge
                    )
                    
                    if gap_description:
                        gap = ResearchGap(
                            gap_type=gap_type,
                            description=gap_description,
                            field=field,
                            related_concepts=related_concepts,
                            potential_impact=self._assess_gap_impact(gap_description, field),
                            feasibility_score=self._assess_gap_feasibility(gap_description, field),
                            required_resources=self._identify_required_resources(gap_description),
                            timeline_estimate=self._estimate_timeline(gap_description),
                            interdisciplinary_connections=self._identify_interdisciplinary_connections(gap_description)
                        )
                        identified_gaps.append(gap)
        
        # Remove duplicates and rank by impact and feasibility
        unique_gaps = self._deduplicate_gaps(identified_gaps)
        ranked_gaps = self._rank_gaps_by_potential(unique_gaps)
        
        return ranked_gaps[:10]  # Return top 10 gaps
    
    def discover_interdisciplinary_connections(self, 
                                             physics_concept: str,
                                             target_fields: Optional[List[str]] = None) -> List[InterdisciplinaryConnection]:
        """
        Discover potential interdisciplinary connections for a physics concept.
        
        Args:
            physics_concept: The physics concept to analyze
            target_fields: Specific fields to explore (optional)
            
        Returns:
            List of interdisciplinary connections with confidence scores
        """
        if target_fields is None:
            target_fields = list(self.interdisciplinary_fields.keys())
        
        connections = []
        
        for field in target_fields:
            field_info = self.interdisciplinary_fields[field]
            
            # Check for shared concepts
            shared_concepts = field_info.get("shared_concepts", [])
            if any(concept.lower() in physics_concept.lower() for concept in shared_concepts):
                connection_type = "shared_concept"
                confidence = 0.8
            else:
                # Look for analogies and structural similarities
                connection_type = "analogy_based"
                confidence = self._assess_analogy_strength(physics_concept, field_info)
            
            if confidence > 0.5:  # Threshold for meaningful connections
                connection = InterdisciplinaryConnection(
                    source_field="physics",
                    target_field=field,
                    connection_type=connection_type,
                    description=self._generate_connection_description(physics_concept, field, field_info),
                    potential_applications=self._identify_potential_applications(physics_concept, field_info),
                    confidence_score=confidence
                )
                connections.append(connection)
        
        # Sort by confidence score
        connections.sort(key=lambda x: x.confidence_score, reverse=True)
        return connections
    
    def generate_creative_hypotheses(self, 
                                   topic: str,
                                   context: str = "",
                                   reasoning_patterns: Optional[List[CreativeReasoningPattern]] = None,
                                   num_hypotheses: int = 5,
                                   thread_id: Optional[str] = None) -> List[CreativeHypothesis]:
        """
        Generate creative hypotheses using specific reasoning patterns.
        
        Args:
            topic: The topic or problem to generate hypotheses for
            context: Additional context or constraints
            reasoning_patterns: Specific patterns to use (optional)
            num_hypotheses: Number of hypotheses to generate
            thread_id: Thread ID for conversation context
            
        Returns:
            List of creative hypotheses with metadata
        """
        if reasoning_patterns is None:
            reasoning_patterns = list(CreativeReasoningPattern)
        
        hypotheses = []
        
        for pattern in reasoning_patterns:
            pattern_info = self.creative_patterns[pattern]
            
            # Generate hypothesis using this pattern
            hypothesis_text = self._apply_creative_pattern(topic, pattern, context)
            
            if hypothesis_text:
                # Assess hypothesis quality
                testability = self._assess_testability(hypothesis_text)
                novelty = self._assess_novelty(hypothesis_text, topic)
                confidence = self._assess_confidence(hypothesis_text, pattern)
                
                # Identify interdisciplinary connections
                interdisciplinary_connections = self._identify_hypothesis_connections(hypothesis_text)
                
                # Propose experimental approaches
                experimental_approaches = self._propose_experimental_approaches(hypothesis_text)
                
                # Identify potential implications
                implications = self._identify_implications(hypothesis_text, topic)
                
                # Get related concepts from knowledge graph
                related_concepts = self._get_related_concepts(hypothesis_text)
                
                # Identify research gaps addressed
                gaps_addressed = self._identify_gaps_addressed(hypothesis_text)
                
                hypothesis = CreativeHypothesis(
                    hypothesis=hypothesis_text,
                    reasoning_pattern=pattern,
                    confidence_level=confidence,
                    testability_score=testability,
                    novelty_score=novelty,
                    interdisciplinary_connections=interdisciplinary_connections,
                    experimental_approaches=experimental_approaches,
                    potential_implications=implications,
                    related_concepts=related_concepts,
                    research_gaps_addressed=gaps_addressed
                )
                hypotheses.append(hypothesis)
        
        # Sort by overall quality score
        hypotheses.sort(key=lambda h: h.confidence_level * h.testability_score * h.novelty_score, reverse=True)
        return hypotheses[:num_hypotheses]
    
    def synthesize_interdisciplinary_insights(self, 
                                            problem: str,
                                            fields: List[str],
                                            thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesize insights from multiple disciplines to address a physics problem.
        
        Args:
            problem: The physics problem to address
            fields: List of fields to draw insights from
            thread_id: Thread ID for conversation context
            
        Returns:
            Dictionary containing synthesized insights and approaches
        """
        synthesis_result = {
            "problem": problem,
            "fields_analyzed": fields,
            "cross_field_insights": [],
            "synthetic_approaches": [],
            "novel_hypotheses": [],
            "experimental_designs": [],
            "collaboration_opportunities": []
        }
        
        # Analyze each field for relevant insights
        field_insights = {}
        for field in fields:
            field_info = self.interdisciplinary_fields.get(field, {})
            insights = self._extract_field_insights(problem, field, field_info)
            field_insights[field] = insights
        
        # Find cross-field patterns and connections
        cross_patterns = self._identify_cross_field_patterns(field_insights)
        synthesis_result["cross_field_insights"] = cross_patterns
        
        # Generate synthetic approaches
        synthetic_approaches = self._generate_synthetic_approaches(problem, field_insights)
        synthesis_result["synthetic_approaches"] = synthetic_approaches
        
        # Create novel hypotheses from synthesis
        novel_hypotheses = self._create_synthesis_hypotheses(problem, cross_patterns)
        synthesis_result["novel_hypotheses"] = novel_hypotheses
        
        # Design interdisciplinary experiments
        experimental_designs = self._design_interdisciplinary_experiments(problem, field_insights)
        synthesis_result["experimental_designs"] = experimental_designs
        
        # Identify collaboration opportunities
        collaborations = self._identify_collaboration_opportunities(problem, fields)
        synthesis_result["collaboration_opportunities"] = collaborations
        
        return synthesis_result
    
    def evaluate_hypothesis_portfolio(self, 
                                    hypotheses: List[CreativeHypothesis],
                                    criteria: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Evaluate a portfolio of hypotheses for diversity, quality, and potential.
        
        Args:
            hypotheses: List of hypotheses to evaluate
            criteria: Evaluation criteria weights (optional)
            
        Returns:
            Portfolio evaluation results
        """
        if criteria is None:
            criteria = {
                "novelty": 0.3,
                "testability": 0.25,
                "confidence": 0.2,
                "interdisciplinary_potential": 0.15,
                "impact_potential": 0.1
            }
        
        evaluation = {
            "total_hypotheses": len(hypotheses),
            "pattern_diversity": self._calculate_pattern_diversity(hypotheses),
            "quality_distribution": self._analyze_quality_distribution(hypotheses),
            "interdisciplinary_coverage": self._assess_interdisciplinary_coverage(hypotheses),
            "risk_profile": self._analyze_risk_profile(hypotheses),
            "top_hypotheses": [],
            "recommendations": []
        }
        
        # Calculate composite scores
        for hypothesis in hypotheses:
            composite_score = (
                hypothesis.novelty_score * criteria["novelty"] +
                hypothesis.testability_score * criteria["testability"] +
                hypothesis.confidence_level * criteria["confidence"] +
                len(hypothesis.interdisciplinary_connections) * 0.1 * criteria["interdisciplinary_potential"] +
                len(hypothesis.potential_implications) * 0.1 * criteria["impact_potential"]
            )
            hypothesis.composite_score = composite_score
        
        # Identify top hypotheses
        top_hypotheses = sorted(hypotheses, key=lambda h: h.composite_score, reverse=True)[:5]
        evaluation["top_hypotheses"] = [
            {
                "hypothesis": h.hypothesis,
                "pattern": h.reasoning_pattern.value,
                "score": h.composite_score,
                "strengths": self._identify_hypothesis_strengths(h),
                "weaknesses": self._identify_hypothesis_weaknesses(h)
            }
            for h in top_hypotheses
        ]
        
        # Generate recommendations
        recommendations = self._generate_portfolio_recommendations(hypotheses, evaluation)
        evaluation["recommendations"] = recommendations
        
        return evaluation
    
    # Helper methods for Phase 3 enhancements
    
    def _pattern_applicable_to_gap(self, pattern: CreativeReasoningPattern, gap_type: ResearchGapType) -> bool:
        """Check if a creative reasoning pattern is applicable to a specific gap type."""
        applicability_map = {
            CreativeReasoningPattern.ANALOGY_BASED: [ResearchGapType.INTERDISCIPLINARY_GAP, ResearchGapType.METHODOLOGICAL_GAP],
            CreativeReasoningPattern.INVERSION_THINKING: [ResearchGapType.THEORETICAL_GAP, ResearchGapType.EXPERIMENTAL_GAP],
            CreativeReasoningPattern.SCALE_BRIDGING: [ResearchGapType.SCALE_GAP, ResearchGapType.THEORETICAL_GAP],
            CreativeReasoningPattern.CONSTRAINT_REMOVAL: [ResearchGapType.THEORETICAL_GAP, ResearchGapType.METHODOLOGICAL_GAP],
            CreativeReasoningPattern.INTERDISCIPLINARY_FUSION: [ResearchGapType.INTERDISCIPLINARY_GAP, ResearchGapType.METHODOLOGICAL_GAP],
            CreativeReasoningPattern.PATTERN_EXTRAPOLATION: [ResearchGapType.THEORETICAL_GAP, ResearchGapType.SCALE_GAP],
            CreativeReasoningPattern.SYMMETRY_BREAKING: [ResearchGapType.THEORETICAL_GAP, ResearchGapType.EXPERIMENTAL_GAP],
            CreativeReasoningPattern.EMERGENT_PROPERTIES: [ResearchGapType.SCALE_GAP, ResearchGapType.THEORETICAL_GAP]
        }
        
        return gap_type in applicability_map.get(pattern, [])
    
    def _generate_gap_description(self, field: str, gap_type: ResearchGapType, 
                                pattern: CreativeReasoningPattern, current_knowledge: str) -> str:
        """Generate a description of a research gap using a specific reasoning pattern."""
        pattern_info = self.creative_patterns[pattern]
        gap_framework = self.gap_analysis_framework[gap_type]
        
        # This would typically involve more sophisticated analysis
        # For now, return a template-based description
        return f"In {field}, applying {pattern.value} reasoning reveals a {gap_type.value} where {pattern_info['description']} could identify unexplored opportunities in {gap_framework['description']}."
    
    def _assess_gap_impact(self, gap_description: str, field: str) -> str:
        """Assess the potential impact of addressing a research gap."""
        # Simplified impact assessment
        if "paradigm" in gap_description.lower() or "fundamental" in gap_description.lower():
            return "High - Potential paradigm shift"
        elif "novel" in gap_description.lower() or "breakthrough" in gap_description.lower():
            return "Medium-High - Significant advancement"
        else:
            return "Medium - Incremental progress"
    
    def _assess_gap_feasibility(self, gap_description: str, field: str) -> float:
        """Assess the feasibility of addressing a research gap."""
        # Simplified feasibility assessment (0.0 to 1.0)
        if "theoretical" in gap_description.lower():
            return 0.7  # Theoretical gaps often more feasible
        elif "experimental" in gap_description.lower():
            return 0.5  # Experimental gaps may require resources
        elif "technological" in gap_description.lower():
            return 0.3  # Technological gaps may be challenging
        else:
            return 0.6  # Default moderate feasibility
    
    def _identify_required_resources(self, gap_description: str) -> List[str]:
        """Identify resources required to address a research gap."""
        resources = []
        
        if "experimental" in gap_description.lower():
            resources.extend(["laboratory equipment", "experimental setup", "measurement tools"])
        if "theoretical" in gap_description.lower():
            resources.extend(["computational resources", "mathematical tools", "theoretical frameworks"])
        if "interdisciplinary" in gap_description.lower():
            resources.extend(["collaborative partnerships", "cross-field expertise", "communication platforms"])
        if "computational" in gap_description.lower():
            resources.extend(["high-performance computing", "software development", "data storage"])
        
        return resources if resources else ["research funding", "time", "expertise"]
    
    def _estimate_timeline(self, gap_description: str) -> str:
        """Estimate timeline for addressing a research gap."""
        if "fundamental" in gap_description.lower() or "paradigm" in gap_description.lower():
            return "5-10 years"
        elif "experimental" in gap_description.lower():
            return "2-5 years"
        elif "theoretical" in gap_description.lower():
            return "1-3 years"
        else:
            return "2-4 years"
    
    def _identify_interdisciplinary_connections(self, gap_description: str) -> List[str]:
        """Identify potential interdisciplinary connections for a research gap."""
        connections = []
        
        for field, field_info in self.interdisciplinary_fields.items():
            shared_concepts = field_info.get("shared_concepts", [])
            if any(concept.lower() in gap_description.lower() for concept in shared_concepts):
                connections.append(field)
        
        return connections
    
    def _deduplicate_gaps(self, gaps: List[ResearchGap]) -> List[ResearchGap]:
        """Remove duplicate research gaps."""
        unique_gaps = []
        seen_descriptions = set()
        
        for gap in gaps:
            if gap.description not in seen_descriptions:
                unique_gaps.append(gap)
                seen_descriptions.add(gap.description)
        
        return unique_gaps
    
    def _rank_gaps_by_potential(self, gaps: List[ResearchGap]) -> List[ResearchGap]:
        """Rank research gaps by their potential impact and feasibility."""
        def gap_score(gap):
            impact_score = {"High": 1.0, "Medium-High": 0.8, "Medium": 0.6, "Low": 0.4}.get(
                gap.potential_impact.split(" - ")[0], 0.5
            )
            return impact_score * gap.feasibility_score
        
        return sorted(gaps, key=gap_score, reverse=True)
    
    def _assess_analogy_strength(self, physics_concept: str, field_info: Dict[str, Any]) -> float:
        """Assess the strength of analogy between physics concept and another field."""
        # Simplified analogy assessment
        shared_concepts = field_info.get("shared_concepts", [])
        research_areas = field_info.get("research_areas", [])
        
        concept_matches = sum(1 for concept in shared_concepts if concept.lower() in physics_concept.lower())
        area_matches = sum(1 for area in research_areas if area.lower() in physics_concept.lower())
        
        total_matches = concept_matches + area_matches
        max_possible = len(shared_concepts) + len(research_areas)
        
        return min(total_matches / max(max_possible, 1), 1.0) if max_possible > 0 else 0.3
    
    def _generate_connection_description(self, physics_concept: str, field: str, field_info: Dict[str, Any]) -> str:
        """Generate a description of an interdisciplinary connection."""
        connection_types = field_info.get("connection_types", [])
        shared_concepts = field_info.get("shared_concepts", [])
        
        primary_connection = connection_types[0] if connection_types else f"physics-{field} interface"
        relevant_concepts = [c for c in shared_concepts if c.lower() in physics_concept.lower()]
        
        if relevant_concepts:
            return f"The physics concept of {physics_concept} connects to {field} through {primary_connection}, sharing fundamental concepts like {', '.join(relevant_concepts[:3])}."
        else:
            return f"The physics concept of {physics_concept} could be explored through {primary_connection}, potentially revealing new insights in {field}."
    
    def _identify_potential_applications(self, physics_concept: str, field_info: Dict[str, Any]) -> List[str]:
        """Identify potential applications of a physics concept in another field."""
        physics_applications = field_info.get("physics_applications", [])
        research_areas = field_info.get("research_areas", [])
        
        # Return a subset of relevant applications
        return physics_applications[:3] + research_areas[:2]
    
    def _apply_creative_pattern(self, topic: str, pattern: CreativeReasoningPattern, context: str) -> str:
        """Apply a specific creative reasoning pattern to generate a hypothesis."""
        pattern_info = self.creative_patterns[pattern]
        
        # This is a simplified implementation - in practice, this would involve
        # more sophisticated pattern application logic
        if pattern == CreativeReasoningPattern.ANALOGY_BASED:
            return f"What if {topic} behaves analogously to {pattern_info['examples'][0]}? This could suggest new mechanisms or principles."
        elif pattern == CreativeReasoningPattern.INVERSION_THINKING:
            return f"What if the conventional understanding of {topic} is inverted? Consider the opposite scenario where {pattern_info['examples'][0]} applies."
        elif pattern == CreativeReasoningPattern.SCALE_BRIDGING:
            return f"What if {topic} exhibits scale-invariant properties similar to {pattern_info['examples'][0]}? This could bridge quantum and classical regimes."
        elif pattern == CreativeReasoningPattern.INTERDISCIPLINARY_FUSION:
            return f"What if {topic} incorporates principles from {pattern_info['examples'][0]}? This interdisciplinary approach could reveal new phenomena."
        else:
            return f"Applying {pattern.value} thinking to {topic} suggests exploring {pattern_info['description']} in this context."
    
    def _assess_testability(self, hypothesis: str) -> float:
        """Assess the testability of a hypothesis."""
        testability_indicators = ["measurable", "observable", "experiment", "test", "detect", "measure"]
        score = sum(1 for indicator in testability_indicators if indicator in hypothesis.lower())
        return min(score / len(testability_indicators), 1.0)
    
    def _assess_novelty(self, hypothesis: str, topic: str) -> float:
        """Assess the novelty of a hypothesis."""
        novelty_indicators = ["novel", "new", "unprecedented", "unexplored", "innovative", "breakthrough"]
        score = sum(1 for indicator in novelty_indicators if indicator in hypothesis.lower())
        return min(score / len(novelty_indicators) + 0.5, 1.0)  # Base novelty of 0.5
    
    def _assess_confidence(self, hypothesis: str, pattern: CreativeReasoningPattern) -> float:
        """Assess confidence in a hypothesis based on the reasoning pattern used."""
        pattern_confidence = {
            CreativeReasoningPattern.ANALOGY_BASED: 0.7,
            CreativeReasoningPattern.PATTERN_EXTRAPOLATION: 0.8,
            CreativeReasoningPattern.INTERDISCIPLINARY_FUSION: 0.6,
            CreativeReasoningPattern.INVERSION_THINKING: 0.5,
            CreativeReasoningPattern.CONSTRAINT_REMOVAL: 0.4,
            CreativeReasoningPattern.SCALE_BRIDGING: 0.7,
            CreativeReasoningPattern.SYMMETRY_BREAKING: 0.6,
            CreativeReasoningPattern.EMERGENT_PROPERTIES: 0.5
        }
        return pattern_confidence.get(pattern, 0.6)
    
    def _identify_hypothesis_connections(self, hypothesis: str) -> List[str]:
        """Identify interdisciplinary connections for a hypothesis."""
        connections = []
        for field, field_info in self.interdisciplinary_fields.items():
            shared_concepts = field_info.get("shared_concepts", [])
            if any(concept.lower() in hypothesis.lower() for concept in shared_concepts):
                connections.append(field)
        return connections
    
    def _propose_experimental_approaches(self, hypothesis: str) -> List[str]:
        """Propose experimental approaches to test a hypothesis."""
        approaches = []
        
        if "quantum" in hypothesis.lower():
            approaches.extend(["quantum interference experiments", "entanglement measurements", "quantum state tomography"])
        if "thermal" in hypothesis.lower() or "temperature" in hypothesis.lower():
            approaches.extend(["calorimetry", "thermal imaging", "temperature-dependent measurements"])
        if "magnetic" in hypothesis.lower():
            approaches.extend(["magnetic field measurements", "magnetometry", "magnetic resonance"])
        if "optical" in hypothesis.lower() or "light" in hypothesis.lower():
            approaches.extend(["spectroscopy", "interferometry", "optical microscopy"])
        
        return approaches[:3] if approaches else ["controlled experiments", "measurement protocols", "data analysis"]
    
    def _identify_implications(self, hypothesis: str, topic: str) -> List[str]:
        """Identify potential implications of a hypothesis."""
        implications = []
        
        if "energy" in hypothesis.lower():
            implications.extend(["energy efficiency improvements", "new energy sources", "conservation implications"])
        if "information" in hypothesis.lower():
            implications.extend(["information processing advances", "communication improvements", "computational applications"])
        if "material" in hypothesis.lower():
            implications.extend(["new materials development", "material properties control", "manufacturing innovations"])
        if "quantum" in hypothesis.lower():
            implications.extend(["quantum technology applications", "fundamental physics insights", "computational advantages"])
        
        return implications[:3] if implications else ["scientific understanding", "technological applications", "theoretical insights"]
    
    def _get_related_concepts(self, hypothesis: str) -> List[str]:
        """Get related concepts from the knowledge graph."""
        if not self.knowledge_graph:
            return []
        
        # Extract key terms from hypothesis
        key_terms = [term.strip() for term in hypothesis.lower().split() if len(term) > 4]
        
        related_concepts = []
        for concept in self.knowledge_graph.concepts.values():
            if any(term in concept.name.lower() for term in key_terms):
                related_concepts.append(concept.name)
        
        return related_concepts[:5]
    
    def _identify_gaps_addressed(self, hypothesis: str) -> List[str]:
        """Identify research gaps that a hypothesis addresses."""
        gaps = []
        
        if "unexplored" in hypothesis.lower() or "missing" in hypothesis.lower():
            gaps.append("theoretical_gap")
        if "untested" in hypothesis.lower() or "experimental" in hypothesis.lower():
            gaps.append("experimental_gap")
        if "interdisciplinary" in hypothesis.lower() or "cross-field" in hypothesis.lower():
            gaps.append("interdisciplinary_gap")
        if "scale" in hypothesis.lower() or "multi-scale" in hypothesis.lower():
            gaps.append("scale_gap")
        
        return gaps
    
    # Additional helper methods for synthesis and evaluation
    
    def _extract_field_insights(self, problem: str, field: str, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant insights from a field for a physics problem."""
        return {
            "field": field,
            "relevant_concepts": field_info.get("shared_concepts", []),
            "applicable_methods": field_info.get("physics_applications", []),
            "research_areas": field_info.get("research_areas", []),
            "connection_strength": self._assess_analogy_strength(problem, field_info)
        }
    
    def _identify_cross_field_patterns(self, field_insights: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify patterns that appear across multiple fields."""
        patterns = []
        
        # Find common concepts across fields
        all_concepts = []
        for field, insights in field_insights.items():
            all_concepts.extend(insights.get("relevant_concepts", []))
        
        from collections import Counter
        concept_counts = Counter(all_concepts)
        
        for concept, count in concept_counts.items():
            if count > 1:  # Concept appears in multiple fields
                fields_with_concept = [
                    field for field, insights in field_insights.items()
                    if concept in insights.get("relevant_concepts", [])
                ]
                patterns.append({
                    "pattern": concept,
                    "fields": fields_with_concept,
                    "frequency": count,
                    "type": "shared_concept"
                })
        
        return patterns
    
    def _generate_synthetic_approaches(self, problem: str, field_insights: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate synthetic approaches combining insights from multiple fields."""
        approaches = []
        
        # Combine methods from different fields
        all_methods = []
        for field, insights in field_insights.items():
            methods = insights.get("applicable_methods", [])
            all_methods.extend([(method, field) for method in methods])
        
        # Create synthetic approaches
        for i, (method1, field1) in enumerate(all_methods):
            for method2, field2 in all_methods[i+1:]:
                if field1 != field2:
                    approaches.append(f"Combine {method1} from {field1} with {method2} from {field2}")
        
        return approaches[:5]  # Return top 5 synthetic approaches
    
    def _create_synthesis_hypotheses(self, problem: str, cross_patterns: List[Dict[str, Any]]) -> List[str]:
        """Create novel hypotheses from cross-field synthesis."""
        hypotheses = []
        
        for pattern in cross_patterns:
            if pattern["type"] == "shared_concept":
                concept = pattern["pattern"]
                fields = pattern["fields"]
                hypothesis = f"What if {problem} exhibits {concept}-like behavior as seen in {' and '.join(fields)}? This could suggest a universal principle governing {concept} across disciplines."
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _design_interdisciplinary_experiments(self, problem: str, field_insights: Dict[str, Dict[str, Any]]) -> List[str]:
        """Design experiments that incorporate insights from multiple fields."""
        experiments = []
        
        for field, insights in field_insights.items():
            methods = insights.get("applicable_methods", [])
            for method in methods[:2]:  # Use top 2 methods per field
                experiment = f"Design {method}-based experiment for {problem} incorporating {field} principles"
                experiments.append(experiment)
        
        return experiments
    
    def _identify_collaboration_opportunities(self, problem: str, fields: List[str]) -> List[str]:
        """Identify specific collaboration opportunities between fields."""
        collaborations = []
        
        for i, field1 in enumerate(fields):
            for field2 in fields[i+1:]:
                collaboration = f"Physics-{field1}-{field2} collaboration on {problem}"
                collaborations.append(collaboration)
        
        return collaborations
    
    def _calculate_pattern_diversity(self, hypotheses: List[CreativeHypothesis]) -> float:
        """Calculate diversity of reasoning patterns in hypothesis portfolio."""
        patterns = [h.reasoning_pattern for h in hypotheses]
        unique_patterns = set(patterns)
        return len(unique_patterns) / len(CreativeReasoningPattern) if patterns else 0.0
    
    def _analyze_quality_distribution(self, hypotheses: List[CreativeHypothesis]) -> Dict[str, float]:
        """Analyze the distribution of quality metrics across hypotheses."""
        if not hypotheses:
            return {"mean_confidence": 0.0, "mean_testability": 0.0, "mean_novelty": 0.0}
        
        return {
            "mean_confidence": np.mean([h.confidence_level for h in hypotheses]),
            "mean_testability": np.mean([h.testability_score for h in hypotheses]),
            "mean_novelty": np.mean([h.novelty_score for h in hypotheses])
        }
    
    def _assess_interdisciplinary_coverage(self, hypotheses: List[CreativeHypothesis]) -> Dict[str, int]:
        """Assess coverage of interdisciplinary connections."""
        all_connections = []
        for h in hypotheses:
            all_connections.extend(h.interdisciplinary_connections)
        
        from collections import Counter
        return dict(Counter(all_connections))
    
    def _analyze_risk_profile(self, hypotheses: List[CreativeHypothesis]) -> Dict[str, int]:
        """Analyze risk profile of hypothesis portfolio."""
        risk_categories = {"low": 0, "medium": 0, "high": 0}
        
        for h in hypotheses:
            if h.confidence_level > 0.7:
                risk_categories["low"] += 1
            elif h.confidence_level > 0.4:
                risk_categories["medium"] += 1
            else:
                risk_categories["high"] += 1
        
        return risk_categories
    
    def _identify_hypothesis_strengths(self, hypothesis: CreativeHypothesis) -> List[str]:
        """Identify strengths of a hypothesis."""
        strengths = []
        
        if hypothesis.testability_score > 0.7:
            strengths.append("highly testable")
        if hypothesis.novelty_score > 0.7:
            strengths.append("highly novel")
        if hypothesis.confidence_level > 0.7:
            strengths.append("high confidence")
        if len(hypothesis.interdisciplinary_connections) > 2:
            strengths.append("strong interdisciplinary potential")
        if len(hypothesis.potential_implications) > 3:
            strengths.append("broad implications")
        
        return strengths
    
    def _identify_hypothesis_weaknesses(self, hypothesis: CreativeHypothesis) -> List[str]:
        """Identify weaknesses of a hypothesis."""
        weaknesses = []
        
        if hypothesis.testability_score < 0.4:
            weaknesses.append("low testability")
        if hypothesis.novelty_score < 0.4:
            weaknesses.append("limited novelty")
        if hypothesis.confidence_level < 0.4:
            weaknesses.append("low confidence")
        if len(hypothesis.interdisciplinary_connections) == 0:
            weaknesses.append("limited interdisciplinary connections")
        if len(hypothesis.experimental_approaches) < 2:
            weaknesses.append("few experimental approaches")
        
        return weaknesses
    
    def _generate_portfolio_recommendations(self, hypotheses: List[CreativeHypothesis], 
                                          evaluation: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving the hypothesis portfolio."""
        recommendations = []
        
        # Check pattern diversity
        if evaluation["pattern_diversity"] < 0.5:
            recommendations.append("Increase diversity of reasoning patterns")
        
        # Check quality distribution
        quality_dist = evaluation["quality_distribution"]
        if quality_dist["mean_testability"] < 0.6:
            recommendations.append("Focus on generating more testable hypotheses")
        if quality_dist["mean_novelty"] < 0.6:
            recommendations.append("Explore more novel and creative approaches")
        
        # Check interdisciplinary coverage
        interdisciplinary_coverage = evaluation["interdisciplinary_coverage"]
        if len(interdisciplinary_coverage) < 3:
            recommendations.append("Expand interdisciplinary connections")
        
        # Check risk profile
        risk_profile = evaluation["risk_profile"]
        if risk_profile["high"] > risk_profile["low"] + risk_profile["medium"]:
            recommendations.append("Balance high-risk hypotheses with safer alternatives")
        
        return recommendations


# Maintain backward compatibility
HypothesisGeneratorAgent = EnhancedHypothesisGeneratorAgent 