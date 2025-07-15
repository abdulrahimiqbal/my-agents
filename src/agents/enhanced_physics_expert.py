"""Enhanced Physics Expert Agent - Advanced physics agent with specialized domains and adaptive intelligence."""

import json
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from .base import BaseAgent
from ..tools.physics_calculator import get_physics_calculator_tools
from ..tools.physics_research import get_physics_research_tools
from ..tools.physics_constants import get_physics_constants_tools
from ..tools.unit_converter import get_unit_converter_tools
from ..database.knowledge_api import KnowledgeAPI


class PhysicsDomain(Enum):
    """Physics specialization domains."""
    GENERAL = "general"
    QUANTUM_MECHANICS = "quantum_mechanics"
    THERMODYNAMICS = "thermodynamics"
    ELECTROMAGNETISM = "electromagnetism"
    RELATIVITY = "relativity"
    MECHANICS = "mechanics"
    OPTICS = "optics"
    PARTICLE_PHYSICS = "particle_physics"
    CONDENSED_MATTER = "condensed_matter"
    ASTROPHYSICS = "astrophysics"


class DifficultyLevel(Enum):
    """Adaptive difficulty levels."""
    HIGH_SCHOOL = "high_school"
    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"
    RESEARCH = "research"
    EXPERT = "expert"


class PhysicsReasoningEngine:
    """Advanced physics reasoning and problem decomposition engine."""
    
    def __init__(self, domain: PhysicsDomain = PhysicsDomain.GENERAL):
        self.domain = domain
        self.reasoning_patterns = self._load_reasoning_patterns()
        self.problem_templates = self._load_problem_templates()
        
    def _load_reasoning_patterns(self) -> Dict[str, Any]:
        """Load domain-specific reasoning patterns."""
        patterns = {
            PhysicsDomain.GENERAL: {
                "key_principles": ["conservation_laws", "symmetry_principles", "dimensional_analysis"],
                "problem_types": ["general_physics", "multi_domain", "conceptual_problems"],
                "mathematical_tools": ["algebra", "calculus", "vector_analysis"],
                "common_mistakes": ["unit_confusion", "sign_errors", "conceptual_misunderstanding"]
            },
            PhysicsDomain.QUANTUM_MECHANICS: {
                "key_principles": ["wave-particle duality", "uncertainty principle", "superposition", "entanglement"],
                "problem_types": ["wave_function", "energy_levels", "tunneling", "measurement"],
                "mathematical_tools": ["schrodinger_equation", "operators", "probability_amplitudes"],
                "common_mistakes": ["classical_thinking", "measurement_confusion", "wavefunction_collapse"]
            },
            PhysicsDomain.THERMODYNAMICS: {
                "key_principles": ["conservation of energy", "entropy", "equilibrium", "statistical mechanics"],
                "problem_types": ["heat_engines", "phase_transitions", "gas_laws", "entropy_calculations"],
                "mathematical_tools": ["boltzmann_distribution", "partition_functions", "free_energy"],
                "common_mistakes": ["reversible_assumption", "equilibrium_confusion", "entropy_misconceptions"]
            },
            PhysicsDomain.ELECTROMAGNETISM: {
                "key_principles": ["maxwell_equations", "electromagnetic_induction", "wave_propagation"],
                "problem_types": ["electric_fields", "magnetic_fields", "circuits", "electromagnetic_waves"],
                "mathematical_tools": ["vector_calculus", "complex_analysis", "fourier_transforms"],
                "common_mistakes": ["field_direction", "boundary_conditions", "gauge_choice"]
            },
            PhysicsDomain.RELATIVITY: {
                "key_principles": ["spacetime", "equivalence_principle", "lorentz_invariance"],
                "problem_types": ["time_dilation", "length_contraction", "energy_momentum", "curvature"],
                "mathematical_tools": ["minkowski_space", "tensor_calculus", "geodesics"],
                "common_mistakes": ["absolute_time", "simultaneity", "coordinate_confusion"]
            },
            PhysicsDomain.MECHANICS: {
                "key_principles": ["newton_laws", "conservation_laws", "lagrangian_mechanics"],
                "problem_types": ["kinematics", "dynamics", "oscillations", "rotational_motion"],
                "mathematical_tools": ["differential_equations", "vector_analysis", "calculus_of_variations"],
                "common_mistakes": ["force_confusion", "reference_frames", "constraint_forces"]
            }
        }
        return patterns.get(self.domain, patterns[PhysicsDomain.GENERAL])
    
    def _load_problem_templates(self) -> Dict[str, Any]:
        """Load problem-solving templates for the domain."""
        return {
            "analysis_steps": [
                "identify_given_information",
                "determine_unknown_quantities",
                "select_relevant_principles",
                "set_up_equations",
                "solve_mathematically",
                "check_dimensional_consistency",
                "verify_physical_reasonableness",
                "interpret_results"
            ],
            "validation_checks": [
                "units_consistency",
                "order_of_magnitude",
                "limiting_cases",
                "symmetry_considerations",
                "conservation_laws"
            ]
        }
    
    def decompose_problem(self, problem: str, difficulty: DifficultyLevel) -> Dict[str, Any]:
        """Decompose a physics problem into structured components."""
        return {
            "problem_type": self._classify_problem_type(problem),
            "difficulty_assessment": difficulty.value,
            "key_concepts": self._extract_key_concepts(problem),
            "mathematical_requirements": self._assess_math_requirements(problem, difficulty),
            "solution_strategy": self._suggest_solution_strategy(problem, difficulty),
            "potential_pitfalls": self._identify_potential_pitfalls(problem),
            "learning_objectives": self._extract_learning_objectives(problem, difficulty)
        }
    
    def _classify_problem_type(self, problem: str) -> str:
        """Classify the type of physics problem."""
        # Simple keyword-based classification (would be enhanced with NLP)
        problem_lower = problem.lower()
        
        if any(word in problem_lower for word in ["quantum", "wave function", "probability", "uncertainty"]):
            return "quantum_mechanics"
        elif any(word in problem_lower for word in ["heat", "temperature", "entropy", "thermodynamic"]):
            return "thermodynamics"
        elif any(word in problem_lower for word in ["electric", "magnetic", "electromagnetic", "field"]):
            return "electromagnetism"
        elif any(word in problem_lower for word in ["relativity", "spacetime", "lorentz", "einstein"]):
            return "relativity"
        elif any(word in problem_lower for word in ["force", "motion", "acceleration", "velocity"]):
            return "mechanics"
        else:
            return "general_physics"
    
    def _extract_key_concepts(self, problem: str) -> List[str]:
        """Extract key physics concepts from the problem."""
        concepts = []
        reasoning = self.reasoning_patterns
        
        problem_lower = problem.lower()
        for concept in reasoning.get("key_principles", []):
            if any(word in problem_lower for word in concept.split("_")):
                concepts.append(concept)
        
        return concepts
    
    def _assess_math_requirements(self, problem: str, difficulty: DifficultyLevel) -> List[str]:
        """Assess mathematical requirements based on problem and difficulty."""
        base_requirements = ["algebra", "basic_calculus"]
        
        if difficulty in [DifficultyLevel.GRADUATE, DifficultyLevel.RESEARCH, DifficultyLevel.EXPERT]:
            base_requirements.extend(["advanced_calculus", "linear_algebra", "differential_equations"])
        
        # Add domain-specific math requirements
        reasoning = self.reasoning_patterns
        domain_math = reasoning.get("mathematical_tools", [])
        
        return base_requirements + domain_math
    
    def _suggest_solution_strategy(self, problem: str, difficulty: DifficultyLevel) -> Dict[str, Any]:
        """Suggest an appropriate solution strategy."""
        return {
            "approach": "systematic_analysis",
            "steps": self.problem_templates["analysis_steps"],
            "difficulty_adaptations": self._get_difficulty_adaptations(difficulty),
            "recommended_tools": self._recommend_tools(problem, difficulty)
        }
    
    def _identify_potential_pitfalls(self, problem: str) -> List[str]:
        """Identify potential pitfalls and common mistakes."""
        reasoning = self.reasoning_patterns
        return reasoning.get("common_mistakes", [])
    
    def _extract_learning_objectives(self, problem: str, difficulty: DifficultyLevel) -> List[str]:
        """Extract learning objectives based on problem and difficulty."""
        objectives = [
            "understand_underlying_physics_principles",
            "apply_mathematical_tools_correctly",
            "interpret_results_physically"
        ]
        
        if difficulty in [DifficultyLevel.GRADUATE, DifficultyLevel.RESEARCH, DifficultyLevel.EXPERT]:
            objectives.extend([
                "connect_to_advanced_concepts",
                "identify_research_implications",
                "evaluate_approximations_and_assumptions"
            ])
        
        return objectives
    
    def _get_difficulty_adaptations(self, difficulty: DifficultyLevel) -> Dict[str, Any]:
        """Get adaptations based on difficulty level."""
        adaptations = {
            DifficultyLevel.HIGH_SCHOOL: {
                "explanation_style": "intuitive_with_analogies",
                "math_level": "basic_algebra_and_trigonometry",
                "detail_level": "conceptual_understanding",
                "examples": "everyday_applications"
            },
            DifficultyLevel.UNDERGRADUATE: {
                "explanation_style": "systematic_with_derivations",
                "math_level": "calculus_and_basic_differential_equations",
                "detail_level": "mathematical_rigor",
                "examples": "textbook_problems"
            },
            DifficultyLevel.GRADUATE: {
                "explanation_style": "advanced_mathematical_treatment",
                "math_level": "advanced_mathematics",
                "detail_level": "theoretical_depth",
                "examples": "research_applications"
            },
            DifficultyLevel.RESEARCH: {
                "explanation_style": "cutting_edge_analysis",
                "math_level": "specialized_mathematics",
                "detail_level": "research_level_rigor",
                "examples": "current_research_problems"
            },
            DifficultyLevel.EXPERT: {
                "explanation_style": "expert_level_discourse",
                "math_level": "advanced_specialized_mathematics",
                "detail_level": "comprehensive_analysis",
                "examples": "frontier_research"
            }
        }
        return adaptations.get(difficulty, adaptations[DifficultyLevel.UNDERGRADUATE])
    
    def _recommend_tools(self, problem: str, difficulty: DifficultyLevel) -> List[str]:
        """Recommend appropriate tools for the problem."""
        tools = ["physics_calculator", "unit_converter", "physics_constants"]
        
        if difficulty in [DifficultyLevel.GRADUATE, DifficultyLevel.RESEARCH, DifficultyLevel.EXPERT]:
            tools.extend(["research_database", "simulation_tools", "advanced_calculators"])
        
        return tools


class EnhancedPhysicsExpertAgent(BaseAgent):
    """Enhanced physics expert agent with specialized domains and adaptive intelligence."""
    
    def __init__(self, 
                 specialization: PhysicsDomain = PhysicsDomain.GENERAL,
                 difficulty_level: DifficultyLevel = DifficultyLevel.UNDERGRADUATE,
                 adaptive_difficulty: bool = True,
                 memory_enabled: bool = True,
                 **kwargs):
        """Initialize the enhanced physics expert agent.
        
        Args:
            specialization: Physics domain specialization
            difficulty_level: Default difficulty level
            adaptive_difficulty: Whether to adapt difficulty based on user interaction
            memory_enabled: Whether to enable memory functionality
            **kwargs: Additional arguments for BaseAgent
        """
        self.specialization = specialization
        self.difficulty_level = difficulty_level
        self.adaptive_difficulty = adaptive_difficulty
        self.memory_enabled = memory_enabled
        
        # Initialize reasoning engine
        self.reasoning_engine = PhysicsReasoningEngine(specialization)
        
        # Initialize knowledge API
        self.knowledge_api = KnowledgeAPI()
        
        # Load specialized tools
        self.specialized_tools = self._load_specialized_tools()
        
        # Create enhanced system message
        self.system_message = self._create_enhanced_system_message()
        
        # Combine all tools
        all_tools = (
            get_physics_calculator_tools() +
            get_physics_research_tools() +
            get_physics_constants_tools() +
            get_unit_converter_tools() +
            self.specialized_tools
        )
        
        super().__init__(tools=all_tools, **kwargs)
    
    def _load_specialized_tools(self) -> List[BaseTool]:
        """Load tools specialized for the physics domain."""
        # This would load domain-specific tools
        # For now, return empty list - to be implemented with actual specialized tools
        return []
    
    def _create_enhanced_system_message(self) -> str:
        """Create an enhanced system message based on specialization and difficulty."""
        base_message = f"""You are an Enhanced PhysicsGPT, a world-class physics expert with specialized knowledge in {self.specialization.value.replace('_', ' ').title()} and adaptive teaching capabilities.

## Your Enhanced Capabilities:

### 🎯 Specialization: {self.specialization.value.replace('_', ' ').title()}
You have deep, specialized knowledge in this domain with access to:
- Advanced domain-specific mathematical tools
- Cutting-edge research and developments
- Specialized problem-solving techniques
- Domain-specific common pitfalls and misconceptions

### 📊 Adaptive Difficulty: {self.difficulty_level.value.replace('_', ' ').title()} Level
You automatically adapt your explanations to the appropriate level:
- **Explanation Style**: {self.reasoning_engine._get_difficulty_adaptations(self.difficulty_level)['explanation_style']}
- **Mathematical Level**: {self.reasoning_engine._get_difficulty_adaptations(self.difficulty_level)['math_level']}
- **Detail Level**: {self.reasoning_engine._get_difficulty_adaptations(self.difficulty_level)['detail_level']}

### 🧠 Advanced Reasoning Engine
You use sophisticated problem decomposition:
1. **Problem Classification**: Identify problem type and domain
2. **Concept Extraction**: Extract key physics principles
3. **Solution Strategy**: Develop systematic approach
4. **Pitfall Identification**: Anticipate common mistakes
5. **Learning Objectives**: Identify educational goals

### 🔧 Your Enhanced Process:
1. **Analyze**: Decompose the problem systematically
2. **Strategize**: Select optimal solution approach
3. **Execute**: Apply domain expertise with mathematical rigor
4. **Validate**: Check consistency and physical reasonableness
5. **Teach**: Explain with appropriate depth and clarity
6. **Connect**: Link to broader physics concepts

### 🎓 Teaching Philosophy:
- **Conceptual Understanding**: Build intuitive grasp of physics
- **Mathematical Rigor**: Maintain appropriate mathematical depth
- **Real-World Connections**: Connect to applications and research
- **Critical Thinking**: Encourage questioning and analysis
- **Progressive Learning**: Build from fundamentals to advanced concepts

### 🔬 Research Integration:
- Stay current with latest developments in your specialization
- Connect problems to ongoing research
- Identify research opportunities and gaps
- Suggest experimental approaches when relevant

## Communication Guidelines:
- Begin with the key physics principle or concept
- Use domain-specific terminology appropriately
- Provide step-by-step solutions with clear reasoning
- Include dimensional analysis and consistency checks
- Offer multiple perspectives when valuable
- Suggest follow-up questions or related concepts
- Adapt complexity to user's demonstrated understanding

Remember: You are not just solving problems - you are advancing physics understanding and inspiring scientific thinking!"""

        # Add domain-specific enhancements
        domain_specific = self._get_domain_specific_message()
        if domain_specific:
            base_message += f"\n\n### 🔬 Domain-Specific Expertise:\n{domain_specific}"
        
        return base_message
    
    def _get_domain_specific_message(self) -> str:
        """Get domain-specific system message enhancements."""
        domain_messages = {
            PhysicsDomain.QUANTUM_MECHANICS: """
- Master quantum mechanical principles: superposition, entanglement, measurement
- Expert in wave function analysis and probability interpretations
- Proficient with quantum operators, commutation relations, and time evolution
- Familiar with quantum computing, quantum information, and quantum technologies
- Skilled in addressing quantum paradoxes and interpretational questions""",
            
            PhysicsDomain.THERMODYNAMICS: """
- Expert in classical and statistical thermodynamics
- Proficient with ensembles, partition functions, and phase transitions
- Skilled in entropy analysis and the second law applications
- Familiar with non-equilibrium thermodynamics and transport phenomena
- Expert in connecting microscopic and macroscopic descriptions""",
            
            PhysicsDomain.ELECTROMAGNETISM: """
- Master of Maxwell's equations and electromagnetic theory
- Expert in electrostatics, magnetostatics, and electromagnetic waves
- Proficient with boundary value problems and Green's functions
- Skilled in antenna theory, waveguides, and electromagnetic compatibility
- Familiar with plasma physics and electromagnetic field applications""",
            
            PhysicsDomain.RELATIVITY: """
- Expert in special and general relativity
- Proficient with spacetime geometry and tensor calculus
- Skilled in solving Einstein field equations and geodesic problems
- Familiar with black holes, cosmology, and gravitational waves
- Expert in relativistic mechanics and field theory""",
            
            PhysicsDomain.MECHANICS: """
- Master of classical mechanics: Newtonian, Lagrangian, and Hamiltonian
- Expert in oscillations, waves, and nonlinear dynamics
- Proficient with rigid body dynamics and continuum mechanics
- Skilled in chaos theory and dynamical systems
- Familiar with fluid mechanics and elasticity theory"""
        }
        
        return domain_messages.get(self.specialization, "")
    
    async def solve_physics_problem(self, 
                                  problem: str, 
                                  context: str = "",
                                  requested_difficulty: Optional[DifficultyLevel] = None,
                                  thread_id: Optional[str] = None) -> str:
        """Solve a physics problem with enhanced capabilities.
        
        Args:
            problem: The physics problem to solve
            context: Additional context or constraints
            requested_difficulty: Specific difficulty level requested
            thread_id: Thread ID for conversation continuity
            
        Returns:
            Enhanced solution with reasoning and explanations
        """
        # Determine effective difficulty level
        effective_difficulty = requested_difficulty or self.difficulty_level
        
        # Decompose the problem using reasoning engine
        problem_analysis = self.reasoning_engine.decompose_problem(problem, effective_difficulty)
        
        # Create enhanced prompt
        enhanced_prompt = f"""
## Physics Problem Analysis

**Problem**: {problem}
{f"**Context**: {context}" if context else ""}

**Problem Classification**: {problem_analysis['problem_type']}
**Difficulty Level**: {effective_difficulty.value}
**Key Concepts**: {', '.join(problem_analysis['key_concepts'])}

**Solution Strategy**: {problem_analysis['solution_strategy']['approach']}
**Mathematical Requirements**: {', '.join(problem_analysis['mathematical_requirements'])}

**Potential Pitfalls to Avoid**: {', '.join(problem_analysis['potential_pitfalls'])}

Please provide a comprehensive solution that:
1. Addresses the specific problem type and key concepts
2. Uses the appropriate mathematical level for {effective_difficulty.value}
3. Follows the systematic solution strategy
4. Avoids the identified potential pitfalls
5. Includes proper validation and interpretation

Adapt your explanation style to: {problem_analysis['solution_strategy']['difficulty_adaptations']['explanation_style']}
"""
        
        # Store problem analysis for future reference
        await self._store_problem_analysis(problem, problem_analysis, thread_id)
        
        # Generate solution using enhanced prompt
        response = await self.achat(enhanced_prompt, thread_id or "default")
        
        return response
    
    async def research_topic(self, 
                           topic: str, 
                           depth: Literal["overview", "detailed", "comprehensive"] = "detailed",
                           thread_id: Optional[str] = None) -> str:
        """Research a physics topic with domain expertise.
        
        Args:
            topic: Physics topic to research
            depth: Level of detail required
            thread_id: Thread ID for conversation continuity
            
        Returns:
            Comprehensive research analysis
        """
        research_prompt = f"""
## Advanced Physics Research Analysis

**Topic**: {topic}
**Specialization Context**: {self.specialization.value.replace('_', ' ').title()}
**Research Depth**: {depth}

Please provide a {depth} analysis that includes:

1. **Fundamental Principles**: Core physics concepts and laws
2. **Mathematical Framework**: Key equations and mathematical tools
3. **Current Understanding**: State of the art and recent developments
4. **Research Frontiers**: Open questions and active research areas
5. **Experimental Aspects**: Key experiments and measurement techniques
6. **Applications**: Practical applications and technological implications
7. **Connections**: Links to other areas of physics
8. **Future Directions**: Promising research directions

Focus on aspects most relevant to {self.specialization.value.replace('_', ' ')} while maintaining broader physics context.
"""
        
        return await self.achat(research_prompt, thread_id or "default")
    
    async def _store_problem_analysis(self, 
                                    problem: str, 
                                    analysis: Dict[str, Any], 
                                    thread_id: Optional[str] = None):
        """Store problem analysis for learning and adaptation."""
        try:
            await self.knowledge_api.store_knowledge(
                title=f"Problem Analysis - {analysis['problem_type']}",
                content=json.dumps({
                    "problem": problem,
                    "analysis": analysis,
                    "specialization": self.specialization.value,
                    "difficulty": self.difficulty_level.value
                }),
                domain=f"problem_analysis_{self.specialization.value}",
                confidence_score=0.9,
                source_type="enhanced_physics_expert",
                metadata={
                    "thread_id": thread_id,
                    "problem_type": analysis['problem_type'],
                    "specialization": self.specialization.value,
                    "difficulty": self.difficulty_level.value
                }
            )
        except Exception as e:
            # Log error but don't fail the main operation
            print(f"Warning: Could not store problem analysis: {e}")
    
    def adapt_difficulty(self, new_difficulty: DifficultyLevel) -> None:
        """Adapt the agent's difficulty level dynamically."""
        if self.adaptive_difficulty:
            self.difficulty_level = new_difficulty
            # Update reasoning engine
            self.reasoning_engine = PhysicsReasoningEngine(self.specialization)
            # Update system message
            self.system_message = self._create_enhanced_system_message()
    
    def get_specialization_info(self) -> Dict[str, Any]:
        """Get information about the agent's specialization."""
        reasoning = self.reasoning_engine.reasoning_patterns
        return {
            "specialization": self.specialization.value,
            "difficulty_level": self.difficulty_level.value,
            "key_principles": reasoning.get("key_principles", []),
            "problem_types": reasoning.get("problem_types", []),
            "mathematical_tools": reasoning.get("mathematical_tools", []),
            "common_mistakes": reasoning.get("common_mistakes", [])
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive information about this enhanced agent."""
        base_info = {
            "name": "EnhancedPhysicsExpertAgent",
            "type": "enhanced_physics_expert",
            "version": "3.0.0",
            "description": f"Enhanced physics expert specialized in {self.specialization.value.replace('_', ' ')} with adaptive intelligence",
            "specialization": self.specialization.value,
            "difficulty_level": self.difficulty_level.value,
            "adaptive_difficulty": self.adaptive_difficulty,
            "capabilities": [
                "Advanced domain-specific problem solving",
                "Adaptive difficulty scaling",
                "Systematic problem decomposition",
                "Advanced reasoning and validation",
                "Research-level topic analysis",
                "Learning objective identification",
                "Pitfall anticipation and avoidance"
            ]
        }
        
        # Add specialization-specific information
        base_info.update(self.get_specialization_info())
        
        return base_info 