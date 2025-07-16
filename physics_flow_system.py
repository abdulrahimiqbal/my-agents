#!/usr/bin/env python3
"""
PhysicsGPT Flow - Modern 10-Agent Physics Laboratory System
Uses CrewAI Flows for proper orchestration instead of broken hierarchical delegation.
"""

import os
import sys
from typing import Dict, Any, List
from pydantic import BaseModel, Field

# Fix SQLite version issue for Streamlit Cloud BEFORE importing anything else
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.flow.flow import Flow, listen, start
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Define Flow State
class PhysicsResearchState(BaseModel):
    question: str = ""
    coordination_plan: str = ""
    theoretical_analysis: str = ""
    creative_hypotheses: str = ""
    mathematical_models: str = ""
    experimental_design: str = ""
    quantum_analysis: str = ""
    relativity_analysis: str = ""
    materials_analysis: str = ""
    computational_simulations: str = ""
    final_synthesis: str = ""

class PhysicsLabFlow(Flow[PhysicsResearchState]):
    """Modern Physics Laboratory Flow with 10 specialized agents."""
    
    def __init__(self):
        super().__init__()
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all 10 specialized physics agents."""
        
        # Initialize LLMs with different personalities for different roles
        model_name = os.getenv("PHYSICS_AGENT_MODEL", "gpt-4o-mini")
        
        # Lab Director - Strategic and decisive
        self.director_llm = ChatOpenAI(temperature=0.2, model=model_name)
        # Senior Researchers - Precise and analytical  
        self.senior_llm = ChatOpenAI(temperature=0.1, model=model_name)
        # Specialists - Creative within their domain
        self.specialist_llm = ChatOpenAI(temperature=0.3, model=model_name)
        # Mathematical - Ultra precise
        self.math_llm = ChatOpenAI(temperature=0.05, model=model_name)
        # Communicator - Creative and accessible
        self.comm_llm = ChatOpenAI(temperature=0.4, model=model_name)
        
        # LAB DIRECTOR - The orchestrator
        self.lab_director = Agent(
            role='Lab Director',
            goal='Coordinate comprehensive physics research by planning the analysis strategy',
            backstory="""You are the director of a world-class physics research laboratory. You coordinate 
            complex research projects by creating strategic analysis plans. You understand each team member's 
            expertise and know how to structure research for maximum insight.""",
            verbose=True,
            allow_delegation=False,
            llm=self.director_llm
        )
        
        # SENIOR PHYSICS EXPERT - The theoretical backbone
        self.physics_expert = Agent(
            role='Senior Physics Expert',
            goal='Provide rigorous theoretical analysis and validate physics principles',
            backstory="""You are the senior theoretical physicist in the lab with deep expertise across 
            all physics domains. You provide theoretical frameworks and ensure all physics is sound and rigorous. 
            You're the go-to person for fundamental physics principles and theoretical validation.""",
            verbose=True,
            allow_delegation=False,
            llm=self.senior_llm
        )
        
        # HYPOTHESIS GENERATOR - The creative mind
        self.hypothesis_generator = Agent(
            role='Hypothesis Generator',
            goal='Generate creative research hypotheses and explore novel approaches',
            backstory="""You are the lab's creative researcher who generates innovative hypotheses and 
            explores unconventional approaches. You provide creative thinking while maintaining scientific rigor, 
            offering fresh perspectives that others might miss.""",
            verbose=True,
            allow_delegation=False,
            llm=self.specialist_llm
        )
        
        # MATHEMATICAL ANALYST - The calculation specialist
        self.mathematical_analyst = Agent(
            role='Mathematical Analyst',
            goal='Perform complex calculations and mathematical modeling',
            backstory="""You are the lab's mathematical specialist who handles all complex calculations, 
            derivations, and numerical analysis. You provide detailed mathematical work, dimensional analysis, 
            and quantitative modeling that supports the research.""",
            verbose=True,
            allow_delegation=False,
            llm=self.math_llm
        )
        
        # EXPERIMENTAL DESIGNER - The practical expert
        self.experimental_designer = Agent(
            role='Experimental Designer',
            goal='Design feasible experiments and practical implementations',
            backstory="""You are the lab's experimental physics specialist who designs practical experiments 
            and implementation strategies. You provide detailed experimental design, feasibility analysis, 
            and practical approaches to testing theoretical predictions.""",
            verbose=True,
            allow_delegation=False,
            llm=self.specialist_llm
        )
        
        # QUANTUM SPECIALIST - The quantum expert
        self.quantum_specialist = Agent(
            role='Quantum Specialist',
            goal='Analyze quantum mechanical aspects and quantum technologies',
            backstory="""You are the lab's quantum mechanics expert specializing in quantum phenomena, 
            quantum computing, and quantum technologies. You provide deep insights into quantum behavior, 
            quantum field theory, and quantum applications.""",
            verbose=True,
            allow_delegation=False,
            llm=self.specialist_llm
        )
        
        # RELATIVITY EXPERT - The spacetime specialist
        self.relativity_expert = Agent(
            role='Relativity Expert',
            goal='Analyze relativistic effects and cosmological implications',
            backstory="""You are the lab's relativity and cosmology expert specializing in special relativity, 
            general relativity, and cosmological phenomena. You provide insights into spacetime effects, 
            gravitational physics, and cosmological implications.""",
            verbose=True,
            allow_delegation=False,
            llm=self.specialist_llm
        )
        
        # CONDENSED MATTER EXPERT - The materials specialist
        self.condensed_matter_expert = Agent(
            role='Condensed Matter Expert',
            goal='Analyze materials properties and solid-state phenomena',
            backstory="""You are the lab's condensed matter physicist specializing in materials science, 
            solid-state physics, and material properties. You provide insights into material behavior, 
            phase transitions, and solid-state phenomena.""",
            verbose=True,
            allow_delegation=False,
            llm=self.specialist_llm
        )
        
        # COMPUTATIONAL PHYSICIST - The simulation expert
        self.computational_physicist = Agent(
            role='Computational Physicist',
            goal='Perform numerical simulations and computational modeling',
            backstory="""You are the lab's computational physics expert specializing in numerical methods, 
            simulations, and computational modeling. You provide computational approaches, simulation strategies, 
            and numerical solutions to complex physics problems.""",
            verbose=True,
            allow_delegation=False,
            llm=self.specialist_llm
        )
        
        # PHYSICS COMMUNICATOR - The science translator
        self.physics_communicator = Agent(
            role='Physics Communicator',
            goal='Synthesize research findings into clear, comprehensive presentations',
            backstory="""You are the lab's science communication specialist who translates complex physics 
            research into clear, comprehensive, and accessible presentations. You integrate insights from all 
            specialists and present them in a coherent, engaging manner.""",
            verbose=True,
            allow_delegation=False,
            llm=self.comm_llm
        )

    @start()
    def coordinate_research(self):
        """Lab Director creates the research coordination plan."""
        print(f"🔬 Lab Director analyzing question: {self.state.question}")
        
        task = Task(
            description=f"""As Lab Director, create a comprehensive research coordination plan for: '{self.state.question}'

Analyze the question and create a strategic plan that identifies:
1. Which of our 9 specialist researchers should be involved
2. What specific analysis each specialist should focus on
3. How their contributions will build toward a complete understanding
4. The key physics domains that need to be addressed

Available specialists:
- Senior Physics Expert: Theoretical frameworks and fundamental principles
- Hypothesis Generator: Creative approaches and novel ideas
- Mathematical Analyst: Calculations, equations, and quantitative models
- Experimental Designer: Practical experiments and detection methods
- Quantum Specialist: Quantum mechanical aspects
- Relativity Expert: Relativistic and cosmological effects
- Condensed Matter Expert: Materials and solid-state physics
- Computational Physicist: Simulations and numerical methods
- Physics Communicator: Final synthesis and presentation

Create a clear coordination plan that will guide the specialists.""",
            expected_output="A detailed coordination plan identifying which specialists to involve and their specific focus areas",
            agent=self.lab_director
        )
        
        crew = Crew(agents=[self.lab_director], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        
        self.state.coordination_plan = result.raw
        print("✅ Coordination plan created")
        return result.raw

    @listen(coordinate_research)
    def theoretical_analysis(self, coordination_plan):
        """Senior Physics Expert provides theoretical foundation."""
        print("🧠 Senior Physics Expert conducting theoretical analysis...")
        
        task = Task(
            description=f"""Based on the coordination plan, provide comprehensive theoretical analysis for: '{self.state.question}'

Coordination Plan: {coordination_plan}

Focus on:
1. Fundamental physics principles involved
2. Theoretical frameworks that apply
3. Key physics laws and concepts
4. Theoretical predictions and implications
5. Connection to established physics theory

Provide rigorous theoretical foundation that other specialists can build upon.""",
            expected_output="Comprehensive theoretical analysis with fundamental principles and frameworks",
            agent=self.physics_expert
        )
        
        crew = Crew(agents=[self.physics_expert], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        
        self.state.theoretical_analysis = result.raw
        print("✅ Theoretical analysis complete")
        return result.raw

    @listen(theoretical_analysis)
    def generate_hypotheses(self, theoretical_analysis):
        """Hypothesis Generator creates novel approaches."""
        print("💡 Hypothesis Generator developing creative approaches...")
        
        task = Task(
            description=f"""Based on the theoretical analysis, generate creative hypotheses and novel approaches for: '{self.state.question}'

Theoretical Foundation: {theoretical_analysis}

Generate:
1. Innovative hypotheses about the phenomenon
2. Novel approaches that haven't been fully explored
3. Creative solutions to challenges identified
4. Alternative perspectives on the problem
5. Unconventional but scientifically sound ideas

Think outside the box while maintaining scientific rigor.""",
            expected_output="Creative hypotheses and novel approaches with scientific justification",
            agent=self.hypothesis_generator
        )
        
        crew = Crew(agents=[self.hypothesis_generator], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        
        self.state.creative_hypotheses = result.raw
        print("✅ Creative hypotheses generated")
        return result.raw

    @listen(generate_hypotheses)
    def mathematical_modeling(self, hypotheses):
        """Mathematical Analyst performs calculations and modeling."""
        print("📊 Mathematical Analyst working on calculations...")
        
        task = Task(
            description=f"""Perform detailed mathematical analysis for: '{self.state.question}'

Based on:
- Theoretical Analysis: {self.state.theoretical_analysis}
- Creative Hypotheses: {hypotheses}

Provide:
1. Relevant mathematical equations and derivations
2. Dimensional analysis and unit considerations
3. Quantitative models and relationships
4. Numerical estimates where possible
5. Mathematical constraints and limitations

Focus on the mathematical foundations that support the physics.""",
            expected_output="Detailed mathematical models, equations, and quantitative analysis",
            agent=self.mathematical_analyst
        )
        
        crew = Crew(agents=[self.mathematical_analyst], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        
        self.state.mathematical_models = result.raw
        print("✅ Mathematical modeling complete")
        return result.raw

    @listen(mathematical_modeling)
    def experimental_design(self, math_models):
        """Experimental Designer creates practical approaches."""
        print("⚗️ Experimental Designer developing practical methods...")
        
        task = Task(
            description=f"""Design practical experimental approaches for: '{self.state.question}'

Building on:
- Theoretical Analysis: {self.state.theoretical_analysis}
- Mathematical Models: {math_models}
- Creative Hypotheses: {self.state.creative_hypotheses}

Design:
1. Practical experimental setups and procedures
2. Detection methods and measurement techniques
3. Required equipment and materials (focus on minimal/accessible options)
4. Expected results and what they would indicate
5. Potential challenges and how to overcome them

Focus on feasible, practical approaches that could actually be implemented.""",
            expected_output="Detailed experimental design with practical implementation strategies",
            agent=self.experimental_designer
        )
        
        crew = Crew(agents=[self.experimental_designer], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        
        self.state.experimental_design = result.raw
        print("✅ Experimental design complete")
        return result.raw

    @listen(experimental_design)
    def quantum_analysis(self, experimental_design):
        """Quantum Specialist analyzes quantum aspects."""
        print("⚛️ Quantum Specialist analyzing quantum mechanical aspects...")
        
        task = Task(
            description=f"""Analyze quantum mechanical aspects of: '{self.state.question}'

Previous work to consider:
- Theoretical Framework: {self.state.theoretical_analysis}
- Experimental Approach: {experimental_design}

Focus on:
1. Quantum mechanical principles involved
2. Quantum effects and phenomena
3. Quantum field theory implications
4. Quantum technologies that might be relevant
5. Quantum limitations and considerations

Provide deep quantum mechanical insights.""",
            expected_output="Comprehensive quantum mechanical analysis and insights",
            agent=self.quantum_specialist
        )
        
        crew = Crew(agents=[self.quantum_specialist], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        
        self.state.quantum_analysis = result.raw
        print("✅ Quantum analysis complete")
        return result.raw

    @listen(quantum_analysis)
    def computational_simulation(self, quantum_analysis):
        """Computational Physicist develops simulation strategies."""
        print("💻 Computational Physicist designing simulations...")
        
        task = Task(
            description=f"""Develop computational approaches and simulation strategies for: '{self.state.question}'

Building on all previous analyses:
- Theoretical: {self.state.theoretical_analysis}
- Mathematical: {self.state.mathematical_models}
- Quantum: {quantum_analysis}

Provide:
1. Computational modeling approaches
2. Simulation strategies and methods
3. Numerical algorithms that could be used
4. Software tools and computational requirements
5. Expected computational challenges and solutions

Focus on practical computational approaches.""",
            expected_output="Comprehensive computational simulation strategies and methods",
            agent=self.computational_physicist
        )
        
        crew = Crew(agents=[self.computational_physicist], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        
        self.state.computational_simulations = result.raw
        print("✅ Computational simulations designed")
        return result.raw

    @listen(computational_simulation)
    def synthesize_research(self, computational_work):
        """Physics Communicator synthesizes all specialist contributions."""
        print("📋 Physics Communicator synthesizing final research...")
        
        task = Task(
            description=f"""Create a comprehensive synthesis of all specialist research for: '{self.state.question}'

Integrate insights from all specialists:

COORDINATION PLAN:
{self.state.coordination_plan}

THEORETICAL ANALYSIS:
{self.state.theoretical_analysis}

CREATIVE HYPOTHESES:
{self.state.creative_hypotheses}

MATHEMATICAL MODELS:
{self.state.mathematical_models}

EXPERIMENTAL DESIGN:
{self.state.experimental_design}

QUANTUM ANALYSIS:
{self.state.quantum_analysis}

COMPUTATIONAL SIMULATIONS:
{computational_work}

Create a comprehensive, well-structured final report that:
1. Integrates all specialist perspectives
2. Presents a complete multi-dimensional analysis
3. Highlights key insights and novel approaches
4. Provides practical implementation strategies
5. Explains complex concepts clearly
6. Demonstrates the full depth of our laboratory's expertise

This should be a masterpiece of collaborative physics research.""",
            expected_output="Comprehensive integrated physics analysis combining all specialist expertise",
            agent=self.physics_communicator
        )
        
        crew = Crew(agents=[self.physics_communicator], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        
        self.state.final_synthesis = result.raw
        print("✅ Final synthesis complete")
        return result.raw

def analyze_physics_question_with_flow(question: str) -> str:
    """
    Analyze a physics question using the modern Flow-based laboratory system.
    
    Args:
        question: The physics question to analyze
        
    Returns:
        Comprehensive physics analysis from the flow-based laboratory team
    """
    
    print(f"🔬 Starting physics laboratory research: {question}")
    print("=" * 80)
    print("🏛️  Physics Laboratory Flow Activated")
    print("👨‍🔬 Modern event-driven orchestration with 10 specialists")
    print("⚛️  Full-spectrum physics analysis in progress...")
    print("=" * 80)
    
    # Initialize and run the flow
    flow = PhysicsLabFlow()
    result = flow.kickoff(inputs={"question": question})
    
    print("=" * 80)
    print("✅ Laboratory research complete!")
    print("📋 Analysis ready for review")
    
    return result

def main():
    """Main entry point for the Physics Laboratory Flow System."""
    
    print("🏛️  PhysicsGPT Flow - 10-Agent Physics Research Laboratory")
    print("=" * 70)
    print("🔬 Modern Event-Driven Physics Lab Orchestration")
    print("👨‍🔬 Lab Director + 9 Specialized Researchers")
    print("")
    print("🧠 Senior Physics Expert (Theoretical Backbone)")
    print("💡 Hypothesis Generator (Creative Mind)")  
    print("📊 Mathematical Analyst (Calculation Specialist)")
    print("⚗️  Experimental Designer (Practical Expert)")
    print("⚛️  Quantum Specialist (Quantum Expert)")
    print("🌌 Relativity Expert (Spacetime Specialist)")
    print("🔧 Condensed Matter Expert (Materials Specialist)")
    print("💻 Computational Physicist (Simulation Expert)")
    print("📝 Physics Communicator (Science Translator)")
    print("=" * 70)
    
    # Check for command line query
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = analyze_physics_question_with_flow(question)
        
        print("\n" + "="*70)
        print("📋 LABORATORY RESEARCH REPORT:")
        print("="*70)
        print(result)
    else:
        print("\n💡 Usage: python physics_flow_system.py 'your physics question'")
        print("💡 Example: python physics_flow_system.py 'how to detect dark matter with minimal stuff?'")

if __name__ == "__main__":
    main() 