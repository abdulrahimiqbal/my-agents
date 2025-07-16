#!/usr/bin/env python3
"""
PhysicsGPT - 10-Agent Physics Laboratory System
Mimics a real physics research lab with specialized roles and natural workflow.
"""

import os
import sys
from typing import Dict, Any, List

# Fix SQLite version issue for Streamlit Cloud BEFORE importing anything else
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

class PhysicsLabSystem:
    """A 10-agent physics laboratory system mimicking real research lab dynamics."""
    
    def __init__(self):
        """Initialize the physics lab with all 10 specialized agents."""
        
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
        
        # Initialize all 10 agents
        self.agents = self._create_physics_lab_team()
        
    def _create_physics_lab_team(self):
        """Create the complete 10-agent physics laboratory team."""
        
        # LAB DIRECTOR - The orchestrator
        lab_director = Agent(
            role='Lab Director',
            goal='Orchestrate comprehensive physics research by coordinating all lab specialists',
            backstory="""You are the director of a world-class physics research laboratory. You coordinate 
            complex research projects by assigning specific tasks to your team of 9 specialized researchers. 
            You understand each team member's expertise and know how to delegate effectively to produce 
            groundbreaking physics research. Your job is to ensure all relevant expertise is brought to bear 
            on each research question.""",
            verbose=True,
            allow_delegation=True,
            llm=self.director_llm
        )
        
        # SENIOR PHYSICS EXPERT - The theoretical backbone
        physics_expert = Agent(
            role='Senior Physics Expert',
            goal='Provide rigorous theoretical analysis and validate physics principles',
            backstory="""You are the senior theoretical physicist in the lab with deep expertise across 
            all physics domains. You work on theoretical frameworks assigned by the Lab Director and 
            ensure all physics is sound and rigorous. You're the go-to person for fundamental physics 
            principles and theoretical validation.""",
            verbose=True,
            allow_delegation=False,
            llm=self.senior_llm
        )
        
        # HYPOTHESIS GENERATOR - The creative mind
        hypothesis_generator = Agent(
            role='Hypothesis Generator',
            goal='Generate creative research hypotheses and explore novel approaches',
            backstory="""You are the lab's creative researcher who generates innovative hypotheses and 
            explores unconventional approaches. When the Lab Director needs fresh ideas or alternative 
            perspectives, you provide creative thinking while maintaining scientific rigor.""",
            verbose=True,
            allow_delegation=False,
            llm=self.specialist_llm
        )
        
        # MATHEMATICAL ANALYST - The calculation specialist
        mathematical_analyst = Agent(
            role='Mathematical Analyst',
            goal='Perform complex calculations and mathematical modeling',
            backstory="""You are the lab's mathematical specialist who handles all complex calculations, 
            derivations, and numerical analysis. When the Lab Director needs precise mathematical work, 
            you provide detailed calculations, dimensional analysis, and quantitative modeling.""",
            verbose=True,
            allow_delegation=False,
            llm=self.math_llm
        )
        
        # EXPERIMENTAL DESIGNER - The practical expert
        experimental_designer = Agent(
            role='Experimental Designer',
            goal='Design feasible experiments and practical implementations',
            backstory="""You are the lab's experimental physics specialist who designs practical experiments 
            and implementation strategies. When the Lab Director needs experimental validation or practical 
            approaches, you provide detailed experimental design and feasibility analysis.""",
            verbose=True,
            allow_delegation=False,
            llm=self.specialist_llm
        )
        
        # QUANTUM SPECIALIST - The quantum expert
        quantum_specialist = Agent(
            role='Quantum Specialist',
            goal='Analyze quantum mechanical aspects and quantum technologies',
            backstory="""You are the lab's quantum mechanics expert specializing in quantum phenomena, 
            quantum computing, and quantum technologies. When the Lab Director encounters quantum-related 
            questions, you provide specialized quantum mechanical analysis and insights.""",
            verbose=True,
            allow_delegation=False,
            llm=self.specialist_llm
        )
        
        # RELATIVITY EXPERT - The spacetime specialist  
        relativity_expert = Agent(
            role='Relativity Expert',
            goal='Analyze relativistic effects and cosmological phenomena',
            backstory="""You are the lab's relativity and cosmology specialist with expertise in special 
            relativity, general relativity, and cosmological phenomena. When the Lab Director needs analysis 
            of spacetime physics or cosmological questions, you provide specialized relativistic insights.""",
            verbose=True,
            allow_delegation=False,
            llm=self.specialist_llm
        )
        
        # CONDENSED MATTER EXPERT - The materials specialist
        condensed_matter_expert = Agent(
            role='Condensed Matter Expert', 
            goal='Analyze materials science and solid-state physics phenomena',
            backstory="""You are the lab's condensed matter physicist specializing in materials science, 
            solid-state physics, and many-body systems. When the Lab Director needs materials or solid-state 
            analysis, you provide specialized insights into material properties and phase behavior.""",
            verbose=True,
            allow_delegation=False,
            llm=self.specialist_llm
        )
        
        # COMPUTATIONAL PHYSICIST - The simulation expert
        computational_physicist = Agent(
            role='Computational Physicist',
            goal='Provide numerical methods and computational analysis',
            backstory="""You are the lab's computational physics specialist who handles numerical methods, 
            simulations, and computational modeling. When the Lab Director needs computational analysis 
            or numerical solutions, you provide algorithmic approaches and simulation strategies.""",
            verbose=True,
            allow_delegation=False,
            llm=self.specialist_llm
        )
        
        # PHYSICS COMMUNICATOR - The translator
        physics_communicator = Agent(
            role='Physics Communicator',
            goal='Synthesize research into clear, accessible explanations',
            backstory="""You are the lab's science communication specialist who translates complex physics 
            research into clear, accessible explanations. When the Lab Director needs final synthesis and 
            communication, you integrate all specialist contributions into coherent, understandable analysis.""",
            verbose=True,
            allow_delegation=False,
            llm=self.comm_llm
        )
        
        return [lab_director, physics_expert, hypothesis_generator, mathematical_analyst,
                experimental_designer, quantum_specialist, relativity_expert, 
                condensed_matter_expert, computational_physicist, physics_communicator]
    
    def analyze_physics_question(self, question: str) -> str:
        """
        Analyze a physics question using the full 10-agent laboratory team.
        
        Args:
            question: The physics question to analyze
            
        Returns:
            Comprehensive physics analysis from the laboratory team
        """
        
        # Create lab research task for the director
        research_task = Task(
            description=f"""As Lab Director, coordinate a comprehensive physics research project to analyze: '{question}'

Your laboratory team consists of:
- Senior Physics Expert: Theoretical analysis and physics validation
- Hypothesis Generator: Creative ideas and novel approaches  
- Mathematical Analyst: Complex calculations and mathematical modeling
- Experimental Designer: Practical experiments and implementation
- Quantum Specialist: Quantum mechanical analysis
- Relativity Expert: Relativistic and cosmological analysis
- Condensed Matter Expert: Materials and solid-state analysis
- Computational Physicist: Numerical methods and simulations
- Physics Communicator: Final synthesis and clear presentation

Your research methodology:
1. Assess the question and identify which specialists are needed
2. Delegate specific research tasks to relevant team members using simple, clear task descriptions
3. For delegation, use the format: task="specific work to do", context="all necessary background info", coworker="exact role name"
4. Collect and integrate specialist contributions
5. Delegate final synthesis to the Physics Communicator
6. Ensure comprehensive coverage of all relevant physics aspects

Remember: When delegating, provide task and context as simple strings, not complex objects.

Coordinate your team to produce a thorough, multi-perspective physics analysis that leverages the full expertise of your laboratory.""",
            agent=self.agents[0],  # Lab Director
            expected_output="Comprehensive laboratory analysis integrating insights from all relevant specialists"
        )
        
        # Create crew with hierarchical lab structure
        lab_crew = Crew(
            agents=self.agents,
            tasks=[research_task],
            process=Process.hierarchical,
            manager_llm=self.director_llm,
            verbose=True,
            share_crew=True,
            max_rpm=30,
            memory=True
        )
        
        # Execute laboratory research
        print(f"🔬 Starting physics laboratory research: {question}")
        print("=" * 80)
        print("🏛️  Physics Laboratory Team Activated")
        print("👨‍🔬 Lab Director coordinating 9 specialist researchers")
        print("⚛️  Full-spectrum physics analysis in progress...")
        print("=" * 80)
        
        result = lab_crew.kickoff()
        
        print("=" * 80)
        print("✅ Laboratory research complete!")
        print("📋 Analysis ready for review")
        
        return result


def main():
    """Main entry point for the Physics Laboratory System."""
    
    print("🏛️  PhysicsGPT - 10-Agent Physics Research Laboratory")
    print("=" * 70)
    print("🔬 Mimicking Real Physics Lab Dynamics")
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
    
    # Initialize the laboratory
    try:
        physics_lab = PhysicsLabSystem()
        print("✅ Physics laboratory initialized successfully")
        print(f"👥 Research team: {len(physics_lab.agents)} total members")
    except Exception as e:
        print(f"❌ Failed to initialize physics laboratory: {e}")
        return
    
    # Check for command line query
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = physics_lab.analyze_physics_question(question)
        
        print("\n" + "="*70)
        print("📋 LABORATORY RESEARCH REPORT:")
        print("="*70)
        print(result)
    else:
        print("\n💡 Usage: python physics_crew_system.py 'your physics question'")
        print("💡 Example: python physics_crew_system.py 'how does quantum entanglement work in many-body systems?'")


if __name__ == "__main__":
    main()