#!/usr/bin/env python3
"""
PhysicsGPT - Clean Multi-Agent System with CrewAI
Professional physics research system with specialized AI agents.
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

class PhysicsGPTCrew:
    """A specialized CrewAI system for comprehensive physics analysis."""
    
    def __init__(self):
        """Initialize the PhysicsGPT crew with specialized agents and enhanced telemetry."""
        
        # Initialize LLMs with different temperatures for different purposes
        model_name = os.getenv("PHYSICS_AGENT_MODEL", "gpt-4o-mini")
        
        # Use standard ChatOpenAI - CrewAI will handle telemetry automatically
        self.precise_llm = ChatOpenAI(temperature=0.1, model=model_name)
        self.creative_llm = ChatOpenAI(temperature=0.3, model=model_name)
        self.mathematical_llm = ChatOpenAI(temperature=0.05, model=model_name)
        
        # Initialize agents
        self.agents = self._create_agents()
        
    def _create_agents(self):
        """Create specialized physics agents with proper delegation setup."""
        
        # Senior Physics Expert - The coordinator
        physics_expert = Agent(
            role='Senior Physics Expert',
            goal='Coordinate comprehensive physics analysis by delegating specialized tasks to team members',
            backstory="""You are a world-renowned physicist and team leader with expertise across all branches 
            of physics. You coordinate complex physics research by delegating specialized tasks to your expert 
            team members. You know when to delegate theoretical work to theorists, experimental design to 
            experimentalists, and mathematical analysis to analysts.""",
            verbose=True,
            allow_delegation=True,  # Enable delegation
            llm=self.precise_llm
        )
        
        # Theoretical Physicist
        theoretical_physicist = Agent(
            role='Theoretical Physicist',
            goal='Develop theoretical frameworks and mathematical models when requested',
            backstory="""You specialize in theoretical physics and mathematical modeling. You work on specific 
            theoretical tasks assigned by the Senior Physics Expert. You derive equations from first principles, 
            develop theoretical frameworks, and connect abstract concepts to observable phenomena.""",
            verbose=True,
            allow_delegation=False,  # Cannot delegate further
            llm=self.mathematical_llm
        )
        
        # Experimental Physics Consultant
        experimental_consultant = Agent(
            role='Experimental Physics Consultant',
            goal='Design experiments and analyze practical implementation when requested',
            backstory="""You are an expert in experimental physics who works on specific experimental tasks 
            assigned by the Senior Physics Expert. You design feasible experiments, identify practical 
            constraints, specify needed equipment, and analyze implementation challenges.""",
            verbose=True,
            allow_delegation=False,
            llm=self.creative_llm
        )
        
        # Mathematical Analyst
        mathematical_analyst = Agent(
            role='Mathematical Analyst',
            goal='Perform rigorous mathematical calculations when requested',
            backstory="""You are a mathematical physicist who handles specific calculation tasks assigned 
            by the Senior Physics Expert. You perform detailed derivations, dimensional analysis, quantitative 
            estimates, and provide mathematical rigor to physics problems.""",
            verbose=True,
            allow_delegation=False,
            llm=self.mathematical_llm
        )
        
        # Science Communicator
        science_communicator = Agent(
            role='Science Communicator',
            goal='Synthesize and communicate physics results when requested',
            backstory="""You excel at making complex physics accessible. You work on communication tasks 
            assigned by the Senior Physics Expert, taking technical analysis from other team members and 
            presenting it clearly while maintaining scientific accuracy.""",
            verbose=True,
            allow_delegation=False,
            llm=self.creative_llm
        )
        
        return [physics_expert, theoretical_physicist, experimental_consultant, 
                mathematical_analyst, science_communicator]
    
    def analyze_physics_query(self, query: str) -> str:
        """
        Analyze a physics query using collaborative agent delegation.
        
        Args:
            query: The physics question or problem to analyze
            
        Returns:
            Comprehensive physics analysis
        """
        
        # Create a single collaborative task for the Senior Physics Expert
        # This agent will delegate specific subtasks to other team members
        collaborative_task = Task(
            description=f"""As Senior Physics Expert, analyze this physics query by coordinating with your team: '{query}'

You have access to specialized team members:
- Theoretical Physicist: For theoretical frameworks and mathematical models
- Experimental Physics Consultant: For experimental design and practical implementation
- Mathematical Analyst: For detailed calculations and mathematical analysis  
- Science Communicator: For clear synthesis and presentation

Your approach should be:
1. Start with your own expert analysis of the core physics
2. Delegate specific theoretical work to the Theoretical Physicist
3. Delegate experimental aspects to the Experimental Physics Consultant  
4. Delegate complex calculations to the Mathematical Analyst
5. Delegate final synthesis to the Science Communicator

Coordinate the team to produce a comprehensive analysis covering:
- Core physics principles and phenomena
- Theoretical frameworks and mathematical models
- Experimental approaches and practical considerations
- Detailed calculations and quantitative analysis
- Clear, accessible explanation of results

Use the 'Delegate work to coworker' tool to assign specific tasks to team members.
Provide clear, specific task descriptions and full context for each delegation.""",
            agent=self.agents[0],  # Senior Physics Expert with delegation enabled
            expected_output="Comprehensive collaborative physics analysis incorporating input from all team members"
        )
        
        # Create crew with the collaborative task
        crew = Crew(
            agents=self.agents,
            tasks=[collaborative_task],  # Single collaborative task
            process=Process.hierarchical,  # Use hierarchical process for delegation
            manager_llm=self.precise_llm,  # LLM for the manager (Senior Physics Expert)
            verbose=True,              # Enable detailed logging
            share_crew=True,          # Enable enhanced telemetry
            max_rpm=30,               # Rate limiting
            memory=True               # Enable crew memory
        )
        
        # Execute the analysis
        print(f"🚀 Starting collaborative physics analysis: {query}")
        print("=" * 80)
        
        result = crew.kickoff()
        
        print("=" * 80)
        print("✅ Collaborative analysis complete!")
        
        return result


def main():
    """Main entry point for PhysicsGPT."""
    
    print("⚛️  PhysicsGPT - Collaborative Physics Research System")
    print("=" * 70)
    print("🤖 Powered by CrewAI with Hierarchical Agent Collaboration")
    print("🧠 Senior Physics Expert (Coordinator)")
    print("🔬 Theoretical Physicist • Experimental Consultant")
    print("📊 Mathematical Analyst • Science Communicator")
    print("=" * 70)
    
    # Initialize the system
    try:
        physics_crew = PhysicsGPTCrew()
        print("✅ PhysicsGPT collaborative system initialized successfully")
        print(f"🤖 Available agents: {len(physics_crew.agents)}")
    except Exception as e:
        print(f"❌ Failed to initialize PhysicsGPT: {e}")
        return
    
    # Check for command line query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = physics_crew.analyze_physics_query(query)
        
        print("\n" + "="*70)
        print("📄 COLLABORATIVE ANALYSIS RESULT:")
        print("="*70)
        print(result)
    else:
        print("\n💡 Usage: python physics_crew_system.py 'your physics question'")
        print("💡 Example: python physics_crew_system.py 'how to detect dark matter in a room?'")


if __name__ == "__main__":
    main()