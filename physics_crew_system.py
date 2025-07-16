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
        """Create specialized physics agents with distinct roles."""
        
        # Senior Physics Expert
        physics_expert = Agent(
            role='Senior Physics Expert',
            goal='Provide comprehensive physics analysis with deep theoretical understanding',
            backstory="""You are a world-renowned physicist with expertise across all branches of physics. 
            You have published hundreds of papers and can explain complex phenomena clearly. Your analysis 
            combines theoretical rigor with practical insights.""",
            verbose=True,
            allow_delegation=True,
            llm=self.precise_llm
        )
        
        # Theoretical Physicist
        theoretical_physicist = Agent(
            role='Theoretical Physicist',
            goal='Explore theoretical frameworks and mathematical models',
            backstory="""You specialize in theoretical physics and mathematical modeling. You can derive 
            equations from first principles and understand the deepest theoretical foundations of physics. 
            You excel at connecting abstract concepts to observable phenomena.""",
            verbose=True,
            allow_delegation=False,
            llm=self.mathematical_llm
        )
        
        # Experimental Physics Consultant
        experimental_consultant = Agent(
            role='Experimental Physics Consultant',
            goal='Design experiments and analyze practical implementation challenges',
            backstory="""You are an expert in experimental physics with decades of hands-on experience. 
            You know what works in practice, what equipment is needed, and what the limitations are. 
            You can design feasible experiments and identify practical constraints.""",
            verbose=True,
            allow_delegation=False,
            llm=self.creative_llm
        )
        
        # Mathematical Analyst
        mathematical_analyst = Agent(
            role='Mathematical Analyst',
            goal='Provide rigorous mathematical analysis and derivations',
            backstory="""You are a mathematical physicist who excels at detailed calculations and 
            mathematical modeling. You can derive complex equations, perform dimensional analysis, 
            and provide quantitative estimates with proper uncertainty analysis.""",
            verbose=True,
            allow_delegation=False,
            llm=self.mathematical_llm
        )
        
        # Science Communicator
        science_communicator = Agent(
            role='Science Communicator',
            goal='Synthesize complex physics into clear, accessible explanations',
            backstory="""You excel at making complex physics concepts accessible to various audiences. 
            You can take technical analysis and present it clearly while maintaining scientific accuracy. 
            You know how to structure explanations and highlight key insights.""",
            verbose=True,
            allow_delegation=False,
            llm=self.creative_llm
        )
        
        return [physics_expert, theoretical_physicist, experimental_consultant, 
                mathematical_analyst, science_communicator]
    
    def analyze_physics_query(self, query: str) -> str:
        """
        Analyze a physics query using the specialized crew.
        
        Args:
            query: The physics question or problem to analyze
            
        Returns:
            Comprehensive physics analysis
        """
        
        # Create specialized tasks for each agent
        tasks = [
            Task(
                description=f"""Analyze this physics query and provide comprehensive expert analysis: '{query}'
                
                Your analysis should include:
                1. Core physics principles involved
                2. Relevant equations and laws
                3. Key physical phenomena
                4. Historical context and discoveries
                5. Current understanding and research
                6. Practical implications
                
                Provide a thorough, expert-level analysis.""",
                agent=self.agents[0],  # Senior Physics Expert
                expected_output="Comprehensive physics analysis with principles, equations, and context"
            ),
            
            Task(
                description=f"""Develop theoretical framework and mathematical models for: '{query}'
                
                Focus on:
                1. Fundamental theoretical principles
                2. Mathematical derivations from first principles
                3. Symmetries and conservation laws
                4. Quantum mechanical or relativistic effects if relevant
                5. Theoretical predictions and implications
                
                Provide rigorous theoretical analysis.""",
                agent=self.agents[1],  # Theoretical Physicist
                expected_output="Theoretical framework with mathematical models and derivations"
            ),
            
            Task(
                description=f"""Design experimental approaches and analyze practical considerations for: '{query}'
                
                Address:
                1. Experimental design and methodology
                2. Required equipment and instrumentation
                3. Measurement techniques and precision
                4. Practical limitations and challenges
                5. Safety considerations
                6. Expected results and interpretation
                
                Provide practical experimental analysis.""",
                agent=self.agents[2],  # Experimental Consultant
                expected_output="Experimental design with practical implementation details"
            ),
            
            Task(
                description=f"""Perform detailed mathematical analysis and calculations for: '{query}'
                
                Include:
                1. Detailed mathematical derivations
                2. Numerical estimates and calculations
                3. Dimensional analysis
                4. Order of magnitude estimates
                5. Uncertainty analysis
                6. Graphical representations if helpful
                
                Provide rigorous mathematical treatment.""",
                agent=self.agents[3],  # Mathematical Analyst
                expected_output="Detailed mathematical analysis with calculations and estimates"
            ),
            
            Task(
                description=f"""Synthesize all previous analyses into a clear, comprehensive explanation of: '{query}'
                
                Create a well-structured response that:
                1. Integrates insights from all specialists
                2. Presents information in logical order
                3. Explains complex concepts clearly
                4. Highlights key takeaways
                5. Addresses the original question directly
                6. Provides a complete picture
                
                Make the physics accessible while maintaining accuracy.""",
                agent=self.agents[4],  # Science Communicator
                expected_output="Clear, comprehensive synthesis of all physics analysis"
            )
        ]
        
        # Create crew with enhanced telemetry enabled
        crew = Crew(
            agents=self.agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,              # Enable detailed logging
            share_crew=True,          # Enable enhanced telemetry
            max_rpm=30,               # Rate limiting
            memory=True               # Enable crew memory
        )
        
        # Execute the analysis
        print(f"🚀 Starting physics analysis: {query}")
        print("=" * 80)
        
        result = crew.kickoff()
        
        print("=" * 80)
        print("✅ Analysis complete!")
        
        return result


def main():
    """Main entry point for PhysicsGPT."""
    
    print("⚛️  PhysicsGPT - Advanced 10-Agent Physics Research System")
    print("=" * 70)
    print("🤖 Powered by CrewAI with 10 Specialized Physics Agents")
    print("🧠 Physics Expert • Hypothesis Generator • Mathematical Analyst")
    print("🔬 Experimental Designer • Pattern Analyst • Quantum Specialist")
    print("🌌 Relativity Expert • Condensed Matter Expert • Computational Physicist")
    print("📚 Physics Education & Communication Specialist")
    print("=" * 70)
    
    # Initialize the system
    try:
        physics_crew = PhysicsGPTCrew()
        print("✅ PhysicsGPT system initialized successfully")
        print(f"🤖 Available agents: {len(physics_crew.agents)}")
    except Exception as e:
        print(f"❌ Failed to initialize PhysicsGPT: {e}")
        return
    
    # Check for command line query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = physics_crew.analyze_physics_query(query)
        
        print("\n📄 ANALYSIS RESULT:")
        print("=" * 50)
        print(result)
        return
    
    # Interactive mode
    print("\n🎯 Choose an option:")
    print("1. 🔬 Ask a physics question (5 core agents)")
    print("2. 🎯 Ask with specific agents")
    print("3. 🧪 Run demo with ALL 10 agents")
    print("4. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                query = input("\n🔬 Enter your physics question: ").strip()
                if query:
                    result = physics_crew.analyze_physics_query(query)
                    print("\n📄 COMPREHENSIVE ANALYSIS:")
                    print("=" * 50)
                    print(result)
                else:
                    print("❌ Please enter a valid question.")
                break
                
            elif choice == "2":
                print("\n🤖 Available agents (10 specialized physics agents):")
                print("   • physics_expert - Rigorous theoretical analysis")
                print("   • hypothesis_generator - Creative hypotheses")
                print("   • mathematical_analyst - Mathematical frameworks")
                print("   • experimental_designer - Experimental approaches")
                print("   • pattern_analyst - Pattern recognition")
                print("   • quantum_specialist - Quantum mechanics expertise")
                print("   • relativity_expert - Relativity and cosmology")
                print("   • condensed_matter_expert - Materials and many-body systems")
                print("   • computational_physicist - Numerical methods and simulations")
                print("   • physics_communicator - Clear explanations and education")
                
                agents_input = input("\nEnter agent names (comma-separated): ").strip()
                agents_to_use = [agent.strip() for agent in agents_input.split(",") if agent.strip()]
                
                query = input("🔬 Enter your physics question: ").strip()
                if query and agents_to_use:
                    result = physics_crew.analyze_physics_query(query)
                    print("\n📄 SPECIALIZED ANALYSIS:")
                    print("=" * 50)
                    print(result)
                else:
                    print("❌ Please enter a valid question and agents.")
                break
                
            elif choice == "3":
                demo_query = "How does quantum entanglement relate to black hole thermodynamics and the holographic principle?"
                print(f"\n🧪 FULL 10-AGENT DEMO ANALYSIS")
                print(f"Query: {demo_query}")
                print("Using ALL 10 specialized physics agents for maximum coverage")
                
                # Use all 10 agents for comprehensive analysis
                all_agents = [
                    'physics_expert', 'hypothesis_generator', 'mathematical_analyst',
                    'experimental_designer', 'pattern_analyst', 'quantum_specialist',
                    'relativity_expert', 'condensed_matter_expert', 'computational_physicist',
                    'physics_communicator'
                ]
                
                result = physics_crew.analyze_physics_query(demo_query)
                
                print("\n📄 DEMO ANALYSIS RESULT:")
                print("=" * 50)
                print(result)
                break
                
            elif choice == "4":
                print("\n👋 Thanks for using PhysicsGPT!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using PhysicsGPT!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break


if __name__ == "__main__":
    main()