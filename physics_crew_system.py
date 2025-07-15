#!/usr/bin/env python3
"""
PhysicsGPT - Clean Multi-Agent System with CrewAI
Professional physics research system with specialized AI agents.
"""

import os
import sys
from typing import Dict, Any, List
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

class PhysicsGPTCrew:
    """Clean, professional multi-agent physics research system using CrewAI."""
    
    def __init__(self):
        """Initialize the PhysicsGPT crew with specialized agents."""
        
        # Initialize LLMs with different temperatures for different purposes
        self.precise_llm = ChatOpenAI(
            model=os.getenv("PHYSICS_AGENT_MODEL", "gpt-4o-mini"),
            temperature=0.1  # Low temperature for accuracy
        )
        
        self.creative_llm = ChatOpenAI(
            model=os.getenv("PHYSICS_AGENT_MODEL", "gpt-4o-mini"),
            temperature=0.7  # High temperature for creativity
        )
        
        self.mathematical_llm = ChatOpenAI(
            model=os.getenv("PHYSICS_AGENT_MODEL", "gpt-4o-mini"),
            temperature=0.05  # Very low for mathematical precision
        )
        
        # Create specialized physics agents
        self.agents = self._create_agents()
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create the complete set of 8+ specialized physics research agents."""
        
        agents = {}
        
        # 1. Senior Physics Expert - Theoretical rigor and established physics
        agents['physics_expert'] = Agent(
            role='Senior Physics Expert',
            goal='Provide rigorous, accurate physics analysis with mathematical precision',
            backstory="""You are a world-renowned theoretical physicist with 20+ years of experience 
            across quantum mechanics, relativity, thermodynamics, electromagnetism, and experimental physics. 
            You provide scientifically accurate analysis with proper mathematical formulations, cite relevant 
            experiments, and acknowledge limitations. Your responses are authoritative yet accessible.""",
            verbose=True,
            allow_delegation=False,
            llm=self.precise_llm
        )
        
        # 2. Creative Physics Researcher - Novel hypotheses and connections
        agents['hypothesis_generator'] = Agent(
            role='Creative Physics Researcher',
            goal='Generate innovative hypotheses and discover novel theoretical connections',
            backstory="""You are a brilliant theoretical physicist known for groundbreaking insights 
            and creative thinking. You excel at finding unexpected connections between different physics 
            domains, proposing testable hypotheses, and thinking outside conventional frameworks. 
            You balance creativity with scientific rigor, always ensuring your ideas are grounded 
            in established physics principles.""",
            verbose=True,
            allow_delegation=False,
            llm=self.creative_llm
        )
        
        # 3. Mathematical Physics Specialist - Quantitative analysis
        agents['mathematical_analyst'] = Agent(
            role='Mathematical Physics Specialist',
            goal='Provide precise mathematical analysis and quantitative frameworks',
            backstory="""You are a mathematical physicist who specializes in translating physical 
            concepts into rigorous mathematical frameworks. You derive equations, perform quantitative 
            analysis, and create mathematical models. Your expertise spans differential equations, 
            linear algebra, complex analysis, and computational physics.""",
            verbose=True,
            allow_delegation=False,
            llm=self.mathematical_llm
        )
        
        # 4. Experimental Physics Designer - Practical testing approaches
        agents['experimental_designer'] = Agent(
            role='Experimental Physics Designer',
            goal='Design feasible experiments to test theoretical predictions',
            backstory="""You are an experienced experimental physicist who designs practical experiments 
            to test theoretical predictions. You consider real-world constraints, measurement techniques, 
            error analysis, and safety protocols. You bridge the gap between theory and practice, 
            ensuring that proposed experiments are both scientifically valuable and practically feasible.""",
            verbose=True,
            allow_delegation=False,
            llm=self.precise_llm
        )
        
        # 5. Pattern Recognition Specialist - Data analysis and relationships
        agents['pattern_analyst'] = Agent(
            role='Pattern Recognition Specialist',
            goal='Identify patterns, correlations, and hidden relationships in physics data',
            backstory="""You are a data-driven physicist who excels at identifying patterns, correlations, 
            and hidden relationships in complex physics data. You use statistical analysis, machine learning 
            techniques, and visualization to uncover insights that might not be immediately obvious. 
            You help connect theoretical predictions with observational evidence.""",
            verbose=True,
            allow_delegation=False,
            llm=self.precise_llm
        )
        
        # 6. Quantum Mechanics Specialist - Deep quantum expertise
        agents['quantum_specialist'] = Agent(
            role='Quantum Mechanics Specialist',
            goal='Provide expert analysis of quantum mechanical phenomena and applications',
            backstory="""You are a quantum mechanics expert with deep knowledge of quantum field theory, 
            quantum information, quantum computing, and quantum foundations. You understand the mathematical 
            formalism of quantum mechanics, from basic postulates to advanced topics like quantum entanglement, 
            decoherence, and quantum measurement theory. You can explain complex quantum phenomena clearly 
            and connect them to practical applications.""",
            verbose=True,
            allow_delegation=False,
            llm=self.precise_llm
        )
        
        # 7. Relativity & Cosmology Expert - Spacetime and universe-scale physics
        agents['relativity_expert'] = Agent(
            role='Relativity & Cosmology Expert',
            goal='Analyze spacetime, gravitational phenomena, and cosmological questions',
            backstory="""You are an expert in general relativity, special relativity, and cosmology. 
            You understand Einstein's field equations, black hole physics, gravitational waves, and 
            the large-scale structure of the universe. You can work with curved spacetime mathematics, 
            cosmological models, and connect relativity to observations from gravitational wave detectors 
            and astronomical surveys.""",
            verbose=True,
            allow_delegation=False,
            llm=self.precise_llm
        )
        
        # 8. Condensed Matter Physicist - Materials and many-body systems
        agents['condensed_matter_expert'] = Agent(
            role='Condensed Matter Physicist',
            goal='Analyze materials, phase transitions, and many-body quantum systems',
            backstory="""You are a condensed matter physicist specializing in materials science, 
            phase transitions, superconductivity, magnetism, and many-body quantum systems. You understand 
            solid state physics, electronic band structure, collective phenomena, and emergent properties 
            in materials. You can connect microscopic physics to macroscopic material properties.""",
            verbose=True,
            allow_delegation=False,
            llm=self.precise_llm
        )
        
        # 9. Computational Physics Specialist - Numerical methods and simulations
        agents['computational_physicist'] = Agent(
            role='Computational Physics Specialist',
            goal='Develop computational approaches and numerical solutions to physics problems',
            backstory="""You are a computational physicist expert in numerical methods, Monte Carlo 
            simulations, finite element analysis, and high-performance computing for physics problems. 
            You can design algorithms, optimize code, and use computational tools to solve complex 
            physics problems that are intractable analytically. You understand both the physics and 
            the computational techniques needed to model physical systems.""",
            verbose=True,
            allow_delegation=False,
            llm=self.precise_llm
        )
        
        # 10. Physics Education & Communication Specialist - Making physics accessible
        agents['physics_communicator'] = Agent(
            role='Physics Education & Communication Specialist',
            goal='Explain complex physics concepts clearly and develop educational approaches',
            backstory="""You are a physics education specialist who excels at making complex physics 
            concepts accessible to different audiences. You understand common misconceptions, can create 
            analogies and visualizations, and know how to structure explanations for maximum clarity. 
            You bridge the gap between expert knowledge and public understanding, ensuring that physics 
            insights are communicated effectively.""",
            verbose=True,
            allow_delegation=False,
            llm=self.creative_llm
        )
        
        return agents
    
    def create_physics_analysis_crew(self, query: str, agents_to_use: List[str] = None) -> Crew:
        """Create a crew for comprehensive physics analysis."""
        
        if agents_to_use is None:
            agents_to_use = ['physics_expert', 'hypothesis_generator', 'mathematical_analyst']
        
        # Filter agents based on request
        selected_agents = [self.agents[agent_name] for agent_name in agents_to_use if agent_name in self.agents]
        
        # Create tasks for each selected agent
        tasks = []
        
        if 'physics_expert' in agents_to_use:
            tasks.append(Task(
                description=f"""
                Provide a comprehensive physics analysis of: {query}
                
                Your analysis should include:
                - Relevant physical principles and fundamental laws
                - Mathematical formulations and key equations
                - Current theoretical understanding and established knowledge
                - Experimental evidence and observational support
                - Known limitations, uncertainties, and open questions
                - Historical context and key contributors
                
                Ensure scientific accuracy and cite relevant experiments or observations.
                """,
                agent=self.agents['physics_expert'],
                expected_output="Detailed physics analysis with equations, evidence, and scientific rigor"
            ))
        
        if 'hypothesis_generator' in agents_to_use:
            tasks.append(Task(
                description=f"""
                Generate creative and innovative hypotheses related to: {query}
                
                Your response should provide:
                - 2-3 novel, scientifically grounded hypotheses
                - Testability criteria for each hypothesis
                - Connections to other physics domains or emerging fields
                - Potential implications for current theoretical frameworks
                - Suggestions for future research directions
                - Creative but scientifically sound speculation
                
                Balance creativity with scientific rigor and feasibility.
                """,
                agent=self.agents['hypothesis_generator'],
                expected_output="Creative hypotheses with testability criteria and interdisciplinary connections"
            ))
        
        if 'mathematical_analyst' in agents_to_use:
            tasks.append(Task(
                description=f"""
                Provide detailed mathematical analysis for: {query}
                
                Your analysis should include:
                - Relevant equations and mathematical derivations
                - Quantitative relationships and scaling laws
                - Mathematical modeling approaches and frameworks
                - Numerical analysis considerations and computational methods
                - Statistical or probabilistic treatments where appropriate
                - Dimensional analysis and unit considerations
                
                Focus on mathematical rigor and quantitative precision.
                """,
                agent=self.agents['mathematical_analyst'],
                expected_output="Mathematical framework with equations, derivations, and quantitative models"
            ))
        
        if 'experimental_designer' in agents_to_use:
            tasks.append(Task(
                description=f"""
                Design experimental approaches to investigate: {query}
                
                Your design should provide:
                - Detailed experimental setup and methodology
                - Required instrumentation and measurement techniques
                - Control variables and experimental protocols
                - Expected results and data analysis methods
                - Error analysis and uncertainty quantification
                - Practical considerations, limitations, and safety protocols
                - Alternative experimental approaches if applicable
                
                Ensure experiments are feasible with current or near-future technology.
                """,
                agent=self.agents['experimental_designer'],
                expected_output="Detailed experimental design with protocols, instrumentation, and analysis methods"
            ))
        
        if 'pattern_analyst' in agents_to_use:
            tasks.append(Task(
                description=f"""
                Analyze patterns and relationships related to: {query}
                
                Your analysis should identify:
                - Statistical patterns and correlations in relevant data
                - Scaling relationships and power laws
                - Symmetries and conservation principles
                - Emergent phenomena and collective behaviors
                - Data visualization and interpretation strategies
                - Connections between seemingly unrelated observations
                - Predictive patterns and trend analysis
                
                Focus on data-driven insights and quantitative pattern recognition.
                """,
                agent=self.agents['pattern_analyst'],
                expected_output="Pattern analysis with statistical insights, correlations, and data-driven conclusions"
            ))
        
        if 'quantum_specialist' in agents_to_use:
            tasks.append(Task(
                description=f"""
                Provide expert quantum mechanical analysis for: {query}
                
                Your analysis should cover:
                - Quantum mechanical principles and formalism
                - Quantum field theory considerations where relevant
                - Quantum information and entanglement aspects
                - Decoherence and measurement theory
                - Quantum computing and quantum technology applications
                - Quantum foundations and interpretational issues
                - Connection to experimental quantum physics
                
                Focus on rigorous quantum mechanical treatment and modern applications.
                """,
                agent=self.agents['quantum_specialist'],
                expected_output="Expert quantum mechanical analysis with rigorous formalism and applications"
            ))
        
        if 'relativity_expert' in agents_to_use:
            tasks.append(Task(
                description=f"""
                Analyze spacetime and gravitational aspects of: {query}
                
                Your analysis should include:
                - General and special relativity considerations
                - Spacetime geometry and curvature effects
                - Black hole physics and thermodynamics
                - Cosmological implications and models
                - Gravitational wave physics
                - Connection to observational astronomy and cosmology
                - Mathematical treatment using tensor calculus
                
                Focus on relativistic physics and cosmological perspectives.
                """,
                agent=self.agents['relativity_expert'],
                expected_output="Relativistic analysis with spacetime geometry and cosmological insights"
            ))
        
        if 'condensed_matter_expert' in agents_to_use:
            tasks.append(Task(
                description=f"""
                Analyze condensed matter and materials aspects of: {query}
                
                Your analysis should cover:
                - Solid state physics and electronic properties
                - Phase transitions and critical phenomena
                - Many-body quantum systems and collective behavior
                - Superconductivity and magnetism
                - Emergent properties and symmetry breaking
                - Materials science applications
                - Connection to experimental condensed matter physics
                
                Focus on many-body physics and emergent phenomena in materials.
                """,
                agent=self.agents['condensed_matter_expert'],
                expected_output="Condensed matter analysis with many-body physics and materials insights"
            ))
        
        if 'computational_physicist' in agents_to_use:
            tasks.append(Task(
                description=f"""
                Develop computational approaches for: {query}
                
                Your analysis should provide:
                - Numerical methods and algorithms
                - Computational modeling strategies
                - Monte Carlo and molecular dynamics approaches
                - High-performance computing considerations
                - Code optimization and parallel processing
                - Visualization and data analysis techniques
                - Computational complexity and scaling
                
                Focus on practical computational solutions and numerical methods.
                """,
                agent=self.agents['computational_physicist'],
                expected_output="Computational framework with numerical methods and algorithmic approaches"
            ))
        
        if 'physics_communicator' in agents_to_use:
            tasks.append(Task(
                description=f"""
                Explain and communicate the physics concepts in: {query}
                
                Your response should provide:
                - Clear, accessible explanations for different audiences
                - Analogies and visualizations to aid understanding
                - Common misconceptions and how to address them
                - Educational approaches and learning strategies
                - Connection to everyday experiences and applications
                - Summary of key insights in plain language
                - Suggestions for further learning and exploration
                
                Focus on making complex physics accessible and engaging.
                """,
                agent=self.agents['physics_communicator'],
                expected_output="Clear educational explanation with analogies and accessible insights"
            ))
        
        # Create and return the crew
        crew = Crew(
            agents=selected_agents,
            tasks=tasks,
            process=Process.sequential,  # Can be changed to hierarchical if needed
            verbose=True,
            memory=False,  # Disable memory to avoid ChromaDB issues on Streamlit Cloud
        )
        
        return crew
    
    def analyze_physics_query(self, query: str, agents_to_use: List[str] = None) -> Dict[str, Any]:
        """Analyze a physics query using the multi-agent crew."""
        
        print(f"🚀 PhysicsGPT Multi-Agent Analysis")
        print(f"=" * 60)
        print(f"📋 Query: {query}")
        
        # Default to using 5 core agents if none specified
        if not agents_to_use:
            agents_to_use = [
                'physics_expert',
                'hypothesis_generator', 
                'mathematical_analyst',
                'quantum_specialist',
                'physics_communicator'
            ]
        
        print(f"🤖 Selected Agents: {', '.join(agents_to_use)}")
        print(f"🔄 Processing with CrewAI...")
        print()
        
        try:
            # Create the crew
            crew = self.create_physics_analysis_crew(query, agents_to_use)
            
            # Execute the analysis
            result = crew.kickoff()
            
            print(f"✅ Analysis Complete!")
            print(f"=" * 60)
            
            return {
                "success": True,
                "query": query,
                "result": result,
                "agents_used": agents_to_use or ['physics_expert', 'hypothesis_generator', 'mathematical_analyst'],
                "crew_size": len(crew.agents)
            }
            
        except Exception as e:
            print(f"❌ Analysis Failed: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "agents_used": agents_to_use or []
            }


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
        
        if result['success']:
            print("\n📄 ANALYSIS RESULT:")
            print("=" * 50)
            print(result['result'])
        else:
            print(f"\n❌ Analysis failed: {result['error']}")
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
                    if result['success']:
                        print("\n📄 COMPREHENSIVE ANALYSIS:")
                        print("=" * 50)
                        print(result['result'])
                    else:
                        print(f"\n❌ Analysis failed: {result['error']}")
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
                    result = physics_crew.analyze_physics_query(query, agents_to_use)
                    if result['success']:
                        print("\n📄 SPECIALIZED ANALYSIS:")
                        print("=" * 50)
                        print(result['result'])
                    else:
                        print(f"\n❌ Analysis failed: {result['error']}")
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
                
                result = physics_crew.analyze_physics_query(demo_query, all_agents)
                
                if result['success']:
                    print("\n📄 DEMO ANALYSIS RESULT:")
                    print("=" * 50)
                    print(result['result'])
                else:
                    print(f"\n❌ Demo failed: {result['error']}")
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