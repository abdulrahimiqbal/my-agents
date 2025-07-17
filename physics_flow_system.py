#!/usr/bin/env python3
"""
PhysicsGPT Flow - Modern 10-Agent Physics Laboratory System
Uses CrewAI Flows for proper orchestration with integrated database & evaluation.
"""

import os
import sys
import time
import asyncio
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

# Import DataAgent components at top level
try:
    from src.agents.data_agent import DataAgent
    from src.agents.data_tools import set_data_agent_instance, DATA_AGENT_TOOLS, register_data_tools_with_agent
    DATA_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DataAgent not available: {e}")
    DATA_AGENT_AVAILABLE = False
    DataAgent = None
    DATA_AGENT_TOOLS = []
    def register_data_tools_with_agent(agent): pass
    def set_data_agent_instance(agent): pass

# Import our CrewAI-compatible database and evaluation framework
from crewai_database_integration import (
    CrewAIKnowledgeAPI, 
    CrewAIEvaluationFramework,
    TaskType
)

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
    
    # DataAgent integration fields
    data_jobs: List[str] = Field(default_factory=list)  # List of active data ingestion job IDs
    data_insights: str = ""    # Extracted insights from data analysis
    data_context: str = ""     # Context about uploaded data files
    has_data: bool = False     # Flag indicating if data is available for analysis

class PhysicsLabFlow(Flow[PhysicsResearchState]):
    """Modern Physics Laboratory Flow with 10 specialized agents and integrated evaluation."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize database and evaluation components
        self.knowledge_api = CrewAIKnowledgeAPI()
        self.evaluation_framework = CrewAIEvaluationFramework(self.knowledge_api)
        
        # Initialize DataAgent
        if DATA_AGENT_AVAILABLE:
            self.data_agent = DataAgent()
            set_data_agent_instance(self.data_agent)  # Set global instance for tools
            self.data_tools = DATA_AGENT_TOOLS
            self.data_available = True
        else:
            print(f"Warning: DataAgent not available")
            self.data_agent = None
            self.data_tools = []
            self.data_available = False
        
        # Initialize agents
        self._initialize_agents()
        
        # Log initialization
        self.knowledge_api.log_event(
            source="physics_lab_flow",
            event_type="system_initialized",
            payload={"agents_count": 11 if self.data_available else 10, "flow_type": "sequential", "data_agent_enabled": self.data_available}
        )
    
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
            goal='Design practical experiments and implementation strategies',
            backstory="""You are the lab's experimental physicist who designs practical approaches to test theories 
            and hypotheses. You focus on feasible experimental setups, especially those requiring minimal resources, 
            while maintaining scientific rigor.""",
            verbose=True,
            allow_delegation=False,
            llm=self.specialist_llm
        )
        
        # QUANTUM SPECIALIST - The quantum expert
        self.quantum_specialist = Agent(
            role='Quantum Specialist',
            goal='Analyze quantum mechanical aspects and quantum technologies',
            backstory="""You are the lab's quantum mechanics expert specializing in quantum phenomena, 
            quantum field theory, and quantum technologies. You provide deep insights into quantum effects 
            and their implications for physics research.""",
            verbose=True,
            allow_delegation=False,
            llm=self.specialist_llm
        )
        
        # RELATIVITY EXPERT - The spacetime specialist
        self.relativity_expert = Agent(
            role='Relativity Expert',
            goal='Analyze relativistic effects and cosmological implications',
            backstory="""You are the lab's relativity specialist with expertise in special and general relativity, 
            cosmology, and spacetime physics. You provide insights into relativistic effects and their implications 
            for physics phenomena.""",
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
        
        # Register data tools with all agents if DataAgent is available
        if self.data_available:
            all_agents = [
                self.lab_director, self.physics_expert, self.hypothesis_generator,
                self.mathematical_analyst, self.experimental_designer, self.quantum_specialist,
                self.relativity_expert, self.condensed_matter_expert, self.computational_physicist,
                self.physics_communicator
            ]
            
            for agent in all_agents:
                register_data_tools_with_agent(agent)
            
            print(f"✅ Registered data analysis tools with {len(all_agents)} physics agents")

    def _log_step_execution(self, step_name: str, agent_role: str, input_data: str, 
                           output_data: str, execution_time: float):
        """Log individual step execution for evaluation."""
        self.knowledge_api.log_event(
            source=agent_role.lower().replace(' ', '_'),
            event_type="step_executed",
            payload={
                "step_name": step_name,
                "input_length": len(input_data),
                "output_length": len(output_data),
                "execution_time": execution_time,
                "question": self.state.question
            }
        )
        
        # Update agent metrics
        self.knowledge_api.update_agent_metrics(
            agent_id=agent_role.lower().replace(' ', '_'),
            task_type="physics_analysis",
            execution_time=execution_time,
            success=bool(output_data and len(output_data) > 50),
            quality_score=min(1.0, len(output_data) / 1000)  # Simple quality heuristic
        )

    @start()
    def coordinate_research(self):
        """Lab Director creates the research coordination plan."""
        start_time = time.time()
        print(f"🔬 Lab Director analyzing question: {self.state.question}")
        
        # Check if data is available for analysis
        data_context = ""
        if self.state.has_data and self.state.data_context:
            data_context = f"""

AVAILABLE DATA CONTEXT:
{self.state.data_context}

Consider how this data can inform the physics analysis."""
        
        task = Task(
            description=f"""As Lab Director, create a comprehensive research coordination plan for: '{self.state.question}'

Analyze the question and create a strategic plan that identifies:
1. Which of our 9 specialist researchers should be involved
2. What specific analysis each specialist should focus on
3. How their contributions will build toward a complete understanding
4. The key physics domains that need to be addressed
5. How to integrate available experimental data (if any) into the analysis{data_context}

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
        
        execution_time = time.time() - start_time
        self.state.coordination_plan = result.raw
        
        # Log execution
        self._log_step_execution(
            "coordinate_research", 
            "Lab Director", 
            self.state.question, 
            result.raw, 
            execution_time
        )
        
        print("✅ Coordination plan created")
        return result.raw

    @listen(coordinate_research)
    def process_uploaded_data(self, coordination_plan):
        """DataAgent processes any uploaded data files for physics analysis."""
        start_time = time.time()
        
        if not self.state.data_jobs or not self.data_available:
            print("📊 No data files to process - proceeding with theoretical analysis")
            self.state.data_insights = "No experimental data available"
            self.state.data_context = "Analysis based on theoretical principles only"
            return "No data files uploaded"
        
        print("📊 DataAgent processing uploaded data files...")
        
        try:
            data_insights = []
            data_contexts = []
            
            for job_id in self.state.data_jobs:
                # Get data status and insights
                status = asyncio.run(self.data_agent.status(job_id))
                
                if status.get("status") == "completed":
                    # Get physics insights
                    insights = self.data_agent.get_physics_insights(job_id)
                    preview = asyncio.run(self.data_agent.preview(job_id, 5))
                    
                    # Build context
                    context_info = f"""
Data File: {status.get('file_path', 'Unknown')}
Rows: {status.get('rows', 0)}, Columns: {status.get('columns', 0)}
File Type: {status.get('mime_type', 'Unknown')}
Physics Insights: {insights.get('physics_patterns', [])}
Detected Units: {insights.get('unit_detection', {}).get('detected_units', {})}
Data Type: {insights.get('data_type', 'experimental')}
"""
                    data_contexts.append(context_info)
                    
                    # Add insights
                    insight_summary = f"Data from {status.get('file_path', 'file')}: {', '.join(insights.get('physics_patterns', ['Standard experimental data']))}"
                    data_insights.append(insight_summary)
                    
                    # Publish data to lab memory
                    try:
                        publish_result = asyncio.run(self.data_agent.publish(job_id))
                        print(f"📈 {publish_result}")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not publish data: {e}")
                
                else:
                    print(f"⚠️ Warning: Job {job_id} status: {status.get('status', 'unknown')}")
            
            # Update state with processed data
            self.state.data_insights = "\n".join(data_insights) if data_insights else "Data processing completed"
            self.state.data_context = "\n".join(data_contexts) if data_contexts else "Processed data available"
            
            execution_time = time.time() - start_time
            
            # Log execution
            self._log_step_execution(
                "process_uploaded_data",
                "DataAgent", 
                coordination_plan,
                self.state.data_insights,
                execution_time
            )
            
            print("✅ Data processing complete")
            return self.state.data_insights
            
        except Exception as e:
            error_msg = f"Data processing failed: {str(e)}"
            print(f"❌ {error_msg}")
            self.state.data_insights = error_msg
            self.state.data_context = "Data processing encountered errors"
            return error_msg

    @listen(process_uploaded_data)
    def theoretical_analysis(self, data_processing_result):
        """Senior Physics Expert provides theoretical foundation."""
        start_time = time.time()
        print("🧠 Senior Physics Expert conducting theoretical analysis...")
        
        # Include data context if available
        data_analysis_context = ""
        if self.state.has_data and self.state.data_insights:
            data_analysis_context = f"""

DATA ANALYSIS CONTEXT:
{self.state.data_insights}

DATA DETAILS:
{self.state.data_context}

Consider how this experimental data relates to the theoretical analysis."""
        
        task = Task(
            description=f"""Based on the coordination plan, provide comprehensive theoretical analysis for: '{self.state.question}'

Coordination Plan: {self.state.coordination_plan}
Data Processing Result: {data_processing_result}{data_analysis_context}

Focus on:
1. Fundamental physics principles involved
2. Theoretical frameworks that apply
3. Key physics laws and concepts
4. Theoretical predictions and implications
5. Connection to established physics theory
6. How experimental data (if available) supports or challenges theory

Provide rigorous theoretical foundation that other specialists can build upon.""",
            expected_output="Comprehensive theoretical analysis with fundamental principles and frameworks",
            agent=self.physics_expert
        )
        
        crew = Crew(agents=[self.physics_expert], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        
        execution_time = time.time() - start_time
        self.state.theoretical_analysis = result.raw
        
        # Log execution
        self._log_step_execution(
            "theoretical_analysis", 
            "Senior Physics Expert", 
            data_processing_result, 
            result.raw, 
            execution_time
        )
        
        print("✅ Theoretical analysis complete")
        return result.raw

    @listen(theoretical_analysis)
    def generate_hypotheses(self, theoretical_analysis):
        """Hypothesis Generator creates novel approaches."""
        start_time = time.time()
        print("💡 Hypothesis Generator developing creative approaches...")
        
        task = Task(
            description=f"""Generate creative hypotheses and novel approaches for: '{self.state.question}'

Building on theoretical foundation: {theoretical_analysis}

Provide:
1. Creative and innovative hypotheses
2. Novel approaches to the problem
3. Unconventional ideas that might be overlooked
4. Speculative but scientifically grounded possibilities
5. Alternative perspectives on the question

Balance creativity with scientific rigor.""",
            expected_output="Creative hypotheses and innovative approaches with scientific basis",
            agent=self.hypothesis_generator
        )
        
        crew = Crew(agents=[self.hypothesis_generator], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()
        
        execution_time = time.time() - start_time
        self.state.creative_hypotheses = result.raw
        
        # Log execution and record hypotheses
        self._log_step_execution(
            "generate_hypotheses", 
            "Hypothesis Generator", 
            theoretical_analysis, 
            result.raw, 
            execution_time
        )
        
        # Record hypothesis in database
        self.knowledge_api.record_hypothesis(
            title=f"Creative approaches for: {self.state.question}",
            description=result.raw[:500],
            created_by="Hypothesis Generator",
            confidence_score=0.7
        )
        
        print("✅ Creative hypotheses generated")
        return result.raw

    @listen(generate_hypotheses)
    def mathematical_modeling(self, hypotheses):
        """Mathematical Analyst performs calculations and modeling."""
        start_time = time.time()
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
        
        execution_time = time.time() - start_time
        self.state.mathematical_models = result.raw
        
        # Log execution
        self._log_step_execution(
            "mathematical_modeling", 
            "Mathematical Analyst", 
            hypotheses, 
            result.raw, 
            execution_time
        )
        
        print("✅ Mathematical modeling complete")
        return result.raw

    @listen(mathematical_modeling)
    def experimental_design(self, math_models):
        """Experimental Designer creates practical approaches."""
        start_time = time.time()
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
        
        execution_time = time.time() - start_time
        self.state.experimental_design = result.raw
        
        # Log execution
        self._log_step_execution(
            "experimental_design", 
            "Experimental Designer", 
            math_models, 
            result.raw, 
            execution_time
        )
        
        print("✅ Experimental design complete")
        return result.raw

    @listen(experimental_design)
    def quantum_analysis(self, experimental_design):
        """Quantum Specialist analyzes quantum aspects."""
        start_time = time.time()
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
        
        execution_time = time.time() - start_time
        self.state.quantum_analysis = result.raw
        
        # Log execution
        self._log_step_execution(
            "quantum_analysis", 
            "Quantum Specialist", 
            experimental_design, 
            result.raw, 
            execution_time
        )
        
        print("✅ Quantum analysis complete")
        return result.raw

    @listen(quantum_analysis)
    def computational_simulation(self, quantum_analysis):
        """Computational Physicist develops simulation strategies."""
        start_time = time.time()
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
        
        execution_time = time.time() - start_time
        self.state.computational_simulations = result.raw
        
        # Log execution
        self._log_step_execution(
            "computational_simulation", 
            "Computational Physicist", 
            quantum_analysis, 
            result.raw, 
            execution_time
        )
        
        print("✅ Computational simulations designed")
        return result.raw

    @listen(computational_simulation)
    def synthesize_research(self, computational_work):
        """Physics Communicator synthesizes all specialist contributions."""
        start_time = time.time()
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
        
        execution_time = time.time() - start_time
        self.state.final_synthesis = result.raw
        
        # Log execution
        self._log_step_execution(
            "synthesize_research", 
            "Physics Communicator", 
            computational_work, 
            result.raw, 
            execution_time
        )
        
        # Add final result to knowledge base
        self.knowledge_api.add_knowledge_entry(
            title=f"Physics Laboratory Analysis: {self.state.question}",
            content=result.raw,
            physics_domain="multi_domain",
            source_agents=["Lab Director", "Senior Physics Expert", "Hypothesis Generator", 
                          "Mathematical Analyst", "Experimental Designer", "Quantum Specialist", 
                          "Computational Physicist", "Physics Communicator"],
            confidence_level=0.85
        )
        
        print("✅ Final synthesis complete")
        return result.raw

def analyze_physics_question_with_flow(question: str) -> str:
    """
    Analyze a physics question using the modern Flow-based laboratory system with evaluation.
    
    Args:
        question: The physics question to analyze
        
    Returns:
        Comprehensive physics analysis from the flow-based laboratory team
    """
    
    print(f"🔬 Starting physics laboratory research: {question}")
    print("=" * 80)
    print("🏛️  Physics Laboratory Flow Activated")
    print("👨‍🔬 Modern event-driven orchestration with 10 specialists")
    print("📊 Integrated database & evaluation framework")
    print("⚛️  Full-spectrum physics analysis in progress...")
    print("=" * 80)
    
    overall_start_time = time.time()
    
    # Initialize and run the flow
    flow = PhysicsLabFlow()
    
    # Start evaluation session
    session_id = f"session_{int(time.time())}"
    flow.evaluation_framework.start_evaluation_session(session_id)
    
    # Execute the flow
    result = flow.kickoff(inputs={"question": question})
    
    overall_execution_time = time.time() - overall_start_time
    
    # Evaluate the overall execution
    # Create a mock crew for evaluation (Flow doesn't directly expose crew)
    mock_crew = type('MockCrew', (), {
        'agents': [
            type('Agent', (), {'role': 'Lab Director'})(),
            type('Agent', (), {'role': 'Senior Physics Expert'})(),
            type('Agent', (), {'role': 'Hypothesis Generator'})(),
            type('Agent', (), {'role': 'Mathematical Analyst'})(),
            type('Agent', (), {'role': 'Experimental Designer'})(),
            type('Agent', (), {'role': 'Quantum Specialist'})(),
            type('Agent', (), {'role': 'Computational Physicist'})(),
            type('Agent', (), {'role': 'Physics Communicator'})()
        ]
    })()
    
    evaluation_result = flow.evaluation_framework.evaluate_crew_execution(
        crew=mock_crew,
        query=question,
        result=result,
        execution_time=overall_execution_time
    )
    
    print("=" * 80)
    print("✅ Laboratory research complete!")
    print("📋 Analysis ready for review")
    print(f"📊 Evaluation Score: {evaluation_result['quality_score']:.2f}")
    print(f"⏱️  Total Execution Time: {overall_execution_time:.2f}s")
    print(f"🤖 Agents Involved: {len(evaluation_result['agents_involved'])}")
    
    return result

def main():
    """Main entry point for the Physics Laboratory Flow System."""
    
    print("🏛️  PhysicsGPT Flow - 10-Agent Physics Research Laboratory")
    print("=" * 70)
    print("🔬 Modern Event-Driven Physics Lab Orchestration")
    print("👨‍🔬 Lab Director + 9 Specialized Researchers")
    print("📊 Integrated Database & Evaluation Framework")
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
        
        # Show evaluation metrics
        print("\n" + "="*70)
        print("📊 SYSTEM PERFORMANCE METRICS:")
        print("="*70)
        
        # Get performance report
        flow = PhysicsLabFlow()
        report = flow.evaluation_framework.get_performance_report()
        
        print("📈 System Analytics:")
        for key, value in report['system_analytics'].items():
            print(f"  - {key}: {value}")
            
    else:
        print("\n💡 Usage: python physics_flow_system.py 'your physics question'")
        print("💡 Example: python physics_flow_system.py 'how to detect dark matter with minimal stuff?'")

if __name__ == "__main__":
    main() 