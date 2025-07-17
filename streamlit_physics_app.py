#!/usr/bin/env python3
"""
PhysicsGPT - Professional Physics Research Laboratory System
Real-time telemetry monitoring with comprehensive knowledge management.
"""

import streamlit as st
import os
import sys
import time
from datetime import datetime
from typing import Optional, List, Dict

# Fix SQLite version issue for Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Flow system with error handling
try:
    from physics_flow_system import analyze_physics_question_with_flow
    FLOW_AVAILABLE = True
except ImportError as e:
    FLOW_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Professional CSS styling
def load_professional_css():
    """Load professional CSS styling."""
    st.markdown("""
    <style>
    /* Main app styling */
    .main > div {
        padding: 1rem 2rem;
    }
    
    /* Header styling */
    .system-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .system-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .system-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Status indicators */
    .status-success {
        background-color: #059669;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .status-warning {
        background-color: #d97706;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .status-error {
        background-color: #dc2626;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .status-info {
        background-color: #2563eb;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 500;
    }
    
    /* Section headers */
    .section-header {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 4px 4px 0;
    }
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e293b;
        margin: 0;
    }
    
    /* Telemetry styling */
    .telemetry-container {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .telemetry-title {
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.5rem;
    }
    
    /* Progress indicators */
    .progress-container {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Card styling */
    .info-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8fafc;
        border-radius: 4px 4px 0 0;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    
    /* Remove streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
    """, unsafe_allow_html=True)

class FlowExecutionMonitor:
    """Monitor flow execution with telemetry."""
    
    def __init__(self):
        self.start_time = time.time()
        self.current_step = 0
        self.total_steps = 8
        self.execution_log = []
        self.agent_outputs = {}
        
    def log_step(self, step_name: str, status: str, details: str):
        """Log a step execution."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.execution_log.append({
            'timestamp': timestamp,
            'step': step_name,
            'status': status,
            'details': details
        })
        
        if status == "started":
            self.current_step += 1
            
    def get_progress(self) -> float:
        """Get current progress as a float between 0.0 and 1.0."""
        return min(1.0, max(0.0, self.current_step / self.total_steps))

def safe_progress_update(progress_bar, value: float):
    """Safely update progress bar with normalized value."""
    normalized_value = max(0.0, min(1.0, value))
    try:
        progress_bar.progress(normalized_value)
    except Exception as e:
        st.error(f"Progress update error: {e}")

def run_physics_analysis_with_telemetry(question: str):
    """Run physics analysis with comprehensive telemetry monitoring."""
    
    # Initialize monitor
    monitor = FlowExecutionMonitor()
    
    # Flow steps for telemetry
    flow_steps = [
        "initialize_lab", "director_analysis", "physics_expert_review",
        "hypothesis_generation", "mathematical_analysis", "experimental_design",
        "specialist_consultation", "final_synthesis"
    ]
    
    step_names = [
        "Laboratory Initialization",
        "Lab Director Analysis", 
        "Senior Physics Expert Review",
        "Hypothesis Generation",
        "Mathematical Analysis",
        "Experimental Design",
        "Specialist Consultation",
        "Final Synthesis & Report"
    ]
    
    # Create telemetry UI
    st.markdown('<div class="section-header"><h3 class="section-title">Physics Laboratory Analysis Execution</h3></div>', unsafe_allow_html=True)
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            progress_bar = st.progress(0.0)
        with col2:
            status_text = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Live monitoring containers
    live_output = st.container()
    
    # Telemetry containers
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="telemetry-container">', unsafe_allow_html=True)
        st.markdown('<div class="telemetry-title">Flow Progress</div>', unsafe_allow_html=True)
        flow_status = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="telemetry-container">', unsafe_allow_html=True)
        st.markdown('<div class="telemetry-title">Execution Log</div>', unsafe_allow_html=True)
        telemetry_log = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Start execution monitoring
    with live_output:
        st.markdown('<div class="status-info">Initializing Physics Laboratory Flow System...</div>', unsafe_allow_html=True)
        
        # Simulate flow step progression for telemetry
        for i, (step_name, display_name) in enumerate(zip(flow_steps, step_names)):
            monitor.log_step(step_name, "started", f"Executing {display_name}")
            
            # Update progress
            progress = monitor.get_progress()
            safe_progress_update(progress_bar, progress)
            status_text.text(f"Step {monitor.current_step}/{monitor.total_steps}")
            
            # Update flow status
            with flow_status:
                flow_data = []
                for j, name in enumerate(step_names):
                    if j < monitor.current_step:
                        status = "COMPLETED"
                    elif j == monitor.current_step:
                        status = "IN PROGRESS"
                    else:
                        status = "PENDING"
                    
                    flow_data.append({
                        'Step': f"{j+1}. {name}",
                        'Status': status
                    })
                
                st.dataframe(flow_data, use_container_width=True, hide_index=True)
            
            # Update telemetry log
            with telemetry_log:
                log_data = []
                for entry in monitor.execution_log[-5:]:  # Show last 5 entries
                    log_data.append({
                        'Time': entry['timestamp'],
                        'Step': entry['step'],
                        'Status': entry['status'].upper(),
                        'Details': entry['details'][:30] + "..." if len(entry['details']) > 30 else entry['details']
                    })
                
                if log_data:
                    st.dataframe(log_data, use_container_width=True, hide_index=True)
            
            # Small delay for UI updates
            time.sleep(0.3)
        
        # Execute the actual flow analysis
        status_text.text("Running Comprehensive Analysis...")
        monitor.log_step("full_execution", "started", "Running complete flow")
        
        with st.spinner("Physics specialists collaborating..."):
            try:
                start_time = time.time()
                result = analyze_physics_question_with_flow(question)
                end_time = time.time()
                execution_success = True
            except Exception as e:
                st.error(f"Flow execution failed: {e}")
                result = f"Flow execution encountered an error: {str(e)}"
                execution_success = False
                end_time = time.time()
        
        monitor.log_step("full_execution", "completed" if execution_success else "failed", "Flow analysis complete")
        
        # Final progress update
        monitor.current_step = len(flow_steps)
        safe_progress_update(progress_bar, 1.0)
        status_text.text("Analysis Complete" if execution_success else "Analysis Failed")
        
        # Display results
        execution_time = end_time - start_time
        
        if execution_success:
            st.markdown(f'<div class="status-success">Analysis completed successfully in {execution_time:.1f} seconds</div>', unsafe_allow_html=True)
            
            # Results section
            st.markdown('<div class="section-header"><h3 class="section-title">Research Report</h3></div>', unsafe_allow_html=True)
            with st.expander("Full Analysis Report", expanded=True):
                st.markdown(result)
            
            # Download option
            st.download_button(
                label="Download Report",
                data=result,
                file_name=f"physics_analysis_{int(time.time())}.txt",
                mime="text/plain"
            )
        else:
            st.markdown('<div class="status-error">Analysis failed - please try again</div>', unsafe_allow_html=True)

def load_knowledge_data():
    """Load knowledge base and hypothesis data."""
    try:
        from crewai_database_integration import CrewAIKnowledgeAPI
        import sqlite3
        
        knowledge_api = CrewAIKnowledgeAPI()
        
        # Get knowledge entries
        knowledge_entries = []
        try:
            with sqlite3.connect(knowledge_api.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, content, physics_domain, 
                           confidence_level, created_at, source_agents
                    FROM knowledge_entries ORDER BY created_at DESC
                """)
                knowledge_entries = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            st.error(f"Error loading knowledge entries: {e}")
        
        # Get hypotheses
        hypotheses = []
        try:
            with sqlite3.connect(knowledge_api.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, description, confidence_score, 
                           validation_status, created_by, created_at
                    FROM hypotheses ORDER BY created_at DESC
                """)
                hypotheses = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            st.error(f"Error loading hypotheses: {e}")
        
        return knowledge_entries, hypotheses
        
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return [], []

def display_knowledge_base():
    """Display the knowledge base tab."""
    st.markdown('<div class="section-header"><h3 class="section-title">Physics Knowledge Base</h3></div>', unsafe_allow_html=True)
    
    knowledge_entries, _ = load_knowledge_data()
    
    if not knowledge_entries:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.info("No knowledge entries found. Run some physics analyses to populate the knowledge base.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Search functionality
    search_term = st.text_input("Search knowledge base", placeholder="Enter search terms...")
    
    if search_term:
        filtered_entries = [
            entry for entry in knowledge_entries 
            if search_term.lower() in entry.get('title', '').lower() or 
               search_term.lower() in entry.get('content', '').lower()
        ]
    else:
        filtered_entries = knowledge_entries
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Entries", len(knowledge_entries))
    with col2:
        domains = set(entry.get('physics_domain', 'unknown') for entry in knowledge_entries)
        st.metric("Physics Domains", len(domains))
    with col3:
        avg_confidence = sum(entry.get('confidence_level', 0) for entry in knowledge_entries) / len(knowledge_entries) if knowledge_entries else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Display entries
    for entry in filtered_entries:
        with st.expander(f"{entry.get('title', 'Untitled')} | Domain: {entry.get('physics_domain', 'Unknown')}"):
            st.write(f"**Content:** {entry.get('content', 'No content')}")
            st.write(f"**Confidence Level:** {entry.get('confidence_level', 0):.2f}")
            st.write(f"**Created:** {entry.get('created_at', 'Unknown')}")
            st.write(f"**Source Agents:** {entry.get('source_agents', 'Unknown')}")

def display_hypothesis_tracker():
    """Display the hypothesis tracker tab."""
    st.markdown('<div class="section-header"><h3 class="section-title">Hypothesis Research Tracker</h3></div>', unsafe_allow_html=True)
    
    _, hypotheses = load_knowledge_data()
    
    if not hypotheses:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.info("No hypotheses found. Run physics analyses to generate and track hypotheses.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Status filter
    status_options = ["All"] + list(set(h.get('validation_status', 'pending') for h in hypotheses))
    selected_status = st.selectbox("Filter by Status", status_options)
    
    if selected_status != "All":
        filtered_hypotheses = [h for h in hypotheses if h.get('validation_status') == selected_status]
    else:
        filtered_hypotheses = hypotheses
    
    # Status distribution
    status_counts = {}
    for h in hypotheses:
        status = h.get('validation_status', 'pending')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Hypotheses", len(hypotheses))
    with col2:
        st.metric("Pending", status_counts.get('pending', 0))
    with col3:
        st.metric("Under Review", status_counts.get('under_review', 0))
    with col4:
        st.metric("Validated", status_counts.get('validated', 0))
    
    # Research phases progress
    phases = ['pending', 'under_review', 'validated', 'refuted', 'promoted']
    phase_progress = {phase: status_counts.get(phase, 0) for phase in phases}
    
    st.markdown("**Research Phase Distribution:**")
    for phase, count in phase_progress.items():
        if count > 0:
            st.write(f"**{phase.replace('_', ' ').title()}:** {count} hypotheses")
    
    # Display hypotheses
    for hypothesis in filtered_hypotheses:
        status = hypothesis.get('validation_status', 'pending')
        confidence = hypothesis.get('confidence_score', 0)
        
        # Status styling
        if status == 'validated':
            status_class = 'status-success'
        elif status == 'under_review':
            status_class = 'status-warning'
        elif status == 'refuted':
            status_class = 'status-error'
        else:
            status_class = 'status-info'
        
        with st.expander(f"{hypothesis.get('title', 'Untitled Hypothesis')} | Confidence: {confidence:.2f}"):
            st.markdown(f'<div class="{status_class}">Status: {status.replace("_", " ").title()}</div>', unsafe_allow_html=True)
            st.write(f"**Description:** {hypothesis.get('description', 'No description')}")
            st.write(f"**Created by:** {hypothesis.get('created_by', 'Unknown')}")
            st.write(f"**Created:** {hypothesis.get('created_at', 'Unknown')}")
            
            # Progress bar for confidence
            st.progress(confidence)

def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="PhysicsGPT Research Laboratory",
        page_icon="⚛",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load professional styling
    load_professional_css()
    
    # Header
    st.markdown("""
    <div class="system-header">
        <h1 class="system-title">PhysicsGPT Research Laboratory</h1>
        <p class="system-subtitle">Advanced Multi-Agent Physics Analysis System | Real-Time Telemetry Monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System status check
    if not FLOW_AVAILABLE:
        st.markdown('<div class="status-error">System Unavailable: Flow system could not be loaded</div>', unsafe_allow_html=True)
        st.error(f"Import Error: {IMPORT_ERROR}")
        st.stop()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Physics Analysis", 
        "System Status", 
        "Knowledge Base", 
        "Hypothesis Tracker"
    ])
    
    with tab1:
        st.markdown('<div class="section-header"><h3 class="section-title">Physics Question Analysis</h3></div>', unsafe_allow_html=True)
        
        # Question input
        question = st.text_area(
            "Enter your physics question:",
            placeholder="Example: What is the relationship between force, mass, and acceleration in classical mechanics?",
            height=100
        )
        
        if st.button("Analyze Physics Question", type="primary"):
            if question.strip():
                run_physics_analysis_with_telemetry(question)
            else:
                st.warning("Please enter a physics question to analyze.")
    
    with tab2:
        st.markdown('<div class="section-header"><h3 class="section-title">System Status & Information</h3></div>', unsafe_allow_html=True)
        
        # System info
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**System Architecture:** 10 Specialized Physics Agents")
        st.markdown("**Flow Engine:** CrewAI Flows with Event-Driven Architecture")  
        st.markdown("**Database:** SQLite with Real-Time Analytics")
        st.markdown("**Telemetry:** Comprehensive Flow Monitoring")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Agent information
        agents_info = [
            ("Lab Director", "Orchestrates research workflow and ensures scientific rigor"),
            ("Senior Physics Expert", "Provides high-level physics expertise and validation"),
            ("Hypothesis Generator", "Creates testable scientific hypotheses"),
            ("Mathematical Analyst", "Performs quantitative analysis and modeling"),
            ("Experimental Designer", "Designs experiments to test hypotheses"),
            ("Quantum Specialist", "Expert in quantum mechanics and phenomena"),
            ("Relativity Expert", "Specialist in special and general relativity"),
            ("Condensed Matter Expert", "Expert in solid-state and condensed matter physics"),
            ("Computational Physicist", "Performs computational modeling and simulations"),
            ("Physics Communicator", "Translates complex physics into clear explanations")
        ]
        
        st.markdown("**Agent Specifications:**")
        for agent, description in agents_info:
            st.markdown(f"• **{agent}:** {description}")
    
    with tab3:
        display_knowledge_base()
    
    with tab4:
        display_hypothesis_tracker()

if __name__ == "__main__":
    main()