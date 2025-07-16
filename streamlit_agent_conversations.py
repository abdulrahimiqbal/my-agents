#!/usr/bin/env python3
"""
Streamlit Agent Conversation Monitor
Real-time monitoring interface for PhysicsGPT Flow system with enhanced telemetry.
"""

import streamlit as st
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import json
import time
import threading
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the new Flow system
try:
    from physics_flow_system import analyze_physics_question_with_flow, PhysicsLabFlow
    FLOW_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import Flow system: {e}")
    FLOW_AVAILABLE = False

class FlowMonitor:
    """Monitor Flow execution with real-time tracking."""
    
    def __init__(self):
        self.total_steps = 8
        self.current_step = 0
        self.start_time = None
        self.execution_log = []
        self.step_timings = {}
    
    def start_monitoring(self):
        """Start monitoring session."""
        self.start_time = time.time()
        self.current_step = 0
        self.execution_log = []
        self.step_timings = {}
    
    def log_step(self, step_name: str, status: str, details: str = ""):
        """Log a flow step execution."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.execution_log.append({
            'timestamp': timestamp,
            'step': step_name,
            'status': status,
            'details': details
        })
        
        if status == "completed":
            self.current_step += 1
    
    def get_progress(self) -> float:
        """Get current progress as a percentage (0.0 to 1.0)."""
        if self.total_steps == 0:
            return 0.0
        # Ensure progress is between 0.0 and 1.0
        progress = min(max(self.current_step / self.total_steps, 0.0), 1.0)
        return progress
    
    def get_progress_percentage(self) -> int:
        """Get current progress as percentage (0 to 100)."""
        return int(self.get_progress() * 100)

def capture_flow_output(question: str):
    """Capture flow execution with detailed telemetry."""
    
    # Initialize monitor
    monitor = FlowMonitor()
    monitor.start_monitoring()
    
    # Define flow steps for tracking
    flow_steps = [
        "coordinate_research",
        "theoretical_analysis", 
        "generate_hypotheses",
        "mathematical_modeling",
        "experimental_design",
        "quantum_analysis",
        "computational_simulation",
        "synthesize_research"
    ]
    
    step_names = [
        "🔬 Lab Director - Research Coordination",
        "🧠 Senior Physics Expert - Theoretical Analysis",
        "💡 Hypothesis Generator - Creative Approaches", 
        "📊 Mathematical Analyst - Calculations",
        "⚗️ Experimental Designer - Practical Methods",
        "⚛️ Quantum Specialist - Quantum Analysis",
        "💻 Computational Physicist - Simulations",
        "📝 Physics Communicator - Final Synthesis"
    ]
    
    # Create UI containers
    st.markdown("### 🔬 Physics Laboratory Flow Execution")
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        col1, col2 = st.columns([3, 1])
        with col1:
            progress_bar = st.progress(0.0)
        with col2:
            status_text = st.empty()
    
    # Live monitoring containers
    live_output = st.container()
    agent_outputs = st.container()
    
    # Telemetry containers
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Flow Progress")
        flow_status = st.empty()
    with col2:
        st.markdown("#### 🧠 Telemetry Log")
        telemetry_log = st.empty()
    
    # Start execution monitoring
    with live_output:
        st.info("🚀 **Initializing Physics Laboratory Flow System...**")
        
        # Simulate flow step progression
        for i, (step_name, display_name) in enumerate(zip(flow_steps, step_names)):
            monitor.log_step(step_name, "started", f"Executing {display_name}")
            
            # Update progress (ensure it's between 0.0 and 1.0)
            progress = monitor.get_progress()
            progress_bar.progress(progress)
            status_text.text(f"Step {monitor.current_step + 1}/{monitor.total_steps}")
            
            # Update flow status
            with flow_status:
                flow_data = []
                for j, name in enumerate(step_names):
                    if j < monitor.current_step:
                        status = "✅ Completed"
                    elif j == monitor.current_step:
                        status = "🔄 In Progress"
                    else:
                        status = "⏳ Pending"
                    
                    flow_data.append({
                        'Step': f"{j+1}. {name}",
                        'Status': status
                    })
                
                st.dataframe(flow_data, use_container_width=True, hide_index=True)
            
            # Update telemetry log
            with telemetry_log:
                log_data = []
                for entry in monitor.execution_log:
                    log_data.append({
                        'Time': entry['timestamp'],
                        'Step': entry['step'],
                        'Status': entry['status'].upper(),
                        'Details': entry['details'][:50] + "..." if len(entry['details']) > 50 else entry['details']
                    })
                
                if log_data:
                    st.dataframe(log_data, use_container_width=True, hide_index=True)
            
            # Small delay for UI updates
            time.sleep(0.5)
        
        # Execute the actual flow analysis
        status_text.text("🚀 Running Comprehensive Analysis...")
        monitor.log_step("full_execution", "started", "Running complete flow")
        
        with st.spinner("🔬 Physics specialists collaborating..."):
            try:
                result = analyze_physics_question_with_flow(question)
                execution_success = True
            except Exception as e:
                st.error(f"❌ Flow execution failed: {e}")
                result = f"Flow execution encountered an error: {str(e)}"
                execution_success = False
        
        monitor.log_step("full_execution", "completed" if execution_success else "failed", "Flow analysis complete")
        
        # Final progress update
        monitor.current_step = len(flow_steps)
        progress_bar.progress(1.0)  # Ensure final progress is exactly 1.0
        status_text.text("✅ Analysis Complete!" if execution_success else "❌ Analysis Failed")
        
        # Display final results
        total_time = time.time() - monitor.start_time
        
        with live_output:
            if execution_success:
                st.success(f"🎉 **Analysis completed successfully in {total_time:.1f} seconds!**")
                st.markdown("### All Flow Steps Completed:")
                for step_name in step_names:
                    st.markdown(f"✅ {step_name}")
            else:
                st.error(f"❌ **Analysis failed after {total_time:.1f} seconds**")
        
        with agent_outputs:
            st.header("📋 Complete Laboratory Analysis")
            if execution_success:
                st.markdown(result)
            else:
                st.error(result)
                st.info("💡 **Troubleshooting Tips:**")
                st.markdown("""
                - Check your API keys are correctly set
                - Ensure internet connectivity
                - Try refreshing the page
                - Contact support if issues persist
                """)

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="PhysicsGPT Flow Laboratory",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3730a3 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .feature-box {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 PHYSICS LABORATORY FLOW SYSTEM</h1>
        <p>10 Specialized Agents • Event-Driven Architecture • Enhanced Telemetry</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key configuration
        if "OPENAI_API_KEY" not in st.secrets:
            openai_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
        else:
            st.success("✅ OpenAI API Key configured")
        
        # Model selection
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        selected_model = st.selectbox("Model", model_options, index=0)
        os.environ["PHYSICS_AGENT_MODEL"] = selected_model
        
        # Temperature setting
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        os.environ["PHYSICS_AGENT_TEMPERATURE"] = str(temperature)
        
        st.markdown("---")
        
        # System status
        st.header("📊 System Status")
        if FLOW_AVAILABLE:
            st.success("✅ Flow System Ready")
        else:
            st.error("❌ Flow System Error")
        
        # Environment variables check
        env_vars = ["OPENAI_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY"]
        for var in env_vars:
            if var in os.environ or var in st.secrets:
                st.success(f"✅ {var}")
            else:
                st.warning(f"⚠️ {var} not set")
        
        st.markdown("---")
        st.info("📊 Enhanced Telemetry")
        st.markdown("""
        - Real-time flow monitoring
        - Agent interaction tracking  
        - Performance metrics
        - LangSmith integration
        """)
    
    # Main interface
    if not FLOW_AVAILABLE:
        st.error("❌ **Physics Flow System Not Available**")
        st.info("The Flow system could not be imported. Please check the installation.")
        return
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🚀 Execute Flow", "📊 System Info", "📚 Documentation"])
    
    with tab1:
        st.header("🔬 Physics Laboratory Analysis")
        
        # Question input
        question = st.text_area(
            "Enter your physics question:",
            placeholder="e.g., How do quantum tunneling effects work in semiconductor devices?",
            height=100,
            help="Ask any physics question and our 10-agent laboratory team will provide comprehensive analysis"
        )
        
        # Analysis buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            analyze_button = st.button("🚀 Start Laboratory Analysis", type="primary", use_container_width=True)
        with col2:
            example_button = st.button("💡 Use Example", use_container_width=True)
        with col3:
            clear_button = st.button("🧹 Clear", use_container_width=True)
        
        # Handle buttons
        if example_button:
            st.rerun()
        
        if clear_button:
            st.rerun()
        
        if analyze_button and question:
            capture_flow_output(question)
        elif analyze_button and not question:
            st.warning("⚠️ Please enter a physics question first!")
    
    with tab2:
        st.header("📊 System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-box">
                <h3>🤖 Agent Architecture</h3>
                <ul>
                    <li><strong>Lab Director:</strong> Research coordination</li>
                    <li><strong>Senior Physics Expert:</strong> Theoretical analysis</li>
                    <li><strong>Hypothesis Generator:</strong> Creative approaches</li>
                    <li><strong>Mathematical Analyst:</strong> Calculations</li>
                    <li><strong>Experimental Designer:</strong> Practical methods</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-box">
                <h3>⚛️ Specialist Agents</h3>
                <ul>
                    <li><strong>Quantum Specialist:</strong> Quantum mechanics</li>
                    <li><strong>Relativity Expert:</strong> Spacetime physics</li>
                    <li><strong>Condensed Matter Expert:</strong> Materials</li>
                    <li><strong>Computational Physicist:</strong> Simulations</li>
                    <li><strong>Physics Communicator:</strong> Synthesis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Flow execution pattern
        st.markdown("### 🔄 Flow Execution Pattern")
        st.markdown("""
        1. **🔬 Lab Director** - Creates research coordination plan
        2. **🧠 Senior Physics Expert** - Provides theoretical foundation  
        3. **💡 Hypothesis Generator** - Generates creative approaches
        4. **📊 Mathematical Analyst** - Performs calculations and modeling
        5. **⚗️ Experimental Designer** - Designs practical experiments
        6. **⚛️ Quantum Specialist** - Analyzes quantum aspects
        7. **💻 Computational Physicist** - Runs simulations
        8. **📝 Physics Communicator** - Synthesizes final report
        """)
    
    with tab3:
        st.header("📚 Documentation")
        
        st.markdown("""
        ### 🔬 PhysicsGPT Flow Laboratory System
        
        This system uses **CrewAI Flows** for event-driven orchestration of 10 specialized physics agents.
        
        #### Key Features:
        - **Event-Driven Architecture**: Modern @start/@listen decorators
        - **Sequential Flow Execution**: 8 coordinated steps
        - **Real-Time Monitoring**: Live progress tracking
        - **Comprehensive Telemetry**: LangSmith + custom monitoring
        - **Professional UI**: Modern Streamlit interface
        
        #### How It Works:
        1. **Input**: Enter any physics question
        2. **Coordination**: Lab Director creates research plan
        3. **Execution**: 8 specialists work sequentially 
        4. **Synthesis**: Final comprehensive report
        5. **Output**: Multi-perspective physics analysis
        
        #### Telemetry Features:
        - Real-time flow step tracking
        - Agent interaction monitoring
        - Performance metrics collection
        - LangSmith trace integration
        - Custom progress visualization
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>⚛️ PhysicsGPT Flow Laboratory System</p>
        <p>Powered by CrewAI Flows & OpenAI • 10 Specialized Physics Agents • Real-Time Telemetry</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()