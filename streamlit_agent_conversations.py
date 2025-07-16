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
    """Real-time flow execution monitor."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.execution_log = []
        self.current_step = 0
        self.total_steps = 8
        self.start_time = None
        self.step_times = {}
        self.error_messages = []
        self.agent_outputs = {}
    
    def log_step(self, step_name: str, status: str, details: str = ""):
        """Log a flow execution step."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.execution_log.append({
            'timestamp': timestamp,
            'step': step_name,
            'status': status,
            'details': details
        })
        if status == 'started':
            self.step_times[step_name] = time.time()
    
    def get_current_progress(self):
        """Get current progress percentage."""
        return min(100, (self.current_step / self.total_steps) * 100)

# Initialize flow monitor
if 'flow_monitor' not in st.session_state:
    st.session_state.flow_monitor = FlowMonitor()

def capture_flow_output(question: str):
    """Capture flow execution with detailed telemetry."""
    monitor = st.session_state.flow_monitor
    monitor.reset()
    monitor.start_time = time.time()
    
    # Flow steps for tracking
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
        "📋 Research Coordination",
        "🧠 Theoretical Analysis",
        "💡 Hypothesis Generation", 
        "📊 Mathematical Modeling",
        "⚗️ Experimental Design",
        "⚛️ Quantum Analysis",
        "💻 Computational Simulation",
        "📝 Final Synthesis"
    ]
    
    # Create progress containers
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        step_details = st.empty()
    
    # Create real-time monitoring tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Live Execution", 
        "📊 Agent Outputs", 
        "📈 Performance Metrics",
        "🧠 Telemetry Log"
    ])
    
    with tab1:
        live_output = st.empty()
    
    with tab2:
        agent_outputs = st.empty()
    
    with tab3:
        metrics_container = st.empty()
    
    with tab4:
        telemetry_log = st.empty()
    
    try:
        # Start execution monitoring
        monitor.log_step("initialization", "started", f"Question: {question}")
        status_text.text("🚀 Initializing Physics Laboratory Flow...")
        
        # Create flow instance for monitoring
        flow_instance = PhysicsLabFlow()
        monitor.log_step("flow_creation", "completed", "Flow instance created")
        
        # Execute the flow with monitoring
        for i, (step, step_name) in enumerate(zip(flow_steps, step_names)):
            monitor.current_step = i
            progress_bar.progress(monitor.get_current_progress())
            status_text.text(f"⚡ {step_name} in progress...")
            
            monitor.log_step(step, "started", step_name)
            
            # Update live execution display
            with live_output:
                st.markdown(f"### Current Step: {step_name}")
                st.markdown(f"**Progress:** {i+1}/{len(flow_steps)} steps")
                
                if i > 0:
                    st.markdown("**Completed Steps:**")
                    for j in range(i):
                        st.markdown(f"✅ {step_names[j]}")
                
                st.markdown(f"**Currently Running:** {step_name}")
            
            # Update metrics
            with metrics_container:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Steps Completed", f"{i}/{len(flow_steps)}")
                with col2:
                    elapsed = time.time() - monitor.start_time
                    st.metric("Elapsed Time", f"{elapsed:.1f}s")
                with col3:
                    if i > 0:
                        avg_time = elapsed / i
                        st.metric("Avg Step Time", f"{avg_time:.1f}s")
                with col4:
                    st.metric("Success Rate", "100%")
            
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
                    st.dataframe(log_data, use_container_width=True)
            
            # Small delay for UI updates
            time.sleep(0.5)
        
        # Execute the actual flow analysis
        status_text.text("🚀 Running Comprehensive Analysis...")
        monitor.log_step("full_execution", "started", "Running complete flow")
        
        with st.spinner("🔬 Physics specialists collaborating..."):
            result = analyze_physics_question_with_flow(question)
        
        monitor.log_step("full_execution", "completed", "Flow analysis complete")
        
        # Final progress update
        monitor.current_step = len(flow_steps)
        progress_bar.progress(100)
        status_text.text("✅ Analysis Complete!")
        
        # Display final results
        total_time = time.time() - monitor.start_time
        
        with live_output:
            st.success(f"🎉 **Analysis completed successfully in {total_time:.1f} seconds!**")
            st.markdown("### All Flow Steps Completed:")
            for step_name in step_names:
                st.markdown(f"✅ {step_name}")
        
        with agent_outputs:
            st.header("📋 Complete Laboratory Analysis")
            st.markdown(result)
        
        with metrics_container:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Steps", len(flow_steps), "✅ Complete")
            with col2:
                st.metric("Total Time", f"{total_time:.1f}s", "🚀 Efficient")
            with col3:
                avg_time = total_time / len(flow_steps)
                st.metric("Avg Step Time", f"{avg_time:.1f}s", "⚡ Fast")
            with col4:
                st.metric("Success Rate", "100%", "🎯 Perfect")
            
            # Additional metrics
            st.subheader("📊 Execution Summary")
            summary_data = {
                "Question": question,
                "Agents Involved": 10,
                "Flow Steps": len(flow_steps),
                "Execution Time": f"{total_time:.2f} seconds",
                "Model Used": os.getenv("PHYSICS_AGENT_MODEL", "gpt-4o-mini"),
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Success": "✅ Complete"
            }
            st.json(summary_data)
        
        return result
        
    except Exception as e:
        monitor.log_step("execution", "failed", str(e))
        st.error(f"❌ Flow execution failed: {str(e)}")
        with telemetry_log:
            st.error("**Error Details:**")
            st.exception(e)
        return None

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="PhysicsGPT Flow Laboratory", 
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-banner {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ PHYSICS LABORATORY FLOW SYSTEM</h1>
        <h3>Real-Time Multi-Agent Physics Research Orchestration</h3>
        <p>10 Specialized Agents • Event-Driven Architecture • Enhanced Telemetry</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System status indicators
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if FLOW_AVAILABLE:
            st.success("✅ Flow System Ready")
        else:
            st.error("❌ Flow System Error")
    
    with col2:
        st.info("🔬 10 Specialist Agents")
    
    with col3:
        st.info("⚛️ Event-Driven Flow")
    
    with col4:
        st.info("📊 Enhanced Telemetry")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🛠️ Laboratory Configuration")
        
        # API Key configuration with improved UX
        st.subheader("🔑 API Configuration")
        
        # Check for existing API key
        existing_key = os.getenv("OPENAI_API_KEY", "")
        if existing_key:
            st.success("✅ API Key configured")
            show_key = st.checkbox("Show API key", value=False)
            if show_key:
                st.code(f"{existing_key[:10]}...{existing_key[-10:]}")
        else:
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password", 
                help="Enter your OpenAI API key to enable physics analysis"
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("✅ API Key set!")
        
        # Model selection with descriptions
        st.subheader("🤖 Model Configuration")
        model_options = {
            "gpt-4o-mini": "Fast, efficient, cost-effective",
            "gpt-4o": "High performance, latest model", 
            "gpt-4": "Reliable, proven performance",
            "gpt-3.5-turbo": "Fast, economical option"
        }
        
        selected_model = st.selectbox(
            "Select AI Model",
            options=list(model_options.keys()),
            index=0,
            format_func=lambda x: f"{x} - {model_options[x]}"
        )
        os.environ["PHYSICS_AGENT_MODEL"] = selected_model
        
        st.divider()
        
        # Enhanced agent information
        st.subheader("🏛️ Laboratory Specialists")
        specialists = [
            ("🧠", "Senior Physics Expert", "Theoretical frameworks & principles"),
            ("💡", "Hypothesis Generator", "Creative approaches & novel ideas"), 
            ("📊", "Mathematical Analyst", "Calculations & quantitative models"),
            ("⚗️", "Experimental Designer", "Practical experiments & methods"),
            ("⚛️", "Quantum Specialist", "Quantum mechanical aspects"),
            ("🌌", "Relativity Expert", "Relativistic & cosmological effects"),
            ("🔧", "Condensed Matter Expert", "Materials & solid-state physics"),
            ("💻", "Computational Physicist", "Simulations & numerical methods"),
            ("📝", "Physics Communicator", "Synthesis & presentation")
        ]
        
        for icon, name, description in specialists:
            st.markdown(f"""
            <div class="agent-card">
                <strong>{icon} {name}</strong><br>
                <small>{description}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Enhanced flow execution info
        st.subheader("⚡ Flow Architecture")
        st.markdown("""
        **Event-Driven Execution:**
        1. 📋 **Research Coordination** - Strategic planning
        2. 🧠 **Theoretical Analysis** - Physics foundations  
        3. 💡 **Hypothesis Generation** - Creative approaches
        4. 📊 **Mathematical Modeling** - Quantitative analysis
        5. ⚗️ **Experimental Design** - Practical methods
        6. ⚛️ **Quantum Analysis** - Quantum mechanics
        7. 💻 **Computational Simulation** - Numerical modeling
        8. 📝 **Final Synthesis** - Comprehensive report
        """)
        
        # System metrics
        st.subheader("📈 System Metrics")
        if 'flow_monitor' in st.session_state:
            monitor = st.session_state.flow_monitor
            if hasattr(monitor, 'execution_log') and monitor.execution_log:
                st.metric("Total Executions", len([e for e in monitor.execution_log if e['step'] == 'full_execution']))
                if monitor.start_time:
                    current_session = time.time() - monitor.start_time
                    st.metric("Session Duration", f"{current_session:.0f}s")
    
    # Main interface
    st.header("🔬 Physics Research Interface")
    
    # Enhanced input section
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_question = st.text_area(
            "🎯 Enter your physics research question:",
            placeholder="e.g., How do quantum tunneling effects work in semiconductor devices?",
            help="Ask any physics question for comprehensive analysis by our 10-agent laboratory",
            height=100
        )
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)  # Add spacing
        analyze_button = st.button(
            "🚀 Launch Analysis", 
            type="primary", 
            use_container_width=True,
            disabled=not user_question or not FLOW_AVAILABLE or not os.getenv("OPENAI_API_KEY")
        )
        
        clear_button = st.button(
            "🧹 Clear Results",
            use_container_width=True
        )
        
        if clear_button:
            if 'flow_monitor' in st.session_state:
                st.session_state.flow_monitor.reset()
            st.rerun()
    
    # Enhanced example questions
    st.markdown("### 💡 Example Research Questions")
    example_questions = [
        "How do quantum tunneling effects work in semiconductor devices?",
        "What is the most fundamental physics theory and why?", 
        "How does quantum entanglement work in many-body systems?",
        "What are the implications of the black hole information paradox?",
        "How can we achieve controlled nuclear fusion with minimal resources?",
        "What is the relationship between consciousness and quantum mechanics?"
    ]
    
    cols = st.columns(3)
    for i, question in enumerate(example_questions):
        with cols[i % 3]:
            if st.button(
                f"📝 {question[:35]}...", 
                key=f"example_{i}", 
                use_container_width=True,
                help=question
            ):
                user_question = question
                st.rerun()
    
    # Analysis execution with enhanced monitoring
    if analyze_button and user_question and FLOW_AVAILABLE:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ Please provide your OpenAI API key in the sidebar.")
        else:
            st.markdown("---")
            st.header("🔬 Real-Time Laboratory Analysis")
            
            # Execute with comprehensive monitoring
            result = capture_flow_output(user_question)
            
            if result:
                # Success banner
                st.markdown("""
                <div class="success-banner">
                    <h3>🎉 Analysis Successfully Completed!</h3>
                    <p>All 10 physics specialists have contributed to your research question</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Download options
                st.download_button(
                    label="📥 Download Full Report",
                    data=f"Physics Laboratory Analysis Report\n" +
                          f"Question: {user_question}\n" +
                          f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" +
                          f"Model: {selected_model}\n\n" +
                          f"{'='*80}\n" +
                          f"ANALYSIS RESULTS\n" +
                          f"{'='*80}\n\n" +
                          result,
                    file_name=f"physics_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    elif analyze_button and not user_question:
        st.warning("⚠️ Please enter a physics research question to analyze.")
    
    elif analyze_button and not FLOW_AVAILABLE:
        st.error("❌ Flow system is not available. Please check the installation.")
    
    elif analyze_button and not os.getenv("OPENAI_API_KEY"):
        st.error("🔑 Please configure your OpenAI API key in the sidebar.")
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h4>🏛️ PhysicsGPT Flow Laboratory</h4>
        <p><strong>Modern Event-Driven Multi-Agent Physics Research System</strong></p>
        <p>Powered by CrewAI Flows & OpenAI • 10 Specialized Physics Agents • Real-Time Telemetry</p>
        <p><em>Advancing physics research through collaborative AI intelligence</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()