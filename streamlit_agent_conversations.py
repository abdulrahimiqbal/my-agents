#!/usr/bin/env python3

"""
Real-time Agent Conversations Viewer
A Streamlit app that shows agent conversations in real-time using CrewAI's built-in telemetry.
"""

# Fix for SQLite issues on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import threading
import time
import os
from datetime import datetime
from physics_crew_system import PhysicsGPTCrew

# Page configuration
st.set_page_config(
    page_title="🚀 PhysicsGPT Agent Conversations",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .agent-card {
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .success-message {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-message {
        background-color: #d1ecf1;
        border-color: #bee5eb;
        color: #0c5460;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    if 'physics_crew' not in st.session_state:
        st.session_state.physics_crew = None
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'analysis_error' not in st.session_state:
        st.session_state.analysis_error = None
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()

def start_analysis(query: str):
    """Start the agent analysis with CrewAI's built-in telemetry."""
    st.session_state.analysis_running = True
    st.session_state.analysis_result = None
    st.session_state.analysis_error = None
    
    def run_analysis():
        try:
            # Run the physics analysis
            result = st.session_state.physics_crew.analyze_physics_query(query)
            st.session_state.analysis_result = result
            st.session_state.analysis_running = False
            st.session_state.last_update = time.time()
        except Exception as e:
            st.session_state.analysis_error = str(e)
            st.session_state.analysis_running = False
            st.session_state.last_update = time.time()
    
    # Start analysis in background thread
    analysis_thread = threading.Thread(target=run_analysis)
    analysis_thread.daemon = True
    analysis_thread.start()

def stop_analysis():
    """Stop the running analysis."""
    st.session_state.analysis_running = False
    st.session_state.analysis_result = None
    st.session_state.analysis_error = None

def main():
    """Main Streamlit application."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header"><h1>🚀 PhysicsGPT Agent Conversations</h1><p>Real-time AI Agent Collaboration with Enhanced Telemetry</p></div>', unsafe_allow_html=True)
    
    # Initialize system if needed
    if not st.session_state.system_ready:
        with st.spinner("🚀 Initializing PhysicsGPT system with enhanced telemetry..."):
            try:
                st.session_state.physics_crew = PhysicsGPTCrew()
                st.session_state.system_ready = True
                st.success("✅ System initialized successfully with CrewAI enhanced telemetry!")
            except Exception as e:
                st.error(f"❌ Failed to initialize system: {e}")
                st.error("Please check your OpenAI API key and internet connection.")
                return
    else:
        # Show a small status indicator when system is ready
        st.success("✅ PhysicsGPT System Ready with Enhanced Telemetry")
    
    # Main layout with two columns
    col1, col2 = st.columns([1, 2])
    
    # Sidebar - Controls
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Query input
        st.subheader("🔬 Physics Query")
        query = st.text_area(
            "Enter your physics question:",
            placeholder="How to detect dark matter in a room?",
            height=100
        )
        
        # Control buttons
        col_start, col_stop = st.columns(2)
        
        with col_start:
            if st.button("🚀 Start Analysis", disabled=st.session_state.analysis_running):
                if query.strip():
                    start_analysis(query.strip())
                    st.rerun()
                else:
                    st.warning("Please enter a physics question first!")
        
        with col_stop:
            if st.button("🛑 Stop Analysis", disabled=not st.session_state.analysis_running):
                stop_analysis()
                st.rerun()
        
        # System status
        st.subheader("📊 System Status")
        if st.session_state.analysis_running:
            st.info("🔄 Analysis in progress...")
        elif st.session_state.analysis_result:
            st.success("✅ Analysis completed")
        elif st.session_state.analysis_error:
            st.error("❌ Analysis failed")
        else:
            st.info("⏳ Ready for analysis")
        
        # Telemetry info
        st.subheader("📡 Telemetry Info")
        st.info("**Enhanced Telemetry**: ✅ Enabled\n\n**Data Collection**: CrewAI automatically collects comprehensive execution data including agent interactions, task progress, and crew performance metrics.")
        
        st.caption("Using CrewAI's built-in telemetry system with `share_crew=True` for enhanced monitoring.")
    
    # Main content area
    with col1:
        st.header("📊 Analysis Status")
        
        if st.session_state.analysis_running:
            st.info("🔄 **Analysis Running**\n\nThe physics crew is analyzing your query. CrewAI's enhanced telemetry is automatically capturing all agent interactions and task progress.")
            
            # Auto-refresh during analysis
            time.sleep(2)
            st.rerun()
            
        elif st.session_state.analysis_error:
            st.error(f"❌ **Analysis Error**\n\n{st.session_state.analysis_error}")
            if st.button("🔄 Clear Error"):
                st.session_state.analysis_error = None
                st.rerun()
                
        elif st.session_state.analysis_result:
            st.success("✅ **Analysis Complete**\n\nThe physics crew has successfully completed the analysis. View the detailed results in the right panel.")
            
        else:
            st.info("👋 **Ready for Analysis**\n\nEnter a physics question in the sidebar and click 'Start Analysis' to begin. The system will use CrewAI's enhanced telemetry to capture all agent conversations.")
    
    with col2:
        st.header("📄 Analysis Results")
        
        if st.session_state.analysis_result:
            st.markdown("### 🎯 Physics Analysis Result")
            
            # Display the result in an expandable section
            with st.expander("📋 Full Analysis", expanded=True):
                st.markdown(st.session_state.analysis_result)
            
            # Show telemetry note
            st.info("📡 **Telemetry Note**: This analysis was performed with CrewAI's enhanced telemetry enabled, capturing detailed agent interactions, task execution data, and crew performance metrics.")
            
        elif st.session_state.analysis_running:
            st.info("🔄 Analysis in progress... Results will appear here when complete.")
            
        else:
            st.info("👋 Start an analysis to see comprehensive physics results here!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🚀 PhysicsGPT with CrewAI Enhanced Telemetry | 
        Built with Streamlit, CrewAI, and OpenAI | 
        <strong>share_crew=True</strong> for comprehensive monitoring
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()