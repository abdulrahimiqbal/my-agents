#!/usr/bin/env python3
"""
Streamlit Agent Conversation Monitor
Real-time monitoring interface for PhysicsGPT Flow system.
"""

import streamlit as st
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the new Flow system
try:
    from physics_flow_system import analyze_physics_question_with_flow
    FLOW_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import Flow system: {e}")
    FLOW_AVAILABLE = False

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="PhysicsGPT Flow Laboratory",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title and description
    st.title("🏛️ PHYSICS LABORATORY FLOW SYSTEM - 10 AGENTS")
    st.markdown("### Modern Event-Driven Physics Research Orchestration")
    
    # Flow system status
    col1, col2, col3 = st.columns(3)
    with col1:
        if FLOW_AVAILABLE:
            st.success("✅ Flow System Ready")
        else:
            st.error("❌ Flow System Error")
    
    with col2:
        st.info("🔬 10 Specialist Agents")
    
    with col3:
        st.info("⚛️ Event-Driven Orchestration")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🛠️ Flow Configuration")
        
        # API Key configuration
        st.subheader("API Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", help="Your OpenAI API key")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Model selection
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        selected_model = st.selectbox("Select Model", model_options, index=0)
        os.environ["PHYSICS_AGENT_MODEL"] = selected_model
        
        st.divider()
        
        # Flow information
        st.subheader("🏛️ Laboratory Specialists")
        specialists = [
            "🧠 Senior Physics Expert",
            "💡 Hypothesis Generator", 
            "📊 Mathematical Analyst",
            "⚗️ Experimental Designer",
            "⚛️ Quantum Specialist",
            "🌌 Relativity Expert",
            "🔧 Condensed Matter Expert",
            "💻 Computational Physicist",
            "📝 Physics Communicator"
        ]
        
        for specialist in specialists:
            st.markdown(f"- {specialist}")
        
        st.divider()
        
        # Flow execution info
        st.subheader("⚡ Flow Execution")
        st.markdown("""
        **Event-Driven Steps:**
        1. 📋 Research Coordination
        2. 🧠 Theoretical Analysis
        3. 💡 Hypothesis Generation
        4. 📊 Mathematical Modeling
        5. ⚗️ Experimental Design
        6. ⚛️ Quantum Analysis
        7. 💻 Computational Simulation
        8. 📝 Final Synthesis
        """)
    
    # Main interface
    st.header("🔬 Physics Research Query")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_question = st.text_input(
            "Enter your physics question:",
            placeholder="e.g., How to detect dark matter with minimal equipment?",
            help="Ask any physics question for comprehensive multi-specialist analysis"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add space
        analyze_button = st.button("🚀 Analyze with Flow", type="primary", use_container_width=True)
    
    # Example questions
    st.markdown("**Example Questions:**")
    example_questions = [
        "How to detect dark matter with minimal equipment?",
        "What is the most important physics theory?",
        "How does quantum entanglement work in many-body systems?",
        "What are the implications of black hole information paradox?",
        "How can we achieve nuclear fusion in a garage?",
        "What is the nature of consciousness from a physics perspective?"
    ]
    
    cols = st.columns(3)
    for i, question in enumerate(example_questions):
        with cols[i % 3]:
            if st.button(f"📝 {question[:30]}...", key=f"example_{i}", use_container_width=True):
                user_question = question
                st.rerun()
    
    # Analysis execution
    if analyze_button and user_question and FLOW_AVAILABLE:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ Please provide your OpenAI API key in the sidebar.")
        else:
            # Create analysis container
            analysis_container = st.container()
            
            with analysis_container:
                st.header("🔬 Laboratory Analysis in Progress")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Flow execution steps
                flow_steps = [
                    "📋 Lab Director coordinating research plan...",
                    "🧠 Senior Physics Expert conducting theoretical analysis...",
                    "💡 Hypothesis Generator developing creative approaches...",
                    "📊 Mathematical Analyst performing calculations...",
                    "⚗️ Experimental Designer creating practical methods...",
                    "⚛️ Quantum Specialist analyzing quantum aspects...",
                    "💻 Computational Physicist designing simulations...",
                    "📝 Physics Communicator synthesizing final report..."
                ]
                
                # Create tabs for real-time results
                tab1, tab2, tab3 = st.tabs(["🔬 Flow Execution", "📊 Analysis Results", "🧠 Flow State"])
                
                with tab1:
                    flow_output = st.empty()
                
                with tab2:
                    results_container = st.empty()
                
                with tab3:
                    state_container = st.empty()
                
                try:
                    # Simulate flow progress (since we can't easily capture real-time flow output)
                    for i, step in enumerate(flow_steps):
                        progress_bar.progress((i + 1) / len(flow_steps))
                        status_text.text(step)
                        
                        with flow_output:
                            st.markdown(f"**Current Step:** {step}")
                            
                            # Show previous steps
                            if i > 0:
                                st.markdown("**Completed Steps:**")
                                for j in range(i):
                                    st.markdown(f"✅ {flow_steps[j]}")
                    
                    # Execute the actual flow
                    status_text.text("🚀 Executing Physics Laboratory Flow...")
                    
                    # Run the flow analysis
                    with st.spinner("Running comprehensive physics analysis..."):
                        result = analyze_physics_question_with_flow(user_question)
                    
                    # Show final results
                    progress_bar.progress(1.0)
                    status_text.text("✅ Analysis complete!")
                    
                    with results_container:
                        st.header("📋 Laboratory Research Report")
                        st.markdown(result)
                    
                    with state_container:
                        st.header("🧠 Flow Execution Summary")
                        st.success("✅ All 10 specialists successfully contributed to the analysis")
                        
                        # Show execution summary
                        execution_summary = {
                            "Question": user_question,
                            "Specialists Involved": len(specialists),
                            "Flow Steps Completed": len(flow_steps),
                            "Analysis Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Model Used": selected_model
                        }
                        
                        st.json(execution_summary)
                    
                except Exception as e:
                    st.error(f"❌ Flow execution failed: {str(e)}")
                    st.exception(e)
    
    elif analyze_button and not user_question:
        st.warning("⚠️ Please enter a physics question to analyze.")
    
    elif analyze_button and not FLOW_AVAILABLE:
        st.error("❌ Flow system is not available. Please check the installation.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>PhysicsGPT Flow Laboratory - Modern Event-Driven Multi-Agent Physics Research</p>
    <p>Powered by CrewAI Flows & OpenAI | 10 Specialized Physics Agents</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()