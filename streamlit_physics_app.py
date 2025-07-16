#!/usr/bin/env python3
"""
PhysicsGPT - Simplified Streamlit App for Cloud Deployment
Optimized for Streamlit Cloud with proper error handling.
"""

import streamlit as st
import os
import sys
import time
from datetime import datetime
from typing import Optional

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

def safe_progress_update(progress_bar, value: float):
    """Safely update progress bar with normalized value."""
    # Ensure value is between 0.0 and 1.0
    normalized_value = max(0.0, min(1.0, value))
    try:
        progress_bar.progress(normalized_value)
    except Exception as e:
        st.error(f"Progress update error: {e}")

def run_physics_analysis(question: str):
    """Run physics analysis with proper error handling."""
    
    # Create containers
    st.markdown("### 🔬 Physics Laboratory Analysis")
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        col1, col2 = st.columns([4, 1])
        with col1:
            progress_bar = st.progress(0.0)
        with col2:
            status_text = st.empty()
    
    # Results container
    results_container = st.container()
    
    # Steps for visualization
    steps = [
        "🔬 Initializing Laboratory",
        "🧠 Theoretical Analysis", 
        "💡 Generating Hypotheses",
        "📊 Mathematical Modeling",
        "⚗️ Experimental Design",
        "⚛️ Quantum Analysis", 
        "💻 Computational Simulation",
        "📝 Final Synthesis"
    ]
    
    try:
        # Initialize
        status_text.text("Initializing...")
        safe_progress_update(progress_bar, 0.0)
        
        # Simulate progress for better UX
        for i, step in enumerate(steps):
            status_text.text(step)
            progress_value = (i + 1) / len(steps)
            safe_progress_update(progress_bar, progress_value)
            time.sleep(0.3)
        
        # Execute actual analysis
        status_text.text("🚀 Running Analysis...")
        
        with st.spinner("Physics specialists working..."):
            result = analyze_physics_question_with_flow(question)
        
        # Complete
        safe_progress_update(progress_bar, 1.0)
        status_text.text("✅ Complete!")
        
        # Display results
        with results_container:
            st.success("🎉 **Analysis Complete!**")
            st.markdown("### 📋 Laboratory Report")
            st.markdown(result)
            
            # Download button
            st.download_button(
                label="📥 Download Report",
                data=result,
                file_name=f"physics_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    except Exception as e:
        safe_progress_update(progress_bar, 1.0)
        status_text.text("❌ Error")
        
        with results_container:
            st.error(f"❌ **Analysis Failed**: {str(e)}")
            st.info("💡 **Troubleshooting:**")
            st.markdown("""
            - Check API keys are set in Streamlit secrets
            - Ensure internet connectivity  
            - Try refreshing the page
            - Try a simpler question
            """)

def main():
    """Main Streamlit application."""
    
    # Page config
    st.set_page_config(
        page_title="PhysicsGPT Laboratory",
        page_icon="🔬", 
        layout="wide"
    )
    
    # Header
    st.markdown("""
    # 🔬 PhysicsGPT Laboratory
    ### 10-Agent Physics Research System
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key management
        api_key_configured = False
        
        if "OPENAI_API_KEY" in st.secrets:
            st.success("✅ OpenAI API Key (from secrets)")
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
            api_key_configured = True
        elif "OPENAI_API_KEY" in os.environ:
            st.success("✅ OpenAI API Key (from env)")
            api_key_configured = True
        else:
            openai_key = st.text_input("OpenAI API Key", type="password")
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
                api_key_configured = True
        
        # Other API keys
        other_keys = ["LANGCHAIN_API_KEY", "TAVILY_API_KEY"]
        for key in other_keys:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
                st.success(f"✅ {key}")
            elif key in os.environ:
                st.success(f"✅ {key}")
            else:
                st.info(f"ℹ️ {key} (optional)")
        
        # Set LangSmith tracing if available
        if "LANGCHAIN_API_KEY" in os.environ:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = "physics-gpt"
        
        st.markdown("---")
        
        # System status
        st.header("📊 Status")
        
        if not FLOW_AVAILABLE:
            st.error("❌ Flow System Error")
            st.error(f"Import error: {IMPORT_ERROR}")
        elif not api_key_configured:
            st.warning("⚠️ API Key Required")
        else:
            st.success("✅ System Ready")
        
        # Model settings
        st.header("⚙️ Settings")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        
        os.environ["PHYSICS_AGENT_MODEL"] = model
        os.environ["PHYSICS_AGENT_TEMPERATURE"] = str(temperature)
    
    # Main interface
    if not FLOW_AVAILABLE:
        st.error("❌ **System Error**")
        st.error(f"Failed to import physics flow system: {IMPORT_ERROR}")
        st.info("Please check the system logs and try refreshing.")
        return
    
    if not api_key_configured:
        st.warning("⚠️ **API Key Required**")
        st.info("Please configure your OpenAI API key in the sidebar to continue.")
        return
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🚀 Analysis", "📊 System Info"])
    
    with tab1:
        # Question input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question = st.text_area(
                "Physics Question:",
                placeholder="e.g., How do quantum tunneling effects work in semiconductor devices?",
                height=100
            )
        
        with col2:
            st.markdown("#### 💡 Examples")
            examples = [
                "How does quantum entanglement work?",
                "Explain dark matter detection methods",
                "What is the physics of black holes?",
                "How do superconductors work?",
                "Explain nuclear fusion reactions"
            ]
            
            for example in examples:
                if st.button(example, key=example, use_container_width=True):
                    question = example
                    st.rerun()
        
        # Analysis button
        if st.button("🚀 **Start Analysis**", type="primary", use_container_width=True):
            if question.strip():
                run_physics_analysis(question)
            else:
                st.warning("Please enter a physics question first!")
    
    with tab2:
        st.header("📊 System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🤖 Core Agents
            - **Lab Director**: Research coordination  
            - **Senior Physics Expert**: Theoretical analysis
            - **Hypothesis Generator**: Creative approaches
            - **Mathematical Analyst**: Calculations
            - **Experimental Designer**: Practical methods
            """)
        
        with col2:
            st.markdown("""
            #### ⚛️ Specialists
            - **Quantum Specialist**: Quantum mechanics
            - **Relativity Expert**: Spacetime physics
            - **Condensed Matter Expert**: Materials science
            - **Computational Physicist**: Simulations
            - **Physics Communicator**: Synthesis
            """)
        
        st.markdown("---")
        st.markdown("""
        #### 🔄 Analysis Process
        1. **Coordination**: Lab Director creates research plan
        2. **Theory**: Senior Expert provides theoretical foundation
        3. **Hypotheses**: Generator creates innovative approaches  
        4. **Mathematics**: Analyst performs calculations
        5. **Experiments**: Designer creates practical methods
        6. **Quantum**: Specialist analyzes quantum aspects
        7. **Computation**: Physicist runs simulations
        8. **Synthesis**: Communicator creates final report
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>⚛️ PhysicsGPT Laboratory System • Powered by CrewAI Flows</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()