#!/usr/bin/env python3
"""
PhysicsGPT - Modern Streamlit Interface
Beautiful, responsive UI for the 10-agent physics research system.
"""

# Fix SQLite version issue for Streamlit Cloud BEFORE importing anything else
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import time
import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import required modules with error handling
try:
    from physics_crew_system import PhysicsGPTCrew
    PHYSICS_SYSTEM_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Failed to import PhysicsGPTCrew: {e}")
    st.error("Please check that all dependencies are installed correctly.")
    PHYSICS_SYSTEM_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="PhysicsGPT - AI Physics Research",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
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
    .result-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'physics_crew' not in st.session_state:
    if PHYSICS_SYSTEM_AVAILABLE:
        with st.spinner("🚀 Initializing PhysicsGPT system..."):
            try:
                st.session_state.physics_crew = PhysicsGPTCrew()
                st.session_state.analysis_history = []
                st.session_state.system_ready = True
            except Exception as e:
                st.error(f"❌ Failed to initialize PhysicsGPT: {e}")
                st.session_state.system_ready = False
    else:
        st.session_state.system_ready = False
        st.session_state.analysis_history = []

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚛️ PhysicsGPT</h1>
        <h3>Advanced 10-Agent Physics Research System</h3>
        <p>Powered by CrewAI with Specialized Physics Agents</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar - Agent Selection
    with st.sidebar:
        st.header("🤖 Agent Configuration")
        
        # Agent selection mode
        mode = st.radio(
            "Analysis Mode:",
            ["🎯 5 Core Agents (Recommended)", "🔬 Custom Selection", "🚀 All 10 Agents"]
        )
        
        # Agent descriptions
        agent_info = {
            'physics_expert': '🧠 Senior Physics Expert - Rigorous theoretical analysis',
            'hypothesis_generator': '💡 Creative Physics Researcher - Novel hypotheses',
            'mathematical_analyst': '📊 Mathematical Physics Specialist - Quantitative frameworks',
            'experimental_designer': '🔬 Experimental Physics Designer - Practical testing',
            'pattern_analyst': '📈 Pattern Recognition Specialist - Data relationships',
            'quantum_specialist': '⚛️ Quantum Mechanics Specialist - Quantum expertise',
            'relativity_expert': '🌌 Relativity & Cosmology Expert - Spacetime physics',
            'condensed_matter_expert': '🔧 Condensed Matter Physicist - Materials science',
            'computational_physicist': '💻 Computational Physics Specialist - Numerical methods',
            'physics_communicator': '📚 Physics Education Specialist - Clear explanations'
        }
        
        selected_agents = []
        
        if mode == "🎯 5 Core Agents (Recommended)":
            selected_agents = [
                'physics_expert', 'hypothesis_generator', 'mathematical_analyst',
                'quantum_specialist', 'physics_communicator'
            ]
            st.info("Using balanced set of 5 core agents for comprehensive analysis")
            
        elif mode == "🚀 All 10 Agents":
            selected_agents = list(agent_info.keys())
            st.warning("Full 10-agent analysis - This will take longer but provide maximum depth")
            
        else:  # Custom selection
            st.subheader("Select Agents:")
            for agent_key, agent_desc in agent_info.items():
                if st.checkbox(agent_desc, key=agent_key):
                    selected_agents.append(agent_key)
        
        # Display selected agents
        if selected_agents:
            st.subheader(f"Selected: {len(selected_agents)} agents")
            for agent in selected_agents:
                st.markdown(f"<div class='agent-card'>{agent_info[agent]}</div>", 
                           unsafe_allow_html=True)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔬 Physics Query Interface")
        
        # Query input
        query = st.text_area(
            "Enter your physics question:",
            placeholder="e.g., What is the relationship between quantum entanglement and black hole thermodynamics?",
            height=100
        )
        
        # Example queries
        st.subheader("💡 Example Queries")
        examples = [
            "What is dark matter and how do we detect it?",
            "How does quantum entanglement work in quantum computing?",
            "What are the best ideas to create a multiagent swarm for physics discovery?",
            "Explain the relationship between entropy and information theory",
            "How do gravitational waves propagate through spacetime?"
        ]
        
        example_cols = st.columns(2)
        for i, example in enumerate(examples):
            with example_cols[i % 2]:
                if st.button(f"📝 {example[:50]}...", key=f"example_{i}"):
                    query = example
                    st.rerun()
        
        # Analysis button
        if st.button("🚀 Analyze Physics Query", type="primary", disabled=not query or not selected_agents):
            if query and selected_agents:
                analyze_query(query, selected_agents)
    
    with col2:
        st.header("📊 System Status")
        
        # System metrics
        metrics_container = st.container()
        with metrics_container:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Available Agents", "10", "🤖")
            with col_b:
                st.metric("Selected Agents", len(selected_agents), "✅")
        
        # Analysis history
        st.subheader("📚 Recent Analyses")
        if st.session_state.analysis_history:
            for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
                with st.expander(f"Query {len(st.session_state.analysis_history)-i}: {analysis['query'][:30]}..."):
                    st.write(f"**Agents Used:** {', '.join(analysis['agents'])}")
                    st.write(f"**Time:** {analysis['timestamp']}")
                    st.write(f"**Success:** {'✅' if analysis['success'] else '❌'}")
        else:
            st.info("No analyses yet. Submit a query to get started!")

def analyze_query(query: str, selected_agents: list):
    """Analyze physics query with selected agents."""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Analysis container
    result_container = st.container()
    
    with result_container:
        status_text.text("🚀 Initializing analysis...")
        progress_bar.progress(10)
        
        start_time = time.time()
        
        try:
            status_text.text("🤖 Agents collaborating...")
            progress_bar.progress(30)
            
            # Run analysis
            result = st.session_state.physics_crew.analyze_physics_query(query, selected_agents)
            
            progress_bar.progress(90)
            status_text.text("✅ Analysis complete!")
            
            # Store in history
            st.session_state.analysis_history.append({
                'query': query,
                'agents': selected_agents,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'success': result['success'],
                'duration': time.time() - start_time
            })
            
            progress_bar.progress(100)
            
            # Display results
            if result['success']:
                st.success(f"✅ Analysis completed successfully in {time.time() - start_time:.1f}s")
                
                # Results display
                st.markdown("""
                <div class="result-container">
                <h3>📄 Analysis Results</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Format and display the result
                st.markdown("### 🔬 Physics Analysis")
                st.markdown(result['result'])
                
                # Analysis metadata
                with st.expander("📊 Analysis Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Agents Used", len(selected_agents))
                    with col2:
                        st.metric("Duration", f"{time.time() - start_time:.1f}s")
                    with col3:
                        st.metric("Success", "✅")
                    
                    st.write("**Agents Involved:**")
                    for agent in selected_agents:
                        st.write(f"• {agent}")
                
                # Download option
                st.download_button(
                    label="📥 Download Analysis",
                    data=f"Query: {query}\n\nAgents: {', '.join(selected_agents)}\n\nResult:\n{result['result']}",
                    file_name=f"physics_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
            else:
                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ System error: {str(e)}")
            status_text.text("❌ Analysis failed")
        
        finally:
            progress_bar.empty()
            status_text.empty()

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>⚛️ PhysicsGPT - Advanced AI Physics Research System</p>
        <p>Powered by CrewAI • Built with Streamlit • 10 Specialized Physics Agents</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()