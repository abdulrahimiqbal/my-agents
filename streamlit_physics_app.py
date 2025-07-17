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
        col1, col2 = st.columns([3, 1])
        with col1:
            progress_bar = st.progress(0.0)
        with col2:
            status_text = st.empty()
    
    # Results container
    results_container = st.container()
    
    try:
        # Update progress
        status_text.text("🚀 Starting analysis...")
        safe_progress_update(progress_bar, 0.1)
        
        # Run the analysis
        start_time = time.time()
        result = analyze_physics_question_with_flow(question)
        end_time = time.time()
        
        # Final progress
        status_text.text("✅ Analysis complete!")
        safe_progress_update(progress_bar, 1.0)
        
        # Display results
        with results_container:
            execution_time = end_time - start_time
            
            st.success(f"✅ **Analysis Complete!** (Execution time: {execution_time:.1f}s)")
            
            # Results
            st.markdown("### 📄 Research Report")
            with st.expander("📋 Full Analysis", expanded=True):
                st.markdown(result)
            
            # Download option
            st.download_button(
                label="📥 Download Report",
                data=result,
                file_name=f"physics_analysis_{int(time.time())}.txt",
                mime="text/plain"
            )
            
    except Exception as e:
        status_text.text("❌ Error occurred")
        safe_progress_update(progress_bar, 0.0)
        
        with results_container:
            st.error(f"❌ **Analysis Failed**: {str(e)}")
            st.info("Please try a different question or check the system status.")

def load_knowledge_data():
    """Load knowledge base and hypothesis data."""
    try:
        from crewai_database_integration import CrewAIKnowledgeAPI
        import sqlite3
        import json
        
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
    st.header("🧠 Physics Knowledge Base")
    
    knowledge_entries, _ = load_knowledge_data()
    
    if not knowledge_entries:
        st.info("📚 No knowledge entries yet. Run some physics analyses to build the knowledge base!")
        st.markdown("""
        **How to build knowledge:**
        1. Go to the 🚀 Analysis tab
        2. Ask complex physics questions
        3. The system will generate and validate hypotheses
        4. Validated findings will appear here as knowledge entries
        """)
        return
    
    # Knowledge base statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Entries", len(knowledge_entries))
    with col2:
        domains = [entry.get('physics_domain', 'Unknown') for entry in knowledge_entries]
        unique_domains = len(set(domains))
        st.metric("Physics Domains", unique_domains)
    with col3:
        avg_confidence = sum(entry.get('confidence_level', 0) for entry in knowledge_entries) / len(knowledge_entries)
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Search functionality
    search_term = st.text_input("🔍 Search Knowledge Base", placeholder="Enter search terms...")
    
    # Filter knowledge entries
    filtered_entries = knowledge_entries
    if search_term:
        filtered_entries = [
            entry for entry in knowledge_entries
            if search_term.lower() in entry.get('title', '').lower() or 
               search_term.lower() in entry.get('content', '').lower() or
               search_term.lower() in entry.get('physics_domain', '').lower()
        ]
    
    # Display knowledge entries
    for entry in filtered_entries:
        with st.expander(f"📖 {entry.get('title', 'Untitled')} (Confidence: {entry.get('confidence_level', 0):.2f})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Content:** {entry.get('content', 'No content')}")
                
            with col2:
                st.markdown(f"**Domain:** {entry.get('physics_domain', 'Unknown')}")
                st.markdown(f"**Created:** {entry.get('created_at', 'Unknown')}")
                
                # Parse source agents
                source_agents = entry.get('source_agents', '[]')
                try:
                    import json
                    agents = json.loads(source_agents) if isinstance(source_agents, str) else source_agents
                    if agents:
                        st.markdown(f"**Contributors:** {', '.join(agents)}")
                except:
                    st.markdown("**Contributors:** Unknown")

def display_hypothesis_tracker():
    """Display the hypothesis progression tracker tab."""
    st.header("🔬 Hypothesis Research Progression")
    
    _, hypotheses = load_knowledge_data()
    
    if not hypotheses:
        st.info("💡 No hypotheses yet. Run some physics analyses to generate hypotheses!")
        st.markdown("""
        **How hypotheses work:**
        1. Ask complex physics questions in the 🚀 Analysis tab
        2. The Hypothesis Generator creates new scientific ideas
        3. Other agents evaluate and validate the hypotheses
        4. Track their progression through research phases here
        """)
        return
    
    # Research phase explanation
    st.markdown("""
    ### 📋 Research Phases
    - **📝 Pending**: Initial hypothesis proposed
    - **🔍 Under Review**: Being evaluated by agents
    - **✅ Validated**: Passed scientific validation
    - **❌ Refuted**: Disproven by evidence
    - **🏆 Promoted**: Accepted into knowledge base
    """)
    
    # Hypothesis statistics
    status_counts = {}
    for hyp in hypotheses:
        status = hyp.get('validation_status', 'pending')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Display status distribution
    col1, col2, col3, col4, col5 = st.columns(5)
    
    phase_info = {
        'pending': {'col': col1, 'emoji': '📝', 'label': 'Pending'},
        'under_review': {'col': col2, 'emoji': '🔍', 'label': 'Under Review'},
        'validated': {'col': col3, 'emoji': '✅', 'label': 'Validated'},
        'refuted': {'col': col4, 'emoji': '❌', 'label': 'Refuted'},
        'promoted': {'col': col5, 'emoji': '🏆', 'label': 'Promoted'}
    }
    
    for status, info in phase_info.items():
        count = status_counts.get(status, 0)
        with info['col']:
            st.metric(f"{info['emoji']} {info['label']}", count)
    
    # Filter by status
    status_filter = st.selectbox(
        "Filter by Status:",
        ["All"] + list(phase_info.keys()),
        format_func=lambda x: f"{phase_info[x]['emoji']} {phase_info[x]['label']}" if x != "All" else "🔬 All Hypotheses"
    )
    
    # Filter hypotheses
    filtered_hypotheses = hypotheses
    if status_filter != "All":
        filtered_hypotheses = [h for h in hypotheses if h.get('validation_status') == status_filter]
    
    # Display hypotheses
    for hyp in filtered_hypotheses:
        status = hyp.get('validation_status', 'pending')
        status_info = phase_info.get(status, {'emoji': '❓', 'label': 'Unknown'})
        
        with st.expander(f"{status_info['emoji']} {hyp.get('title', 'Untitled Hypothesis')} (Confidence: {hyp.get('confidence_score', 0):.2f})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Description:** {hyp.get('description', 'No description')}")
                
            with col2:
                st.markdown(f"**Status:** {status_info['emoji']} {status_info['label']}")
                st.markdown(f"**Created by:** {hyp.get('created_by', 'Unknown')}")
                st.markdown(f"**Created:** {hyp.get('created_at', 'Unknown')}")
                
                # Progress indicator
                if status == 'pending':
                    progress = 0.2
                elif status == 'under_review':
                    progress = 0.4
                elif status == 'validated':
                    progress = 0.8
                elif status == 'promoted':
                    progress = 1.0
                else:  # refuted
                    progress = 0.1
                
                st.progress(progress)

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
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Analysis", "📊 System Info", "🧠 Knowledge Base", "🔬 Hypothesis Tracker"])
    
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
    
    with tab3:
        display_knowledge_base()
    
    with tab4:
        display_hypothesis_tracker()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>⚛️ PhysicsGPT Laboratory System • Powered by CrewAI Flows</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()