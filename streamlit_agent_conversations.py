#!/usr/bin/env python3
"""
PhysicsGPT - Agent Conversations UI
Shows real-time agent thoughts, decisions, and progress.
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
import threading
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Import our modules
from physics_crew_system import PhysicsGPTCrew
from agent_monitor import agent_monitor, simulate_agent_progress

# Page configuration
st.set_page_config(
    page_title="PhysicsGPT - Agent Conversations",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for conversation UI
st.markdown("""
<style>
    .conversation-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .agent-active {
        border-left: 4px solid #10B981;
        background: #ecfdf5;
    }
    .agent-completed {
        border-left: 4px solid #6B7280;
        background: #f9fafb;
        opacity: 0.8;
    }
    .thought-bubble {
        background: #e0f2fe;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        border-left: 3px solid #0288d1;
    }
    .decision-bubble {
        background: #f3e5f5;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        border-left: 3px solid #7b1fa2;
    }
    .question-bubble {
        background: #fff3e0;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        border-left: 3px solid #f57c00;
    }
    .progress-container {
        background: #ffffff;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .agent-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .status-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-thinking {
        background: #dbeafe;
        color: #1e40af;
    }
    .status-completed {
        background: #dcfce7;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'physics_crew' not in st.session_state:
        st.session_state.physics_crew = None
        st.session_state.system_ready = False
        
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
        
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
        
    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}
        
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()

def update_conversations_callback(conversations):
    """Callback to update conversations in session state."""
    st.session_state.conversations = conversations
    st.session_state.last_update = time.time()

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%); 
                padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;">
        <h1>🧠 PhysicsGPT Agent Conversations</h1>
        <h3>Real-time Agent Thoughts & Decision Making</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system if needed
    if not st.session_state.system_ready:
        with st.spinner("🚀 Initializing PhysicsGPT system..."):
            try:
                st.session_state.physics_crew = PhysicsGPTCrew()
                agent_monitor.add_update_callback(update_conversations_callback)
                st.session_state.system_ready = True
                st.success("✅ System initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize system: {e}")
                st.error("Please check your OpenAI API key and internet connection.")
                return
    else:
        # Show a small status indicator when system is ready
        st.success("✅ PhysicsGPT System Ready")
    
    # Sidebar - Controls
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Add prominent diagnostic section at the top
        st.subheader("🔧 Debug Tools")
        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            if st.button("🔧 Diagnostics", type="secondary"):
                run_diagnostics()
        with col_diag2:
            if st.button("🧪 Test Monitor", type="secondary"):
                test_monitoring_system()
        
        st.divider()
        
        # Agent selection
        st.subheader("🤖 Select Agents")
        agent_options = {
            'physics_expert': '🧠 Physics Expert',
            'hypothesis_generator': '💡 Hypothesis Generator', 
            'mathematical_analyst': '📊 Mathematical Analyst',
            'quantum_specialist': '⚛️ Quantum Specialist',
            'experimental_designer': '🔬 Experimental Designer',
            'pattern_analyst': '📈 Pattern Analyst',
            'relativity_expert': '🌌 Relativity Expert',
            'condensed_matter_expert': '🔧 Condensed Matter Expert',
            'computational_physicist': '💻 Computational Physicist',
            'physics_communicator': '📚 Physics Communicator'
        }
        
        # Quick selection modes
        mode = st.radio(
            "Selection Mode:",
            ["🎯 5 Core Agents", "🚀 All 10 Agents", "🔧 Custom Selection"]
        )
        
        if mode == "🎯 5 Core Agents":
            selected_agents = ['physics_expert', 'hypothesis_generator', 'mathematical_analyst', 
                             'quantum_specialist', 'physics_communicator']
        elif mode == "🚀 All 10 Agents":
            selected_agents = list(agent_options.keys())
        else:
            selected_agents = []
            for key, label in agent_options.items():
                if st.checkbox(label, key=f"agent_{key}"):
                    selected_agents.append(key)
        
        st.write(f"**Selected:** {len(selected_agents)} agents")
        
        # Demo mode toggle
        st.subheader("🎮 Demo Mode")
        demo_mode = st.checkbox("Enable Demo Mode (Simulated Conversations)")
        
        if demo_mode:
            st.info("Demo mode will simulate agent conversations for testing the UI")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("🔬 Physics Query")
        
        # Query input
        query = st.text_area(
            "Enter your physics question:",
            placeholder="e.g., How does quantum entanglement relate to black hole thermodynamics?",
            height=100
        )
        
        # Example queries
        with st.expander("💡 Example Queries"):
            examples = [
                "What is dark matter and how do we detect it?",
                "How does quantum entanglement work in quantum computing?",
                "Explain the relationship between entropy and information theory",
                "How do gravitational waves propagate through spacetime?",
                "What are the implications of the holographic principle?"
            ]
            
            for i, example in enumerate(examples):
                if st.button(f"📝 {example}", key=f"example_{i}"):
                    query = example
                    st.rerun()
        
        # Analysis controls
        st.subheader("🚀 Analysis Controls")
        
        if not st.session_state.analysis_running:
            if st.button("🧠 Start Agent Analysis", type="primary", 
                        disabled=not query or not selected_agents):
                if query and selected_agents:
                    start_analysis(query, selected_agents, demo_mode)
        else:
            if st.button("⏹️ Stop Analysis", type="secondary"):
                stop_analysis()
                
        # System status
        st.subheader("📊 System Status")
        summary = agent_monitor.get_conversation_summary()
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Active Agents", summary.get('active_agents', 0))
            st.metric("Total Interactions", summary.get('total_interactions', 0))
        with col_b:
            st.metric("Completed Agents", summary.get('completed_agents', 0))
            st.metric("Total Agents", summary.get('total_agents', 0))
        
        # Add diagnostic button
        if st.button("🔧 Run Diagnostics"):
            run_diagnostics()
        
        # Add test monitoring button
        if st.button("🧪 Test Monitoring"):
            test_monitoring_system()
    
    with col2:
        st.header("🤖 Agent Conversations")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh (every 2 seconds)", value=True)
        
        if auto_refresh and st.session_state.analysis_running:
            time.sleep(2)
            st.rerun()
        
        # Check for analysis errors and display them
        if hasattr(st.session_state, 'analysis_error') and st.session_state.analysis_error:
            st.error(f"❌ Analysis Error: {st.session_state.analysis_error}")
            if st.button("🔄 Clear Error"):
                del st.session_state.analysis_error
                st.rerun()
        
        # Check for analysis results and display them
        if hasattr(st.session_state, 'analysis_result') and st.session_state.analysis_result:
            result = st.session_state.analysis_result
            if result.get('success'):
                st.success("✅ Analysis completed successfully!")
                with st.expander("📄 View Analysis Result", expanded=True):
                    st.markdown(result.get('result', 'No result available'))
            else:
                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        
        # Display conversations
        conversations = st.session_state.conversations
        
        if not conversations:
            if st.session_state.analysis_running:
                st.info("🔄 Analysis is running... Waiting for agent conversations to appear...")
                # Add some debugging info
                st.caption(f"Monitoring active: {agent_monitor.monitoring}")
                st.caption(f"Last update: {st.session_state.last_update}")
            else:
                st.info("👋 Start an analysis to see agent conversations in real-time!")
        else:
            # Sort conversations by start time
            sorted_conversations = sorted(
                conversations.values(), 
                key=lambda x: x.get('start_time', 0)
            )
            
            for conv_data in sorted_conversations:
                display_agent_conversation(conv_data)

def display_agent_conversation(conv_data):
    """Display a single agent conversation."""
    agent_name = conv_data['agent_name']
    status = conv_data['status']
    progress = conv_data['progress_percentage']
    current_step = conv_data['current_step']
    
    # Determine card style
    card_class = "conversation-card"
    if status == "thinking":
        card_class += " agent-active"
    elif status == "completed":
        card_class += " agent-completed"
    
    # Agent header
    st.markdown(f"""
    <div class="{card_class}">
        <div class="agent-header">
            <h4>🤖 {agent_name.replace('_', ' ').title()}</h4>
            <span class="status-badge status-{status}">{status.title()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    if status == "thinking":
        st.progress(progress / 100.0)
        st.caption(f"📍 {current_step}")
    
    # Conversation content
    with st.expander(f"💭 Conversation Details ({conv_data['total_interactions']} interactions)", 
                     expanded=(status == "thinking")):
        
        # Thoughts
        thoughts = conv_data.get('thoughts', [])
        if thoughts:
            st.subheader("💭 Thoughts")
            for thought in thoughts[-5:]:  # Show last 5 thoughts
                timestamp = datetime.fromtimestamp(thought['timestamp']).strftime("%H:%M:%S")
                st.markdown(f"""
                <div class="thought-bubble">
                    <small>{timestamp}</small><br>
                    {thought['content']}
                </div>
                """, unsafe_allow_html=True)
        
        # Decisions
        decisions = conv_data.get('decisions', [])
        if decisions:
            st.subheader("🎯 Decisions")
            for decision in decisions[-3:]:  # Show last 3 decisions
                timestamp = datetime.fromtimestamp(decision['timestamp']).strftime("%H:%M:%S")
                st.markdown(f"""
                <div class="decision-bubble">
                    <small>{timestamp}</small><br>
                    <strong>Decision:</strong> {decision['decision']}<br>
                    {f"<strong>Reasoning:</strong> {decision['reasoning']}" if decision.get('reasoning') else ""}
                </div>
                """, unsafe_allow_html=True)
        
        # Questions
        questions = conv_data.get('questions', [])
        if questions:
            st.subheader("❓ Questions")
            for question in questions[-3:]:  # Show last 3 questions
                timestamp = datetime.fromtimestamp(question['timestamp']).strftime("%H:%M:%S")
                st.markdown(f"""
                <div class="question-bubble">
                    <small>{timestamp}</small><br>
                    {question['question']}
                </div>
                """, unsafe_allow_html=True)
        
        # Final output
        if status == "completed" and conv_data.get('final_output'):
            st.subheader("📄 Final Output")
            st.markdown(conv_data['final_output'][:500] + "..." if len(conv_data['final_output']) > 500 else conv_data['final_output'])

def start_analysis(query, selected_agents, demo_mode):
    """Start the agent analysis."""
    st.session_state.analysis_running = True
    
    # Clear previous errors and results
    if hasattr(st.session_state, 'analysis_error'):
        del st.session_state.analysis_error
    if hasattr(st.session_state, 'analysis_result'):
        del st.session_state.analysis_result
    
    agent_monitor.start_monitoring()
    
    if demo_mode:
        # Start demo simulations
        for agent in selected_agents:
            task_desc = f"Analyzing: {query[:50]}..."
            thread = threading.Thread(
                target=simulate_agent_progress,
                args=(agent, task_desc, 20)  # 20 second simulation
            )
            thread.daemon = True
            thread.start()
    else:
        # Start real analysis with monitored CrewAI
        def run_real_analysis():
            try:
                print(f"🚀 Starting real analysis with monitoring enabled")
                print(f"📋 Query: {query}")
                print(f"🤖 Selected agents: {selected_agents}")
                
                # Create physics crew with monitoring enabled
                monitored_crew = PhysicsGPTCrew(enable_monitoring=True)
                print(f"✅ Physics crew created successfully")
                
                # Run the analysis
                print(f"🔄 Running analysis...")
                result = monitored_crew.analyze_physics_query(query, selected_agents)
                print(f"📊 Analysis result: {result}")
                
                # Store result in session state
                st.session_state.analysis_result = result
                st.session_state.analysis_running = False
                print(f"✅ Analysis completed and stored in session state")
                
            except Exception as e:
                print(f"❌ Analysis failed with error: {str(e)}")
                import traceback
                traceback.print_exc()
                st.session_state.analysis_error = str(e)
                st.session_state.analysis_running = False
        
        # Start analysis in background thread
        analysis_thread = threading.Thread(target=run_real_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        print(f"🔄 Analysis thread started")
        
    st.rerun()

def stop_analysis():
    """Stop the agent analysis."""
    st.session_state.analysis_running = False
    agent_monitor.stop_monitoring()
    st.rerun()

def run_diagnostics():
    """Run diagnostic functions to check monitoring system."""
    st.info("🔧 Running Diagnostics...")
    
    # Check if monitoring is active
    if agent_monitor.monitoring:
        st.success("✅ Monitoring is active")
        st.caption(f"Conversations: {len(agent_monitor.conversations)}")
        st.caption(f"Active agents: {len(agent_monitor.active_agents)}")
    else:
        st.warning("⚠️ Monitoring is not active")
    
    # Check if callbacks are registered
    if agent_monitor.update_callbacks:
        st.success(f"✅ {len(agent_monitor.update_callbacks)} update callbacks registered")
    else:
        st.warning("⚠️ No update callbacks registered - conversations won't update")
    
    # Check if CrewAI is initialized
    if st.session_state.physics_crew:
        st.success("✅ Physics crew is initialized")
    else:
        st.error("❌ Physics crew is not initialized")
    
    # Test monitored LLM creation
    try:
        from monitored_llm import create_monitored_llm
        test_llm = create_monitored_llm("test_agent", temperature=0.1)
        st.success("✅ Monitored LLM creation works")
    except Exception as e:
        st.error(f"❌ Monitored LLM creation failed: {e}")
    
    # Check session state
    st.subheader("Session State Debug")
    st.json({
        "analysis_running": st.session_state.analysis_running,
        "system_ready": st.session_state.system_ready,
        "monitoring_active": st.session_state.monitoring_active,
        "last_update": st.session_state.last_update,
        "conversations_count": len(st.session_state.conversations),
        "has_analysis_error": hasattr(st.session_state, 'analysis_error'),
        "has_analysis_result": hasattr(st.session_state, 'analysis_result')
    })
    
    st.info("🔧 Diagnostics complete")

def test_monitoring_system():
    """Manually trigger a test of the monitoring system."""
    st.info("🧪 Testing Monitoring System...")
    try:
        # Simulate a simple interaction
        print("Simulating a simple interaction...")
        test_agent = "test_agent"
        test_task = "Testing the monitoring system."
        test_reasoning = "This is a test to ensure the monitoring system is working correctly."
        
        # Create a dummy conversation data
        dummy_conversation = {
            "agent_name": test_agent,
            "status": "thinking",
            "progress_percentage": 0,
            "current_step": "Initializing",
            "total_interactions": 0,
            "start_time": time.time(),
            "thoughts": [],
            "decisions": [],
            "questions": [],
            "final_output": "Test completed successfully."
        }
        
        # Add to session state for display
        st.session_state.conversations[f"{test_agent}_{time.time()}"] = dummy_conversation
        st.session_state.last_update = time.time()
        
        # Simulate a thought
        print("Simulating a thought...")
        dummy_conversation["thoughts"].append({
            "timestamp": time.time(),
            "content": f"Thought: {test_task} (Reasoning: {test_reasoning})"
        })
        st.session_state.last_update = time.time()
        
        # Simulate a decision
        print("Simulating a decision...")
        dummy_conversation["decisions"].append({
            "timestamp": time.time(),
            "decision": "Decision: Test decision.",
            "reasoning": "Reasoning: This is a test reasoning."
        })
        st.session_state.last_update = time.time()
        
        # Simulate a question
        print("Simulating a question...")
        dummy_conversation["questions"].append({
            "timestamp": time.time(),
            "question": "Question: Test question."
        })
        st.session_state.last_update = time.time()
        
        # Simulate completion
        print("Simulating completion...")
        dummy_conversation["status"] = "completed"
        dummy_conversation["progress_percentage"] = 100
        dummy_conversation["current_step"] = "Final Output"
        dummy_conversation["final_output"] = "Test completed successfully."
        st.session_state.last_update = time.time()
        
        st.success("✅ Monitoring system test completed successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Monitoring system test failed: {e}")
        import traceback
        traceback.print_exc()
        st.rerun()

if __name__ == "__main__":
    main()