"""
Collaborative PhysicsGPT - Multi-Agent Physics Research System
A comprehensive Streamlit interface for collaborative physics research using specialized AI agents.
"""

import streamlit as st
import os
import sys
import traceback
from typing import Optional, Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime
import uuid
import json

# Simple path setup for Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Import with fallback error handling
try:
    from src.agents import CollaborativePhysicsSystem
    from src.config.settings import get_settings
    from src.memory.stores import get_memory_store
except ImportError as e1:
    try:
        from agents import CollaborativePhysicsSystem
        from config.settings import get_settings
        from memory.stores import get_memory_store
    except ImportError as e2:
        st.error("❌ **Import Error**: Unable to load Collaborative PhysicsGPT modules")
        st.error(f"Error: {e2}")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Collaborative PhysicsGPT - Multi-Agent AI Research",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/abdulrahimiqbal/my-agents',
        'Report a bug': "https://github.com/abdulrahimiqbal/my-agents/issues",
        'About': "Collaborative PhysicsGPT - Multi-Agent Physics Research System"
    }
)

# Custom CSS for collaborative theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #7c3aed 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
    }
    
    .physics-expert-card {
        border-left: 4px solid #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    }
    
    .hypothesis-generator-card {
        border-left: 4px solid #f59e0b;
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }
    
    .supervisor-card {
        border-left: 4px solid #8b5cf6;
        background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
    }
    
    .collaboration-mode {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        color: #1e40af;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        position: relative;
    }
    
    .user-message {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
    }
    
    .physics-expert-message {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left: 4px solid #10b981;
    }
    
    .hypothesis-generator-message {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid #f59e0b;
    }
    
    .supervisor-message {
        background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
        border-left: 4px solid #8b5cf6;
    }
    
    .synthesis-message {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #0ea5e9;
        border: 2px dashed #0ea5e9;
    }
    
    .agent-indicator {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .physics-expert-indicator {
        background: #10b981;
        color: white;
    }
    
    .hypothesis-generator-indicator {
        background: #f59e0b;
        color: white;
    }
    
    .supervisor-indicator {
        background: #8b5cf6;
        color: white;
    }
    
    .session-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .session-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .active-session {
        border-left: 4px solid #3b82f6;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    }
    
    .collaboration-metrics {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        min-width: 120px;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(124, 58, 237, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables for collaborative system."""
    if 'collaborative_system' not in st.session_state:
        st.session_state.collaborative_system = None
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'active_sessions' not in st.session_state:
        st.session_state.active_sessions = {}
    if 'collaboration_mode' not in st.session_state:
        st.session_state.collaboration_mode = "research"
    if 'agent_settings' not in st.session_state:
        st.session_state.agent_settings = {
            'difficulty_level': 'undergraduate',
            'creativity_level': 'high',
            'collaboration_style': 'balanced'
        }

def load_collaborative_system() -> Optional[CollaborativePhysicsSystem]:
    """Load and initialize the collaborative physics system."""
    try:
        system = CollaborativePhysicsSystem(
            difficulty_level=st.session_state.agent_settings['difficulty_level'],
            creativity_level=st.session_state.agent_settings['creativity_level'],
            collaboration_style=st.session_state.agent_settings['collaboration_style'],
            memory_enabled=True
        )
        return system
    except Exception as e:
        st.error(f"Error loading collaborative system: {e}")
        st.error(traceback.format_exc())
        return None

def create_header():
    """Create the main header for the collaborative interface."""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Collaborative PhysicsGPT</h1>
        <p>Multi-Agent Physics Research System</p>
        <p style="font-size: 1.1em; margin-top: 1rem;">
            🔬 Physics Expert • 💡 Hypothesis Generator • 🤝 Supervisor Agent
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create the enhanced sidebar with collaboration controls."""
    with st.sidebar:
        st.markdown("### 🎛️ Collaboration Controls")
        
        # Collaboration Mode Selection
        st.markdown("#### 🔄 Collaboration Mode")
        mode = st.selectbox(
            "Select collaboration mode:",
            ["research", "debate", "brainstorm", "teaching"],
            index=["research", "debate", "brainstorm", "teaching"].index(st.session_state.collaboration_mode),
            help="Choose how the agents should collaborate"
        )
        
        if mode != st.session_state.collaboration_mode:
            st.session_state.collaboration_mode = mode
            st.rerun()
        
        # Mode descriptions
        mode_descriptions = {
            "research": "🔬 Systematic investigation with balanced analysis and exploration",
            "debate": "⚡ Structured discussion where agents challenge ideas",
            "brainstorm": "🧠 Creative exploration with minimal constraints",
            "teaching": "📚 Educational explanations with expert knowledge and analogies"
        }
        
        st.markdown(f"*{mode_descriptions[mode]}*")
        
        st.markdown("---")
        
        # Agent Settings
        st.markdown("#### ⚙️ Agent Settings")
        
        # Physics Expert Settings
        with st.expander("🔬 Physics Expert", expanded=False):
            difficulty = st.selectbox(
                "Difficulty Level:",
                ["high_school", "undergraduate", "graduate", "research"],
                index=["high_school", "undergraduate", "graduate", "research"].index(
                    st.session_state.agent_settings['difficulty_level']
                )
            )
            
            specialty = st.selectbox(
                "Specialty (optional):",
                [None, "quantum_mechanics", "thermodynamics", "electromagnetism", 
                 "particle_physics", "astrophysics", "condensed_matter"],
                index=0
            )
        
        # Hypothesis Generator Settings
        with st.expander("💡 Hypothesis Generator", expanded=False):
            creativity = st.selectbox(
                "Creativity Level:",
                ["conservative", "moderate", "high", "bold"],
                index=["conservative", "moderate", "high", "bold"].index(
                    st.session_state.agent_settings['creativity_level']
                )
            )
            
            exploration = st.selectbox(
                "Exploration Scope:",
                ["focused", "broad", "interdisciplinary"],
                index=1
            )
            
            risk_tolerance = st.selectbox(
                "Risk Tolerance:",
                ["low", "medium", "high"],
                index=1
            )
        
        # Supervisor Settings
        with st.expander("🤝 Supervisor", expanded=False):
            collaboration_style = st.selectbox(
                "Collaboration Style:",
                ["balanced", "expert_led", "creative_led"],
                index=["balanced", "expert_led", "creative_led"].index(
                    st.session_state.agent_settings['collaboration_style']
                )
            )
            
            max_iterations = st.slider(
                "Max Iterations:",
                min_value=2,
                max_value=10,
                value=5,
                help="Maximum number of agent interactions per session"
            )
        
        # Apply Settings Button
        if st.button("🔄 Apply Settings", type="primary"):
            new_settings = {
                'difficulty_level': difficulty,
                'creativity_level': creativity,
                'collaboration_style': collaboration_style
            }
            
            if new_settings != st.session_state.agent_settings:
                st.session_state.agent_settings = new_settings
                st.session_state.collaborative_system = None  # Force reload
                st.success("Settings updated! System will reload on next interaction.")
                st.rerun()
        
        st.markdown("---")
        
        # Session Management
        st.markdown("#### 📝 Session Management")
        
        if st.button("🆕 New Session"):
            st.session_state.current_session_id = None
            st.session_state.chat_history = []
            st.success("New session started!")
            st.rerun()
        
        # Active Sessions List
        if st.session_state.collaborative_system and st.session_state.collaborative_system.active_sessions:
            st.markdown("**Active Sessions:**")
            for session_id, session_info in st.session_state.collaborative_system.active_sessions.items():
                truncated_topic = session_info['topic'][:30] + "..." if len(session_info['topic']) > 30 else session_info['topic']
                if st.button(f"📂 {truncated_topic}", key=f"session_{session_id}"):
                    st.session_state.current_session_id = session_id
                    st.rerun()
        
        st.markdown("---")
        
        # System Status
        st.markdown("#### 📊 System Status")
        
        if st.session_state.collaborative_system:
            status = st.session_state.collaborative_system.get_agent_status()
            
            st.markdown("**Agents Status:**")
            st.markdown(f"🔬 Physics Expert: {status['physics_expert']['difficulty_level']}")
            st.markdown(f"💡 Hypothesis Generator: {status['hypothesis_generator']['creativity_level']}")
            st.markdown(f"🤝 Supervisor: {status['supervisor']['collaboration_style']}")
            st.markdown(f"📝 Active Sessions: {status['system']['active_sessions']}")

def display_agent_status():
    """Display real-time agent status indicators."""
    if not st.session_state.collaborative_system:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🔬 Physics Expert</h4>
            <p>Ready</p>
            <small>Rigorous Analysis</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>💡 Hypothesis Generator</h4>
            <p>Ready</p>
            <small>Creative Thinking</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🤝 Supervisor</h4>
            <p>Ready</p>
            <small>Orchestrating</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Sessions</h4>
            <p>{len(st.session_state.active_sessions)}</p>
            <small>Active</small>
        </div>
        """, unsafe_allow_html=True)

def display_collaboration_interface():
    """Display the main collaboration interface."""
    st.markdown(f"""
    <div class="collaboration-mode">
        Current Mode: {st.session_state.collaboration_mode.upper()}
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Start Options
    st.markdown("### 🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔬 Research a Topic", type="primary"):
            st.session_state.quick_action = "research"
    
    with col2:
        if st.button("💡 Generate Hypotheses", type="primary"):
            st.session_state.quick_action = "hypotheses"
    
    with col3:
        if st.button("⚡ Start a Debate", type="primary"):
            st.session_state.quick_action = "debate"
    
    # Handle quick actions
    if hasattr(st.session_state, 'quick_action'):
        if st.session_state.quick_action == "research":
            st.markdown("#### 🔬 Research Topic")
            topic = st.text_input("Enter a physics topic to research collaboratively:")
            context = st.text_area("Additional context (optional):")
            
            if st.button("Start Research") and topic:
                start_collaborative_session(topic, "research", context)
                del st.session_state.quick_action
                st.rerun()
        
        elif st.session_state.quick_action == "hypotheses":
            st.markdown("#### 💡 Generate Hypotheses")
            topic = st.text_input("Enter a topic for hypothesis generation:")
            num_hypotheses = st.slider("Number of hypotheses:", 1, 10, 3)
            
            if st.button("Generate Hypotheses") and topic:
                generate_hypotheses_session(topic, num_hypotheses)
                del st.session_state.quick_action
                st.rerun()
        
        elif st.session_state.quick_action == "debate":
            st.markdown("#### ⚡ Start Debate")
            hypothesis = st.text_area("Enter a hypothesis to debate:")
            topic = st.text_input("Topic context:")
            
            if st.button("Start Debate") and hypothesis and topic:
                start_debate_session(hypothesis, topic)
                del st.session_state.quick_action
                st.rerun()

def start_collaborative_session(topic: str, mode: str, context: str = ""):
    """Start a new collaborative session."""
    if not st.session_state.collaborative_system:
        st.session_state.collaborative_system = load_collaborative_system()
    
    if st.session_state.collaborative_system:
        try:
            session = st.session_state.collaborative_system.start_collaborative_session(
                topic=topic,
                mode=mode,
                context=context
            )
            
            st.session_state.current_session_id = session["session_info"]["session_id"]
            st.session_state.active_sessions[st.session_state.current_session_id] = session["session_info"]
            
            # Add to chat history
            st.session_state.chat_history = [
                {"role": "user", "content": f"Research topic: {topic}", "timestamp": datetime.now()},
                {"role": "collaborative", "content": session["response"], "timestamp": datetime.now()}
            ]
            
            st.success(f"Started {mode} session: {topic}")
            
        except Exception as e:
            st.error(f"Error starting session: {e}")

def generate_hypotheses_session(topic: str, num_hypotheses: int):
    """Generate hypotheses for a topic."""
    if not st.session_state.collaborative_system:
        st.session_state.collaborative_system = load_collaborative_system()
    
    if st.session_state.collaborative_system:
        try:
            hypotheses = st.session_state.collaborative_system.generate_hypotheses(
                topic=topic,
                num_hypotheses=num_hypotheses
            )
            
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "user", 
                "content": f"Generate {num_hypotheses} hypotheses for: {topic}", 
                "timestamp": datetime.now()
            })
            st.session_state.chat_history.append({
                "role": "hypothesis_generator", 
                "content": hypotheses, 
                "timestamp": datetime.now()
            })
            
            st.success("Hypotheses generated!")
            
        except Exception as e:
            st.error(f"Error generating hypotheses: {e}")

def start_debate_session(hypothesis: str, topic: str):
    """Start a debate session."""
    if not st.session_state.collaborative_system:
        st.session_state.collaborative_system = load_collaborative_system()
    
    if st.session_state.collaborative_system:
        try:
            debate_result = st.session_state.collaborative_system.facilitate_debate(
                hypothesis=hypothesis,
                topic=topic
            )
            
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "user", 
                "content": f"Debate hypothesis: {hypothesis} (Topic: {topic})", 
                "timestamp": datetime.now()
            })
            st.session_state.chat_history.append({
                "role": "supervisor", 
                "content": debate_result, 
                "timestamp": datetime.now()
            })
            
            st.success("Debate session started!")
            
        except Exception as e:
            st.error(f"Error starting debate: {e}")

def display_chat_interface():
    """Display the collaborative chat interface."""
    st.markdown("### 💬 Collaborative Chat")
    
    # Chat input
    user_input = st.chat_input("Ask a question or continue the collaboration...")
    
    if user_input:
        handle_user_input(user_input)
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(message)

def handle_user_input(user_input: str):
    """Handle user input and generate collaborative response."""
    if not st.session_state.collaborative_system:
        st.session_state.collaborative_system = load_collaborative_system()
    
    if not st.session_state.collaborative_system:
        st.error("Could not load collaborative system")
        return
    
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now()
    })
    
    try:
        if st.session_state.current_session_id:
            # Continue existing session
            response = st.session_state.collaborative_system.continue_collaboration(
                session_id=st.session_state.current_session_id,
                user_input=user_input,
                mode=st.session_state.collaboration_mode
            )
            
            if "error" in response:
                st.error(response["error"])
                return
            
            # Add response to history
            st.session_state.chat_history.append({
                "role": "collaborative",
                "content": response["response"],
                "timestamp": datetime.now()
            })
        else:
            # Start new session
            session = st.session_state.collaborative_system.start_collaborative_session(
                topic=user_input,
                mode=st.session_state.collaboration_mode
            )
            
            st.session_state.current_session_id = session["session_info"]["session_id"]
            st.session_state.active_sessions[st.session_state.current_session_id] = session["session_info"]
            
            # Add response to history
            st.session_state.chat_history.append({
                "role": "collaborative",
                "content": session["response"],
                "timestamp": datetime.now()
            })
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing input: {e}")
        st.session_state.chat_history.append({
            "role": "error",
            "content": f"Error: {str(e)}",
            "timestamp": datetime.now()
        })

def display_chat_message(message: Dict[str, Any]):
    """Display a chat message with appropriate styling."""
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", datetime.now())
    
    # Determine message styling based on role
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="agent-indicator">👤 You</div>
            <div>{content}</div>
            <small style="color: #6b7280;">{timestamp.strftime("%H:%M:%S")}</small>
        </div>
        """, unsafe_allow_html=True)
    
    elif role == "physics_expert" or "🔬" in content:
        st.markdown(f"""
        <div class="chat-message physics-expert-message">
            <div class="agent-indicator physics-expert-indicator">🔬 Physics Expert</div>
            <div>{content}</div>
            <small style="color: #6b7280;">{timestamp.strftime("%H:%M:%S")}</small>
        </div>
        """, unsafe_allow_html=True)
    
    elif role == "hypothesis_generator" or "💡" in content:
        st.markdown(f"""
        <div class="chat-message hypothesis-generator-message">
            <div class="agent-indicator hypothesis-generator-indicator">💡 Hypothesis Generator</div>
            <div>{content}</div>
            <small style="color: #6b7280;">{timestamp.strftime("%H:%M:%S")}</small>
        </div>
        """, unsafe_allow_html=True)
    
    elif role == "supervisor" or "🤝" in content:
        st.markdown(f"""
        <div class="chat-message supervisor-message">
            <div class="agent-indicator supervisor-indicator">🤝 Supervisor</div>
            <div>{content}</div>
            <small style="color: #6b7280;">{timestamp.strftime("%H:%M:%S")}</small>
        </div>
        """, unsafe_allow_html=True)
    
    elif role == "collaborative" or "synthesis" in role:
        st.markdown(f"""
        <div class="chat-message synthesis-message">
            <div class="agent-indicator" style="background: #0ea5e9; color: white;">🤝 Collaborative Synthesis</div>
            <div>{content}</div>
            <small style="color: #6b7280;">{timestamp.strftime("%H:%M:%S")}</small>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div class="chat-message">
            <div>{content}</div>
            <small style="color: #6b7280;">{timestamp.strftime("%H:%M:%S")}</small>
        </div>
        """, unsafe_allow_html=True)

def display_example_collaborations():
    """Display example collaboration scenarios."""
    st.markdown("### 📚 Example Collaborations")
    
    examples = [
        {
            "title": "🔬 Quantum Entanglement Research",
            "description": "Collaborative research on quantum entanglement applications",
            "topic": "quantum entanglement applications in quantum computing",
            "mode": "research"
        },
        {
            "title": "⚡ Dark Matter Debate",
            "description": "Debate about novel dark matter detection methods",
            "hypothesis": "Quantum sensors could detect dark matter through gravitational wave coupling",
            "topic": "dark matter detection",
            "mode": "debate"
        },
        {
            "title": "💡 Fusion Energy Brainstorm",
            "description": "Creative exploration of fusion energy challenges",
            "topic": "room-temperature fusion possibilities",
            "mode": "brainstorm"
        },
        {
            "title": "📚 Quantum Mechanics Teaching",
            "description": "Collaborative explanation of quantum mechanics concepts",
            "topic": "wave-particle duality explanation for beginners",
            "mode": "teaching"
        }
    ]
    
    cols = st.columns(2)
    
    for i, example in enumerate(examples):
        with cols[i % 2]:
            with st.expander(example["title"]):
                st.markdown(f"**Description:** {example['description']}")
                
                if example["mode"] == "debate":
                    st.markdown(f"**Hypothesis:** {example['hypothesis']}")
                    st.markdown(f"**Topic:** {example['topic']}")
                    if st.button(f"Start Debate", key=f"example_debate_{i}"):
                        start_debate_session(example["hypothesis"], example["topic"])
                        st.rerun()
                else:
                    st.markdown(f"**Topic:** {example['topic']}")
                    if st.button(f"Start {example['mode'].title()}", key=f"example_{i}"):
                        start_collaborative_session(example["topic"], example["mode"])
                        st.rerun()

def main():
    """Main application function."""
    initialize_session_state()
    
    # Create header
    create_header()
    
    # Create sidebar
    create_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display agent status
        display_agent_status()
        
        # Display collaboration interface
        display_collaboration_interface()
        
        # Display chat interface
        display_chat_interface()
    
    with col2:
        # Display example collaborations
        display_example_collaborations()
        
        # Display session info if active
        if st.session_state.current_session_id and st.session_state.collaborative_system:
            st.markdown("### 📊 Current Session")
            session_info = st.session_state.active_sessions.get(st.session_state.current_session_id)
            if session_info:
                st.markdown(f"**Topic:** {session_info['topic']}")
                st.markdown(f"**Mode:** {session_info['mode']}")
                st.markdown(f"**Iterations:** {session_info.get('iteration_count', 0)}")
                
                if st.button("📋 Get Session Summary"):
                    try:
                        summary = st.session_state.collaborative_system.get_session_summary(
                            st.session_state.current_session_id
                        )
                        st.markdown("**Session Summary:**")
                        st.markdown(summary["summary"])
                    except Exception as e:
                        st.error(f"Error getting summary: {e}")
                
                if st.button("🔒 Close Session"):
                    try:
                        st.session_state.collaborative_system.close_session(
                            st.session_state.current_session_id
                        )
                        st.session_state.current_session_id = None
                        st.session_state.chat_history = []
                        st.success("Session closed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error closing session: {e}")

if __name__ == "__main__":
    main() 