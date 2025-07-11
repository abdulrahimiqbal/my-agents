"""
🚀 PhysicsGPT - Simplified Main Interface
Clean, focused interface for physics research with optional collaboration.
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Simple path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Page config
st.set_page_config(
    page_title="PhysicsGPT",
    page_icon="⚛️",
    layout="wide"
)

# Clean CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .mode-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .mode-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
    }
    
    .mode-card.selected {
        border-color: #3b82f6;
        background: #f8fafc;
    }
    
    .agent-response {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state."""
    if 'mode' not in st.session_state:
        st.session_state.mode = 'single'
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'physics_expert' not in st.session_state:
        st.session_state.physics_expert = None
    if 'collaborative_system' not in st.session_state:
        st.session_state.collaborative_system = None

def load_single_agent():
    """Load single physics expert agent."""
    if st.session_state.physics_expert is None:
        try:
            from src.agents import PhysicsExpertAgent
            st.session_state.physics_expert = PhysicsExpertAgent()
            return True
        except Exception as e:
            st.error(f"Failed to load physics expert: {e}")
            return False
    return True

def load_collaborative_system():
    """Load collaborative physics system."""
    if st.session_state.collaborative_system is None:
        try:
            from src.agents import CollaborativePhysicsSystem
            st.session_state.collaborative_system = CollaborativePhysicsSystem()
            return True
        except Exception as e:
            st.error(f"Failed to load collaborative system: {e}")
            return False
    return True

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚛️ PhysicsGPT</h1>
        <p>AI-powered physics research and problem solving</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode selection
    st.markdown("### 🎯 Choose Your Research Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        single_selected = st.session_state.mode == 'single'
        card_class = "mode-card selected" if single_selected else "mode-card"
        
        st.markdown(f"""
        <div class="{card_class}">
            <h3>🔬 Single Expert Mode</h3>
            <p>Work with one specialized physics expert agent</p>
            <ul style="text-align: left; margin: 1rem 0;">
                <li>Deep physics knowledge</li>
                <li>Step-by-step problem solving</li>
                <li>Educational explanations</li>
                <li>Fast and focused responses</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Single Expert", key="single_mode", use_container_width=True):
            st.session_state.mode = 'single'
            st.rerun()
    
    with col2:
        collab_selected = st.session_state.mode == 'collaborative'
        card_class = "mode-card selected" if collab_selected else "mode-card"
        
        st.markdown(f"""
        <div class="{card_class}">
            <h3>🤖 Collaborative Mode</h3>
            <p>Multiple AI agents working together</p>
            <ul style="text-align: left; margin: 1rem 0;">
                <li>Physics Expert + Hypothesis Generator</li>
                <li>Supervised collaboration</li>
                <li>Creative problem solving</li>
                <li>Research-grade analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Collaborative", key="collab_mode", use_container_width=True):
            st.session_state.mode = 'collaborative'
            st.rerun()
    
    st.markdown("---")
    
    # Interface based on selected mode
    if st.session_state.mode == 'single':
        display_single_mode()
    elif st.session_state.mode == 'collaborative':
        display_collaborative_mode()

def display_single_mode():
    """Display single agent interface."""
    st.markdown("### 🔬 Physics Expert Chat")
    
    if not load_single_agent():
        return
    
    # Simple chat interface
    user_question = st.text_area(
        "Ask your physics question:",
        placeholder="e.g., Explain quantum tunneling, solve a mechanics problem, or derive an equation",
        height=120
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        difficulty = st.selectbox("Difficulty:", ["undergraduate", "graduate", "research"])
    
    with col2:
        if st.button("🚀 Ask Expert", type="primary", disabled=not user_question):
            if user_question:
                with st.spinner("🔬 Physics expert is thinking..."):
                    try:
                        response = st.session_state.physics_expert.chat(user_question)
                        
                        # Display response
                        st.markdown("### 📝 Expert Response:")
                        st.markdown(f"""
                        <div class="agent-response">
                            {response}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Save to history
                        st.session_state.chat_history.append({
                            'question': user_question,
                            'response': response,
                            'mode': 'single',
                            'timestamp': datetime.now()
                        })
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # Quick examples
    with col3:
        st.markdown("**Quick Examples:**")
        examples = [
            "What is quantum entanglement?",
            "Derive Newton's second law",
            "Explain black hole formation"
        ]
        for example in examples:
            if st.button(example, key=f"single_{example}"):
                st.session_state.temp_question = example
                st.rerun()

def display_collaborative_mode():
    """Display collaborative interface."""
    st.markdown("### 🤖 Multi-Agent Collaboration")
    
    if not load_collaborative_system():
        return
    
    # Collaboration settings
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.selectbox(
            "Collaboration Style:",
            ["research", "debate", "brainstorm", "teaching"],
            help="How should the agents work together?"
        )
    
    with col2:
        st.markdown("**Active Agents:**")
        st.markdown("🔬 Physics Expert • 💡 Hypothesis Generator • 🤝 Supervisor")
    
    # Question input
    user_question = st.text_area(
        "Research question or problem:",
        placeholder="e.g., How might quantum computing impact cryptography? or Explore new approaches to fusion energy",
        height=120
    )
    
    if st.button("🚀 Start Collaboration", type="primary", disabled=not user_question):
        if user_question:
            with st.spinner("🤖 Agents are collaborating..."):
                try:
                    session = st.session_state.collaborative_system.start_collaborative_session(
                        topic=user_question,
                        mode=mode
                    )
                    
                    # Display collaborative response
                    st.markdown("### 🎯 Collaborative Analysis:")
                    st.markdown(f"""
                    <div class="agent-response">
                        <strong>🤝 Multi-Agent Research Result:</strong><br><br>
                        {session["response"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show suggested next steps
                    if "next_actions" in session:
                        st.markdown("### 💡 Suggested Next Steps:")
                        for action in session["next_actions"]:
                            st.markdown(f"• {action}")
                    
                    # Save to history
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'response': session["response"],
                        'mode': 'collaborative',
                        'collaboration_mode': mode,
                        'timestamp': datetime.now()
                    })
                    
                except Exception as e:
                    st.error(f"Collaboration error: {e}")
    
    # Quick research topics
    st.markdown("**Research Topics:**")
    topics = [
        "Quantum computing breakthroughs",
        "Dark matter detection methods", 
        "Fusion energy challenges"
    ]
    
    cols = st.columns(len(topics))
    for i, topic in enumerate(topics):
        with cols[i]:
            if st.button(topic, key=f"collab_{topic}"):
                st.session_state.temp_question = topic
                st.rerun()

# Display recent conversations
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 📚 Recent Conversations")
    
    # Show last 3 conversations
    for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):
        with st.expander(f"💬 {chat['question'][:60]}... ({chat['mode']})"):
            st.markdown(f"**Question:** {chat['question']}")
            st.markdown(f"**Response:** {chat['response'][:300]}...")
            if len(chat['response']) > 300:
                st.markdown("*[Response truncated - click to see full conversation]*")

if __name__ == "__main__":
    main() 