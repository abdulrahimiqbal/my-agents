"""
🤖 Collaborative PhysicsGPT - Ultra Simple Interface
The simplest way to experience multi-agent physics research.
"""

import streamlit as st
import sys
import os

# Simple path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Page config
st.set_page_config(
    page_title="Collaborative PhysicsGPT",
    page_icon="🤖",
    layout="centered"
)

# Simple CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .agent-response {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .physics-expert {
        background: #f0f9ff;
        border-color: #0ea5e9;
    }
    
    .hypothesis-generator {
        background: #fffbeb;
        border-color: #f59e0b;
    }
    
    .supervisor {
        background: #f5f3ff;
        border-color: #8b5cf6;
    }
</style>
""", unsafe_allow_html=True)

def initialize_system():
    """Initialize the collaborative system."""
    if 'collaborative_system' not in st.session_state:
        try:
            from src.agents import CollaborativePhysicsSystem
            st.session_state.collaborative_system = CollaborativePhysicsSystem()
            return True
        except Exception as e:
            st.error(f"Failed to initialize system: {e}")
            return False
    return True

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Collaborative PhysicsGPT</h1>
        <p>Three AI agents working together to solve physics problems</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    if not initialize_system():
        st.stop()
    
    # Simple explanation
    st.markdown("""
    ### How it works:
    - 🔬 **Physics Expert**: Provides rigorous scientific analysis
    - 💡 **Hypothesis Generator**: Creates innovative ideas and approaches  
    - 🤝 **Supervisor**: Coordinates the collaboration and synthesizes insights
    
    Just ask a physics question and watch the agents collaborate!
    """)
    
    # User input
    user_question = st.text_area(
        "Ask a physics question:",
        placeholder="e.g., Explain quantum entanglement or How do black holes form?",
        height=100
    )
    
    # Collaboration mode
    col1, col2 = st.columns(2)
    with col1:
        mode = st.selectbox(
            "Collaboration mode:",
            ["research", "debate", "brainstorm", "teaching"],
            help="How should the agents work together?"
        )
    
    with col2:
        if st.button("🚀 Start Collaboration", type="primary", disabled=not user_question):
            if user_question:
                with st.spinner("🤖 Agents are collaborating..."):
                    try:
                        # Start collaborative session
                        session = st.session_state.collaborative_system.start_collaborative_session(
                            topic=user_question,
                            mode=mode
                        )
                        
                        # Display result
                        st.markdown("### 🎯 Collaborative Result:")
                        st.markdown(f"""
                        <div class="agent-response supervisor">
                            <strong>🤝 Collaborative Analysis:</strong><br>
                            {session["response"]}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show next actions
                        if "next_actions" in session:
                            st.markdown("### 💡 Suggested Next Steps:")
                            for action in session["next_actions"]:
                                st.markdown(f"• {action}")
                                
                    except Exception as e:
                        st.error(f"Error during collaboration: {e}")
    
    # Chat history (simple)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display recent chats
    if st.session_state.chat_history:
        st.markdown("### 📝 Recent Conversations")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):  # Show last 3
            with st.expander(f"💬 {chat['question'][:50]}..."):
                st.markdown(chat['response'])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
        🚀 Powered by LangChain + LangGraph | 🧠 Multi-Agent AI Research
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 