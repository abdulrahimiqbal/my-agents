"""
Main entry point for Collaborative PhysicsGPT on Streamlit Cloud.
This file serves as the primary app entry point for deployment.
"""

import streamlit as st
import os
import sys

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Set page config first
st.set_page_config(
    page_title="Collaborative PhysicsGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main app selection
st.markdown("""
# 🚀 PhysicsGPT - AI Physics Research Platform

Choose your physics research experience:
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ## 🔬 Single Agent Mode
    
    Traditional physics expert agent with:
    - Comprehensive problem solving
    - Educational explanations  
    - Physics calculations and tools
    - Step-by-step guidance
    """)
    
    if st.button("Launch Single Agent PhysicsGPT", type="secondary", use_container_width=True):
        st.switch_page("streamlit_app.py")

with col2:
    st.markdown("""
    ## 🤖 Collaborative Mode ⭐ NEW!
    
    Multi-agent physics research system with:
    - **Physics Expert** + **Hypothesis Generator** + **Supervisor**
    - 4 collaboration modes (Research, Debate, Brainstorm, Teaching)
    - Advanced session management
    - Real-time agent interactions
    """)
    
    if st.button("Launch Collaborative PhysicsGPT", type="primary", use_container_width=True):
        st.switch_page("streamlit_collaborative.py")

# Information section
st.markdown("---")

st.markdown("""
### 🌟 What's New in Collaborative Mode?

The collaborative system features multiple AI agents working together:

- **🔬 Physics Expert Agent**: Provides rigorous scientific analysis and validates hypotheses
- **💡 Hypothesis Generator Agent**: Generates creative ideas and identifies research gaps  
- **🤝 Supervisor Agent**: Orchestrates collaboration between agents

**Collaboration Modes:**
- **Research**: Systematic investigation with balanced analysis
- **Debate**: Structured discussion where agents challenge ideas
- **Brainstorm**: Creative exploration with minimal constraints
- **Teaching**: Educational explanations with expert knowledge

Try the collaborative mode to see AI agents debate quantum mechanics, brainstorm fusion energy solutions, or research dark matter detection methods!
""")

# Usage instructions
with st.expander("📚 How to Use"):
    st.markdown("""
    **For Single Agent Mode:**
    1. Ask physics questions or describe problems
    2. Use the sidebar tools for calculations
    3. Adjust difficulty level for your needs
    
    **For Collaborative Mode:**
    1. Choose a collaboration mode in the sidebar
    2. Start with a physics topic or question
    3. Watch agents work together to provide insights
    4. Use quick-start options for common scenarios
    5. Manage sessions and get summaries
    """)

# Footer
st.markdown("---")
st.markdown("*Built with LangChain, LangGraph, and Streamlit • Powered by OpenAI GPT-4*") 