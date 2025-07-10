"""
PhysicsGPT - Interactive Physics Expert Chat Agent
A comprehensive Streamlit interface for physics problem solving and education.
"""

import streamlit as st
import os
import sys
import traceback
from typing import Optional, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime

# Simple path setup for Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Import with fallback error handling
try:
    # Try direct imports first
    import src.agents.physics_expert as physics_expert_module
    import src.config.settings as settings_module
    import src.memory.stores as memory_module
    
    PhysicsExpertAgent = physics_expert_module.PhysicsExpertAgent
    Settings = settings_module.Settings
    MemoryStore = memory_module.MemoryStore
    
except ImportError as e1:
    try:
        # Fallback to relative imports
        from src.agents.physics_expert import PhysicsExpertAgent
        from src.config.settings import Settings
        from src.memory.stores import MemoryStore
    except ImportError as e2:
        try:
            # Last resort: direct module imports
            from agents.physics_expert import PhysicsExpertAgent
            from config.settings import Settings
            from memory.stores import MemoryStore
        except ImportError as e3:
            st.error("❌ **Import Error**: Unable to load PhysicsGPT modules")
            st.error("**Debug Information:**")
            st.code(f"Error 1: {e1}")
            st.code(f"Error 2: {e2}")
            st.code(f"Error 3: {e3}")
            st.code(f"Current directory: {current_dir}")
            st.code(f"Python path: {sys.path[:5]}")
            st.code(f"Directory contents: {os.listdir(current_dir)}")
            if os.path.exists(os.path.join(current_dir, 'src')):
                st.code(f"Src directory contents: {os.listdir(os.path.join(current_dir, 'src'))}")
            st.stop()

# Page configuration
st.set_page_config(
    page_title="PhysicsGPT - AI Physics Expert",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/abdulrahimiqbal/my-agents',
        'Report a bug': "https://github.com/abdulrahimiqbal/my-agents/issues",
        'About': "PhysicsGPT - Your AI Physics Expert powered by LangChain and OpenAI"
    }
)

# Custom CSS for physics theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .physics-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .equation-display {
        background: #1e293b;
        color: #f1f5f9;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 1.2em;
        text-align: center;
        margin: 1rem 0;
    }
    
    .physics-concept {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e40af 0%, #3b82f6 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #06b6d4 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    
    .user-message {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 4px solid #10b981;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'physics_agent' not in st.session_state:
        st.session_state.physics_agent = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'settings' not in st.session_state:
        st.session_state.settings = None
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = f"physics_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if 'problem_count' not in st.session_state:
        st.session_state.problem_count = 0
    if 'concept_count' not in st.session_state:
        st.session_state.concept_count = 0

def load_physics_agent() -> Optional[PhysicsExpertAgent]:
    """Load and initialize the physics expert agent."""
    try:
        # Load settings
        settings = Settings()
        st.session_state.settings = settings
        
        # Initialize memory store
        memory_store = MemoryStore(db_path="physics_memory.db")
        
        # Create physics agent
        agent = PhysicsExpertAgent(
            difficulty_level=st.session_state.get('difficulty_level', 'undergraduate'),
            specialty=st.session_state.get('specialty', None),
            memory_enabled=True,
            memory=memory_store.get_checkpointer()
        )
        
        return agent
    except Exception as e:
        st.error(f"Error loading physics agent: {e}")
        return None

def create_header():
    """Create the main header with physics theme."""
    st.markdown("""
    <div class="main-header">
        <h1>⚛️ PhysicsGPT</h1>
        <p style="font-size: 1.2em; margin: 0;">Your AI Physics Expert & Problem Solver</p>
        <p style="opacity: 0.9; margin: 0.5rem 0 0 0;">Powered by LangChain • OpenAI • Advanced Physics Tools</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create the sidebar with agent configuration."""
    with st.sidebar:
        st.markdown("## ⚙️ Physics Agent Settings")
        
        # Difficulty level selection
        difficulty_level = st.selectbox(
            "🎓 Difficulty Level",
            ["high_school", "undergraduate", "graduate", "research"],
            index=1,
            help="Select the appropriate difficulty level for explanations and problems"
        )
        
        # Physics specialty
        specialty = st.selectbox(
            "🔬 Physics Specialty",
            [None, "mechanics", "electromagnetism", "quantum", "thermodynamics", 
             "relativity", "optics", "particle_physics", "cosmology", "condensed_matter"],
            help="Optional specialty focus for the agent"
        )
        
        # Update session state
        if st.session_state.get('difficulty_level') != difficulty_level:
            st.session_state.difficulty_level = difficulty_level
            st.session_state.physics_agent = None  # Reload agent
        
        if st.session_state.get('specialty') != specialty:
            st.session_state.specialty = specialty
            st.session_state.physics_agent = None  # Reload agent
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("## 🚀 Quick Actions")
        
        if st.button("🧮 Physics Calculator", use_container_width=True):
            st.session_state.quick_action = "calculator"
        
        if st.button("📚 Physics Constants", use_container_width=True):
            st.session_state.quick_action = "constants"
        
        if st.button("🔄 Unit Converter", use_container_width=True):
            st.session_state.quick_action = "converter"
        
        if st.button("📖 Equation Lookup", use_container_width=True):
            st.session_state.quick_action = "equations"
        
        if st.button("🔍 ArXiv Search", use_container_width=True):
            st.session_state.quick_action = "arxiv"
        
        st.markdown("---")
        
        # Session statistics
        st.markdown("## 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Problems Solved", st.session_state.problem_count)
        with col2:
            st.metric("Concepts Explored", st.session_state.concept_count)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.problem_count = 0
            st.session_state.concept_count = 0
            st.rerun()

def display_quick_action_interface():
    """Display interface for quick actions."""
    if 'quick_action' not in st.session_state:
        return
    
    action = st.session_state.quick_action
    
    if action == "calculator":
        st.markdown('<div class="physics-card">', unsafe_allow_html=True)
        st.markdown("### 🧮 Physics Calculator")
        
        calc_type = st.selectbox(
            "Calculator Type",
            ["Scientific Expression", "Vector Operations", "Quadratic Solver", "Physics Functions"]
        )
        
        if calc_type == "Scientific Expression":
            expression = st.text_input("Enter mathematical expression:", placeholder="sin(pi/4) + sqrt(2)")
            if st.button("Calculate") and expression:
                if st.session_state.physics_agent:
                    try:
                        result = st.session_state.physics_agent.run(
                            f"Calculate this expression: {expression}",
                            thread_id=st.session_state.thread_id
                        )
                        st.success(result)
                    except Exception as e:
                        st.error(f"Calculation error: {e}")
        
        elif calc_type == "Vector Operations":
            col1, col2 = st.columns(2)
            with col1:
                vector1 = st.text_input("Vector 1 (comma-separated):", placeholder="1,2,3")
                operation = st.selectbox("Operation", ["magnitude", "normalize", "dot", "cross", "add", "subtract"])
            with col2:
                vector2 = st.text_input("Vector 2 (if needed):", placeholder="4,5,6")
            
            if st.button("Calculate Vector Operation") and vector1:
                if st.session_state.physics_agent:
                    try:
                        prompt = f"Perform vector operation: {operation} on vector {vector1}"
                        if vector2 and operation in ["dot", "cross", "add", "subtract"]:
                            prompt += f" and vector {vector2}"
                        result = st.session_state.physics_agent.run(prompt, thread_id=st.session_state.thread_id)
                        st.success(result)
                    except Exception as e:
                        st.error(f"Vector calculation error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif action == "constants":
        st.markdown('<div class="physics-card">', unsafe_allow_html=True)
        st.markdown("### 📚 Physics Constants")
        
        category = st.selectbox(
            "Category",
            ["fundamental", "particles", "electromagnetic", "thermodynamic", "astronomical", "all"]
        )
        
        if st.button("List Constants"):
            if st.session_state.physics_agent:
                try:
                    result = st.session_state.physics_agent.run(
                        f"List physics constants in category: {category}",
                        thread_id=st.session_state.thread_id
                    )
                    st.info(result)
                except Exception as e:
                    st.error(f"Error: {e}")
        
        constant_name = st.text_input("Look up specific constant:", placeholder="speed_of_light")
        if st.button("Get Constant") and constant_name:
            if st.session_state.physics_agent:
                try:
                    result = st.session_state.physics_agent.run(
                        f"Get physics constant: {constant_name}",
                        thread_id=st.session_state.thread_id
                    )
                    st.success(result)
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif action == "converter":
        st.markdown('<div class="physics-card">', unsafe_allow_html=True)
        st.markdown("### 🔄 Unit Converter")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            value = st.number_input("Value:", value=1.0)
            from_unit = st.text_input("From unit:", placeholder="m")
        with col2:
            to_unit = st.text_input("To unit:", placeholder="ft")
        with col3:
            quantity = st.selectbox("Quantity (optional)", ["auto", "length", "mass", "time", "energy", "force"])
        
        if st.button("Convert Units") and from_unit and to_unit:
            if st.session_state.physics_agent:
                try:
                    result = st.session_state.physics_agent.run(
                        f"Convert {value} {from_unit} to {to_unit}",
                        thread_id=st.session_state.thread_id
                    )
                    st.success(result)
                except Exception as e:
                    st.error(f"Conversion error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif action == "equations":
        st.markdown('<div class="physics-card">', unsafe_allow_html=True)
        st.markdown("### 📖 Equation Lookup")
        
        equation_name = st.text_input("Equation name:", placeholder="newton's second law")
        if st.button("Look Up Equation") and equation_name:
            if st.session_state.physics_agent:
                try:
                    result = st.session_state.physics_agent.run(
                        f"Look up physics equation: {equation_name}",
                        thread_id=st.session_state.thread_id
                    )
                    st.info(result)
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif action == "arxiv":
        st.markdown('<div class="physics-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 ArXiv Research")
        
        col1, col2 = st.columns(2)
        with col1:
            query = st.text_input("Search query:", placeholder="quantum entanglement")
            max_results = st.slider("Max results:", 1, 10, 5)
        with col2:
            category = st.selectbox("Category", ["physics", "math", "cs", "all"])
        
        if st.button("Search ArXiv") and query:
            if st.session_state.physics_agent:
                try:
                    result = st.session_state.physics_agent.run(
                        f"Search ArXiv for: {query} (max {max_results} results, category: {category})",
                        thread_id=st.session_state.thread_id
                    )
                    st.info(result)
                except Exception as e:
                    st.error(f"Search error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Clear the quick action
    if st.button("❌ Close Quick Action"):
        del st.session_state.quick_action
        st.rerun()

def display_chat_interface():
    """Display the main chat interface."""
    st.markdown("## 💬 Physics Chat")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>PhysicsGPT:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask me anything about physics...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get agent response
        if st.session_state.physics_agent:
            try:
                with st.spinner("PhysicsGPT is thinking..."):
                    response = st.session_state.physics_agent.run(
                        user_input,
                        thread_id=st.session_state.thread_id
                    )
                
                # Add response to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Update counters
                if any(word in user_input.lower() for word in ['solve', 'calculate', 'problem']):
                    st.session_state.problem_count += 1
                if any(word in user_input.lower() for word in ['explain', 'what is', 'concept']):
                    st.session_state.concept_count += 1
                
                st.rerun()
            
            except Exception as e:
                st.error(f"Error getting response: {e}")
                st.error(f"Traceback: {traceback.format_exc()}")

def display_example_problems():
    """Display example physics problems for quick testing."""
    st.markdown("## 🎯 Example Problems")
    
    examples = [
        {
            "title": "Classical Mechanics",
            "problem": "A ball is thrown upward with an initial velocity of 20 m/s. How high does it go and how long does it take to return to the ground?",
            "icon": "🏀"
        },
        {
            "title": "Electromagnetism", 
            "problem": "What is the electric field strength 2 meters away from a point charge of 5 μC?",
            "icon": "⚡"
        },
        {
            "title": "Quantum Mechanics",
            "problem": "Calculate the wavelength of an electron moving at 10% the speed of light.",
            "icon": "🌊"
        },
        {
            "title": "Thermodynamics",
            "problem": "An ideal gas undergoes isothermal expansion from 1 L to 3 L at 300 K. If the initial pressure is 2 atm, what is the final pressure?",
            "icon": "🌡️"
        }
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            st.markdown(f'<div class="physics-concept">', unsafe_allow_html=True)
            st.markdown(f"### {example['icon']} {example['title']}")
            st.markdown(f"*{example['problem']}*")
            if st.button(f"Try this problem", key=f"example_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": example['problem']})
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function."""
    initialize_session_state()
    
    # Load physics agent if not already loaded
    if st.session_state.physics_agent is None:
        with st.spinner("Initializing PhysicsGPT..."):
            st.session_state.physics_agent = load_physics_agent()
    
    # Check if agent loaded successfully
    if st.session_state.physics_agent is None:
        st.error("Failed to initialize PhysicsGPT. Please check your configuration.")
        st.stop()
    
    # Create UI
    create_header()
    create_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display quick action interface if active
        if 'quick_action' in st.session_state:
            display_quick_action_interface()
        else:
            display_chat_interface()
    
    with col2:
        display_example_problems()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.7; padding: 1rem;">
        <p>PhysicsGPT v1.0 | Built with ❤️ using Streamlit & LangChain</p>
        <p>🔬 Advancing Physics Education Through AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 