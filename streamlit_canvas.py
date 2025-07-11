"""
PhysicsGPT Agent Canvas - Modern Agent-Centric Interface
A visual, interactive canvas where users can directly engage with AI agents.
"""

import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import time

# Simple path setup for imports
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Import our components
try:
    from src.agents import CollaborativePhysicsSystem, PhysicsExpertAgent, HypothesisGeneratorAgent, SupervisorAgent
    from src.database import KnowledgeAPI, DatabaseMigrator
    from src.config import get_settings
    from src.memory import get_memory_store
except ImportError as e:
    st.error(f"❌ **Import Error**: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="PhysicsGPT Agent Canvas",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern agent canvas design
st.markdown("""
<style>
    /* Global Styles */
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        color: white;
        text-align: center;
    }
    
    .canvas-area {
        background: #f8fafc;
        border-radius: 15px;
        padding: 2rem;
        min-height: 600px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    /* Agent Cards */
    .agent-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 2px solid transparent;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .agent-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        border-color: #3b82f6;
    }
    
    .agent-card.selected {
        border-color: #3b82f6;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1);
    }
    
    .agent-card.active::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #10b981, #3b82f6);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .agent-avatar {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        margin: 0 auto 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        color: white;
        font-weight: bold;
    }
    
    .physics-expert {
        background: linear-gradient(135deg, #10b981, #059669);
    }
    
    .hypothesis-generator {
        background: linear-gradient(135deg, #f59e0b, #d97706);
    }
    
    .supervisor {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
    }
    
    .agent-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .agent-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    .status-active {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-inactive {
        background: #f3f4f6;
        color: #6b7280;
    }
    
    .status-thinking {
        background: #fef3c7;
        color: #92400e;
        animation: thinking 1.5s infinite;
    }
    
    @keyframes thinking {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .agent-stats {
        display: flex;
        justify-content: space-between;
        margin: 1rem 0;
        font-size: 0.8rem;
        color: #6b7280;
    }
    
    .chat-button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .chat-button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: translateY(-2px);
    }
    
    /* Chat Interface */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        min-height: 400px;
    }
    
    .chat-header {
        display: flex;
        align-items: center;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f1f5f9;
        margin-bottom: 1rem;
    }
    
    .chat-messages {
        max-height: 300px;
        overflow-y: auto;
        margin-bottom: 1rem;
        padding: 0.5rem;
    }
    
    .message {
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 12px;
        max-width: 80%;
    }
    
    .message.user {
        background: #eff6ff;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .message.agent {
        background: #f0fdf4;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    
    /* Knowledge Sidebar */
    .knowledge-sidebar {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        height: fit-content;
        position: sticky;
        top: 1rem;
    }
    
    .knowledge-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .knowledge-metric:last-child {
        border-bottom: none;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #3b82f6;
    }
    
    .recent-activity {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 2px solid #f1f5f9;
    }
    
    .activity-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        font-size: 0.9rem;
        color: #6b7280;
    }
    
    .activity-icon {
        margin-right: 0.5rem;
        font-size: 1rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .agent-card {
            margin: 0.5rem 0;
        }
        
        .canvas-area {
            padding: 1rem;
        }
    }
    
    /* Collaboration Mode */
    .collaboration-active {
        background: linear-gradient(135deg, #fef3c7, #f59e0b);
        border: 2px solid #f59e0b;
    }
    
    .collaboration-indicator {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: #f59e0b;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 8px;
        font-size: 0.7rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'knowledge_api' not in st.session_state:
        st.session_state.knowledge_api = None
    if 'agents' not in st.session_state:
        st.session_state.agents = {}
    if 'selected_agent' not in st.session_state:
        st.session_state.selected_agent = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}
    if 'collaboration_mode' not in st.session_state:
        st.session_state.collaboration_mode = False
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = {
            'physics_expert': 'inactive',
            'hypothesis_generator': 'inactive', 
            'supervisor': 'inactive'
        }

def initialize_knowledge_api():
    """Initialize the Knowledge API."""
    if st.session_state.knowledge_api is None:
        try:
            with st.spinner("🔄 Initializing knowledge management system..."):
                migrator = DatabaseMigrator("./data/memory.db")
                migration_success = migrator.migrate()
                
                if migration_success:
                    from src.database.knowledge_api import get_knowledge_api
                    st.session_state.knowledge_api = get_knowledge_api()
                    return True
                else:
                    st.error("❌ Failed to initialize knowledge management system")
                    return False
        except Exception as e:
            st.error(f"❌ Knowledge API initialization failed: {e}")
            return False
    return True

def initialize_agents():
    """Initialize the agent instances."""
    if not st.session_state.agents:
        try:
            st.session_state.agents = {
                'physics_expert': PhysicsExpertAgent(
                    difficulty_level="undergraduate",
                    specialty="general",
                    memory_enabled=True
                ),
                'hypothesis_generator': HypothesisGeneratorAgent(
                    creativity_level="high",
                    exploration_scope="broad",
                    memory_enabled=True
                ),
                'supervisor': SupervisorAgent(
                    collaboration_style="balanced",
                    memory_enabled=True
                )
            }
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize agents: {e}")
            return False
    return True

def create_main_header():
    """Create the main header."""
    st.markdown("""
    <div class="main-container">
        <h1>🧪 PhysicsGPT Agent Canvas</h1>
        <p>Interactive Multi-Agent Physics Research Platform</p>
    </div>
    """, unsafe_allow_html=True)

def render_agent_card(agent_id: str, agent_name: str, agent_icon: str, agent_class: str):
    """Render an individual agent card."""
    status = st.session_state.agent_status.get(agent_id, 'inactive')
    
    # Get agent stats
    chat_count = len(st.session_state.chat_history.get(agent_id, []))
    
    # Determine status display
    status_class = f"status-{status}"
    status_text = status.title()
    
    # Card HTML
    card_classes = f"agent-card {agent_class}"
    if st.session_state.selected_agent == agent_id:
        card_classes += " selected"
    if status == 'active':
        card_classes += " active"
    
    card_html = f"""
    <div class="{card_classes}" onclick="selectAgent('{agent_id}')">
        <div class="agent-avatar {agent_class}">
            {agent_icon}
        </div>
        <div class="agent-name">{agent_name}</div>
        <div class="agent-status {status_class}">{status_text}</div>
        <div class="agent-stats">
            <span>💬 {chat_count} chats</span>
            <span>⚡ Ready</span>
        </div>
    </div>
    """
    
    return card_html

def render_agent_canvas():
    """Render the main agent canvas."""
    st.markdown('<div class="canvas-area">', unsafe_allow_html=True)
    
    # Agent cards in a row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔬 Physics Expert", key="select_physics", use_container_width=True):
            st.session_state.selected_agent = 'physics_expert'
            st.session_state.agent_status['physics_expert'] = 'active'
        
        card_html = render_agent_card(
            'physics_expert', 
            'Physics Expert', 
            '🔬', 
            'physics-expert'
        )
        st.markdown(card_html, unsafe_allow_html=True)
    
    with col2:
        if st.button("💡 Hypothesis Generator", key="select_hypothesis", use_container_width=True):
            st.session_state.selected_agent = 'hypothesis_generator'
            st.session_state.agent_status['hypothesis_generator'] = 'active'
        
        card_html = render_agent_card(
            'hypothesis_generator', 
            'Hypothesis Generator', 
            '💡', 
            'hypothesis-generator'
        )
        st.markdown(card_html, unsafe_allow_html=True)
    
    with col3:
        if st.button("🤝 Supervisor", key="select_supervisor", use_container_width=True):
            st.session_state.selected_agent = 'supervisor'
            st.session_state.agent_status['supervisor'] = 'active'
        
        card_html = render_agent_card(
            'supervisor', 
            'Supervisor', 
            '🤝', 
            'supervisor'
        )
        st.markdown(card_html, unsafe_allow_html=True)
    
    # Collaboration toggle
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        collaboration = st.toggle("🤝 Collaboration Mode", value=st.session_state.collaboration_mode)
        if collaboration != st.session_state.collaboration_mode:
            st.session_state.collaboration_mode = collaboration
            if collaboration:
                st.session_state.selected_agent = 'supervisor'
                st.session_state.agent_status['supervisor'] = 'active'
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_chat_interface():
    """Render the chat interface for the selected agent."""
    if not st.session_state.selected_agent:
        st.info("👆 Select an agent above to start chatting!")
        return
    
    agent_id = st.session_state.selected_agent
    agent_names = {
        'physics_expert': '🔬 Physics Expert',
        'hypothesis_generator': '💡 Hypothesis Generator', 
        'supervisor': '🤝 Supervisor'
    }
    
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Chat header
    st.markdown(f"""
    <div class="chat-header">
        <h3>💬 Chat with {agent_names[agent_id]}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat messages
    if agent_id not in st.session_state.chat_history:
        st.session_state.chat_history[agent_id] = []
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history[agent_id]:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="message user">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message agent">
                    <strong>{agent_names[agent_id]}:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input(f"Ask {agent_names[agent_id]} anything...")
    
    if user_input:
        # Add user message
        st.session_state.chat_history[agent_id].append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })
        
        # Update agent status
        st.session_state.agent_status[agent_id] = 'thinking'
        
        # Get agent response
        try:
            agent = st.session_state.agents[agent_id]
            
            with st.spinner(f"{agent_names[agent_id]} is thinking..."):
                if st.session_state.collaboration_mode and agent_id == 'supervisor':
                    # Use collaborative system
                    from src.agents import CollaborativePhysicsSystem
                    collab_system = CollaborativePhysicsSystem()
                    response = collab_system.start_research_session(
                        topic=user_input,
                        session_id=f"canvas_{int(time.time())}"
                    )
                else:
                    # Direct agent chat
                    response = agent.chat(user_input, thread_id=f"canvas_{agent_id}")
            
            # Add agent response
            st.session_state.chat_history[agent_id].append({
                'role': 'agent',
                'content': response,
                'timestamp': datetime.now()
            })
            
            # Update agent status
            st.session_state.agent_status[agent_id] = 'active'
            
            # Log interaction
            if st.session_state.knowledge_api:
                try:
                    st.session_state.knowledge_api.log_event_sync(
                        source=f"canvas_{agent_id}",
                        event_type="agent_interaction",
                        payload={
                            "user_input": user_input,
                            "response_length": len(response),
                            "agent_type": agent_id,
                            "collaboration_mode": st.session_state.collaboration_mode
                        }
                    )
                except Exception as e:
                    st.error(f"Failed to log interaction: {e}")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error getting response: {e}")
            st.session_state.agent_status[agent_id] = 'inactive'
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_knowledge_sidebar():
    """Render the knowledge management sidebar."""
    st.markdown('<div class="knowledge-sidebar">', unsafe_allow_html=True)
    
    st.markdown("### 📊 Knowledge Hub")
    
    # Get analytics if available
    if st.session_state.knowledge_api:
        try:
            analytics = asyncio.run(st.session_state.knowledge_api.get_system_analytics())
            
            # Metrics
            st.markdown(f"""
            <div class="knowledge-metric">
                <span>📚 Knowledge</span>
                <span class="metric-value">{analytics.get('total_knowledge', 0)}</span>
            </div>
            <div class="knowledge-metric">
                <span>🧪 Hypotheses</span>
                <span class="metric-value">{analytics.get('total_hypotheses', 0)}</span>
            </div>
            <div class="knowledge-metric">
                <span>📋 Events</span>
                <span class="metric-value">{analytics.get('total_events', 0)}</span>
            </div>
            <div class="knowledge-metric">
                <span>🎯 Success Rate</span>
                <span class="metric-value">{analytics.get('promotion_rate', 0):.0%}</span>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Failed to load analytics: {e}")
    
    # Quick search
    st.markdown("### 🔍 Quick Search")
    search_query = st.text_input("Search knowledge...", placeholder="quantum mechanics")
    
    if search_query and st.session_state.knowledge_api:
        try:
            results = asyncio.run(st.session_state.knowledge_api.search_knowledge(search_query, limit=3))
            if results:
                st.markdown("**Results:**")
                for result in results:
                    st.markdown(f"• {result['statement'][:60]}...")
            else:
                st.info("No results found")
        except Exception as e:
            st.error(f"Search failed: {e}")
    
    # Recent activity
    st.markdown("""
    <div class="recent-activity">
        <h4>📈 Recent Activity</h4>
        <div class="activity-item">
            <span class="activity-icon">💬</span>
            <span>Agent interaction logged</span>
        </div>
        <div class="activity-item">
            <span class="activity-icon">🧪</span>
            <span>Hypothesis generated</span>
        </div>
        <div class="activity-item">
            <span class="activity-icon">📚</span>
            <span>Knowledge promoted</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Full knowledge management
    if st.button("📖 Open Full Knowledge Lab", use_container_width=True):
        st.switch_page("streamlit_knowledge.py")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Initialize systems
    if not initialize_knowledge_api():
        st.stop()
    
    if not initialize_agents():
        st.stop()
    
    # Create header
    create_main_header()
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Agent canvas
        render_agent_canvas()
        
        # Chat interface
        render_chat_interface()
    
    with col2:
        # Knowledge sidebar
        render_knowledge_sidebar()

if __name__ == "__main__":
    main() 