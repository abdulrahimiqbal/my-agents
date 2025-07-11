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
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        border-color: #3b82f6;
    }
    
    .agent-card.selected {
        border-color: #3b82f6;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1);
        transform: scale(1.05);
    }
    
    .agent-card.active::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6);
        background-size: 200% 100%;
        animation: gradient-flow 3s ease infinite;
    }
    
    .agent-card.thinking {
        animation: thinking-pulse 2s ease-in-out infinite;
    }
    
    @keyframes gradient-flow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes thinking-pulse {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        50% { 
            transform: scale(1.02);
            box-shadow: 0 12px 40px rgba(59, 130, 246, 0.2);
        }
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
        transition: transform 0.3s ease;
    }
    
    .agent-card:hover .agent-avatar {
        transform: rotate(5deg) scale(1.1);
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
        transition: color 0.3s ease;
    }
    
    .agent-card:hover .agent-name {
        color: #3b82f6;
    }
    
    .agent-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .status-active {
        background: #d1fae5;
        color: #065f46;
        animation: status-glow 2s ease-in-out infinite;
    }
    
    .status-inactive {
        background: #f3f4f6;
        color: #6b7280;
    }
    
    .status-thinking {
        background: #fef3c7;
        color: #92400e;
        animation: thinking-dots 1.5s infinite;
    }
    
    @keyframes status-glow {
        0%, 100% { box-shadow: 0 0 5px rgba(16, 185, 129, 0.3); }
        50% { box-shadow: 0 0 15px rgba(16, 185, 129, 0.6); }
    }
    
    @keyframes thinking-dots {
        0%, 20% { opacity: 1; }
        50% { opacity: 0.7; }
        80%, 100% { opacity: 1; }
    }
    
    .agent-stats {
        display: flex;
        justify-content: space-between;
        margin: 1rem 0;
        font-size: 0.8rem;
        color: #6b7280;
        transition: opacity 0.3s ease;
    }
    
    .agent-card:hover .agent-stats {
        opacity: 0.8;
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
        position: relative;
        overflow: hidden;
    }
    
    .chat-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .chat-button:hover::before {
        left: 100%;
    }
    
    .chat-button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
    }
    
    /* Chat Interface */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        min-height: 400px;
        animation: slideInUp 0.5s ease-out;
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
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
        animation: messageSlide 0.3s ease-out;
    }
    
    @keyframes messageSlide {
        from {
            opacity: 0;
            transform: translateX(-10px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .message.user {
        background: #eff6ff;
        margin-left: auto;
        border-bottom-right-radius: 4px;
        border-left: 3px solid #3b82f6;
    }
    
    .message.agent {
        background: #f0fdf4;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        border-right: 3px solid #10b981;
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
        animation: slideInRight 0.5s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .knowledge-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid #f1f5f9;
        transition: background-color 0.3s ease;
    }
    
    .knowledge-metric:hover {
        background-color: #f8fafc;
        border-radius: 8px;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    
    .knowledge-metric:last-child {
        border-bottom: none;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #3b82f6;
        transition: transform 0.3s ease;
    }
    
    .knowledge-metric:hover .metric-value {
        transform: scale(1.1);
    }
    
    /* Collaboration Mode Styles */
    .collaboration-active {
        background: linear-gradient(135deg, #fef3c7, #f59e0b);
        border: 2px solid #f59e0b;
        animation: collaboration-glow 3s ease-in-out infinite;
    }
    
    @keyframes collaboration-glow {
        0%, 100% { 
            box-shadow: 0 0 20px rgba(245, 158, 11, 0.3);
        }
        50% { 
            box-shadow: 0 0 30px rgba(245, 158, 11, 0.6);
        }
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
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-5px); }
        60% { transform: translateY(-3px); }
    }
    
    /* Responsive Design */
    @media (max-width: 1200px) {
        .canvas-area {
            padding: 1.5rem;
        }
        
        .agent-card {
            margin: 0.5rem;
            padding: 1rem;
        }
        
        .agent-avatar {
            width: 60px;
            height: 60px;
            font-size: 2rem;
        }
    }
    
    @media (max-width: 768px) {
        .main-container {
            padding: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .canvas-area {
            padding: 1rem;
            min-height: 400px;
        }
        
        .agent-card {
            margin: 0.5rem 0;
            padding: 1rem;
        }
        
        .agent-avatar {
            width: 50px;
            height: 50px;
            font-size: 1.5rem;
        }
        
        .agent-name {
            font-size: 1rem;
        }
        
        .chat-container {
            margin-top: 1rem;
            padding: 1rem;
            min-height: 300px;
        }
        
        .knowledge-sidebar {
            margin-top: 1rem;
            position: static;
        }
        
        .message {
            max-width: 95%;
            padding: 0.75rem;
        }
    }
    
    @media (max-width: 480px) {
        .agent-card {
            text-align: center;
        }
        
        .agent-stats {
            flex-direction: column;
            gap: 0.25rem;
            text-align: center;
        }
        
        .chat-button {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
        
        .knowledge-metric {
            flex-direction: column;
            text-align: center;
            gap: 0.25rem;
        }
    }
    
    /* Loading States */
    .loading-shimmer {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Custom Scrollbar */
    .chat-messages::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-messages::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 10px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Accessibility */
    .agent-card:focus {
        outline: 3px solid #3b82f6;
        outline-offset: 2px;
    }
    
    .chat-button:focus {
        outline: 3px solid #ffffff;
        outline-offset: 2px;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .canvas-area {
            background: #1f2937;
            color: #f9fafb;
        }
        
        .agent-card {
            background: #374151;
            color: #f9fafb;
        }
        
        .knowledge-sidebar {
            background: #374151;
            color: #f9fafb;
        }
        
        .chat-container {
            background: #374151;
            color: #f9fafb;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'agents' not in st.session_state:
        st.session_state.agents = {}
    
    if 'knowledge_api' not in st.session_state:
        st.session_state.knowledge_api = None
    
    if 'selected_agent' not in st.session_state:
        st.session_state.selected_agent = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {
            'physics_expert': [],
            'hypothesis_generator': [],
            'supervisor': []
        }
    
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = {
            'physics_expert': 'inactive',
            'hypothesis_generator': 'inactive',
            'supervisor': 'inactive'
        }
    
    if 'collaboration_mode' not in st.session_state:
        st.session_state.collaboration_mode = False
    
    # User preferences system
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'theme': 'light',  # light, dark, auto
            'animation_speed': 'normal',  # slow, normal, fast
            'auto_scroll': True,
            'sound_enabled': False,
            'compact_mode': False,
            'default_agent': 'physics_expert',
            'collaboration_auto_start': False,
            'notification_level': 'normal',  # minimal, normal, verbose
            'preferred_layout': 'standard',  # standard, compact, wide
            'quick_actions_visible': True,
            'analytics_visible': True,
            'recent_activity_count': 5
        }
    
    # Quick actions history
    if 'quick_actions_history' not in st.session_state:
        st.session_state.quick_actions_history = []
    
    # Session analytics
    if 'session_analytics' not in st.session_state:
        st.session_state.session_analytics = {
            'interactions': 0,
            'hypotheses_generated': 0,
            'knowledge_created': 0,
            'session_start': datetime.now(),
            'active_time': 0,
            'favorite_agents': {}
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
    """Render the main agent canvas with advanced animations."""
    st.markdown('<div class="canvas-area">', unsafe_allow_html=True)
    
    # Collaboration mode indicator
    if st.session_state.collaboration_mode:
        st.markdown("""
        <div class="collaboration-indicator">
            🤝 Collaboration Active
        </div>
        """, unsafe_allow_html=True)
    
    # Agent connection visualization
    if st.session_state.collaboration_mode:
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <div class="agent-connections">
                <div class="connection-line physics-to-hyp"></div>
                <div class="connection-line hyp-to-supervisor"></div>
                <div class="connection-line physics-to-supervisor"></div>
                <div class="collaboration-pulse"></div>
            </div>
        </div>
        
        <style>
        .agent-connections {
            position: relative;
            height: 60px;
            margin: 1rem 0;
        }
        
        .connection-line {
            position: absolute;
            height: 2px;
            background: linear-gradient(90deg, #3b82f6, #10b981);
            animation: data-flow 2s infinite;
            border-radius: 1px;
        }
        
        .physics-to-hyp {
            top: 20px;
            left: 20%;
            width: 25%;
            animation-delay: 0s;
        }
        
        .hyp-to-supervisor {
            top: 20px;
            right: 20%;
            width: 25%;
            animation-delay: 0.7s;
        }
        
        .physics-to-supervisor {
            top: 40px;
            left: 30%;
            width: 40%;
            animation-delay: 1.4s;
        }
        
        .collaboration-pulse {
            position: absolute;
            top: 50%;
            left: 50%;
            width: 20px;
            height: 20px;
            background: #f59e0b;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            animation: collaboration-pulse 2s infinite;
        }
        
        @keyframes data-flow {
            0% {
                opacity: 0;
                transform: scaleX(0);
            }
            50% {
                opacity: 1;
                transform: scaleX(1);
            }
            100% {
                opacity: 0;
                transform: scaleX(1);
            }
        }
        
        @keyframes collaboration-pulse {
            0%, 100% {
                transform: translate(-50%, -50%) scale(1);
                opacity: 1;
            }
            50% {
                transform: translate(-50%, -50%) scale(1.5);
                opacity: 0.7;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Agent cards in a row with enhanced animations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔬 Physics Expert", key="select_physics", use_container_width=True):
            st.session_state.selected_agent = 'physics_expert'
            st.session_state.agent_status['physics_expert'] = 'active'
            
            # Update session analytics
            st.session_state.session_analytics['interactions'] += 1
            if 'physics_expert' not in st.session_state.session_analytics['favorite_agents']:
                st.session_state.session_analytics['favorite_agents']['physics_expert'] = 0
            st.session_state.session_analytics['favorite_agents']['physics_expert'] += 1
        
        # Enhanced agent card with status animations
        status = st.session_state.agent_status.get('physics_expert', 'inactive')
        thinking_class = " thinking" if status == 'thinking' else ""
        
        card_html = f"""
        <div class="agent-card physics-expert{thinking_class}" id="physics-expert-card">
            <div class="agent-avatar physics-expert">
                🔬
            </div>
            <div class="agent-name">Physics Expert</div>
            <div class="agent-status status-{status}">{status.title()}</div>
            <div class="agent-stats">
                <span>💬 {len(st.session_state.chat_history.get('physics_expert', []))} chats</span>
                <span>⚡ Ready</span>
            </div>
            {f'<div class="collaboration-indicator">🤝 Active</div>' if st.session_state.collaboration_mode else ''}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    
    with col2:
        if st.button("💡 Hypothesis Generator", key="select_hypothesis", use_container_width=True):
            st.session_state.selected_agent = 'hypothesis_generator'
            st.session_state.agent_status['hypothesis_generator'] = 'active'
            
            # Update session analytics
            st.session_state.session_analytics['interactions'] += 1
            if 'hypothesis_generator' not in st.session_state.session_analytics['favorite_agents']:
                st.session_state.session_analytics['favorite_agents']['hypothesis_generator'] = 0
            st.session_state.session_analytics['favorite_agents']['hypothesis_generator'] += 1
        
        # Enhanced agent card with status animations
        status = st.session_state.agent_status.get('hypothesis_generator', 'inactive')
        thinking_class = " thinking" if status == 'thinking' else ""
        
        card_html = f"""
        <div class="agent-card hypothesis-generator{thinking_class}" id="hypothesis-generator-card">
            <div class="agent-avatar hypothesis-generator">
                💡
            </div>
            <div class="agent-name">Hypothesis Generator</div>
            <div class="agent-status status-{status}">{status.title()}</div>
            <div class="agent-stats">
                <span>💬 {len(st.session_state.chat_history.get('hypothesis_generator', []))} chats</span>
                <span>⚡ Ready</span>
            </div>
            {f'<div class="collaboration-indicator">🤝 Active</div>' if st.session_state.collaboration_mode else ''}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    
    with col3:
        if st.button("🤝 Supervisor", key="select_supervisor", use_container_width=True):
            st.session_state.selected_agent = 'supervisor'
            st.session_state.agent_status['supervisor'] = 'active'
            
            # Update session analytics
            st.session_state.session_analytics['interactions'] += 1
            if 'supervisor' not in st.session_state.session_analytics['favorite_agents']:
                st.session_state.session_analytics['favorite_agents']['supervisor'] = 0
            st.session_state.session_analytics['favorite_agents']['supervisor'] += 1
        
        # Enhanced agent card with status animations
        status = st.session_state.agent_status.get('supervisor', 'inactive')
        thinking_class = " thinking" if status == 'thinking' else ""
        
        card_html = f"""
        <div class="agent-card supervisor{thinking_class}" id="supervisor-card">
            <div class="agent-avatar supervisor">
                🤝
            </div>
            <div class="agent-name">Supervisor</div>
            <div class="agent-status status-{status}">{status.title()}</div>
            <div class="agent-stats">
                <span>💬 {len(st.session_state.chat_history.get('supervisor', []))} chats</span>
                <span>⚡ Ready</span>
            </div>
            {f'<div class="collaboration-indicator">🤝 Active</div>' if st.session_state.collaboration_mode else ''}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    
    # Collaboration toggle with enhanced styling
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        collaboration = st.toggle("🤝 Collaboration Mode", value=st.session_state.collaboration_mode)
        if collaboration != st.session_state.collaboration_mode:
            st.session_state.collaboration_mode = collaboration
            if collaboration:
                st.balloons()  # Celebration animation
                st.success("🎉 Collaboration mode activated! Agents will work together.")
                
                # Auto-start collaboration if preference is set
                if st.session_state.user_preferences.get('collaboration_auto_start', False):
                    st.session_state.selected_agent = 'supervisor'
                    st.session_state.agent_status['supervisor'] = 'active'
            else:
                st.info("🔄 Collaboration mode deactivated. Agents will work independently.")
            st.rerun()
    
    # Session analytics display (if enabled in preferences)
    if st.session_state.user_preferences.get('analytics_visible', True):
        st.markdown("---")
        st.markdown("### 📊 Session Analytics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Interactions", 
                st.session_state.session_analytics['interactions'],
                delta="+1" if st.session_state.session_analytics['interactions'] > 0 else None
            )
        
        with col2:
            st.metric(
                "Hypotheses", 
                st.session_state.session_analytics['hypotheses_generated']
            )
        
        with col3:
            st.metric(
                "Knowledge", 
                st.session_state.session_analytics['knowledge_created']
            )
        
        with col4:
            session_duration = datetime.now() - st.session_state.session_analytics['session_start']
            minutes = int(session_duration.total_seconds() / 60)
            st.metric("Session Time", f"{minutes}m")
        
        # Favorite agents chart
        if st.session_state.session_analytics['favorite_agents']:
            st.markdown("#### 🌟 Agent Usage")
            favorite_data = st.session_state.session_analytics['favorite_agents']
            
            # Simple bar chart using metrics
            for agent, count in favorite_data.items():
                agent_name = agent.replace('_', ' ').title()
                percentage = (count / st.session_state.session_analytics['interactions']) * 100 if st.session_state.session_analytics['interactions'] > 0 else 0
                st.progress(percentage / 100, text=f"{agent_name}: {count} interactions ({percentage:.0f}%)")
    
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
    """Render the enhanced knowledge management sidebar."""
    st.markdown('<div class="knowledge-sidebar">', unsafe_allow_html=True)
    
    # Collapsible sections
    with st.expander("📊 Knowledge Metrics", expanded=True):
        if st.session_state.knowledge_api:
            try:
                analytics = asyncio.run(st.session_state.knowledge_api.get_system_analytics())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📚 Knowledge", analytics.get('total_knowledge', 0))
                    st.metric("🧪 Hypotheses", analytics.get('total_hypotheses', 0))
                with col2:
                    st.metric("📋 Events", analytics.get('total_events', 0))
                    st.metric("🎯 Success", f"{analytics.get('promotion_rate', 0):.0%}")
                
            except Exception as e:
                st.error(f"Failed to load analytics: {e}")
    
    # Enhanced search with filters
    with st.expander("🔍 Knowledge Search", expanded=True):
        search_query = st.text_input("Search knowledge...", placeholder="quantum mechanics", key="kb_search")
        
        col1, col2 = st.columns(2)
        with col1:
            domain_filter = st.selectbox("Domain", ["All", "quantum", "classical", "thermodynamics"], key="domain_filter")
        with col2:
            confidence_filter = st.slider("Min Confidence", 0.0, 1.0, 0.5, key="conf_filter")
        
        if search_query and st.session_state.knowledge_api:
            try:
                results = asyncio.run(st.session_state.knowledge_api.search_knowledge(
                    search_query, 
                    domain=domain_filter if domain_filter != "All" else None,
                    confidence_threshold=confidence_filter,
                    limit=5
                ))
                
                if results:
                    st.markdown("**Search Results:**")
                    for i, result in enumerate(results):
                        with st.container():
                            st.markdown(f"**{i+1}.** {result['statement'][:80]}...")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("💬 Discuss", key=f"discuss_{result['id']}", help="Discuss with agents"):
                                    # Add to chat as context
                                    if st.session_state.selected_agent:
                                        context_msg = f"Let's discuss this knowledge: {result['statement']}"
                                        st.session_state.chat_history[st.session_state.selected_agent].append({
                                            'role': 'user',
                                            'content': context_msg,
                                            'timestamp': datetime.now()
                                        })
                                        st.rerun()
                            with col2:
                                if st.button("🧪 Hypothesis", key=f"hyp_{result['id']}", help="Generate hypothesis"):
                                    # Generate hypothesis based on this knowledge
                                    if st.session_state.selected_agent == 'hypothesis_generator':
                                        hyp_msg = f"Generate a creative hypothesis based on: {result['statement']}"
                                        st.session_state.chat_history['hypothesis_generator'].append({
                                            'role': 'user',
                                            'content': hyp_msg,
                                            'timestamp': datetime.now()
                                        })
                                        st.session_state.selected_agent = 'hypothesis_generator'
                                        st.rerun()
                            with col3:
                                st.markdown(f"*{result.get('confidence', 0):.2f}*")
                else:
                    st.info("No results found")
            except Exception as e:
                st.error(f"Search failed: {e}")
    
    # Recent activity with real data
    with st.expander("📈 Recent Activity", expanded=True):
        if st.session_state.knowledge_api:
            try:
                # Get recent events
                recent_events = asyncio.run(st.session_state.knowledge_api.get_events_by_type("agent_interaction", limit=5))
                
                if recent_events:
                    for event in recent_events[-3:]:  # Show last 3
                        timestamp = event.get('timestamp', 'Unknown')
                        source = event.get('source', 'Unknown')
                        st.markdown(f"• 💬 {source} - {timestamp[:10]}")
                else:
                    st.markdown("• 💬 Agent interaction logged")
                    st.markdown("• 🧪 Hypothesis generated") 
                    st.markdown("• 📚 Knowledge promoted")
                    
            except Exception as e:
                st.markdown("• 💬 Agent interaction logged")
                st.markdown("• 🧪 Hypothesis generated")
                st.markdown("• 📚 Knowledge promoted")
    
    # Quick actions
    with st.expander("⚡ Quick Actions", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧪 New Hypothesis", use_container_width=True):
                st.session_state.selected_agent = 'hypothesis_generator'
                st.session_state.agent_status['hypothesis_generator'] = 'active'
                st.rerun()
        with col2:
            if st.button("🔬 Ask Expert", use_container_width=True):
                st.session_state.selected_agent = 'physics_expert'
                st.session_state.agent_status['physics_expert'] = 'active'
                st.rerun()
        
        if st.button("📖 Full Knowledge Lab", use_container_width=True):
            st.info("Opening Knowledge Lab... (would switch to streamlit_knowledge.py)")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_preferences_panel():
    """Render user preferences panel."""
    with st.expander("⚙️ Preferences", expanded=False):
        st.markdown("### 🎨 Interface")
        
        col1, col2 = st.columns(2)
        with col1:
            theme = st.selectbox(
                "Theme",
                ["light", "dark", "auto"],
                index=["light", "dark", "auto"].index(st.session_state.user_preferences['theme']),
                key="pref_theme"
            )
            st.session_state.user_preferences['theme'] = theme
            
            animation_speed = st.selectbox(
                "Animation Speed",
                ["slow", "normal", "fast"],
                index=["slow", "normal", "fast"].index(st.session_state.user_preferences['animation_speed']),
                key="pref_anim_speed"
            )
            st.session_state.user_preferences['animation_speed'] = animation_speed
            
        with col2:
            layout = st.selectbox(
                "Layout",
                ["standard", "compact", "wide"],
                index=["standard", "compact", "wide"].index(st.session_state.user_preferences['preferred_layout']),
                key="pref_layout"
            )
            st.session_state.user_preferences['preferred_layout'] = layout
            
            notification_level = st.selectbox(
                "Notifications",
                ["minimal", "normal", "verbose"],
                index=["minimal", "normal", "verbose"].index(st.session_state.user_preferences['notification_level']),
                key="pref_notifications"
            )
            st.session_state.user_preferences['notification_level'] = notification_level
        
        st.markdown("### 🤖 Agent Behavior")
        
        col1, col2 = st.columns(2)
        with col1:
            default_agent = st.selectbox(
                "Default Agent",
                ["physics_expert", "hypothesis_generator", "supervisor"],
                index=["physics_expert", "hypothesis_generator", "supervisor"].index(st.session_state.user_preferences['default_agent']),
                key="pref_default_agent"
            )
            st.session_state.user_preferences['default_agent'] = default_agent
            
            auto_scroll = st.checkbox(
                "Auto-scroll chat",
                value=st.session_state.user_preferences['auto_scroll'],
                key="pref_auto_scroll"
            )
            st.session_state.user_preferences['auto_scroll'] = auto_scroll
            
        with col2:
            collaboration_auto = st.checkbox(
                "Auto-start collaboration",
                value=st.session_state.user_preferences['collaboration_auto_start'],
                key="pref_collab_auto"
            )
            st.session_state.user_preferences['collaboration_auto_start'] = collaboration_auto
            
            compact_mode = st.checkbox(
                "Compact mode",
                value=st.session_state.user_preferences['compact_mode'],
                key="pref_compact"
            )
            st.session_state.user_preferences['compact_mode'] = compact_mode
        
        st.markdown("### 📊 Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            analytics_visible = st.checkbox(
                "Show analytics",
                value=st.session_state.user_preferences['analytics_visible'],
                key="pref_analytics"
            )
            st.session_state.user_preferences['analytics_visible'] = analytics_visible
            
        with col2:
            activity_count = st.slider(
                "Recent activity items",
                1, 10,
                value=st.session_state.user_preferences['recent_activity_count'],
                key="pref_activity_count"
            )
            st.session_state.user_preferences['recent_activity_count'] = activity_count
        
        # Save preferences button
        if st.button("💾 Save Preferences", use_container_width=True):
            # Here you could save to database or local storage
            st.success("✅ Preferences saved!")
            st.rerun()

def render_quick_actions_menu():
    """Render quick actions floating menu."""
    if st.session_state.user_preferences.get('quick_actions_visible', True):
        with st.expander("⚡ Quick Actions", expanded=False):
            st.markdown("### 🚀 Common Tasks")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧪 Generate Hypothesis", use_container_width=True, key="qa_hypothesis"):
                    # Quick hypothesis generation
                    st.session_state.selected_agent = 'hypothesis_generator'
                    st.session_state.agent_status['hypothesis_generator'] = 'active'
                    
                    # Add quick prompt
                    quick_prompt = "Generate an innovative hypothesis about quantum mechanics applications"
                    st.session_state.chat_history['hypothesis_generator'].append({
                        'role': 'user',
                        'content': quick_prompt,
                        'timestamp': datetime.now()
                    })
                    
                    # Log quick action
                    st.session_state.quick_actions_history.append({
                        'action': 'generate_hypothesis',
                        'timestamp': datetime.now(),
                        'agent': 'hypothesis_generator'
                    })
                    
                    st.rerun()
                
                if st.button("🔬 Physics Analysis", use_container_width=True, key="qa_analysis"):
                    # Quick physics analysis
                    st.session_state.selected_agent = 'physics_expert'
                    st.session_state.agent_status['physics_expert'] = 'active'
                    
                    # Add quick prompt
                    quick_prompt = "Analyze the physics behind electromagnetic wave propagation"
                    st.session_state.chat_history['physics_expert'].append({
                        'role': 'user',
                        'content': quick_prompt,
                        'timestamp': datetime.now()
                    })
                    
                    # Log quick action
                    st.session_state.quick_actions_history.append({
                        'action': 'physics_analysis',
                        'timestamp': datetime.now(),
                        'agent': 'physics_expert'
                    })
                    
                    st.rerun()
                
                if st.button("📊 Research Summary", use_container_width=True, key="qa_summary"):
                    # Quick research summary
                    st.session_state.selected_agent = 'supervisor'
                    st.session_state.agent_status['supervisor'] = 'active'
                    
                    # Add quick prompt
                    quick_prompt = "Provide a comprehensive summary of our recent research progress"
                    st.session_state.chat_history['supervisor'].append({
                        'role': 'user',
                        'content': quick_prompt,
                        'timestamp': datetime.now()
                    })
                    
                    # Log quick action
                    st.session_state.quick_actions_history.append({
                        'action': 'research_summary',
                        'timestamp': datetime.now(),
                        'agent': 'supervisor'
                    })
                    
                    st.rerun()
            
            with col2:
                if st.button("🤝 Start Collaboration", use_container_width=True, key="qa_collab"):
                    # Start collaboration mode
                    st.session_state.collaboration_mode = True
                    st.session_state.selected_agent = 'supervisor'
                    st.session_state.agent_status['supervisor'] = 'active'
                    
                    # Add collaboration prompt
                    quick_prompt = "Let's start a collaborative research session on quantum entanglement"
                    st.session_state.chat_history['supervisor'].append({
                        'role': 'user',
                        'content': quick_prompt,
                        'timestamp': datetime.now()
                    })
                    
                    # Log quick action
                    st.session_state.quick_actions_history.append({
                        'action': 'start_collaboration',
                        'timestamp': datetime.now(),
                        'agent': 'supervisor'
                    })
                    
                    st.rerun()
                
                if st.button("🔍 Knowledge Search", use_container_width=True, key="qa_search"):
                    # Focus on knowledge search
                    st.info("💡 Use the knowledge search in the sidebar to find relevant information!")
                    
                    # Log quick action
                    st.session_state.quick_actions_history.append({
                        'action': 'knowledge_search',
                        'timestamp': datetime.now(),
                        'agent': 'knowledge_system'
                    })
                
                if st.button("📈 View Analytics", use_container_width=True, key="qa_analytics"):
                    # Show analytics
                    if st.session_state.knowledge_api:
                        try:
                            analytics = asyncio.run(st.session_state.knowledge_api.get_system_analytics())
                            
                            st.markdown("### 📊 Quick Analytics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Knowledge", analytics.get('total_knowledge', 0))
                            with col2:
                                st.metric("Hypotheses", analytics.get('total_hypotheses', 0))
                            with col3:
                                st.metric("Events", analytics.get('total_events', 0))
                                
                        except Exception as e:
                            st.error(f"Failed to load analytics: {e}")
                    
                    # Log quick action
                    st.session_state.quick_actions_history.append({
                        'action': 'view_analytics',
                        'timestamp': datetime.now(),
                        'agent': 'analytics_system'
                    })
            
            # Quick action history
            if st.session_state.quick_actions_history:
                st.markdown("### 📝 Recent Actions")
                recent_actions = st.session_state.quick_actions_history[-3:]  # Show last 3
                for action in reversed(recent_actions):
                    timestamp = action['timestamp'].strftime("%H:%M")
                    st.markdown(f"• {timestamp} - {action['action'].replace('_', ' ').title()}")

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
    
    # Apply user preferences
    if st.session_state.user_preferences['preferred_layout'] == 'wide':
        layout_ratio = [3, 1]
    elif st.session_state.user_preferences['preferred_layout'] == 'compact':
        layout_ratio = [1.5, 1]
    else:
        layout_ratio = [2, 1]
    
    # Main layout
    col1, col2 = st.columns(layout_ratio)
    
    with col1:
        # Agent canvas
        render_agent_canvas()
        
        # Chat interface
        render_chat_interface()
        
        # Quick actions (if not compact mode)
        if not st.session_state.user_preferences.get('compact_mode', False):
            render_quick_actions_menu()
    
    with col2:
        # Knowledge sidebar
        render_knowledge_sidebar()
        
        # Preferences panel
        render_preferences_panel()

if __name__ == "__main__":
    main() 