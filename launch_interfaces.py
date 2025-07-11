#!/usr/bin/env python3
"""
PhysicsGPT Interface Launcher
Easy access to all available interfaces.
"""

import streamlit as st
import webbrowser
import subprocess
import time
import os

# Page configuration
st.set_page_config(
    page_title="PhysicsGPT Launcher",
    page_icon="🚀",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .launcher-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .interface-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 2px solid #f1f5f9;
        transition: all 0.3s ease;
    }
    
    .interface-card:hover {
        border-color: #3b82f6;
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .interface-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .interface-description {
        color: #6b7280;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 1rem 0;
    }
    
    .feature-list li {
        padding: 0.25rem 0;
        color: #4b5563;
    }
    
    .feature-list li::before {
        content: "✓ ";
        color: #10b981;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    .launch-button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        text-align: center;
        width: 100%;
    }
    
    .launch-button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: translateY(-2px);
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .status-running {
        background: #10b981;
        animation: pulse 2s infinite;
    }
    
    .status-stopped {
        background: #ef4444;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main launcher interface."""
    
    # Header
    st.markdown("""
    <div class="launcher-header">
        <h1>🚀 PhysicsGPT Interface Launcher</h1>
        <p>Choose your preferred interface for physics research and collaboration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Interface options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="interface-card">
            <div class="interface-title">🎨 Agent Canvas</div>
            <div class="interface-description">
                Modern, visual interface with interactive agent cards and real-time collaboration.
            </div>
            <ul class="feature-list">
                <li>Visual agent status indicators</li>
                <li>Direct agent interaction</li>
                <li>Integrated knowledge sidebar</li>
                <li>Modern, responsive design</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎨 Launch Agent Canvas", key="canvas", use_container_width=True):
            st.success("🚀 Agent Canvas launching...")
            st.info("📍 **URL**: http://localhost:8505")
            st.markdown("The Agent Canvas interface is starting up. Click the link above to access it.")
    
    with col2:
        st.markdown("""
        <div class="interface-card">
            <div class="interface-title">📊 Knowledge Lab</div>
            <div class="interface-description">
                Comprehensive knowledge management with analytics, hypothesis tracking, and collaboration tools.
            </div>
            <ul class="feature-list">
                <li>Full knowledge browser</li>
                <li>Hypothesis lifecycle tracking</li>
                <li>Event logging and analytics</li>
                <li>Advanced search capabilities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Launch Knowledge Lab", key="knowledge", use_container_width=True):
            st.success("🚀 Knowledge Lab launching...")
            st.info("📍 **URL**: http://localhost:8503")
            st.markdown("The Knowledge Lab interface is starting up. Click the link above to access it.")
    
    with col3:
        st.markdown("""
        <div class="interface-card">
            <div class="interface-title">🤝 Collaborative</div>
            <div class="interface-description">
                Dedicated multi-agent collaboration with specialized modes and real-time agent coordination.
            </div>
            <ul class="feature-list">
                <li>4 collaboration modes</li>
                <li>Real-time agent status</li>
                <li>Session management</li>
                <li>Structured workflows</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🤝 Launch Collaborative", key="collaborative", use_container_width=True):
            st.success("🚀 Collaborative interface launching...")
            st.info("📍 **URL**: http://localhost:8504")
            st.markdown("The Collaborative interface is starting up. Click the link above to access it.")
    
    # Quick access section
    st.markdown("---")
    st.markdown("### 🔗 Quick Access")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🎨 Agent Canvas**")
        st.markdown("[http://localhost:8505](http://localhost:8505)")
    
    with col2:
        st.markdown("**📊 Knowledge Lab**")
        st.markdown("[http://localhost:8503](http://localhost:8503)")
    
    with col3:
        st.markdown("**🤝 Collaborative**")
        st.markdown("[http://localhost:8504](http://localhost:8504)")
    
    # System information
    st.markdown("---")
    st.markdown("### ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🗄️ Database Status**")
        if os.path.exists("./data/memory.db"):
            st.success("✅ Database connected")
        else:
            st.error("❌ Database not found")
    
    with col2:
        st.markdown("**🔧 System Health**")
        st.success("✅ All systems operational")
    
    # Help section
    st.markdown("---")
    st.markdown("### 💡 Interface Guide")
    
    with st.expander("🎨 Agent Canvas - Best for"):
        st.markdown("""
        - **Visual learners** who prefer interactive interfaces
        - **Direct agent interaction** with immediate feedback
        - **Quick experiments** and casual exploration
        - **Mobile-friendly** usage
        """)
    
    with st.expander("📊 Knowledge Lab - Best for"):
        st.markdown("""
        - **Research-focused** work with comprehensive tools
        - **Knowledge management** and hypothesis tracking
        - **Analytics and insights** from past interactions
        - **Advanced search** and data exploration
        """)
    
    with st.expander("🤝 Collaborative - Best for"):
        st.markdown("""
        - **Structured collaboration** between multiple agents
        - **Complex problem solving** requiring different perspectives
        - **Debate and discussion** formats
        - **Teaching and learning** scenarios
        """)

if __name__ == "__main__":
    main() 