#!/usr/bin/env python3
"""
🚀 PhysicsGPT Interface Launcher
A comprehensive launcher for all physics agent interfaces.
"""

import streamlit as st
import subprocess
import sys
import os
from datetime import datetime
import webbrowser
import time

# Configure page
st.set_page_config(
    page_title="PhysicsGPT Launcher",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for launcher
st.markdown("""
<style>
    .launcher-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .interface-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 2px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .interface-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px rgba(0,0,0,0.15);
        border-color: #3b82f6;
    }
    
    .interface-card.recommended {
        border-color: #f59e0b;
        background: linear-gradient(135deg, #fef3c7 0%, #ffffff 100%);
    }
    
    .interface-card.experimental {
        border-color: #8b5cf6;
        background: linear-gradient(135deg, #f3e8ff 0%, #ffffff 100%);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .card-icon {
        font-size: 3rem;
        margin-right: 1rem;
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f2937;
    }
    
    .card-subtitle {
        color: #6b7280;
        font-size: 0.9rem;
    }
    
    .feature-list {
        margin: 1rem 0;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        color: #374151;
    }
    
    .feature-icon {
        margin-right: 0.5rem;
        color: #10b981;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: auto;
    }
    
    .status-live {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-recommended {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-experimental {
        background: #f3e8ff;
        color: #6b21a8;
    }
    
    .status-development {
        background: #e5e7eb;
        color: #374151;
    }
    
    .quick-stats {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stat-item {
        display: flex;
        justify-content: space-between;
        margin: 0.25rem 0;
    }
    
    .launch-button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .launch-button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
    }
    
    .sidebar-info {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main launcher interface."""
    
    # Header
    st.markdown("""
    <div class="launcher-header">
        <h1>🚀 PhysicsGPT Interface Launcher</h1>
        <p>Choose your preferred physics agent interface</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with system info
    with st.sidebar:
        st.markdown("### 🔧 System Information")
        
        st.markdown(f"""
        <div class="sidebar-info">
            <strong>🕒 Current Time:</strong><br>
            {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br><br>
            
            <strong>🐍 Python Version:</strong><br>
            {sys.version.split()[0]}<br><br>
            
            <strong>📁 Working Directory:</strong><br>
            {os.getcwd()}<br><br>
            
            <strong>🌐 Available Ports:</strong><br>
            8501 (Default Streamlit)<br>
            8502 (Knowledge Lab)<br>
            8503 (Collaborative)<br>
            8504 (Canvas)<br>
            8505 (Simple)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📚 Quick Links")
        if st.button("📖 Documentation", use_container_width=True):
            st.info("📄 Documentation would open here")
        
        if st.button("🐛 Report Issue", use_container_width=True):
            st.info("🔗 GitHub issues would open here")
        
        if st.button("💬 Community", use_container_width=True):
            st.info("💭 Community forum would open here")
    
    # Main interface selection
    col1, col2 = st.columns(2)
    
    with col1:
        # Canvas Interface (Recommended)
        st.markdown("""
        <div class="interface-card recommended">
            <div class="card-header">
                <div class="card-icon">🎨</div>
                <div>
                    <div class="card-title">Agent Canvas</div>
                    <div class="card-subtitle">Interactive multi-agent workspace</div>
                </div>
                <div class="status-badge status-recommended">Recommended</div>
            </div>
            
            <div class="feature-list">
                <div class="feature-item">
                    <span class="feature-icon">✨</span>
                    Visual agent interaction canvas
                </div>
                <div class="feature-item">
                    <span class="feature-icon">🤝</span>
                    Real-time collaboration mode
                </div>
                <div class="feature-item">
                    <span class="feature-icon">📊</span>
                    Integrated knowledge management
                </div>
                <div class="feature-item">
                    <span class="feature-icon">⚡</span>
                    Quick actions & preferences
                </div>
                <div class="feature-item">
                    <span class="feature-icon">📱</span>
                    Responsive design
                </div>
            </div>
            
            <div class="quick-stats">
                <div class="stat-item">
                    <span>🎯 Best for:</span>
                    <span>Research & Exploration</span>
                </div>
                <div class="stat-item">
                    <span>🔧 Complexity:</span>
                    <span>Medium</span>
                </div>
                <div class="stat-item">
                    <span>⚡ Performance:</span>
                    <span>Optimized</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Canvas Interface", key="launch_canvas", use_container_width=True):
            st.info("🎨 Launching Agent Canvas on port 8504...")
            try:
                # Launch streamlit_canvas.py
                subprocess.Popen([
                    sys.executable, "-m", "streamlit", "run", 
                    "streamlit_canvas.py", 
                    "--server.port", "8504",
                    "--server.headless", "true"
                ])
                time.sleep(2)
                st.success("✅ Canvas Interface launched! Opening in browser...")
                webbrowser.open("http://localhost:8504")
            except Exception as e:
                st.error(f"❌ Failed to launch: {e}")
        
        # Knowledge Lab Interface
        st.markdown("""
        <div class="interface-card">
            <div class="card-header">
                <div class="card-icon">🧪</div>
                <div>
                    <div class="card-title">Knowledge Lab</div>
                    <div class="card-subtitle">Advanced knowledge management</div>
                </div>
                <div class="status-badge status-live">Live</div>
            </div>
            
            <div class="feature-list">
                <div class="feature-item">
                    <span class="feature-icon">📚</span>
                    Comprehensive knowledge browser
                </div>
                <div class="feature-item">
                    <span class="feature-icon">🔍</span>
                    Advanced search & filtering
                </div>
                <div class="feature-item">
                    <span class="feature-icon">📈</span>
                    Analytics & insights
                </div>
                <div class="feature-item">
                    <span class="feature-icon">🧬</span>
                    Hypothesis tracking
                </div>
            </div>
            
            <div class="quick-stats">
                <div class="stat-item">
                    <span>🎯 Best for:</span>
                    <span>Knowledge Management</span>
                </div>
                <div class="stat-item">
                    <span>🔧 Complexity:</span>
                    <span>High</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🧪 Launch Knowledge Lab", key="launch_knowledge", use_container_width=True):
            st.info("🧪 Launching Knowledge Lab on port 8502...")
            try:
                subprocess.Popen([
                    sys.executable, "-m", "streamlit", "run", 
                    "streamlit_knowledge.py", 
                    "--server.port", "8502",
                    "--server.headless", "true"
                ])
                time.sleep(2)
                st.success("✅ Knowledge Lab launched! Opening in browser...")
                webbrowser.open("http://localhost:8502")
            except Exception as e:
                st.error(f"❌ Failed to launch: {e}")
    
    with col2:
        # Collaborative Interface
        st.markdown("""
        <div class="interface-card">
            <div class="card-header">
                <div class="card-icon">🤖</div>
                <div>
                    <div class="card-title">Collaborative Agents</div>
                    <div class="card-subtitle">Multi-agent collaboration</div>
                </div>
                <div class="status-badge status-live">Live</div>
            </div>
            
            <div class="feature-list">
                <div class="feature-item">
                    <span class="feature-icon">🤝</span>
                    Agent-to-agent communication
                </div>
                <div class="feature-item">
                    <span class="feature-icon">🎭</span>
                    Multiple collaboration modes
                </div>
                <div class="feature-item">
                    <span class="feature-icon">💬</span>
                    Real-time chat interface
                </div>
                <div class="feature-item">
                    <span class="feature-icon">📊</span>
                    Session management
                </div>
            </div>
            
            <div class="quick-stats">
                <div class="stat-item">
                    <span>🎯 Best for:</span>
                    <span>Team Research</span>
                </div>
                <div class="stat-item">
                    <span>🔧 Complexity:</span>
                    <span>Medium</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🤖 Launch Collaborative Interface", key="launch_collab", use_container_width=True):
            st.info("🤖 Launching Collaborative Interface on port 8503...")
            try:
                subprocess.Popen([
                    sys.executable, "-m", "streamlit", "run", 
                    "streamlit_collaborative.py", 
                    "--server.port", "8503",
                    "--server.headless", "true"
                ])
                time.sleep(2)
                st.success("✅ Collaborative Interface launched! Opening in browser...")
                webbrowser.open("http://localhost:8503")
            except Exception as e:
                st.error(f"❌ Failed to launch: {e}")
        
        # Simple Interface
        st.markdown("""
        <div class="interface-card">
            <div class="card-header">
                <div class="card-icon">⚡</div>
                <div>
                    <div class="card-title">Simple Chat</div>
                    <div class="card-subtitle">Lightweight single-agent chat</div>
                </div>
                <div class="status-badge status-live">Live</div>
            </div>
            
            <div class="feature-list">
                <div class="feature-item">
                    <span class="feature-icon">💬</span>
                    Clean chat interface
                </div>
                <div class="feature-item">
                    <span class="feature-icon">🔬</span>
                    Single physics expert agent
                </div>
                <div class="feature-item">
                    <span class="feature-icon">⚡</span>
                    Fast & lightweight
                </div>
                <div class="feature-item">
                    <span class="feature-icon">📱</span>
                    Mobile-friendly
                </div>
            </div>
            
            <div class="quick-stats">
                <div class="stat-item">
                    <span>🎯 Best for:</span>
                    <span>Quick Questions</span>
                </div>
                <div class="stat-item">
                    <span>🔧 Complexity:</span>
                    <span>Low</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("⚡ Launch Simple Chat", key="launch_simple", use_container_width=True):
            st.info("⚡ Launching Simple Chat on port 8505...")
            try:
                subprocess.Popen([
                    sys.executable, "-m", "streamlit", "run", 
                    "streamlit_simple.py", 
                    "--server.port", "8505",
                    "--server.headless", "true"
                ])
                time.sleep(2)
                st.success("✅ Simple Chat launched! Opening in browser...")
                webbrowser.open("http://localhost:8505")
            except Exception as e:
                st.error(f"❌ Failed to launch: {e}")
    
    # Footer with tips
    st.markdown("---")
    st.markdown("""
    ### 💡 Pro Tips
    
    - **🎨 Agent Canvas**: Best for interactive research and exploration with visual feedback
    - **🧪 Knowledge Lab**: Perfect for managing and analyzing your research knowledge base
    - **🤖 Collaborative**: Ideal when you need multiple agents working together on complex problems
    - **⚡ Simple Chat**: Great for quick physics questions and lightweight interactions
    
    ### 🔧 Technical Notes
    
    - All interfaces run on different ports to avoid conflicts
    - Interfaces will open automatically in your default browser
    - Use `Ctrl+C` in terminal to stop any running interface
    - Environment variables should be set for optimal performance
    """)

if __name__ == "__main__":
    main() 