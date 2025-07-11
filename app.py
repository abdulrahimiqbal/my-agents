"""
🚀 PhysicsGPT - Main Entry Point
Choose between single-agent physics expert or collaborative multi-agent research.
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
    page_title="PhysicsGPT",
    page_icon="⚛️",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .option-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e5e7eb;
        margin: 1rem;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .option-card:hover {
        border-color: #3b82f6;
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(59, 130, 246, 0.15);
    }
    
    .feature-list {
        text-align: left;
        margin: 1.5rem 0;
    }
    
    .feature-list li {
        margin: 0.5rem 0;
        color: #4b5563;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚛️ PhysicsGPT</h1>
        <p style="font-size: 1.3em; margin-top: 1rem;">
            AI-Powered Physics Research & Problem Solving Platform
        </p>
        <p style="opacity: 0.9; margin-top: 0.5rem;">
            Choose your research experience below
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Main options
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="option-card">
            <div style="text-align: center;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🔬</div>
                <h2>Single Expert Mode</h2>
                <p style="color: #6b7280; margin-bottom: 1.5rem;">
                    Work with one specialized physics expert agent
                </p>
            </div>
            
            <div class="feature-list">
                <strong>Perfect for:</strong>
                <ul>
                    <li>🎓 Learning physics concepts</li>
                    <li>📝 Solving homework problems</li>
                    <li>🧮 Step-by-step calculations</li>
                    <li>📚 Quick explanations</li>
                    <li>⚡ Fast, focused responses</li>
                </ul>
            </div>
            
            <div style="text-align: center; margin-top: 2rem;">
                <p style="color: #059669; font-weight: 600;">
                    ✅ Recommended for beginners
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Single Expert", type="secondary", use_container_width=True):
            st.switch_page("streamlit_app.py")

    with col2:
        st.markdown("""
        <div class="option-card">
            <div style="text-align: center;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
                <h2>Collaborative Mode</h2>
                <p style="color: #6b7280; margin-bottom: 1.5rem;">
                    Multiple AI agents working together
                </p>
            </div>
            
            <div class="feature-list">
                <strong>Perfect for:</strong>
                <ul>
                    <li>🔬 Research-grade analysis</li>
                    <li>💡 Creative hypothesis generation</li>
                    <li>🎭 Multi-perspective insights</li>
                    <li>🤝 Collaborative problem solving</li>
                    <li>🚀 Advanced physics exploration</li>
                </ul>
            </div>
            
            <div style="text-align: center; margin-top: 2rem;">
                <p style="color: #dc2626; font-weight: 600;">
                    ⭐ NEW! Advanced features
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🤖 Launch Collaborative", type="primary", use_container_width=True):
            st.switch_page("streamlit_collaborative.py")

    # Quick info section
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 How It Works
        1. Choose your mode above
        2. Ask physics questions
        3. Get AI-powered answers
        4. Learn and explore!
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 Physics Coverage
        - Classical & Quantum Mechanics
        - Electromagnetism & Optics  
        - Thermodynamics & Relativity
        - Particle Physics & Cosmology
        """)
    
    with col3:
        st.markdown("""
        ### 🚀 Powered By
        - **LangChain** for AI orchestration
        - **LangGraph** for multi-agent workflows
        - **OpenAI GPT** for physics expertise
        - **Streamlit** for user interface
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem;">
        <p>🧠 Built with advanced AI • 🔬 Scientifically rigorous • 🎓 Educational focused</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            Need help? Try the Single Expert mode first, then explore Collaborative mode for advanced research.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 