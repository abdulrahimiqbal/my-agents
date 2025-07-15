#!/usr/bin/env python3
"""
PhysicsGPT - Simple Streamlit Interface
Fallback version that works even if CrewAI has issues.
"""

# Fix SQLite version issue for Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="PhysicsGPT - AI Physics Research",
    page_icon="⚛️",
    layout="wide",
)

def main():
    # Header
    st.markdown("""
    # ⚛️ PhysicsGPT
    ### Advanced 10-Agent Physics Research System
    
    **Status:** Testing deployment on Streamlit Cloud
    """)
    
    # Test system status
    st.subheader("🔧 System Status")
    
    # Test imports
    status_container = st.container()
    
    with status_container:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Core Dependencies:**")
            
            # Test basic imports
            try:
                import openai
                st.success("✅ OpenAI")
            except ImportError:
                st.error("❌ OpenAI")
            
            try:
                import streamlit
                st.success("✅ Streamlit")
            except ImportError:
                st.error("❌ Streamlit")
                
            try:
                import pandas
                st.success("✅ Pandas")
            except ImportError:
                st.error("❌ Pandas")
        
        with col2:
            st.write("**AI Framework:**")
            
            try:
                import crewai
                st.success("✅ CrewAI")
                crew_status = True
            except ImportError as e:
                st.error(f"❌ CrewAI: {e}")
                crew_status = False
            
            try:
                import langchain
                st.success("✅ LangChain")
            except ImportError:
                st.error("❌ LangChain")
            
            try:
                from physics_crew_system import PhysicsGPTCrew
                st.success("✅ PhysicsGPT System")
                system_ready = True
            except ImportError as e:
                st.error(f"❌ PhysicsGPT System: {e}")
                system_ready = False
    
    # Environment check
    st.subheader("🔑 Environment Configuration")
    
    env_status = st.container()
    with env_status:
        if "OPENAI_API_KEY" in st.secrets:
            st.success("✅ OpenAI API Key configured")
        else:
            st.warning("⚠️ OpenAI API Key not found in secrets")
            st.info("Add your OPENAI_API_KEY in Streamlit Cloud secrets")
    
    # System ready check
    if crew_status and system_ready:
        st.success("🎉 **System Ready!** All components loaded successfully.")
        
        # Simple query interface
        st.subheader("🔬 Physics Query Interface")
        
        query = st.text_area(
            "Enter your physics question:",
            placeholder="e.g., What is quantum entanglement?",
            height=100
        )
        
        if st.button("🚀 Analyze with PhysicsGPT", type="primary"):
            if query:
                with st.spinner("🤖 Analyzing with 5 core physics agents..."):
                    try:
                        from physics_crew_system import PhysicsGPTCrew
                        physics_crew = PhysicsGPTCrew()
                        result = physics_crew.analyze_physics_query(query)
                        
                        if result['success']:
                            st.success("✅ Analysis Complete!")
                            st.markdown("### 📄 Results:")
                            st.markdown(result['result'])
                        else:
                            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ System error: {str(e)}")
            else:
                st.warning("Please enter a physics question")
    else:
        st.error("❌ **System Not Ready** - Please check the errors above")
        st.info("The system is still deploying. Try refreshing in a few minutes.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>⚛️ PhysicsGPT - Advanced AI Physics Research System</p>
        <p>Powered by CrewAI • 10 Specialized Physics Agents</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()