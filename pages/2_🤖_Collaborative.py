"""
Collaborative PhysicsGPT Page
"""

import os
import sys

# Add src to path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Import and run the collaborative app
import streamlit as st

st.set_page_config(
    page_title="Collaborative PhysicsGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the main function from streamlit_collaborative
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("streamlit_collaborative", os.path.join(current_dir, "streamlit_collaborative.py"))
    streamlit_collaborative = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(streamlit_collaborative)
    
    # Run the main function
    if hasattr(streamlit_collaborative, 'main'):
        streamlit_collaborative.main()
    else:
        st.error("Could not load collaborative app")
        
except Exception as e:
    st.error(f"Error loading collaborative app: {e}")
    st.markdown("Please check that all dependencies are properly installed.") 