"""
Single Agent PhysicsGPT Page
"""

import os
import sys

# Add src to path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Import and run the single agent app
import streamlit as st

st.set_page_config(
    page_title="Single Agent PhysicsGPT",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the main function from streamlit_app
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("streamlit_app", os.path.join(current_dir, "streamlit_app.py"))
    streamlit_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(streamlit_app)
    
    # Run the main function
    if hasattr(streamlit_app, 'main'):
        streamlit_app.main()
    else:
        st.error("Could not load single agent app")
        
except Exception as e:
    st.error(f"Error loading single agent app: {e}")
    st.markdown("Please check that all dependencies are properly installed.") 