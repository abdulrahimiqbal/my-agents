#!/usr/bin/env python3

"""
Raw Physics Analysis System
Basic interface for PhysicsGPT with telemetry.
"""

# Fix for SQLite issues on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import threading
import time
from datetime import datetime
from physics_crew_system import PhysicsGPTCrew

# Basic page configuration
st.set_page_config(
    page_title="Physics Analysis System",
    layout="wide"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'telemetry_logs' not in st.session_state:
    st.session_state.telemetry_logs = []
if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False
if 'analysis_error' not in st.session_state:
    st.session_state.analysis_error = None

def log_telemetry(message):
    """Add telemetry message to logs"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.telemetry_logs.append(f"[{timestamp}] {message}")
    # Keep only last 50 entries
    if len(st.session_state.telemetry_logs) > 50:
        st.session_state.telemetry_logs = st.session_state.telemetry_logs[-50:]

def run_analysis(query):
    """Run physics analysis in background thread"""
    try:
        log_telemetry("Starting physics analysis")
        
        # Initialize crew
        crew = PhysicsGPTCrew()
        log_telemetry("Physics crew initialized")
        
        # Run analysis
        log_telemetry(f"Analyzing query: {query}")
        result = crew.analyze_physics_question(query)
        
        # Store results
        st.session_state.analysis_results = result
        st.session_state.analysis_error = None
        log_telemetry("Analysis completed successfully")
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        st.session_state.analysis_error = error_msg
        st.session_state.analysis_results = None
        log_telemetry(error_msg)
    
    finally:
        st.session_state.is_analyzing = False

# Main interface
st.write("PHYSICS ANALYSIS SYSTEM")
st.write("=" * 50)

# Input section
st.write("Enter your physics question:")
query = st.text_input("Query", placeholder="e.g., how to detect dark matter in a room?")

if st.button("Analyze"):
    if query.strip():
        st.session_state.is_analyzing = True
        st.session_state.analysis_results = None
        st.session_state.analysis_error = None
        st.session_state.telemetry_logs = []
        
        # Start analysis in background thread
        thread = threading.Thread(target=run_analysis, args=(query,))
        thread.daemon = True
        thread.start()
        
        log_telemetry("Analysis started")
    else:
        st.write("Please enter a physics question")

# Status section
st.write("")
st.write("STATUS:")
if st.session_state.is_analyzing:
    st.write("ANALYZING...")
elif st.session_state.analysis_results:
    st.write("ANALYSIS COMPLETE")
elif st.session_state.analysis_error:
    st.write("ANALYSIS FAILED")
else:
    st.write("READY")

# Results section
st.write("")
st.write("ANALYSIS RESULTS:")
st.write("-" * 30)

if st.session_state.analysis_results:
    st.text_area("Results", value=st.session_state.analysis_results, height=300)
elif st.session_state.analysis_error:
    st.text_area("Error", value=st.session_state.analysis_error, height=100)
elif st.session_state.is_analyzing:
    st.write("Analysis in progress...")
else:
    st.write("No results yet")

# Telemetry section
st.write("")
st.write("TELEMETRY DATA:")
st.write("-" * 30)

if st.session_state.telemetry_logs:
    telemetry_text = "\n".join(st.session_state.telemetry_logs)
    st.text_area("Telemetry", value=telemetry_text, height=200)
else:
    st.write("No telemetry data")

# Auto-refresh during analysis
if st.session_state.is_analyzing:
    time.sleep(1)
    st.rerun()