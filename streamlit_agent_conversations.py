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
import time
import io
import contextlib
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
if 'crew_output' not in st.session_state:
    st.session_state.crew_output = []
if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False
if 'analysis_error' not in st.session_state:
    st.session_state.analysis_error = None
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

def log_telemetry(message):
    """Add telemetry message to logs"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.telemetry_logs.append(f"[{timestamp}] {message}")
    # Keep only last 100 entries
    if len(st.session_state.telemetry_logs) > 100:
        st.session_state.telemetry_logs = st.session_state.telemetry_logs[-100:]

def log_crew_output(message):
    """Add crew output message to logs"""
    if message.strip():  # Only log non-empty messages
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.crew_output.append(f"[{timestamp}] {message.strip()}")
        # Keep only last 200 entries
        if len(st.session_state.crew_output) > 200:
            st.session_state.crew_output = st.session_state.crew_output[-200:]

@contextlib.contextmanager
def capture_crew_output():
    """Capture stdout from CrewAI for telemetry"""
    old_stdout = sys.stdout
    stdout_capture = io.StringIO()
    
    class TeeOutput:
        def write(self, text):
            old_stdout.write(text)  # Still show in console
            stdout_capture.write(text)
            # Log to our telemetry in real-time
            if text.strip():
                log_crew_output(text)
        
        def flush(self):
            old_stdout.flush()
            stdout_capture.flush()
    
    try:
        sys.stdout = TeeOutput()
        yield stdout_capture
    finally:
        sys.stdout = old_stdout

def run_analysis(query):
    """Run physics analysis synchronously with enhanced telemetry capture"""
    try:
        log_telemetry("Starting physics analysis")
        
        # Initialize crew
        log_telemetry("Initializing physics crew")
        crew = PhysicsGPTCrew()
        log_telemetry("Physics crew initialized")
        
        # Run analysis with output capture
        log_telemetry(f"Analyzing query: {query}")
        log_telemetry("Capturing CrewAI agent conversations...")
        
        with capture_crew_output() as output:
            result = crew.analyze_physics_query(query)
        
        # Store results
        st.session_state.analysis_results = result
        st.session_state.analysis_error = None
        log_telemetry("Analysis completed successfully")
        log_telemetry(f"Captured {len(st.session_state.crew_output)} crew output messages")
        
        return True
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        st.session_state.analysis_error = error_msg
        st.session_state.analysis_results = None
        log_telemetry(error_msg)
        return False

# Main interface
st.write("PHYSICS ANALYSIS SYSTEM")
st.write("=" * 50)

# Input section
st.write("Enter your physics question:")
query = st.text_input("Query", placeholder="e.g., how to detect dark matter in a room?")

if st.button("Analyze"):
    if query.strip():
        # Clear previous results
        st.session_state.analysis_results = None
        st.session_state.analysis_error = None
        st.session_state.telemetry_logs = []
        st.session_state.crew_output = []
        st.session_state.current_query = query.strip()
        
        log_telemetry("Analysis request received")
        
        # Show analysis in progress
        with st.spinner("Analyzing physics question..."):
            success = run_analysis(query.strip())
        
        if success:
            st.success("Analysis completed!")
        else:
            st.error("Analysis failed!")
            
    else:
        st.warning("Please enter a physics question")

# Status section
st.write("")
st.write("STATUS:")
if st.session_state.analysis_results:
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
else:
    st.write("No results yet")

# Telemetry section
st.write("")
st.write("TELEMETRY DATA:")
st.write("-" * 30)

if st.session_state.telemetry_logs:
    telemetry_text = "\n".join(st.session_state.telemetry_logs)
    st.text_area("System Telemetry", value=telemetry_text, height=150)
else:
    st.write("No telemetry data")

# Crew Output section (the actual agent conversations)
st.write("")
st.write("CREW AGENT CONVERSATIONS:")
st.write("-" * 30)

if st.session_state.crew_output:
    crew_text = "\n".join(st.session_state.crew_output)
    st.text_area("Agent Output", value=crew_text, height=400)
else:
    st.write("No crew output yet")

# Query info
if st.session_state.current_query:
    st.write("")
    st.write(f"Current query: {st.session_state.current_query}")