#!/usr/bin/env python3
"""
Simple Knowledge Management Dashboard
Minimal UI for viewing hypothesis progression and research analytics.
"""

import streamlit as st
import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Fix SQLite version issue for Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from crewai_database_integration import CrewAIKnowledgeAPI, CrewAIEvaluationFramework

def load_data():
    """Load data from the database."""
    try:
        knowledge_api = CrewAIKnowledgeAPI()
        evaluation_framework = CrewAIEvaluationFramework()
        
        # Get data
        events = knowledge_api.get_recent_events(limit=1000)
        
        # Get hypotheses with error handling
        hypotheses = []
        try:
            with sqlite3.connect(knowledge_api.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, description, confidence_score, 
                           validation_status, created_by, created_at
                    FROM hypotheses ORDER BY created_at DESC
                """)
                hypotheses = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            st.error(f"Error loading hypotheses: {e}")
        
        # Get knowledge entries
        knowledge_entries = []
        try:
            with sqlite3.connect(knowledge_api.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, content, physics_domain, 
                           confidence_level, created_at
                    FROM knowledge_entries ORDER BY created_at DESC
                """)
                knowledge_entries = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            st.error(f"Error loading knowledge entries: {e}")
        
        # Get system analytics
        analytics = knowledge_api.get_system_analytics()
        
        return events, hypotheses, knowledge_entries, analytics
        
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return [], [], [], {}

def display_hypothesis_progression(hypotheses):
    """Display hypothesis progression through research phases."""
    st.header("🔬 Hypothesis Research Progression")
    
    if not hypotheses:
        st.info("No hypotheses found. Run some physics analyses to generate hypotheses!")
        return
    
    # Status distribution
    status_counts = {}
    for hyp in hypotheses:
        status = hyp.get('validation_status', 'pending')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Create progress visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if status_counts:
            # Status distribution pie chart
            fig = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title="Hypothesis Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Status summary cards
        st.markdown("#### Research Phase Summary")
        
        phases = {
            'pending': {'emoji': '📋', 'label': 'Proposed'},
            'under_review': {'emoji': '🔍', 'label': 'Under Review'},
            'validated': {'emoji': '✅', 'label': 'Validated'},
            'refuted': {'emoji': '❌', 'label': 'Refuted'},
            'promoted': {'emoji': '🏆', 'label': 'Promoted'}
        }
        
        for status, info in phases.items():
            count = status_counts.get(status, 0)
            st.metric(
                label=f"{info['emoji']} {info['label']}",
                value=count
            )
    
    # Recent hypotheses table
    st.markdown("#### Recent Hypotheses")
    
    df_hypotheses = pd.DataFrame(hypotheses)
    if not df_hypotheses.empty:
        # Display recent hypotheses
        display_df = df_hypotheses[['title', 'validation_status', 'confidence_score', 'created_by', 'created_at']].copy()
        display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        display_df = display_df.rename(columns={
            'title': 'Hypothesis',
            'validation_status': 'Status',
            'confidence_score': 'Confidence',
            'created_by': 'Created By',
            'created_at': 'Created'
        })
        st.dataframe(display_df, use_container_width=True)

def display_knowledge_base(knowledge_entries):
    """Display validated knowledge base."""
    st.header("🧠 Validated Knowledge Base")
    
    if not knowledge_entries:
        st.info("No validated knowledge entries yet. Complete some research workflows!")
        return
    
    # Domain distribution
    domain_counts = {}
    for entry in knowledge_entries:
        domain = entry.get('physics_domain', 'Unknown')
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Knowledge by Domain")
        for domain, count in sorted(domain_counts.items()):
            st.metric(label=domain.replace('_', ' ').title(), value=count)
    
    with col2:
        if domain_counts:
            # Domain distribution chart
            fig = px.bar(
                x=list(domain_counts.keys()),
                y=list(domain_counts.values()),
                title="Knowledge Entries by Physics Domain"
            )
            fig.update_xaxes(title="Physics Domain")
            fig.update_yaxes(title="Number of Entries")
            st.plotly_chart(fig, use_container_width=True)
    
    # Knowledge entries table
    st.markdown("#### Recent Knowledge Entries")
    
    df_knowledge = pd.DataFrame(knowledge_entries)
    if not df_knowledge.empty:
        display_df = df_knowledge[['title', 'physics_domain', 'confidence_level', 'created_at']].copy()
        display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        display_df = display_df.rename(columns={
            'title': 'Knowledge Entry',
            'physics_domain': 'Domain',
            'confidence_level': 'Confidence',
            'created_at': 'Created'
        })
        st.dataframe(display_df, use_container_width=True)

def display_system_analytics(events, analytics):
    """Display system performance analytics."""
    st.header("📊 System Analytics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Events", analytics.get('total_events', 0))
    with col2:
        st.metric("Total Hypotheses", analytics.get('total_hypotheses', 0))
    with col3:
        st.metric("Knowledge Entries", analytics.get('total_knowledge_entries', 0))
    with col4:
        success_rate = analytics.get('success_rate', 0)
        st.metric("Success Rate", f"{success_rate:.1%}")
    
    # Activity timeline
    if events:
        st.markdown("#### Recent Activity Timeline")
        
        # Convert events to DataFrame
        df_events = pd.DataFrame(events)
        if not df_events.empty and 'timestamp' in df_events.columns:
            df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
            df_events['date'] = df_events['timestamp'].dt.date
            
            # Daily activity counts
            daily_activity = df_events.groupby('date').size().reset_index(name='events')
            
            fig = px.line(
                daily_activity,
                x='date',
                y='events',
                title='Daily System Activity',
                markers=True
            )
            fig.update_xaxes(title="Date")
            fig.update_yaxes(title="Number of Events")
            st.plotly_chart(fig, use_container_width=True)
            
            # Event type distribution
            event_type_counts = df_events['event_type'].value_counts()
            if len(event_type_counts) > 0:
                fig = px.bar(
                    x=event_type_counts.values,
                    y=event_type_counts.index,
                    orientation='h',
                    title='Event Types Distribution'
                )
                fig.update_xaxes(title="Count")
                fig.update_yaxes(title="Event Type")
                st.plotly_chart(fig, use_container_width=True)

def display_research_workflow():
    """Display the research workflow explanation."""
    st.header("🔄 Research Workflow")
    
    st.markdown("""
    ### Scientific Research Phases in PhysicsGPT
    
    **1. 📋 PROPOSED** - Initial hypothesis generation
    - Hypothesis Generator creates new scientific ideas
    - Initial confidence score assigned
    - Available for peer review
    
    **2. 🔍 UNDER_REVIEW** - Active scientific evaluation  
    - Multiple agents analyze the hypothesis
    - Mathematical modeling applied
    - Experimental designs created
    - Evidence gathering begins
    
    **3. ✅ VALIDATED** - Scientific validation achieved
    - Hypothesis passes rigorous testing
    - Mathematical models confirmed
    - Experimental evidence supports theory
    - Peer consensus achieved
    
    **4. ❌ REFUTED** - Hypothesis disproven
    - Evidence contradicts hypothesis
    - Mathematical inconsistencies found
    - Stored for learning purposes
    
    **5. 🏆 PROMOTED** - Accepted scientific knowledge
    - Validated hypothesis promoted to knowledge base
    - Becomes part of system's permanent knowledge
    - Used to inform future research
    """)

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="PhysicsGPT Knowledge Dashboard",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 PhysicsGPT Knowledge Management Dashboard")
    st.markdown("Track hypothesis progression through scientific research phases")
    
    # Load data
    with st.spinner("Loading knowledge management data..."):
        events, hypotheses, knowledge_entries, analytics = load_data()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("📋 Navigation")
        
        page = st.radio(
            "Select View:",
            [
                "🔬 Hypothesis Progression",
                "🧠 Knowledge Base", 
                "📊 System Analytics",
                "🔄 Research Workflow"
            ]
        )
        
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        st.metric("Active Hypotheses", len(hypotheses))
        st.metric("Knowledge Entries", len(knowledge_entries))
        st.metric("Recent Events", len(events))
    
    # Display selected page
    if page == "🔬 Hypothesis Progression":
        display_hypothesis_progression(hypotheses)
    elif page == "🧠 Knowledge Base":
        display_knowledge_base(knowledge_entries)
    elif page == "📊 System Analytics":
        display_system_analytics(events, analytics)
    elif page == "🔄 Research Workflow":
        display_research_workflow()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🧠 PhysicsGPT Knowledge Management • Real-time Research Tracking</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 