"""
Collaborative PhysicsGPT with Knowledge Management - Advanced Streamlit Interface
Multi-agent physics research system with empirical event logging and knowledge tracking.
"""

import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import traceback

# Simple path setup for imports
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Import our components
try:
    from src.agents import CollaborativePhysicsSystem
    from src.database import KnowledgeAPI, DatabaseMigrator
    from src.config import get_settings
    from src.memory import get_memory_store
except ImportError as e:
    st.error(f"❌ **Import Error**: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="PhysicsGPT Knowledge Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for knowledge management theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .knowledge-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
        transition: all 0.3s ease;
    }
    
    .knowledge-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    
    .hypothesis-card {
        background: linear-gradient(135deg, #fef3c7 0%, #f59e0b 20%, #fef3c7 100%);
        border-left: 4px solid #f59e0b;
    }
    
    .knowledge-promoted {
        background: linear-gradient(135deg, #d1fae5 0%, #10b981 20%, #d1fae5 100%);
        border-left: 4px solid #10b981;
    }
    
    .event-card {
        background: linear-gradient(135deg, #e0f2fe 0%, #0ea5e9 20%, #e0f2fe 100%);
        border-left: 4px solid #0ea5e9;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #3b82f6;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .agent-indicator {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .physics-expert {
        background: #10b981;
        color: white;
    }
    
    .hypothesis-generator {
        background: #f59e0b;
        color: white;
    }
    
    .supervisor {
        background: #8b5cf6;
        color: white;
    }
    
    .confidence-high {
        background: #10b981;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    
    .confidence-medium {
        background: #f59e0b;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    
    .confidence-low {
        background: #ef4444;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'knowledge_api' not in st.session_state:
        st.session_state.knowledge_api = None
    if 'collaborative_system' not in st.session_state:
        st.session_state.collaborative_system = None
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'selected_hypothesis_id' not in st.session_state:
        st.session_state.selected_hypothesis_id = None
    if 'selected_knowledge_id' not in st.session_state:
        st.session_state.selected_knowledge_id = None

def initialize_knowledge_api():
    """Initialize the Knowledge API."""
    if st.session_state.knowledge_api is None:
        try:
            # Run migration first
            with st.spinner("🔄 Initializing knowledge management system..."):
                migrator = DatabaseMigrator("./data/memory.db")
                migration_success = migrator.migrate()
                
                if migration_success:
                    st.session_state.knowledge_api = KnowledgeAPI()
                    st.success("✅ Knowledge management system initialized!")
                else:
                    st.error("❌ Failed to initialize knowledge management system")
                    return False
        except Exception as e:
            st.error(f"❌ Knowledge API initialization failed: {e}")
            return False
    return True

def create_main_header():
    """Create the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🧪 PhysicsGPT Knowledge Lab</h1>
        <p>Advanced Multi-Agent Physics Research with Knowledge Management</p>
        <p style="font-size: 1.1em; margin-top: 1rem;">
            🔬 Physics Expert • 💡 Hypothesis Generator • 🤝 Supervisor • 📊 Knowledge Tracker
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_system_metrics():
    """Display system-wide metrics."""
    if st.session_state.knowledge_api:
        try:
            # Get system analytics
            analytics = asyncio.run(st.session_state.knowledge_api.get_system_analytics())
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{analytics.get('total_knowledge', 0)}</div>
                    <div class="metric-label">Knowledge Entries</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{analytics.get('total_hypotheses', 0)}</div>
                    <div class="metric-label">Hypotheses</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{analytics.get('total_events', 0)}</div>
                    <div class="metric-label">Events Logged</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                promotion_rate = analytics.get('promotion_rate', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{promotion_rate:.1%}</div>
                    <div class="metric-label">Promotion Rate</div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Failed to load system metrics: {e}")

def display_knowledge_browser():
    """Display the knowledge browser interface."""
    st.markdown("### 📚 Knowledge Base Browser")
    
    # Search and filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_query = st.text_input("🔍 Search Knowledge", placeholder="quantum entanglement")
    
    with col2:
        domain_filter = st.selectbox("📂 Domain", ["All", "quantum", "classical", "thermodynamics", "electromagnetism", "relativity", "particle", "astrophysics", "condensed_matter"])
    
    with col3:
        confidence_filter = st.slider("🎯 Min Confidence", 0.0, 1.0, 0.5, 0.1)
    
    if st.button("🔍 Search Knowledge") or search_query:
        try:
            if search_query:
                # Search with query
                results = asyncio.run(st.session_state.knowledge_api.search_knowledge(
                    query=search_query,
                    domain=domain_filter if domain_filter != "All" else None,
                    confidence_threshold=confidence_filter
                ))
            else:
                # Get all knowledge for domain
                if domain_filter != "All":
                    results = asyncio.run(st.session_state.knowledge_api.get_knowledge_by_domain(domain_filter))
                else:
                    results = []
            
            if results:
                st.markdown(f"Found {len(results)} knowledge entries:")
                
                for knowledge in results:
                    with st.expander(f"📖 {knowledge['statement'][:100]}..."):
                        st.markdown(f"**Statement:** {knowledge['statement']}")
                        
                        # Display confidence
                        confidence = knowledge.get('confidence', 0)
                        if confidence > 0.8:
                            confidence_class = "confidence-high"
                        elif confidence > 0.5:
                            confidence_class = "confidence-medium"
                        else:
                            confidence_class = "confidence-low"
                        
                        st.markdown(f'<span class="{confidence_class}">Confidence: {confidence:.2f}</span>', unsafe_allow_html=True)
                        
                        # Display domain and creation date
                        st.markdown(f"**Domain:** {knowledge.get('domain', 'General')}")
                        st.markdown(f"**Created:** {knowledge.get('created_at', 'Unknown')}")
                        
                        # Display provenance
                        provenance = knowledge.get('provenance', {})
                        if provenance:
                            st.markdown("**🏆 Promotion History:**")
                            
                            if 'hypothesis_id' in provenance:
                                st.markdown(f"- Promoted from Hypothesis #{provenance['hypothesis_id']}")
                            
                            if 'promoted_by' in provenance:
                                st.markdown(f"- Promoted by: {provenance['promoted_by']}")
                            
                            if 'validation_events' in provenance:
                                st.markdown(f"- Validation events: {len(provenance['validation_events'])}")
                            
                            if 'original_creator' in provenance:
                                st.markdown(f"- Original creator: {provenance['original_creator']}")
                        
                        # Actions
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"📊 View Details", key=f"knowledge_details_{knowledge['id']}"):
                                st.session_state.selected_knowledge_id = knowledge['id']
                        
                        with col2:
                            if st.button(f"🔗 Related Hypotheses", key=f"knowledge_related_{knowledge['id']}"):
                                # Show related hypotheses
                                st.info("Feature coming soon: Related hypotheses viewer")
            
            else:
                st.info("No knowledge entries found matching your criteria.")
                
        except Exception as e:
            st.error(f"Knowledge search failed: {e}")

def display_hypothesis_tracker():
    """Display the hypothesis tracker interface."""
    st.markdown("### 🧪 Hypothesis Tracker")
    
    # Status tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔄 Proposed", "🔬 Under Review", "✅ Validated", "🚀 Promoted", "❌ Refuted"])
    
    with tab1:  # Proposed hypotheses
        hypotheses = asyncio.run(st.session_state.knowledge_api.get_hypotheses_by_status("proposed"))
        display_hypotheses_list(hypotheses, "proposed")
    
    with tab2:  # Under review
        hypotheses = asyncio.run(st.session_state.knowledge_api.get_hypotheses_by_status("under_review"))
        display_hypotheses_list(hypotheses, "under_review")
    
    with tab3:  # Validated
        hypotheses = asyncio.run(st.session_state.knowledge_api.get_hypotheses_by_status("validated"))
        display_hypotheses_list(hypotheses, "validated")
    
    with tab4:  # Promoted
        hypotheses = asyncio.run(st.session_state.knowledge_api.get_hypotheses_by_status("promoted"))
        display_hypotheses_list(hypotheses, "promoted")
    
    with tab5:  # Refuted
        hypotheses = asyncio.run(st.session_state.knowledge_api.get_hypotheses_by_status("refuted"))
        display_hypotheses_list(hypotheses, "refuted")

def display_hypotheses_list(hypotheses: List[Dict], status: str):
    """Display a list of hypotheses with actions."""
    if hypotheses:
        st.markdown(f"Found {len(hypotheses)} {status} hypotheses:")
        
        for hyp in hypotheses:
            with st.expander(f"💡 {hyp['statement'][:80]}... (Confidence: {hyp['confidence']:.2f})"):
                st.markdown(f"**Statement:** {hyp['statement']}")
                
                # Display confidence with color coding
                confidence = hyp['confidence']
                if confidence > 0.8:
                    confidence_class = "confidence-high"
                elif confidence > 0.5:
                    confidence_class = "confidence-medium"
                else:
                    confidence_class = "confidence-low"
                
                st.markdown(f'<span class="{confidence_class}">Confidence: {confidence:.2f}</span>', unsafe_allow_html=True)
                
                # Display metadata
                st.markdown(f"**Created by:** {hyp.get('created_by', 'Unknown')}")
                st.markdown(f"**Created:** {hyp.get('created_at', 'Unknown')}")
                st.markdown(f"**Last updated:** {hyp.get('updated_at', 'Unknown')}")
                
                # Display supporting evidence count
                support_count = len(hyp.get('support_ids', []))
                st.markdown(f"**Supporting evidence:** {support_count} events")
                
                # Actions based on status
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"📊 Analytics", key=f"analytics_{hyp['id']}"):
                        display_hypothesis_analytics(hyp['id'])
                
                with col2:
                    if status in ["proposed", "under_review", "validated"]:
                        if st.button(f"🚀 Promote", key=f"promote_{hyp['id']}"):
                            promote_hypothesis_to_knowledge(hyp['id'])
                
                with col3:
                    if status in ["proposed", "under_review"]:
                        if st.button(f"❌ Refute", key=f"refute_{hyp['id']}"):
                            refute_hypothesis(hyp['id'])
    else:
        st.info(f"No {status} hypotheses found.")

def display_hypothesis_analytics(hypothesis_id: int):
    """Display detailed analytics for a hypothesis."""
    try:
        analytics = asyncio.run(st.session_state.knowledge_api.get_hypothesis_analytics(hypothesis_id))
        
        if 'error' in analytics:
            st.error(f"Analytics error: {analytics['error']}")
            return
        
        st.markdown("#### 📊 Hypothesis Analytics")
        
        # Basic info
        hypothesis = analytics['hypothesis']
        st.markdown(f"**Statement:** {hypothesis['statement']}")
        st.markdown(f"**Current Confidence:** {hypothesis['confidence']:.2f}")
        st.markdown(f"**Age:** {analytics['age_days']} days")
        st.markdown(f"**Supporting Events:** {analytics['support_count']}")
        
        # Confidence history
        if analytics['confidence_history']:
            st.markdown("#### 📈 Confidence Evolution")
            
            confidence_df = pd.DataFrame(analytics['confidence_history'])
            confidence_df['timestamp'] = pd.to_datetime(confidence_df['timestamp'])
            
            fig = px.line(confidence_df, x='timestamp', y='confidence', 
                         title='Confidence Over Time',
                         labels={'timestamp': 'Time', 'confidence': 'Confidence'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Supporting events
        if analytics['supporting_events']:
            st.markdown("#### 🎯 Supporting Events")
            
            events_df = pd.DataFrame(analytics['supporting_events'])
            st.dataframe(events_df[['timestamp', 'source', 'event_type']], use_container_width=True)
        
    except Exception as e:
        st.error(f"Failed to load hypothesis analytics: {e}")

def promote_hypothesis_to_knowledge(hypothesis_id: int):
    """Promote a hypothesis to knowledge."""
    try:
        # Get supporting events for validation
        hypothesis = asyncio.run(st.session_state.knowledge_api.get_hypothesis_analytics(hypothesis_id))
        
        if 'error' in hypothesis:
            st.error(f"Cannot promote hypothesis: {hypothesis['error']}")
            return
        
        # Extract validation events
        validation_events = [event['id'] for event in hypothesis.get('supporting_events', [])]
        
        # Promote to knowledge
        knowledge_id = asyncio.run(st.session_state.knowledge_api.promote_to_knowledge(
            hypothesis_id=hypothesis_id,
            validation_events=validation_events,
            domain="general",  # Could be extracted from hypothesis
            promoted_by="user"
        ))
        
        st.success(f"✅ Hypothesis promoted to knowledge! Knowledge ID: {knowledge_id}")
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to promote hypothesis: {e}")

def refute_hypothesis(hypothesis_id: int):
    """Mark a hypothesis as refuted."""
    try:
        success = asyncio.run(st.session_state.knowledge_api.update_hypothesis_status(
            hypothesis_id=hypothesis_id,
            new_status="refuted",
            reason="User decision",
            updated_by="user"
        ))
        
        if success:
            st.success("✅ Hypothesis marked as refuted")
            st.rerun()
        else:
            st.error("❌ Failed to update hypothesis status")
            
    except Exception as e:
        st.error(f"Failed to refute hypothesis: {e}")

def display_events_log():
    """Display the events log interface."""
    st.markdown("### 📋 Events Log")
    
    # Event type filter
    col1, col2, col3 = st.columns(3)
    
    with col1:
        event_type = st.selectbox("Event Type", ["All", "hypothesis_proposed", "hypothesis_evaluation", "confidence_updated", "hypothesis_promoted", "problem_solving", "concept_explanation"])
    
    with col2:
        source_filter = st.selectbox("Source", ["All", "physics_expert", "hypothesis_generator", "supervisor", "user"])
    
    with col3:
        days_back = st.slider("Days Back", 1, 30, 7)
    
    if st.button("🔍 Load Events"):
        try:
            since_date = datetime.now() - timedelta(days=days_back)
            
            if event_type == "All":
                # Get recent events
                events = asyncio.run(st.session_state.knowledge_api.get_events_by_type(
                    event_type="hypothesis_proposed",  # This is a limitation - we'd need a get_all_events method
                    since=since_date,
                    source=source_filter if source_filter != "All" else None
                ))
            else:
                events = asyncio.run(st.session_state.knowledge_api.get_events_by_type(
                    event_type=event_type,
                    since=since_date,
                    source=source_filter if source_filter != "All" else None
                ))
            
            if events:
                st.markdown(f"Found {len(events)} events:")
                
                for event in events:
                    with st.expander(f"🎯 {event['source']} - {event['event_type']} ({event['timestamp']})"):
                        st.markdown(f"**Source:** {event['source']}")
                        st.markdown(f"**Type:** {event['event_type']}")
                        st.markdown(f"**Timestamp:** {event['timestamp']}")
                        
                        # Display payload
                        if event.get('payload_json'):
                            payload = json.loads(event['payload_json'])
                            st.markdown("**Details:**")
                            st.json(payload)
            else:
                st.info("No events found matching your criteria.")
                
        except Exception as e:
            st.error(f"Failed to load events: {e}")

def display_collaboration_interface():
    """Display the enhanced collaborative interface with full multi-agent capabilities."""
    st.markdown("### 🤖 Collaborative Physics Research")
    
    # Initialize enhanced session state for collaboration
    initialize_collaborative_session_state()
    
    # Load collaborative system if needed
    if st.session_state.collaborative_system is None:
        st.session_state.collaborative_system = load_collaborative_system()
    
    if not st.session_state.collaborative_system:
        st.error("Failed to initialize collaborative system")
        return
    
    # Display agent status
    display_agent_status()
    
    # Collaboration controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        collaboration_mode = st.selectbox(
            "🔄 Collaboration Mode",
            ["research", "debate", "brainstorm", "teaching"],
            index=["research", "debate", "brainstorm", "teaching"].index(st.session_state.collaboration_mode),
            help="Choose how the agents should collaborate"
        )
        if collaboration_mode != st.session_state.collaboration_mode:
            st.session_state.collaboration_mode = collaboration_mode
    
    with col2:
        if st.button("🆕 New Session"):
            st.session_state.current_session_id = None
            st.session_state.chat_history = []
            st.success("New session started!")
            st.rerun()
    
    with col3:
        session_count = len(st.session_state.active_sessions) if hasattr(st.session_state, 'active_sessions') else 0
        st.metric("Active Sessions", session_count)
    
    # Mode descriptions
    mode_descriptions = {
        "research": "🔬 Systematic investigation with balanced analysis and exploration",
        "debate": "⚡ Structured discussion where agents challenge ideas",
        "brainstorm": "🧠 Creative exploration with minimal constraints",
        "teaching": "📚 Educational explanations with expert knowledge and analogies"
    }
    st.info(mode_descriptions[collaboration_mode])
    
    # Quick Start Options
    st.markdown("#### 🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔬 Research a Topic", type="primary"):
            st.session_state.quick_action = "research"
    
    with col2:
        if st.button("💡 Generate Hypotheses", type="primary"):
            st.session_state.quick_action = "hypotheses"
    
    with col3:
        if st.button("⚡ Start a Debate", type="primary"):
            st.session_state.quick_action = "debate"
    
    # Handle quick actions
    if hasattr(st.session_state, 'quick_action'):
        if st.session_state.quick_action == "research":
            st.markdown("#### 🔬 Research Topic")
            topic = st.text_input("Enter a physics topic to research collaboratively:")
            context = st.text_area("Additional context (optional):")
            
            if st.button("Start Research") and topic:
                start_collaborative_session(topic, "research", context)
                del st.session_state.quick_action
                st.rerun()
        
        elif st.session_state.quick_action == "hypotheses":
            st.markdown("#### 💡 Generate Hypotheses")
            topic = st.text_input("Enter a topic for hypothesis generation:")
            num_hypotheses = st.slider("Number of hypotheses:", 1, 10, 3)
            
            if st.button("Generate Hypotheses") and topic:
                generate_hypotheses_session(topic, num_hypotheses)
                del st.session_state.quick_action
                st.rerun()
        
        elif st.session_state.quick_action == "debate":
            st.markdown("#### ⚡ Start Debate")
            hypothesis = st.text_area("Enter a hypothesis to debate:")
            topic = st.text_input("Topic context:")
            
            if st.button("Start Debate") and hypothesis and topic:
                start_debate_session(hypothesis, topic)
                del st.session_state.quick_action
                st.rerun()
    
    # Chat Interface
    display_chat_interface()

def initialize_collaborative_session_state():
    """Initialize session state variables for enhanced collaboration."""
    if 'collaborative_system' not in st.session_state:
        st.session_state.collaborative_system = None
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'active_sessions' not in st.session_state:
        st.session_state.active_sessions = {}
    if 'collaboration_mode' not in st.session_state:
        st.session_state.collaboration_mode = "research"
    if 'agent_settings' not in st.session_state:
        st.session_state.agent_settings = {
            'difficulty_level': 'undergraduate',
            'creativity_level': 'high',
            'collaboration_style': 'balanced'
        }

def load_collaborative_system() -> Optional[CollaborativePhysicsSystem]:
    """Load and initialize the collaborative physics system."""
    try:
        system = CollaborativePhysicsSystem(
            difficulty_level=st.session_state.agent_settings['difficulty_level'],
            creativity_level=st.session_state.agent_settings['creativity_level'],
            collaboration_style=st.session_state.agent_settings['collaboration_style'],
            memory_enabled=True
        )
        return system
    except Exception as e:
        st.error(f"Error loading collaborative system: {e}")
        st.error(traceback.format_exc())
        return None

def display_agent_status():
    """Display real-time agent status indicators."""
    if not st.session_state.collaborative_system:
        return
    
    st.markdown("#### 🤖 Agent Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="padding: 1rem; border: 1px solid #e5e7eb; border-radius: 0.5rem; background: #f9fafb;">
            <h4 style="margin: 0; color: #059669;">🔬 Physics Expert</h4>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">Ready</p>
            <small style="color: #9ca3af;">Rigorous Analysis</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 1rem; border: 1px solid #e5e7eb; border-radius: 0.5rem; background: #f9fafb;">
            <h4 style="margin: 0; color: #7c3aed;">💡 Hypothesis Generator</h4>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">Ready</p>
            <small style="color: #9ca3af;">Creative Thinking</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="padding: 1rem; border: 1px solid #e5e7eb; border-radius: 0.5rem; background: #f9fafb;">
            <h4 style="margin: 0; color: #0ea5e9;">🤝 Supervisor</h4>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">Ready</p>
            <small style="color: #9ca3af;">Orchestrating</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        session_count = len(st.session_state.active_sessions)
        st.markdown(f"""
        <div style="padding: 1rem; border: 1px solid #e5e7eb; border-radius: 0.5rem; background: #f9fafb;">
            <h4 style="margin: 0; color: #dc2626;">📊 Sessions</h4>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">{session_count}</p>
            <small style="color: #9ca3af;">Active</small>
        </div>
        """, unsafe_allow_html=True)

def display_chat_interface():
    """Display the collaborative chat interface."""
    st.markdown("#### 💬 Collaborative Chat")
    
    # Chat input
    user_input = st.chat_input("Ask a question or continue the collaboration...")
    
    if user_input:
        handle_user_input(user_input)
    
    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            display_chat_message(message)
    else:
        st.info("Start a conversation by typing a message above or using the Quick Start buttons!")

def handle_user_input(user_input: str):
    """Handle user input and generate collaborative response."""
    if not st.session_state.collaborative_system:
        st.session_state.collaborative_system = load_collaborative_system()
    
    if not st.session_state.collaborative_system:
        st.error("Could not load collaborative system")
        return
    
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now()
    })
    
    try:
        if st.session_state.current_session_id:
            # Continue existing session
            response = st.session_state.collaborative_system.continue_collaboration(
                session_id=st.session_state.current_session_id,
                user_input=user_input,
                mode=st.session_state.collaboration_mode
            )
            
            if "error" in response:
                st.error(response["error"])
                return
            
            # Add response to history
            st.session_state.chat_history.append({
                "role": "collaborative",
                "content": response["response"],
                "timestamp": datetime.now()
            })
        else:
            # Start new session
            session = st.session_state.collaborative_system.start_collaborative_session(
                topic=user_input,
                mode=st.session_state.collaboration_mode
            )
            
            st.session_state.current_session_id = session["session_info"]["session_id"]
            st.session_state.active_sessions[st.session_state.current_session_id] = session["session_info"]
            
            # Add response to history
            st.session_state.chat_history.append({
                "role": "collaborative",
                "content": session["response"],
                "timestamp": datetime.now()
            })
        
        # Log the collaboration to knowledge database
        if st.session_state.knowledge_api:
            asyncio.run(st.session_state.knowledge_api.log_event(
                source="user",
                event_type="collaboration_message",
                payload={
                    "user_input": user_input,
                    "mode": st.session_state.collaboration_mode,
                    "session_id": st.session_state.current_session_id
                }
            ))
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing input: {e}")
        st.session_state.chat_history.append({
            "role": "error",
            "content": f"Error: {str(e)}",
            "timestamp": datetime.now()
        })

def display_chat_message(message: Dict[str, Any]):
    """Display a chat message with appropriate styling."""
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", datetime.now())
    
    # Determine message styling based on role
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
            st.caption(timestamp.strftime("%H:%M:%S"))
    
    elif role == "physics_expert" or "🔬" in str(content):
        with st.chat_message("assistant", avatar="🔬"):
            st.markdown(f"**Physics Expert**: {content}")
            st.caption(timestamp.strftime("%H:%M:%S"))
    
    elif role == "hypothesis_generator" or "💡" in str(content):
        with st.chat_message("assistant", avatar="💡"):
            st.markdown(f"**Hypothesis Generator**: {content}")
            st.caption(timestamp.strftime("%H:%M:%S"))
    
    elif role == "supervisor" or "🤝" in str(content):
        with st.chat_message("assistant", avatar="🤝"):
            st.markdown(f"**Supervisor**: {content}")
            st.caption(timestamp.strftime("%H:%M:%S"))
    
    elif role == "collaborative" or "synthesis" in role:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"**Collaborative Team**: {content}")
            st.caption(timestamp.strftime("%H:%M:%S"))
    
    elif role == "error":
        st.error(content)
    
    else:
        with st.chat_message("assistant"):
            st.markdown(content)
            st.caption(timestamp.strftime("%H:%M:%S"))

def start_collaborative_session(topic: str, mode: str, context: str = ""):
    """Start a new collaborative session."""
    if not st.session_state.collaborative_system:
        st.session_state.collaborative_system = load_collaborative_system()
    
    if st.session_state.collaborative_system:
        try:
            session = st.session_state.collaborative_system.start_collaborative_session(
                topic=topic,
                mode=mode,
                context=context
            )
            
            st.session_state.current_session_id = session["session_info"]["session_id"]
            st.session_state.active_sessions[st.session_state.current_session_id] = session["session_info"]
            
            # Add to chat history
            st.session_state.chat_history = [
                {"role": "user", "content": f"Research topic: {topic}", "timestamp": datetime.now()},
                {"role": "collaborative", "content": session["response"], "timestamp": datetime.now()}
            ]
            
            # Log to knowledge database
            if st.session_state.knowledge_api:
                asyncio.run(st.session_state.knowledge_api.log_event(
                    source="user",
                    event_type="collaboration_started",
                    payload={
                        "topic": topic,
                        "mode": mode,
                        "context": context,
                        "session_id": st.session_state.current_session_id
                    }
                ))
            
            st.success(f"Started {mode} session: {topic}")
            
        except Exception as e:
            st.error(f"Error starting session: {e}")

def generate_hypotheses_session(topic: str, num_hypotheses: int):
    """Generate hypotheses for a topic."""
    if not st.session_state.collaborative_system:
        st.session_state.collaborative_system = load_collaborative_system()
    
    if st.session_state.collaborative_system:
        try:
            hypotheses = st.session_state.collaborative_system.generate_hypotheses(
                topic=topic,
                num_hypotheses=num_hypotheses
            )
            
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "user", 
                "content": f"Generate {num_hypotheses} hypotheses for: {topic}", 
                "timestamp": datetime.now()
            })
            st.session_state.chat_history.append({
                "role": "hypothesis_generator", 
                "content": hypotheses, 
                "timestamp": datetime.now()
            })
            
            # Log to knowledge database
            if st.session_state.knowledge_api:
                asyncio.run(st.session_state.knowledge_api.log_event(
                    source="hypothesis_generator",
                    event_type="hypothesis_generation_session",
                    payload={
                        "topic": topic,
                        "num_hypotheses": num_hypotheses,
                        "hypotheses": hypotheses
                    }
                ))
            
            st.success("Hypotheses generated!")
            
        except Exception as e:
            st.error(f"Error generating hypotheses: {e}")

def start_debate_session(hypothesis: str, topic: str):
    """Start a debate session."""
    if not st.session_state.collaborative_system:
        st.session_state.collaborative_system = load_collaborative_system()
    
    if st.session_state.collaborative_system:
        try:
            debate_result = st.session_state.collaborative_system.facilitate_debate(
                hypothesis=hypothesis,
                topic=topic
            )
            
            # Add to chat history
            st.session_state.chat_history.append({
                "role": "user", 
                "content": f"Debate hypothesis: {hypothesis} (Topic: {topic})", 
                "timestamp": datetime.now()
            })
            st.session_state.chat_history.append({
                "role": "supervisor", 
                "content": debate_result, 
                "timestamp": datetime.now()
            })
            
            # Log to knowledge database
            if st.session_state.knowledge_api:
                asyncio.run(st.session_state.knowledge_api.log_event(
                    source="supervisor",
                    event_type="debate_session",
                    payload={
                        "hypothesis": hypothesis,
                        "topic": topic,
                        "result": debate_result
                    }
                ))
            
            st.success("Debate session started!")
            
        except Exception as e:
            st.error(f"Error starting debate: {e}")

def main():
    """Main application function with interface selection."""
    initialize_session_state()
    
    # Initialize Knowledge API
    if not initialize_knowledge_api():
        st.stop()
    
    # Interface selection at the top
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; text-align: center; margin: 0;">🚀 PhysicsGPT Interface Hub</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Interface tabs
    interface_tab1, interface_tab2, interface_tab3 = st.tabs([
        "🧪 Knowledge Lab", 
        "🎨 Agent Canvas", 
        "🤖 Collaborative"
    ])
    
    with interface_tab1:
        # Original Knowledge Lab Interface
        # Create main header
        create_main_header()
        
        # Display system metrics
        display_system_metrics()
        
        # Main interface tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📚 Knowledge Base", 
            "🧪 Hypotheses", 
            "📋 Events Log", 
            "💬 Collaborate",
            "📊 Analytics"
        ])
        
        with tab1:
            display_knowledge_browser()
        
        with tab2:
            display_hypothesis_tracker()
        
        with tab3:
            display_events_log()
        
        with tab4:
            display_collaboration_interface()
        
        with tab5:
            st.markdown("### 📊 System Analytics")
            st.info("Advanced analytics dashboard coming soon!")
            
            # For now, show basic system info
            if st.session_state.knowledge_api:
                try:
                    analytics = asyncio.run(st.session_state.knowledge_api.get_system_analytics())
                    
                    # Status distribution
                    if analytics.get('status_distribution'):
                        st.markdown("#### Hypothesis Status Distribution")
                        status_df = pd.DataFrame(list(analytics['status_distribution'].items()), 
                                               columns=['Status', 'Count'])
                        fig = px.pie(status_df, values='Count', names='Status', 
                                   title='Hypothesis Status Distribution')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Domain distribution
                    if analytics.get('domain_distribution'):
                        st.markdown("#### Knowledge Domain Distribution")
                        domain_df = pd.DataFrame(list(analytics['domain_distribution'].items()), 
                                               columns=['Domain', 'Count'])
                        fig = px.bar(domain_df, x='Domain', y='Count', 
                                   title='Knowledge by Domain')
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Failed to load analytics: {e}")
    
    with interface_tab2:
        # Agent Canvas Interface
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
            <h1>🎨 Agent Canvas</h1>
            <p>Interactive Multi-Agent Physics Research Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple agent canvas implementation
        st.markdown("### 🤖 Physics Agents")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 16px rgba(0,0,0,0.1);">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
                <h3>Physics Expert</h3>
                <p>Rigorous scientific analysis and validation</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("💬 Chat with Physics Expert", use_container_width=True):
                st.info("🔬 Physics Expert ready for questions!")
        
        with col2:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 16px rgba(0,0,0,0.1);">
                <div style="font-size: 3rem; margin-bottom: 1rem;">💡</div>
                <h3>Hypothesis Generator</h3>
                <p>Creative idea generation and exploration</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🧪 Generate Hypothesis", use_container_width=True):
                st.info("💡 Hypothesis Generator ready to create!")
        
        with col3:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 16px rgba(0,0,0,0.1);">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🤝</div>
                <h3>Supervisor</h3>
                <p>Orchestrates collaboration and coordination</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("👥 Start Collaboration", use_container_width=True):
                st.info("🤝 Supervisor ready to coordinate!")
        
        # Quick actions
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧪 Quick Hypothesis", use_container_width=True):
                st.balloons()
                st.success("🎉 Generated hypothesis about quantum entanglement applications!")
        
        with col2:
            if st.button("🔬 Physics Analysis", use_container_width=True):
                st.success("📊 Analyzing electromagnetic wave propagation...")
        
        with col3:
            if st.button("📊 Research Summary", use_container_width=True):
                st.success("📈 Compiling research progress summary...")
        
        # Note about full canvas
        st.markdown("---")
        st.info("""
        💡 **Note**: This is a simplified Agent Canvas. For the full interactive experience with:
        - Advanced animations and visual effects
        - Real-time collaboration visualization  
        - User preferences and customization
        - Session analytics and tracking
        
        The complete Agent Canvas is available in `streamlit_canvas.py` for local testing.
        """)
    
    with interface_tab3:
        # Collaborative Interface
        st.markdown("""
        <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
            <h1>🤖 Collaborative Agents</h1>
            <p>Multi-Agent Physics Research Collaboration</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use the existing collaboration interface from the Knowledge Lab
        display_collaboration_interface()

if __name__ == "__main__":
    main() 