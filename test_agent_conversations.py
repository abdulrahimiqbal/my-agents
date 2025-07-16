#!/usr/bin/env python3
"""
Test script to verify agent conversation monitoring works.
"""

import time
from physics_crew_system import PhysicsGPTCrew
from agent_monitor import agent_monitor

def test_agent_conversations():
    """Test the agent conversation monitoring system."""
    
    print("🧪 Testing Agent Conversation Monitoring")
    print("=" * 50)
    
    # Start monitoring
    agent_monitor.start_monitoring()
    
    # Create physics crew with monitoring enabled
    print("🚀 Creating monitored PhysicsGPT crew...")
    crew = PhysicsGPTCrew(enable_monitoring=True)
    
    # Test query
    query = "What is quantum entanglement?"
    selected_agents = ['physics_expert', 'quantum_specialist']
    
    print(f"📋 Query: {query}")
    print(f"🤖 Selected agents: {selected_agents}")
    print()
    
    # Add callback to see updates
    def print_conversations(conversations):
        print(f"📊 Active conversations: {len(conversations)}")
        for name, conv in conversations.items():
            data = conv.to_dict()
            print(f"  - {name}: {data['status']} ({data['progress_percentage']}%)")
    
    agent_monitor.add_update_callback(print_conversations)
    
    # Run analysis
    print("🔄 Starting analysis...")
    result = crew.analyze_physics_query(query, selected_agents)
    
    print("\n✅ Analysis complete!")
    print("📄 Result:", result.get('result', 'No result')[:200] + "...")
    
    # Show final conversation summary
    summary = agent_monitor.get_conversation_summary()
    print(f"\n📊 Final Summary:")
    print(f"  - Total agents: {summary['total_agents']}")
    print(f"  - Completed agents: {summary['completed_agents']}")
    print(f"  - Total interactions: {summary['total_interactions']}")
    
    # Stop monitoring
    agent_monitor.stop_monitoring()

if __name__ == "__main__":
    test_agent_conversations()