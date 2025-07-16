#!/usr/bin/env python3
"""
Test script to verify agent conversation monitoring works in demo mode.
"""

import time
import threading
from agent_monitor import agent_monitor, simulate_agent_progress

def test_demo_conversations():
    """Test the agent conversation monitoring system in demo mode."""
    
    print("🧪 Testing Agent Conversation Monitoring (Demo Mode)")
    print("=" * 60)
    
    # Start monitoring
    agent_monitor.start_monitoring()
    
    # Test agents
    selected_agents = ['physics_expert', 'quantum_specialist', 'mathematical_analyst']
    query = "What is quantum entanglement?"
    
    print(f"📋 Query: {query}")
    print(f"🤖 Selected agents: {selected_agents}")
    print()
    
    # Add callback to see updates
    def print_conversations(conversations):
        print(f"📊 Active conversations: {len(conversations)}")
        for name, conv in conversations.items():
            data = conv.to_dict()
            print(f"  - {name}: {data['status']} ({data['progress_percentage']}%) - {data['current_step']}")
            if data['total_interactions'] > 0:
                print(f"    💭 {data['total_interactions']} interactions")
    
    agent_monitor.add_update_callback(print_conversations)
    
    # Start demo simulations
    print("🔄 Starting demo agent simulations...")
    threads = []
    
    for agent in selected_agents:
        task_desc = f"Analyzing: {query}"
        thread = threading.Thread(
            target=simulate_agent_progress,
            args=(agent, task_desc, 15)  # 15 second simulation
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Wait for all agents to complete
    for thread in threads:
        thread.join()
    
    print("\n✅ Demo analysis complete!")
    
    # Show final conversation summary
    summary = agent_monitor.get_conversation_summary()
    print(f"\n📊 Final Summary:")
    print(f"  - Total agents: {summary['total_agents']}")
    print(f"  - Completed agents: {summary['completed_agents']}")
    print(f"  - Total interactions: {summary['total_interactions']}")
    
    # Show detailed conversations
    print(f"\n💭 Detailed Conversations:")
    print("=" * 40)
    
    conversations = agent_monitor.get_active_conversations()
    for conv_data in conversations:
        print(f"\n🤖 {conv_data['agent_name'].replace('_', ' ').title()}:")
        print(f"   Status: {conv_data['status']}")
        print(f"   Duration: {conv_data['duration']:.1f}s")
        print(f"   Interactions: {conv_data['total_interactions']}")
        
        # Show some thoughts
        if conv_data['thoughts']:
            print(f"   💭 Sample thoughts:")
            for thought in conv_data['thoughts'][:3]:
                print(f"      - {thought['content']}")
        
        # Show decisions
        if conv_data['decisions']:
            print(f"   🎯 Decisions:")
            for decision in conv_data['decisions']:
                print(f"      - {decision['decision']}")
    
    # Stop monitoring
    agent_monitor.stop_monitoring()
    print(f"\n🏁 Test completed successfully!")

if __name__ == "__main__":
    test_demo_conversations()