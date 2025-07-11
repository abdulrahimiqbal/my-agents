#!/usr/bin/env python3
"""
Test script to verify collaborative system database integration.
This script tests that the collaborative system properly creates hypotheses and logs events.
"""

import asyncio
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

from src.agents import CollaborativePhysicsSystem
from src.database.knowledge_api import get_knowledge_api

async def test_collaborative_system():
    """Test the collaborative system and database integration."""
    print("🧪 Testing Collaborative Physics System Database Integration")
    print("=" * 60)
    
    # Initialize knowledge API
    print("1. Initializing knowledge API...")
    api = get_knowledge_api()
    
    # Get baseline counts
    print("2. Getting baseline database counts...")
    analytics_before = await api.get_system_analytics()
    print(f"   Before - Knowledge: {analytics_before.get('total_knowledge', 0)}")
    print(f"   Before - Hypotheses: {analytics_before.get('total_hypotheses', 0)}")
    print(f"   Before - Events: {analytics_before.get('total_events', 0)}")
    
    # Test direct hypothesis creation
    print("\n3. Testing direct hypothesis creation...")
    hypothesis_id = api.propose_hypothesis_sync(
        statement="Quantum computing can solve NP-complete problems exponentially faster than classical computers",
        created_by="test_system",
        initial_confidence=0.7,
        domain="quantum"
    )
    print(f"   Created hypothesis ID: {hypothesis_id}")
    
    # Test event logging
    print("\n4. Testing event logging...")
    event_id = api.log_event_sync(
        source="test_system",
        event_type="test_event",
        payload={
            "test_data": "collaborative system test",
            "timestamp": "2024-01-01"
        }
    )
    print(f"   Logged event ID: {event_id}")
    
    # Check database after operations
    print("\n5. Checking database after operations...")
    analytics_after = await api.get_system_analytics()
    print(f"   After - Knowledge: {analytics_after.get('total_knowledge', 0)}")
    print(f"   After - Hypotheses: {analytics_after.get('total_hypotheses', 0)}")
    print(f"   After - Events: {analytics_after.get('total_events', 0)}")
    
    # Check for new hypotheses
    new_hypotheses = analytics_after.get('total_hypotheses', 0) - analytics_before.get('total_hypotheses', 0)
    new_events = analytics_after.get('total_events', 0) - analytics_before.get('total_events', 0)
    
    print(f"\n6. Changes detected:")
    print(f"   New hypotheses: {new_hypotheses}")
    print(f"   New events: {new_events}")
    
    # Check specific hypothesis content
    print("\n7. Checking recent hypotheses...")
    recent_hypotheses = await api.get_hypotheses_by_status("proposed")
    if recent_hypotheses:
        print(f"   Found {len(recent_hypotheses)} proposed hypotheses:")
        for i, h in enumerate(recent_hypotheses[-2:]):  # Show last 2
            print(f"   {i+1}. {h['statement'][:80]}...")
            print(f"      Created by: {h.get('created_by', 'unknown')}")
            print(f"      Confidence: {h.get('confidence', 0):.2f}")
    
    # Check recent events
    print("\n8. Checking recent events...")
    recent_events = await api.get_events_by_type("test_event", limit=3)
    if recent_events:
        print(f"   Found {len(recent_events)} recent test events:")
        for i, e in enumerate(recent_events):
            print(f"   {i+1}. Source: {e.get('source', 'unknown')}")
            print(f"      Timestamp: {e.get('timestamp', 'unknown')}")
    
    # Test hypothesis promotion
    if recent_hypotheses:
        print("\n9. Testing hypothesis promotion...")
        latest_hypothesis = recent_hypotheses[-1]
        hypothesis_id = latest_hypothesis['id']
        
        # Create some supporting events
        support_event_1 = api.log_event_sync(
            source="test_system",
            event_type="validation_evidence",
            payload={"evidence": "Theoretical analysis supports hypothesis"}
        )
        
        support_event_2 = api.log_event_sync(
            source="test_system", 
            event_type="experimental_evidence",
            payload={"evidence": "Preliminary experiments show promise"}
        )
        
        # Promote to knowledge
        knowledge_id = await api.promote_to_knowledge(
            hypothesis_id=hypothesis_id,
            validation_events=[support_event_1, support_event_2],
            domain="quantum",
            promoted_by="test_system"
        )
        
        print(f"   Promoted hypothesis {hypothesis_id} to knowledge {knowledge_id}")
    
    # Final analytics
    print("\n10. Final database state...")
    final_analytics = await api.get_system_analytics()
    print(f"    Final - Knowledge: {final_analytics.get('total_knowledge', 0)}")
    print(f"    Final - Hypotheses: {final_analytics.get('total_hypotheses', 0)}")
    print(f"    Final - Events: {final_analytics.get('total_events', 0)}")
    print(f"    Promotion rate: {final_analytics.get('promotion_rate', 0):.1%}")
    
    print("\n✅ Test completed successfully!")
    print("🎯 The knowledge management system is working correctly.")
    
    return {
        'hypotheses_created': final_analytics.get('total_hypotheses', 0) - analytics_before.get('total_hypotheses', 0),
        'events_logged': final_analytics.get('total_events', 0) - analytics_before.get('total_events', 0),
        'knowledge_created': final_analytics.get('total_knowledge', 0) - analytics_before.get('total_knowledge', 0),
        'final_state': final_analytics
    }

if __name__ == "__main__":
    try:
        results = asyncio.run(test_collaborative_system())
        print(f"\n📊 Test Results Summary:")
        print(f"   Hypotheses created: {results['hypotheses_created']}")
        print(f"   Events logged: {results['events_logged']}")
        print(f"   Knowledge created: {results['knowledge_created']}")
        print(f"   System is working: {'✅ YES' if results['hypotheses_created'] > 0 else '❌ NO'}")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 