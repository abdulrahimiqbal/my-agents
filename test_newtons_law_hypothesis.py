#!/usr/bin/env python3
"""
Test Newton's Law Hypothesis Progression
Tests the hypothesis tracking system by proposing Newton's second law and watching it progress.
"""

import os
import sys
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_newtons_law_hypothesis():
    """Test hypothesis progression with Newton's second law."""
    
    print("🧪 TESTING NEWTON'S LAW HYPOTHESIS PROGRESSION")
    print("=" * 60)
    print("📋 Testing how hypotheses progress through research phases")
    print("🎯 Target: Newton's Second Law of Motion")
    print()
    
    # Step 1: Ask a question that should generate Newton's law as a hypothesis
    newtons_law_question = """
    What is the fundamental relationship between force, mass, and acceleration? 
    Please analyze this from first principles and propose any mathematical 
    relationships you discover.
    """
    
    print("📝 Step 1: Asking physics question about force, mass, and acceleration")
    print(f"Question: {newtons_law_question.strip()}")
    print()
    
    try:
        # Import and run the physics analysis
        from physics_flow_system import analyze_physics_question_with_flow
        
        print("🚀 Starting physics analysis...")
        start_time = time.time()
        
        result = analyze_physics_question_with_flow(newtons_law_question)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✅ Analysis completed in {execution_time:.2f} seconds")
        print()
        print("📄 ANALYSIS RESULT:")
        print("-" * 40)
        print(result[:500] + "..." if len(result) > 500 else result)
        print()
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False
    
    # Step 2: Check the database for new hypotheses
    print("🔍 Step 2: Checking database for new hypotheses...")
    
    try:
        from crewai_database_integration import CrewAIKnowledgeAPI
        import sqlite3
        
        knowledge_api = CrewAIKnowledgeAPI()
        
        # Get recent hypotheses
        with sqlite3.connect(knowledge_api.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get hypotheses from the last 5 minutes
            cursor.execute("""
                SELECT id, title, description, confidence_score, 
                       validation_status, created_by, created_at
                FROM hypotheses 
                WHERE datetime(created_at) >= datetime('now', '-5 minutes')
                ORDER BY created_at DESC
            """)
            
            recent_hypotheses = [dict(row) for row in cursor.fetchall()]
        
        print(f"📊 Found {len(recent_hypotheses)} recent hypotheses")
        
        if recent_hypotheses:
            print("\n🧠 Recent Hypotheses:")
            for i, hyp in enumerate(recent_hypotheses, 1):
                print(f"  {i}. {hyp['title']}")
                print(f"     Status: {hyp['validation_status']}")
                print(f"     Confidence: {hyp['confidence_score']:.2f}")
                print(f"     Created by: {hyp['created_by']}")
                print(f"     Description: {hyp['description'][:100]}...")
                print()
        else:
            print("⚠️ No recent hypotheses found")
            print("💡 The system might not have generated hypotheses for this question")
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False
    
    # Step 3: Check knowledge base for validated findings
    print("🔍 Step 3: Checking knowledge base for validated findings...")
    
    try:
        with sqlite3.connect(knowledge_api.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, title, content, physics_domain, 
                       confidence_level, created_at
                FROM knowledge_entries 
                WHERE datetime(created_at) >= datetime('now', '-5 minutes')
                ORDER BY created_at DESC
            """)
            
            recent_knowledge = [dict(row) for row in cursor.fetchall()]
        
        print(f"📚 Found {len(recent_knowledge)} recent knowledge entries")
        
        if recent_knowledge:
            print("\n📖 Recent Knowledge Entries:")
            for i, entry in enumerate(recent_knowledge, 1):
                print(f"  {i}. {entry['title']}")
                print(f"     Domain: {entry['physics_domain']}")
                print(f"     Confidence: {entry['confidence_level']:.2f}")
                print(f"     Content: {entry['content'][:100]}...")
                print()
        
    except Exception as e:
        print(f"❌ Knowledge base check failed: {e}")
    
    # Step 4: Provide next steps
    print("🎯 Step 4: Next Steps for Testing")
    print("-" * 40)
    print("1. 🌐 Open Streamlit app: streamlit run streamlit_physics_app.py")
    print("2. 🔬 Go to 'Hypothesis Tracker' tab")
    print("3. 👀 Look for Newton's law-related hypotheses")
    print("4. 📊 Check their progression through research phases")
    print("5. 🧠 Go to 'Knowledge Base' tab to see validated knowledge")
    print()
    print("🔄 To test hypothesis progression:")
    print("   - Ask follow-up questions about force and acceleration")
    print("   - The system should validate and promote hypotheses")
    print("   - Watch them move from 'pending' → 'validated' → 'promoted'")
    
    return True

def manual_hypothesis_test():
    """Manually create a Newton's law hypothesis for testing."""
    print("\n🧪 MANUAL HYPOTHESIS TEST")
    print("=" * 40)
    print("Creating Newton's Second Law hypothesis manually...")
    
    try:
        from crewai_database_integration import CrewAIKnowledgeAPI
        
        knowledge_api = CrewAIKnowledgeAPI()
        
        # Create Newton's second law hypothesis
        hypothesis_id = knowledge_api.record_hypothesis(
            title="Newton's Second Law of Motion",
            description="The acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass. Mathematically: F = ma, where F is force, m is mass, and a is acceleration.",
            created_by="test_system",
            confidence_score=0.95
        )
        
        print(f"✅ Created hypothesis with ID: {hypothesis_id}")
        print("📋 Title: Newton's Second Law of Motion")
        print("📝 Description: F = ma relationship")
        print("🎯 Confidence: 0.95")
        print()
        print("🔄 This hypothesis will start in 'pending' status")
        print("👀 Check the Streamlit app to see it in the Hypothesis Tracker!")
        
        return True
        
    except Exception as e:
        print(f"❌ Manual hypothesis creation failed: {e}")
        return False

def main():
    """Main test function."""
    print("🔬 NEWTON'S LAW HYPOTHESIS PROGRESSION TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if we can run the analysis test
    try:
        from physics_flow_system import analyze_physics_question_with_flow
        analysis_available = True
    except ImportError:
        analysis_available = False
        print("⚠️ Physics flow system not available for analysis test")
    
    # Check if we can run database tests
    try:
        from crewai_database_integration import CrewAIKnowledgeAPI
        database_available = True
    except ImportError:
        database_available = False
        print("⚠️ Database integration not available")
    
    if not database_available:
        print("❌ Cannot run tests without database integration")
        return
    
    success = True
    
    if analysis_available:
        print("🧪 Running full analysis test...")
        success = test_newtons_law_hypothesis()
    else:
        print("🧪 Running manual hypothesis test only...")
        success = manual_hypothesis_test()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TEST COMPLETED SUCCESSFULLY!")
        print("👀 Now check the Streamlit app to see the results:")
        print("   streamlit run streamlit_physics_app.py")
        print("   → Go to 🔬 Hypothesis Tracker tab")
        print("   → Go to 🧠 Knowledge Base tab")
    else:
        print("❌ TEST FAILED - Check the errors above")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 