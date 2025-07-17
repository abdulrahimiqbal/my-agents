#!/usr/bin/env python3
"""
Test the Enhanced Physics Flow System with Database Integration
Tests the 10-agent CrewAI Flow system with evaluation and database logging.
"""

import os
import sys
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_system_imports():
    """Test that all required components can be imported."""
    print("🔍 Testing System Imports...")
    print("-" * 50)
    
    try:
        print("Testing CrewAI Flow system...")
        from physics_flow_system import analyze_physics_question_with_flow
        print("✅ Physics Flow system imported successfully")
        
        print("Testing database integration...")
        from crewai_database_integration import CrewAIKnowledgeAPI, CrewAIEvaluationFramework
        print("✅ Database integration imported successfully")
        
        print("Testing CrewAI core...")
        import crewai
        print(f"✅ CrewAI version: {crewai.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_api_key_setup():
    """Test API key configuration."""
    print("\n🔑 Testing API Key Setup...")
    print("-" * 50)
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print(f"✅ OpenAI API key found (length: {len(openai_key)})")
        return True
    else:
        print("❌ OPENAI_API_KEY environment variable not found")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-key-here'")
        return False

def test_database_creation():
    """Test database and evaluation framework creation."""
    print("\n🗄️ Testing Database Creation...")
    print("-" * 50)
    
    try:
        from crewai_database_integration import CrewAIKnowledgeAPI, CrewAIEvaluationFramework
        
        # Test database creation
        knowledge_api = CrewAIKnowledgeAPI()
        print("✅ Knowledge API created successfully")
        
        # Test evaluation framework
        evaluation_framework = CrewAIEvaluationFramework()
        print("✅ Evaluation framework created successfully")
        
        # Test database tables
        tables = knowledge_api.get_all_events()  # This will create tables if they don't exist
        print("✅ Database tables initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Database creation failed: {e}")
        return False

def test_simple_physics_question():
    """Test the system with a simple physics question."""
    print("\n🧪 Testing Simple Physics Question...")
    print("-" * 50)
    
    try:
        from physics_flow_system import analyze_physics_question_with_flow
        
        # Simple test question
        test_question = "What is Newton's second law of motion?"
        
        print(f"Question: {test_question}")
        print("\n🚀 Starting physics analysis flow...")
        
        start_time = time.time()
        
        # Run the analysis
        result = analyze_physics_question_with_flow(test_question)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\n✅ Flow completed successfully!")
        print(f"⏱️ Execution time: {execution_time:.2f} seconds")
        print(f"📝 Result length: {len(result)} characters")
        print(f"🎯 First 200 chars: {result[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Physics question test failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

def test_complex_physics_question():
    """Test the system with a complex physics question."""
    print("\n🔬 Testing Complex Physics Question...")
    print("-" * 50)
    
    try:
        from physics_flow_system import analyze_physics_question_with_flow
        
        # Complex test question
        test_question = "How does quantum entanglement enable quantum computing, and what are the main challenges in maintaining coherence?"
        
        print(f"Question: {test_question}")
        print("\n🚀 Starting complex physics analysis flow...")
        
        start_time = time.time()
        
        # Run the analysis
        result = analyze_physics_question_with_flow(test_question)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\n✅ Complex flow completed successfully!")
        print(f"⏱️ Execution time: {execution_time:.2f} seconds")
        print(f"📝 Result length: {len(result)} characters")
        print(f"🎯 First 300 chars: {result[:300]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Complex physics question test failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

def test_database_analytics():
    """Test database analytics and reporting."""
    print("\n📊 Testing Database Analytics...")
    print("-" * 50)
    
    try:
        from crewai_database_integration import CrewAIKnowledgeAPI, CrewAIEvaluationFramework
        
        knowledge_api = CrewAIKnowledgeAPI()
        evaluation_framework = CrewAIEvaluationFramework()
        
        # Get analytics
        events = knowledge_api.get_all_events()
        hypotheses = knowledge_api.get_all_hypotheses()
        knowledge_entries = knowledge_api.get_all_knowledge_entries()
        
        print(f"📝 Total events logged: {len(events)}")
        print(f"💡 Total hypotheses: {len(hypotheses)}")
        print(f"🧠 Total knowledge entries: {len(knowledge_entries)}")
        
        # Get system report
        report = evaluation_framework.generate_system_report()
        print(f"📋 System report generated: {len(report)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Database analytics test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 ENHANCED PHYSICS FLOW SYSTEM TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Track test results
    tests_passed = 0
    total_tests = 6
    
    # Run tests
    if test_system_imports():
        tests_passed += 1
    
    if test_api_key_setup():
        tests_passed += 1
    
    if test_database_creation():
        tests_passed += 1
    
    if test_simple_physics_question():
        tests_passed += 1
    
    if test_complex_physics_question():
        tests_passed += 1
    
    if test_database_analytics():
        tests_passed += 1
    
    # Final results
    print("\n" + "=" * 60)
    print("🏁 TEST RESULTS")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Your enhanced physics system is working perfectly!")
        print("\n🚀 Ready for deployment or further testing!")
    else:
        print(f"⚠️ {total_tests - tests_passed} test(s) failed. Please check the errors above.")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 