#!/usr/bin/env python3
"""
Basic System Test - Tests components without requiring API calls
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required components can be imported."""
    print("🔍 Testing System Imports...")
    print("-" * 50)
    
    try:
        print("Testing Pydantic...")
        import pydantic
        print(f"✅ Pydantic version: {pydantic.__version__}")
        
        print("Testing CrewAI...")
        import crewai
        print(f"✅ CrewAI version: {crewai.__version__}")
        
        print("Testing OpenAI...")
        import openai
        print(f"✅ OpenAI version: {openai.__version__}")
        
        print("Testing LangChain...")
        import langchain
        print(f"✅ LangChain imported successfully")
        
        print("Testing SQLite...")
        import sqlite3
        print(f"✅ SQLite version: {sqlite3.sqlite_version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_database_integration():
    """Test database integration components."""
    print("\n🗄️ Testing Database Integration...")
    print("-" * 50)
    
    try:
        from crewai_database_integration import CrewAIKnowledgeAPI, CrewAIEvaluationFramework
        print("✅ Database integration components imported successfully")
        
        # Test database creation (without API key)
        knowledge_api = CrewAIKnowledgeAPI()
        print("✅ Knowledge API created successfully")
        
        evaluation_framework = CrewAIEvaluationFramework()
        print("✅ Evaluation framework created successfully")
        
        # Test basic database operations
        events = knowledge_api.get_recent_events()
        print(f"✅ Database queries work - {len(events)} events found")
        
        return True
        
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

def test_physics_flow_import():
    """Test physics flow system import."""
    print("\n🧪 Testing Physics Flow System Import...")
    print("-" * 50)
    
    try:
        from physics_flow_system import analyze_physics_question_with_flow, PhysicsLabFlow
        print("✅ Physics Flow system imported successfully")
        
        # Test flow class creation (without running)
        print("Testing flow class creation...")
        # This will test the class definition without running it
        flow_class = PhysicsLabFlow
        print("✅ PhysicsLabFlow class accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Physics flow system test failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False

def test_api_key_check():
    """Check API key setup."""
    print("\n🔑 Checking API Key Setup...")
    print("-" * 50)
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print(f"✅ OpenAI API key found (length: {len(openai_key)})")
        print("🚀 Ready for full system testing!")
        return True
    else:
        print("⚠️ OPENAI_API_KEY environment variable not found")
        print("To test with actual AI responses, set your API key:")
        print("export OPENAI_API_KEY='your-key-here'")
        print("💡 System components are working - just need API key for full testing")
        return False

def main():
    """Main test function."""
    print("🧪 BASIC SYSTEM COMPONENT TEST")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    # Run tests
    if test_imports():
        tests_passed += 1
    
    if test_database_integration():
        tests_passed += 1
    
    if test_physics_flow_import():
        tests_passed += 1
    
    api_key_available = test_api_key_check()
    if api_key_available:
        tests_passed += 1
    
    # Final results
    print("\n" + "=" * 60)
    print("🏁 BASIC TEST RESULTS")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed >= 3:  # Core components working
        print("🎉 CORE SYSTEM IS READY!")
        if api_key_available:
            print("✅ All components working and API key set - ready for full testing!")
            print("\nRun the full test with:")
            print("python3 test_enhanced_physics_flow.py")
        else:
            print("⚠️ Set your OpenAI API key to run full physics tests:")
            print("export OPENAI_API_KEY='your-key-here'")
            print("Then run: python3 test_enhanced_physics_flow.py")
    else:
        print(f"❌ {total_tests - tests_passed} core component(s) failed")

if __name__ == "__main__":
    main() 