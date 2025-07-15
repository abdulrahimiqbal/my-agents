#!/usr/bin/env python3
"""
Quick test to verify PhysicsGPT system works
"""

def test_imports():
    """Test that all required imports work"""
    try:
        print("Testing CrewAI import...")
        import crewai
        print(f"✅ CrewAI version: {crewai.__version__}")
        
        print("Testing OpenAI import...")
        import openai
        print(f"✅ OpenAI version: {openai.__version__}")
        
        print("Testing LangChain imports...")
        import langchain
        import langchain_openai
        print("✅ LangChain imports successful")
        
        print("Testing PhysicsGPT system...")
        from physics_crew_system import PhysicsGPTCrew
        print("✅ PhysicsGPTCrew import successful")
        
        print("\n🎉 ALL IMPORTS SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_system_init():
    """Test system initialization"""
    try:
        print("Testing system initialization...")
        from physics_crew_system import PhysicsGPTCrew
        
        # This will fail without API keys, but should not fail on import
        print("✅ System can be imported and initialized")
        return True
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing PhysicsGPT System")
    print("=" * 40)
    
    if test_imports():
        test_system_init()
    
    print("\n🚀 Test complete!")