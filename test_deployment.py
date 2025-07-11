#!/usr/bin/env python3
"""Test script to verify deployment readiness."""

def test_imports():
    """Test all critical imports."""
    try:
        from src.agents.collaborative_system import CollaborativePhysicsSystem
        from src.agents import PhysicsExpertAgent, HypothesisGeneratorAgent, SupervisorAgent
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_agent_compliance():
    """Test that all agents implement required methods."""
    from src.agents import PhysicsExpertAgent, HypothesisGeneratorAgent, SupervisorAgent
    
    agents = [PhysicsExpertAgent, HypothesisGeneratorAgent, SupervisorAgent]
    for agent_class in agents:
        if not hasattr(agent_class, 'get_agent_info'):
            print(f"❌ {agent_class.__name__} missing get_agent_info method")
            return False
    
    print("✅ All agents implement required methods")
    return True

def test_system_creation():
    """Test that the collaborative system can be created."""
    try:
        from src.agents.collaborative_system import CollaborativePhysicsSystem
        system = CollaborativePhysicsSystem()
        print("✅ Collaborative system creation successful")
        return True
    except Exception as e:
        print(f"❌ System creation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running deployment readiness tests...")
    
    tests = [test_imports, test_agent_compliance, test_system_creation]
    results = [test() for test in tests]
    
    if all(results):
        print("🎉 All tests passed! Ready for deployment.")
        exit(0)
    else:
        print("❌ Some tests failed. Fix issues before deploying.")
        exit(1)
