#!/usr/bin/env python3
"""
Test script for PhysicsGPT implementation
Run this to verify the physics agent is working correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from agents.physics_expert import PhysicsExpertAgent
        print("✅ PhysicsExpertAgent imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PhysicsExpertAgent: {e}")
        return False
    
    try:
        from tools.physics_calculator import get_physics_calculator_tools
        from tools.physics_constants import get_physics_constants_tools
        from tools.unit_converter import get_unit_converter_tools
        from tools.physics_research import get_physics_research_tools
        print("✅ All physics tools imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import physics tools: {e}")
        return False
    
    try:
        from config.settings import Settings
        from memory.memory_store import MemoryStore
        print("✅ Configuration and memory modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import config/memory modules: {e}")
        return False
    
    return True


def test_physics_tools():
    """Test physics tools functionality."""
    print("\n🔧 Testing physics tools...")
    
    try:
        from tools.physics_calculator import scientific_calculator, vector_operations
        from tools.physics_constants import get_physics_constant
        from tools.unit_converter import convert_units
        
        # Test calculator
        calc_result = scientific_calculator("2 + 3 * 4")
        print(f"✅ Calculator test: {calc_result}")
        
        # Test vector operations
        vector_result = vector_operations("magnitude", "3,4,5")
        print(f"✅ Vector test: {vector_result}")
        
        # Test constants
        constant_result = get_physics_constant("speed_of_light")
        print(f"✅ Constants test: {constant_result[:50]}...")
        
        # Test unit conversion
        conversion_result = convert_units(1.0, "m", "ft", "length")
        print(f"✅ Unit conversion test: {conversion_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Physics tools test failed: {e}")
        return False


def test_physics_agent_creation():
    """Test creating a physics agent."""
    print("\n🤖 Testing physics agent creation...")
    
    try:
        from agents.physics_expert import PhysicsExpertAgent
        
        # Create agent without memory for testing
        agent = PhysicsExpertAgent(
            difficulty_level="undergraduate",
            specialty="mechanics",
            memory_enabled=False
        )
        
        print("✅ PhysicsExpertAgent created successfully")
        print(f"   - Difficulty level: {agent.difficulty_level}")
        print(f"   - Specialty: {agent.specialty}")
        print(f"   - Memory enabled: {agent.memory_enabled}")
        
        return True
        
    except Exception as e:
        print(f"❌ Physics agent creation failed: {e}")
        return False


def test_environment():
    """Test environment setup."""
    print("\n🌍 Testing environment...")
    
    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
        
        # Check for OpenAI API key
        with open(env_file) as f:
            content = f.read()
            if "OPENAI_API_KEY=" in content and "your_openai_api_key_here" not in content:
                print("✅ OpenAI API key appears to be configured")
            else:
                print("⚠️  OpenAI API key not configured (needed for full functionality)")
    else:
        print("⚠️  .env file not found (will be created when running the app)")
    
    return True


def main():
    """Run all tests."""
    print("⚛️  PhysicsGPT Implementation Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_physics_tools,
        test_physics_agent_creation,
        test_environment
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PhysicsGPT is ready to use.")
        print("\n🚀 To run PhysicsGPT:")
        print("   python run_physics_gpt.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n💡 Common issues:")
        print("   - Missing dependencies: pip install -r requirements.txt")
        print("   - Python version: Requires Python 3.9+")
        print("   - API keys: Configure .env file")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 