#!/usr/bin/env python3
"""
Setup script for Collaborative PhysicsGPT system.
Ensures all dependencies and configurations are properly set up.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    """Print setup header."""
    print("=" * 80)
    print("🔧 Collaborative PhysicsGPT Setup")
    print("=" * 80)
    print("Setting up your multi-agent physics research system...")
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        print("⚠️  Python 3.8+ is required for the collaborative system")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_and_install_dependencies():
    """Check and install required dependencies."""
    print("\n📦 Checking dependencies...")
    
    # Core requirements
    requirements = [
        "streamlit>=1.28.0",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langgraph>=0.1.0",
        "plotly>=5.0.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0"
    ]
    
    missing_packages = []
    
    for requirement in requirements:
        package_name = requirement.split(">=")[0]
        try:
            __import__(package_name.replace("-", "_"))
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} (missing)")
            missing_packages.append(requirement)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {len(missing_packages)}")
        print("Installing missing dependencies...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    return True

def check_directory_structure():
    """Check if all required files and directories exist."""
    print("\n📁 Checking directory structure...")
    
    required_files = [
        "src/agents/__init__.py",
        "src/agents/collaborative_system.py",
        "src/agents/physics_expert.py",
        "src/agents/hypothesis_generator.py",
        "src/agents/supervisor.py",
        "src/tools/hypothesis_tools.py",
        "streamlit_collaborative.py",
        "demo_collaborative_physics.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {len(missing_files)}")
        print("Please ensure all collaborative system files are present.")
        return False
    
    return True

def create_env_file():
    """Create or check .env file for configuration."""
    print("\n🔧 Checking environment configuration...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("📋 Creating .env file from .env.example...")
            env_file.write_text(env_example.read_text())
            print("✅ .env file created")
        else:
            print("📋 Creating basic .env file...")
            env_content = """# Collaborative PhysicsGPT Configuration

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
MODEL_NAME=gpt-4
TEMPERATURE=0.7

# Memory Configuration
MEMORY_ENABLED=true
MEMORY_DB_PATH=physics_memory.db

# Collaboration Settings
DEFAULT_DIFFICULTY_LEVEL=undergraduate
DEFAULT_CREATIVITY_LEVEL=high
DEFAULT_COLLABORATION_STYLE=balanced

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
"""
            env_file.write_text(env_content)
            print("✅ Basic .env file created")
        
        print("⚠️  Please edit .env file and add your OpenAI API key")
    else:
        print("✅ .env file exists")
    
    return True

def test_imports():
    """Test if all collaborative system modules can be imported."""
    print("\n🧪 Testing module imports...")
    
    try:
        # Test basic imports
        from src.agents import CollaborativePhysicsSystem
        print("✅ CollaborativePhysicsSystem")
        
        from src.agents import PhysicsExpertAgent, HypothesisGeneratorAgent, SupervisorAgent
        print("✅ Individual agents")
        
        from src.tools.hypothesis_tools import get_hypothesis_tools
        print("✅ Hypothesis tools")
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("⚠️  Some modules may not be properly configured")
        return False

def create_test_script():
    """Create a simple test script to verify the system works."""
    print("\n📝 Creating test script...")
    
    test_script = """#!/usr/bin/env python3
'''
Quick test script for Collaborative PhysicsGPT.
Run this to verify the system is working properly.
'''

def test_collaborative_system():
    try:
        from src.agents import CollaborativePhysicsSystem
        
        print("🧪 Testing Collaborative PhysicsGPT...")
        
        # Initialize system
        system = CollaborativePhysicsSystem(
            difficulty_level="undergraduate",
            creativity_level="moderate",
            collaboration_style="balanced",
            memory_enabled=False  # Disable memory for testing
        )
        
        print("✅ System initialized successfully!")
        
        # Test agent status
        status = system.get_agent_status()
        print(f"✅ Agent status retrieved: {len(status)} agents ready")
        
        print("🎉 Collaborative PhysicsGPT is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_collaborative_system()
"""
    
    test_file = Path("test_collaborative.py")
    test_file.write_text(test_script)
    print("✅ Test script created: test_collaborative.py")

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 80)
    print("🎉 Setup Complete!")
    print("=" * 80)
    print("Next steps:")
    print()
    print("1. 🔑 Configure your API key:")
    print("   - Edit .env file and add your OpenAI API key")
    print("   - OPENAI_API_KEY=your_actual_api_key_here")
    print()
    print("2. 🧪 Test the system:")
    print("   - Run: python test_collaborative.py")
    print()
    print("3. 🚀 Launch the application:")
    print("   - Option A: python run_physics_app.py (launcher with menu)")
    print("   - Option B: streamlit run streamlit_collaborative.py (direct)")
    print()
    print("4. 📚 Explore features:")
    print("   - Try different collaboration modes")
    print("   - Experiment with agent settings")
    print("   - Test hypothesis generation and debates")
    print()
    print("🎯 Enjoy your multi-agent physics research system!")

def main():
    """Main setup function."""
    print_header()
    
    # Run all setup checks
    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_and_install_dependencies),
        ("Directory structure", check_directory_structure),
        ("Environment configuration", create_env_file),
        ("Module imports", test_imports),
    ]
    
    all_passed = True
    
    for check_name, check_function in checks:
        try:
            if not check_function():
                all_passed = False
                print(f"⚠️  {check_name} check failed")
        except Exception as e:
            print(f"❌ Error in {check_name} check: {e}")
            all_passed = False
    
    # Create test script regardless of check results
    create_test_script()
    
    if all_passed:
        print_next_steps()
    else:
        print("\n⚠️  Some setup checks failed. Please resolve the issues above.")
        print("You can still try running the system, but some features may not work.")

if __name__ == "__main__":
    main() 