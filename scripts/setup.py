#!/usr/bin/env python3
"""
Setup script for My Agents Project.

This script automates the initial setup of the agents project including:
- Environment setup
- Database initialization
- Dependency verification
- Configuration validation
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")


def print_step(text):
    """Print a formatted step."""
    print(f"\n🔧 {text}")


def print_success(text):
    """Print a success message."""
    print(f"✅ {text}")


def print_warning(text):
    """Print a warning message."""
    print(f"⚠️  {text}")


def print_error(text):
    """Print an error message."""
    print(f"❌ {text}")


def check_python_version():
    """Check if Python version is compatible."""
    print_step("Checking Python version...")
    
    if sys.version_info < (3, 11):
        print_error(f"Python 3.11+ required, but you have {sys.version_info.major}.{sys.version_info.minor}")
        return False
    
    print_success(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} is compatible")
    return True


def create_directories():
    """Create necessary directories."""
    print_step("Creating necessary directories...")
    
    directories = [
        "data",
        "logs", 
        "data/chroma",
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print_success(f"Created directory: {directory}")


def check_environment_file():
    """Check if .env file exists and help create it."""
    print_step("Checking environment configuration...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print_warning(".env file not found. Copying from .env.example...")
            with open(env_example, 'r') as src, open(env_file, 'w') as dst:
                dst.write(src.read())
            print_success("Created .env file from template")
            print_warning("Please edit .env file with your actual API keys!")
        else:
            print_error(".env.example file not found!")
            return False
    else:
        print_success(".env file exists")
    
    return True


def install_dependencies():
    """Install Python dependencies."""
    print_step("Installing Python dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        print_success("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False


def verify_imports():
    """Verify that key packages can be imported."""
    print_step("Verifying package imports...")
    
    packages = [
        ("langchain", "LangChain"),
        ("langgraph", "LangGraph"),
        ("langchain_openai", "LangChain OpenAI"),
        ("sqlite3", "SQLite3"),
        ("pydantic", "Pydantic"),
    ]
    
    all_good = True
    for package, name in packages:
        try:
            __import__(package)
            print_success(f"{name} import successful")
        except ImportError:
            print_error(f"{name} import failed")
            all_good = False
    
    return all_good


def initialize_database():
    """Initialize the database."""
    print_step("Initializing database...")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from src.memory import MemoryStore
        from src.config import get_settings
        
        # Initialize memory store (this will create the database)
        settings = get_settings()
        memory = MemoryStore()
        
        print_success("Database initialized successfully")
        return True
    except Exception as e:
        print_error(f"Database initialization failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality."""
    print_step("Testing basic functionality...")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from src.agents import ChatAgent
        from src.tools import get_basic_tools
        
        # Test agent creation
        agent = ChatAgent(enable_basic_tools=False)  # Disable tools to avoid API calls
        print_success("Agent creation successful")
        
        # Test tools
        tools = get_basic_tools()
        print_success(f"Tools loaded successfully ({len(tools)} tools)")
        
        return True
    except Exception as e:
        print_error(f"Basic functionality test failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print_header("Setup Complete! Next Steps")
    
    print("""
🎉 Your agents project is ready to use!

Next steps:
1. Edit .env file with your API keys:
   - OPENAI_API_KEY (required for LLM)
   - TAVILY_API_KEY (optional, for web search)
   - LANGCHAIN_API_KEY (optional, for tracing)

2. Try the basic usage:
   ```python
   from src.agents import ChatAgent
   agent = ChatAgent()
   response = agent.chat("Hello!")
   print(response)
   ```

3. Use LangGraph Studio for visual debugging:
   ```bash
   cd studio
   langgraph dev
   ```

4. Run tests:
   ```bash
   pytest tests/
   ```

5. Explore notebooks:
   ```bash
   jupyter lab notebooks/
   ```

📚 Check the README.md and docs/ folder for more information!
    """)


def main():
    """Main setup function."""
    print_header("My Agents Project Setup")
    
    # Change to project directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print(f"Setting up project in: {project_root.absolute()}")
    
    # Run setup steps
    steps = [
        ("Python version check", check_python_version),
        ("Directory creation", create_directories),
        ("Environment file check", check_environment_file),
        ("Dependency installation", install_dependencies),
        ("Import verification", verify_imports),
        ("Database initialization", initialize_database),
        ("Basic functionality test", test_basic_functionality),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        if not step_func():
            failed_steps.append(step_name)
    
    if failed_steps:
        print_header("Setup Issues")
        print_error("The following steps failed:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nPlease resolve these issues and run setup again.")
        sys.exit(1)
    else:
        print_next_steps()


if __name__ == "__main__":
    main() 