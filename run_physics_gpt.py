#!/usr/bin/env python3
"""
PhysicsGPT Runner Script
Simple script to set up and run the PhysicsGPT Streamlit application.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")


def check_environment():
    """Check if .env file exists and has required variables."""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("⚠️  .env file not found. Creating template...")
        create_env_template()
        print("📝 Please edit .env file with your API keys and run again.")
        return False
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    with open(env_file) as f:
        content = f.read()
        for var in required_vars:
            if f"{var}=" not in content or f"{var}=your_" in content:
                missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing or incomplete environment variables: {', '.join(missing_vars)}")
        print("📝 Please update your .env file with valid API keys.")
        return False
    
    print("✅ Environment variables configured")
    return True


def create_env_template():
    """Create a template .env file."""
    template = """# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Tavily for web search
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: LangSmith for tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=physics-gpt

# Agent Configuration
PHYSICS_AGENT_TEMPERATURE=0.1
PHYSICS_AGENT_MODEL=gpt-4o-mini
"""
    
    with open(".env", "w") as f:
        f.write(template)


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def run_streamlit():
    """Run the Streamlit application."""
    print("🚀 Starting PhysicsGPT...")
    print("📱 Opening in your default browser...")
    print("🔗 URL: http://localhost:8501")
    print("\n" + "="*50)
    print("🎯 PhysicsGPT Features:")
    print("• 🧮 Advanced physics calculator")
    print("• 📚 Physics constants database")
    print("• 🔄 Comprehensive unit converter")
    print("• 📖 Physics equation lookup")
    print("• 🔍 ArXiv research search")
    print("• 💬 Interactive physics chat")
    print("• 🎓 Multi-level explanations")
    print("="*50 + "\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], 
                      check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 PhysicsGPT stopped by user")


def main():
    """Main function to set up and run PhysicsGPT."""
    print("⚛️  PhysicsGPT Setup & Runner")
    print("="*40)
    
    # Check Python version
    check_python_version()
    
    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies. Please check the error messages above.")
        sys.exit(1)
    
    # Run the application
    run_streamlit()


if __name__ == "__main__":
    main() 