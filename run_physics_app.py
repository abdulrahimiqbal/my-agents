#!/usr/bin/env python3
"""
PhysicsGPT Launcher - Choose between single-agent and collaborative systems.
"""

import subprocess
import sys
import os
from pathlib import Path

def print_banner():
    """Print the application banner."""
    print("=" * 80)
    print("🚀 PhysicsGPT Launcher")
    print("=" * 80)
    print("Choose your physics research experience:")
    print()

def print_options():
    """Print available options."""
    print("1. 🔬 Single Agent PhysicsGPT")
    print("   - Traditional single-agent physics expert")
    print("   - Comprehensive physics problem solving")
    print("   - Educational explanations and calculations")
    print("   - Command: streamlit run streamlit_app.py")
    print()
    
    print("2. 🤖 Collaborative PhysicsGPT (NEW!)")
    print("   - Multi-agent physics research system")
    print("   - Physics Expert + Hypothesis Generator + Supervisor")
    print("   - Multiple collaboration modes (research, debate, brainstorm)")
    print("   - Advanced session management and synthesis")
    print("   - Command: streamlit run streamlit_collaborative.py")
    print()
    
    print("3. 📊 Demo Collaborative System")
    print("   - Run demonstration of collaborative features")
    print("   - Shows system capabilities without UI")
    print("   - Command: python demo_collaborative_physics.py")
    print()
    
    print("4. ❌ Exit")
    print()

def run_streamlit_app(app_file: str, title: str):
    """Run a Streamlit app."""
    print(f"\n🚀 Launching {title}...")
    print(f"Command: streamlit run {app_file}")
    print("=" * 50)
    print("📝 Instructions:")
    print("- The app will open in your default web browser")
    print("- Use Ctrl+C to stop the application")
    print("- The app will be available at: http://localhost:8501")
    print("=" * 50)
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_file], check=True)
    except KeyboardInterrupt:
        print("\n\n✅ Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running Streamlit: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")
    except FileNotFoundError:
        print(f"\n❌ Error: {app_file} not found")
        print("Make sure you're running this script from the correct directory")

def run_demo():
    """Run the collaborative demo."""
    print("\n🚀 Running Collaborative PhysicsGPT Demo...")
    print("Command: python demo_collaborative_physics.py")
    print("=" * 50)
    print("📝 Note: This demo shows system architecture and capabilities")
    print("Some features may require proper environment setup and API keys")
    print("=" * 50)
    print()
    
    try:
        subprocess.run([sys.executable, "demo_collaborative_physics.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n✅ Demo stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running demo: {e}")
    except FileNotFoundError:
        print("\n❌ Error: demo_collaborative_physics.py not found")

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = ["streamlit", "langchain", "plotly"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available!")
    return True

def main():
    """Main launcher function."""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before continuing.")
        return
    
    print()
    
    while True:
        print_options()
        
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                run_streamlit_app("streamlit_app.py", "Single Agent PhysicsGPT")
                
            elif choice == "2":
                run_streamlit_app("streamlit_collaborative.py", "Collaborative PhysicsGPT")
                
            elif choice == "3":
                run_demo()
                
            elif choice == "4":
                print("\n👋 Thanks for using PhysicsGPT!")
                break
                
            else:
                print("\n❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                continue
            
            # Ask if user wants to continue
            print("\n" + "=" * 50)
            continue_choice = input("Would you like to run another option? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                print("\n👋 Thanks for using PhysicsGPT!")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using PhysicsGPT!")
            break
        except EOFError:
            print("\n\n👋 Thanks for using PhysicsGPT!")
            break

if __name__ == "__main__":
    main() 