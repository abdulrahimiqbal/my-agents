#!/bin/bash

echo "🧹 Cleaning up PhysicsGPT codebase for deployment..."
echo "📚 Keeping: Learning files, Arts2Survive docs, and core agent system"

# Remove backup and old projects (but keep main projects)
echo "Removing backup project files..."
rm -rf my-agents-project-backup-*/

# Remove test files
echo "Removing test files..."
rm -f test_*.py debug_*.py run_comprehensive_tests.py

# Remove old/unused main files (keep physics_crew_system.py)
echo "Removing old main files..."
rm -f main.py main_real.py physics_crew_concept.py explore_database.py

# Remove result and log files
echo "Removing result and log files..."
rm -f *.json *.log

# Remove extra documentation (keep README.md and learnings.md)
echo "Removing extra documentation..."
rm -f DATABASE_EXPLORATION_GUIDE.md DEPLOYMENT_BEST_PRACTICES.md "handy commands.md"

# Remove UI files (we'll create new ones)
echo "Removing old UI files..."
rm -f streamlit_*.py

# Remove shell scripts
echo "Removing shell scripts..."
rm -f setup_advanced_system.sh

# Remove cache and system files
echo "Removing cache files..."
rm -rf __pycache__/ .DS_Store
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Remove data directory (we'll recreate if needed)
echo "Removing data directory..."
rm -rf data/

# Rename .env.local to .env
echo "Setting up environment file..."
if [ -f ".env.local" ]; then
    mv .env.local .env
    echo "✅ Renamed .env.local to .env"
fi

# Create deployment README
echo "Creating deployment README..."
cat > DEPLOYMENT_README.md << 'EOF'
# PhysicsGPT - 10-Agent Physics Research System

## Quick Start

1. **Setup Environment:**
   ```bash
   source physics-env/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure API Keys:**
   Edit `.env` file with your API keys:
   - OPENAI_API_KEY
   - TAVILY_API_KEY (optional)
   - LANGCHAIN_API_KEY (optional)

3. **Run the System:**
   ```bash
   python physics_crew_system.py
   ```

## Usage Options

- **Command Line:** `python physics_crew_system.py "Your physics question"`
- **Interactive Mode:** Run without arguments for menu
- **5 Core Agents:** Default balanced analysis
- **10 Agent Demo:** Full comprehensive analysis
- **Custom Agents:** Select specific agents for targeted analysis

## Available Agents

1. **physics_expert** - Rigorous theoretical analysis
2. **hypothesis_generator** - Creative hypotheses
3. **mathematical_analyst** - Mathematical frameworks
4. **experimental_designer** - Experimental approaches
5. **pattern_analyst** - Pattern recognition
6. **quantum_specialist** - Quantum mechanics expertise
7. **relativity_expert** - Relativity and cosmology
8. **condensed_matter_expert** - Materials and many-body systems
9. **computational_physicist** - Numerical methods and simulations
10. **physics_communicator** - Clear explanations and education

## System Requirements

- Python 3.8+
- OpenAI API access
- 2GB+ RAM for full 10-agent analysis
- Internet connection for API calls
EOF

echo ""
echo "✅ Cleanup complete! Your deployment-ready structure:"
echo ""
ls -la
echo ""
echo "📦 Ready for deployment with minimal footprint!"
echo "🚀 Run: source physics-env/bin/activate && python physics_crew_system.py"