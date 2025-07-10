# My Agents Project

A production-ready, independent project for building sophisticated AI agents using LangChain and LangGraph. This project features both traditional single-agent systems and cutting-edge multi-agent collaboration frameworks, building upon the solid foundations from LangChain Academy modules.

## 🌟 Featured Systems

### 🔬 Single-Agent PhysicsGPT
Traditional physics expert agent with comprehensive problem-solving capabilities.

### 🤖 Collaborative PhysicsGPT (NEW!)
Multi-agent physics research system featuring:
- **Physics Expert Agent**: Rigorous scientific analysis and validation
- **Hypothesis Generator Agent**: Creative idea generation and research gap identification  
- **Supervisor Agent**: Orchestrates collaboration between agents
- **Multiple Collaboration Modes**: Research, debate, brainstorm, and teaching modes
- **Advanced Session Management**: Persistent context and synthesis capabilities

## 🏗️ Architecture

This project is designed with modularity and scalability in mind, incorporating best practices from LangChain Academy:

### Core Components

- **`src/agents/`** - Agent implementations with different specializations
- **`src/tools/`** - Reusable tool implementations for agent capabilities
- **`src/memory/`** - Persistent memory systems for conversation and context
- **`src/utils/`** - Common utilities and helper functions
- **`src/config/`** - Configuration management and settings
- **`studio/`** - LangGraph Studio integration for visual debugging
- **`tests/`** - Comprehensive test suite
- **`notebooks/`** - Jupyter notebooks for experimentation
- **`docs/`** - Documentation and guides

### LangChain Academy Integration

This project references and builds upon concepts from:

- **Module 1**: Basic chains, agents, and routing patterns
- **Module 2**: State management and memory systems
- **Module 3**: Breakpoints and human-in-the-loop workflows
- **Module 4**: Parallelization and sub-graph patterns
- **Module 5**: Advanced memory and personalization
- **Module 6**: Production deployment considerations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- LangChain/LangGraph
- OpenAI API key (or other LLM provider)

### Option 1: Easy Launcher (Recommended)
```bash
# Clone or copy this project folder
cd my-agents-project

# Install dependencies and set up the system
pip install -r requirements.txt
python setup_collaborative.py

# Launch with interactive menu
python run_physics_app.py
```

### Option 2: Direct Launch
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env file with your OpenAI API key

# Choose your experience:
# Single-agent PhysicsGPT
streamlit run streamlit_app.py

# Multi-agent Collaborative PhysicsGPT (NEW!)
streamlit run streamlit_collaborative.py

# Or try the collaborative demo
python demo_collaborative_physics.py
```

### Basic Usage Examples

#### Single-Agent System
```python
from src.agents import PhysicsExpertAgent

# Initialize physics expert
agent = PhysicsExpertAgent(difficulty_level="undergraduate")

# Solve a physics problem
response = agent.run("Explain quantum entanglement")
print(response)
```

#### Collaborative System
```python
from src.agents import CollaborativePhysicsSystem

# Initialize collaborative system
system = CollaborativePhysicsSystem(
    difficulty_level="undergraduate",
    creativity_level="high",
    collaboration_style="balanced"
)

# Start collaborative session
session = system.start_collaborative_session(
    topic="quantum computing applications",
    mode="research"
)

print(session["response"])
```

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### LangGraph Studio
```bash
cd studio
langgraph dev
```

### Jupyter Notebooks
```bash
jupyter lab notebooks/
```

## 📚 Documentation

- [Architecture Guide](docs/architecture.md)
- [Agent Development](docs/agents.md)
- [Tool Creation](docs/tools.md)
- [Memory Systems](docs/memory.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

This project follows the patterns and best practices established in LangChain Academy. When contributing:

1. Follow the established architecture patterns
2. Add comprehensive tests
3. Update documentation
4. Reference relevant Academy modules in comments

## 📄 License

MIT License - feel free to use this as a foundation for your own agent projects. 