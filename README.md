# My Agents Project

A production-ready, independent project for building sophisticated AI agents using LangChain and LangGraph. This project builds upon the solid foundations from LangChain Academy modules while providing a clean, extensible architecture for real-world agent applications.

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

- Python 3.11+
- LangChain/LangGraph
- OpenAI API key (or other LLM provider)

### Installation

1. Clone or copy this project folder
2. Install dependencies:
   ```bash
   cd my-agents-project
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Run the setup script:
   ```bash
   python scripts/setup.py
   ```

### Basic Usage

```python
from src.agents import ChatAgent
from src.memory import MemoryStore

# Initialize agent with memory
memory = MemoryStore()
agent = ChatAgent(memory=memory)

# Chat with the agent
response = agent.chat("Hello! Can you help me with a task?")
print(response)
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