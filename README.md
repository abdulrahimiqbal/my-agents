# ⚛️ PhysicsGPT - Multi-Agent Physics Research System

A sophisticated multi-agent system powered by CrewAI that brings together 10 specialized AI physics experts to analyze complex physics questions and generate comprehensive research insights.

## 🌟 Features

### 🤖 10 Specialized Physics Agents
- **Physics Expert** - Senior theoretical physicist for rigorous analysis
- **Hypothesis Generator** - Creative researcher generating novel hypotheses
- **Mathematical Analyst** - Quantitative frameworks and mathematical modeling
- **Experimental Designer** - Practical experimental design and testing
- **Pattern Analyst** - Data relationships and pattern recognition
- **Quantum Specialist** - Quantum mechanics and quantum computing expertise
- **Relativity Expert** - Spacetime physics and cosmology
- **Condensed Matter Expert** - Materials science and solid-state physics
- **Computational Physicist** - Numerical methods and simulations
- **Physics Communicator** - Clear explanations and educational content

### 🎯 Multiple Analysis Modes
- **5 Core Agents** - Balanced analysis with key specialists
- **Custom Selection** - Choose specific agents for targeted analysis
- **Full 10-Agent Swarm** - Maximum depth with all specialists

### 📊 Real-Time Visualization
- **Agent Timeline** - See when each agent is active
- **Interaction Cards** - View agent thoughts and outputs
- **Live Updates** - Watch the analysis unfold in real-time

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Streamlit

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/physicsgpt.git
cd physicsgpt
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

4. **Run the application**
```bash
# Simple interface
streamlit run streamlit_simple_app.py

# Advanced physics interface
streamlit run streamlit_physics_app.py

# Agent visualization interface
streamlit run streamlit_agent_viz.py
```

## 🎮 Usage

### Basic Physics Analysis
1. Open the Streamlit app
2. Enter your physics question
3. Select analysis mode (5 agents, custom, or all 10)
4. Click "Analyze" and watch the agents collaborate

### Example Questions
- "What is dark matter and how do we detect it?"
- "How does quantum entanglement work in quantum computing?"
- "Explain the relationship between entropy and information theory"
- "How do gravitational waves propagate through spacetime?"
- "What are the best ideas to create a multiagent swarm for physics discovery?"

### Agent Visualization
The visualization interface shows:
- **Timeline Chart** - When each agent starts and finishes
- **Agent Cards** - Real-time status and thoughts
- **Output Sections** - Detailed results from each agent

## 🏗️ Architecture

### Core Components
- `physics_crew_system.py` - Main CrewAI orchestration
- `streamlit_*.py` - User interfaces
- `agent_callback.py` - Real-time interaction tracking
- `src/agents/` - Individual agent implementations
- `src/config/` - Configuration and settings

### Agent Workflow
1. **Query Processing** - Parse and understand the physics question
2. **Agent Selection** - Choose appropriate specialists
3. **Parallel Analysis** - Agents work simultaneously on different aspects
4. **Synthesis** - Combine insights into comprehensive analysis
5. **Communication** - Present results in accessible format

## 🔧 Configuration

### Agent Selection
Customize which agents participate in analysis:
```python
# Use specific agents
selected_agents = ['physics_expert', 'quantum_specialist', 'mathematical_analyst']

# Or use predefined sets
crew.analyze_physics_query(query, agents_to_use=selected_agents)
```

### Model Configuration
Adjust AI model settings in `src/config/semantic_config.py`:
```python
MODEL_CONFIG = {
    "model": "gpt-4",
    "temperature": 0.1,
    "max_tokens": 2000
}
```

## 📁 Project Structure

```
physicsgpt/
├── README.md
├── requirements.txt
├── .env.example
├── physics_crew_system.py      # Main system
├── agent_callback.py           # Interaction tracking
├── streamlit_simple_app.py     # Basic interface
├── streamlit_physics_app.py    # Advanced interface
├── streamlit_agent_viz.py      # Visualization interface
├── src/
│   ├── agents/                 # Individual agent implementations
│   └── config/                 # Configuration files
└── docs/                       # Documentation
```

## 🚀 Deployment

### Streamlit Cloud
1. Fork this repository
2. Connect to Streamlit Cloud
3. Add your OpenAI API key to secrets
4. Deploy with one click

### Docker
```bash
docker build -t physicsgpt .
docker run -p 8501:8501 physicsgpt
```

### Railway/Heroku
See `DEPLOYMENT_GUIDE.md` for detailed deployment instructions.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest

# Format code
black .
```

## 📊 Performance

- **5-Agent Mode**: ~30-60 seconds analysis time
- **10-Agent Mode**: ~60-120 seconds analysis time
- **Memory Usage**: ~500MB-1GB depending on mode
- **API Calls**: 10-50 OpenAI API calls per analysis

## 🔒 Security & Privacy

- API keys stored securely in environment variables
- No physics data stored permanently
- All processing happens in real-time
- Open source and transparent

## 📚 Documentation

- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Detailed deployment instructions
- [Agent Architecture](docs/agents.md) - How agents work together
- [API Reference](docs/api.md) - Programming interface
- [Examples](docs/examples.md) - Sample analyses and use cases

## 🐛 Troubleshooting

### Common Issues

**"Failed to initialize PhysicsGPT"**
- Check your OpenAI API key is set correctly
- Ensure all dependencies are installed

**"Analysis taking too long"**
- Try using fewer agents (5-agent mode)
- Check your internet connection
- Verify API rate limits

**"Import errors"**
- Run `pip install -r requirements.txt`
- Check Python version (3.8+ required)

## 📈 Roadmap

- [ ] Add more specialized physics domains
- [ ] Implement agent memory and learning
- [ ] Add export functionality for analyses
- [ ] Create API endpoints for integration
- [ ] Add support for physics equations and LaTeX
- [ ] Implement collaborative filtering for agent selection

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [CrewAI](https://github.com/joaomdmoura/crewAI)
- Powered by [OpenAI GPT-4](https://openai.com)
- UI created with [Streamlit](https://streamlit.io)
- Visualization using [Plotly](https://plotly.com)

## 📞 Support

- 📧 Email: support@physicsgpt.com
- 💬 Discord: [Join our community](https://discord.gg/physicsgpt)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/physicsgpt/issues)
- 📖 Docs: [Full Documentation](https://docs.physicsgpt.com)

---

**Made with ⚛️ by the PhysicsGPT Team**

*Advancing physics research through collaborative AI*