# PhysicsGPT - Simplified Multi-Agent Physics Research System

A streamlined, production-ready multi-agent physics research system focused on core agent functionality without UI complexity. This system demonstrates sophisticated AI agent collaboration through a simple command-line interface.

## 🌟 **Core Features**

### 🤖 **Advanced Multi-Agent Architecture**
- **AdvancedSupervisorAgent**: Intelligent orchestration with routing algorithms and consensus detection
- **EnhancedPhysicsExpertAgent**: Specialized physics analysis with domain expertise
- **HypothesisGeneratorAgent**: Creative hypothesis generation and research gap identification
- **MathematicalAnalysisAgent**: Advanced mathematical computations and modeling
- **PatternRecognitionAgent**: Pattern analysis and relationship discovery

### 📊 **Comprehensive Journey Logging**
- Complete query journey tracking from input to consensus
- Agent activity monitoring and performance metrics
- Tool usage tracking and resource utilization
- Detailed execution timelines and collaboration patterns

### 🧠 **Intelligent Systems**
- **Knowledge Graph**: Physics concepts and relationships
- **Semantic Search**: Intelligent knowledge retrieval
- **Learning System**: Adaptive behavior based on interaction history
- **Memory Store**: Persistent conversation and context management

### 🛠️ **Physics Tools Suite**
- Advanced physics calculations and constants
- Unit conversion and dimensional analysis
- ArXiv research integration
- Web search capabilities
- Hypothesis generation and evaluation tools

## 🚀 **Quick Start**

### **Installation**
```bash
# Clone the repository
git clone https://github.com/abdulrahimiqbal/my-agents.git
cd my-agents-project

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your OpenAI API key and other configurations
```

### **Basic Usage**

#### **1. Interactive Mode**
```bash
python main.py
```

#### **2. Direct Query**
```bash
python main.py "Explain quantum entanglement and its applications"
```

#### **3. Demo Queries**
```bash
python test_query_system.py --demo
```

#### **4. Advanced Testing**
```bash
python test_query_system.py "How do black holes form?" --log-level verbose
```

## 🏗️ **System Architecture**

```
my-agents-project/
├── main.py                    # ✨ Simple entry point
├── test_query_system.py       # 🧪 Comprehensive test system
├── requirements.txt           # 📦 Core dependencies
├── src/                       # 🏗️ Core agent architecture
│   ├── agents/               # 🤖 Agent implementations
│   │   ├── advanced_supervisor.py    # Main orchestrator
│   │   ├── enhanced_physics_expert.py
│   │   ├── hypothesis_generator.py
│   │   ├── mathematical_analysis.py
│   │   ├── pattern_recognition.py
│   │   └── parallel_orchestrator.py
│   ├── tools/                # 🛠️ Physics tools
│   │   ├── physics_calculator.py
│   │   ├── physics_constants.py
│   │   ├── unit_converter.py
│   │   └── physics_research.py
│   ├── knowledge/            # 🧠 Knowledge systems
│   │   ├── knowledge_graph.py
│   │   └── semantic_search.py
│   ├── memory/               # 💾 Memory systems
│   │   └── stores.py
│   ├── config/               # ⚙️ Configuration
│   │   └── settings.py
│   └── database/             # 🗄️ Data management
│       ├── knowledge_api.py
│       └── migrations.py
├── tests/                    # 🧪 Core tests
│   ├── test_advanced_supervisor.py
│   ├── test_enhanced_physics_expert.py
│   ├── test_knowledge_graph.py
│   └── test_agents.py
└── data/                     # 📊 Data storage
    ├── memory.db
    └── knowledge_graph.db
```

## 🎯 **Usage Examples**

### **Example 1: Basic Physics Query**
```bash
python main.py "What is the uncertainty principle?"
```

**Output:**
```
🔬 Testing Physics Query: What is the uncertainty principle?
📊 Agent Activities: 3 agents collaborating
🛠️ Tools Used: physics_concept_search, generate_creative_hypotheses
📊 Consensus Confidence: 0.87
🤝 Agreement Level: 0.85
```

### **Example 2: Advanced Query with Detailed Logging**
```bash
python test_query_system.py "How does quantum tunneling work?" --log-level verbose
```

**Shows detailed journey:**
- Agent selection and routing decisions
- Tool usage by each agent
- Inter-agent collaboration patterns
- Consensus building process
- Performance metrics and timings

### **Example 3: Demo Mode**
```bash
python test_query_system.py --demo
```

**Runs 5 demonstration queries:**
- Quantum entanglement applications
- Black hole formation
- Energy-mass relationship
- Wave-particle duality
- Uncertainty principle effects

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Required
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here

# Optional
LANGCHAIN_API_KEY=your_langchain_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=physics-agents

# System Configuration
DEFAULT_MODEL=gpt-4o-mini
DEFAULT_TEMPERATURE=0.7
DEBUG=false
```

### **Advanced Configuration**
Edit `src/config/settings.py` for:
- Model selection and parameters
- Database configurations
- Memory settings
- Tool configurations

## 🧪 **Testing**

### **Run Core Tests**
```bash
# Test advanced supervisor
python -m pytest tests/test_advanced_supervisor.py -v

# Test physics expert
python -m pytest tests/test_enhanced_physics_expert.py -v

# Test knowledge graph
python -m pytest tests/test_knowledge_graph.py -v

# Run all tests
python -m pytest tests/ -v
```

### **Manual Testing**
```bash
# Test specific functionality
python test_query_system.py "your test query here"

# Test with different log levels
python test_query_system.py "query" --log-level basic|detailed|verbose
```

## 📊 **Performance Metrics**

The system tracks comprehensive metrics:
- **Execution Time**: Total query processing time
- **Agent Utilization**: Number of agents used
- **Tool Usage**: Tools called by each agent
- **Consensus Metrics**: Agreement levels and confidence scores
- **Memory Usage**: Database operations and storage
- **Learning Metrics**: Interaction recording and adaptation

## 🔄 **Agent Collaboration Flow**

1. **Query Analysis**: AdvancedSupervisorAgent analyzes incoming query
2. **Agent Selection**: Intelligent routing based on query type and agent capabilities
3. **Parallel Processing**: Multiple agents work simultaneously on different aspects
4. **Tool Utilization**: Agents use specialized physics tools as needed
5. **Consensus Building**: System combines agent responses using various methods
6. **Learning**: Interaction recorded for system improvement
7. **Response**: Synthesized answer with journey log and metrics

## 🛠️ **Development**

### **Adding New Agents**
1. Extend `BaseAgent` class
2. Implement specialized tools
3. Register with `AdvancedSupervisorAgent`
4. Add appropriate tests

### **Adding New Tools**
1. Create tool in `src/tools/`
2. Follow LangChain tool patterns
3. Register with appropriate agents
4. Add documentation and tests

### **Database Management**
```bash
# Run migrations
python -c "from src.database.migrations import run_migration; run_migration()"

# Check database status
python -c "from src.database.knowledge_api import KnowledgeAPI; api = KnowledgeAPI(); print(api.get_system_analytics())"
```

## 📈 **System Capabilities**

### **Physics Domains Covered**
- Quantum Mechanics
- Thermodynamics
- Electromagnetism
- Classical Mechanics
- Relativity
- Statistical Physics
- Condensed Matter Physics

### **Analysis Types**
- Theoretical analysis
- Mathematical modeling
- Hypothesis generation
- Experimental design
- Literature review
- Concept explanation

### **Advanced Features**
- Multi-agent consensus building
- Parallel processing
- Adaptive learning
- Knowledge graph integration
- Semantic search
- Performance optimization

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 **Related Projects**

- [LangChain Academy](https://github.com/langchain-ai/langchain-academy) - Foundation patterns
- [LangGraph](https://github.com/langchain-ai/langgraph) - Multi-agent framework
- [Physics Research Tools](https://github.com/abdulrahimiqbal/physics-tools) - Specialized tools

---

**Built with ❤️ using LangChain, LangGraph, and modern AI agent patterns**
