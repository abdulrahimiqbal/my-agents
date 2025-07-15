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
