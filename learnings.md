# LangChain Academy - Setup & Learnings

## Quick Reference Commands

### Environment Setup (Required Every Time)
```bash
# Navigate to project root
cd /Users/rahim/Desktop/langchain-academy

# Activate virtual environment
source lc-academy-env/bin/activate

# Set required environment variables
export OPENAI_API_KEY="your-openai-api-key-here"
export LANGSMITH_API_KEY="your-langsmith-api-key-here"
export LANGSMITH_TRACING=true
```

### Starting LangGraph for Different Modules

#### Module 1 (Simple Graph, Router, Agent)
```bash
cd module-1/studio
langgraph dev
```

#### Module 2 (Chatbot, State Management)
```bash
cd module-2/studio
langgraph dev
```

#### Module 3 (Breakpoints, Human Feedback)
```bash
cd module-3/studio
langgraph dev
```

#### Module 4 (Map-Reduce, Parallelization)
```bash
cd module-4/studio
langgraph dev
```

#### Module 5 (Memory Agent)
```bash
cd module-5/studio
langgraph dev
```

#### Module 6 (Task Maistro - Advanced)
```bash
cd module-6/deployment
langgraph dev
```

## Key Learnings

### 1. Environment Variables Are Critical
- **MUST** be set in the same shell session where you run `langgraph dev`
- Environment variables don't persist across shell sessions
- LangGraph processes inherit environment variables from the parent shell

### 2. Directory Structure Matters
- LangGraph **MUST** be run from directories containing `langgraph.json`
- Each module has its own configuration in `/studio/` or `/deployment/` folders
- Running from wrong directory gives: `Error: Invalid value for '--config': Path 'langgraph.json' does not exist.`

### 3. Virtual Environment Required
- Always activate `lc-academy-env` before running LangGraph
- Without activation: `zsh: command not found: langgraph`

### 4. API Key Validation
- OpenAI keys start with `your-openai-api-key-here` or `sk-`
- LangSmith keys start with `your-langsmith-api-key-here`
- Test connection: `python3 -c "from openai import OpenAI; OpenAI().chat.completions.create(model='gpt-4o', messages=[{'role': 'user', 'content': 'test'}], max_tokens=10)"`

## Available Endpoints When Running
- 🚀 **API**: http://127.0.0.1:2024
- 🎨 **Studio UI**: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 **API Docs**: http://127.0.0.1:2024/docs

## Common Issues & Solutions

### Issue: "command not found: langgraph"
**Solution**: Activate virtual environment first
```bash
source lc-academy-env/bin/activate
```

### Issue: "Path 'langgraph.json' does not exist"
**Solution**: Navigate to correct module directory
```bash
cd module-X/studio  # or module-6/deployment
```

### Issue: "OpenAI API key not set"
**Solution**: Export environment variables in same shell
```bash
export OPENAI_API_KEY="your-key-here"
```

### Issue: "Failed to fetch" in Studio UI
**Solution**: 
1. Kill existing processes: `ps aux | grep langgraph | awk '{print $2}' | xargs kill`
2. Set environment variables
3. Start from correct directory

### Issue: "You exceeded your current quota"
**Solution**: Add credits to OpenAI account at https://platform.openai.com/account/billing

## Module Configurations

### Module 1 - Available Graphs:
- `simple_graph` - Basic LangGraph example
- `router` - Routing logic
- `agent` - Agent implementation

### Module 6 - Available Graphs:
- `task_maistro` - Advanced task management system

## Process Management

### Check if LangGraph is running:
```bash
ps aux | grep "langgraph dev"
```

### Kill existing LangGraph processes:
```bash
ps aux | grep "langgraph dev" | grep -v grep | awk '{print $2}' | xargs kill
```

### Test if server is responding:
```bash
curl -s http://127.0.0.1:2024/docs | head -n 1
```

## Complete Startup Sequence (Template)

```bash
# 1. Navigate to project
cd /Users/rahim/Desktop/langchain-academy

# 2. Kill any existing processes
ps aux | grep "langgraph dev" | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null

# 3. Navigate to desired module
cd module-1/studio  # Change as needed

# 4. Activate environment and set variables
source ../../lc-academy-env/bin/activate
export OPENAI_API_KEY="your-openai-api-key-here"
export LANGSMITH_API_KEY="your-langsmith-api-key-here"
export LANGSMITH_TRACING=true

# 5. Start LangGraph
langgraph dev
```

## Success Indicators

When LangGraph starts successfully, you'll see:
```
Welcome to
╦  ┌─┐┌┐┌┌─┐╔═╗┬─┐┌─┐┌─┐┬ ┬
║  ├─┤││││ ┬║ ╦├┬┘├─┤├─┘├─┤
╩═╝┴ ┴┘└┘└─┘╚═╝┴└─┴ ┴┴  ┴ ┴

- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs

Server started in X.XXs
🎨 Opening Studio in your browser...
```

## Notes
- Environment variables must be re-exported in each new shell session
- Each module can only run one at a time (they all use port 2024)
- LangSmith integration is working when you see metadata submission logs
- Studio UI automatically opens in browser when server starts successfully 