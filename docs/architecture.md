# Architecture Documentation

## Overview

This project implements a production-ready LangGraph agents system with modular architecture, comprehensive testing, and easy extensibility.

## Project Structure

### Source Code (`src/`)

```
src/
├── agents/          # Agent implementations
├── tools/           # Custom tools for agents
├── memory/          # Memory management and persistence
├── utils/           # Utility functions and helpers
└── config/          # Configuration and settings
```

### Key Components

#### 1. Agents (`src/agents/`)

- **BaseAgent**: Abstract base class providing common functionality
  - LLM initialization and configuration
  - Memory management (SQLite checkpointing)
  - Synchronous and asynchronous execution
  - Streaming capabilities
  - Thread-based conversation management

- **ChatAgent**: Simple conversational agent with tool capabilities
  - Calculator tools integration
  - Customizable system messages
  - Memory persistence

#### 2. Tools (`src/tools/`)

- **Calculator Tools**: Mathematical operations (add, subtract, multiply, divide, power, square_root)
- **Web Search Tools**: Tavily-powered web search capabilities
- Extensible design for adding new tools

#### 3. Memory (`src/memory/`)

- **MemoryStore**: Enhanced memory management beyond basic checkpointing
  - Conversation summaries
  - User preferences storage
  - Conversation metadata
  - SQLite-based persistence

#### 4. Configuration (`src/config/`)

- **Settings**: Pydantic-based configuration management
  - Environment variable loading
  - API key management
  - Model configuration
  - Database settings

## Design Patterns

### 1. Abstract Base Class Pattern

The `BaseAgent` class provides a template for all agents, ensuring consistent:
- Initialization procedures
- Graph building workflows
- Execution interfaces
- Memory management

### 2. Tool Composition Pattern

Tools are modular and composable:
```python
tools = get_calculator_tools() + get_web_search_tools()
```

### 3. Configuration Pattern

Centralized configuration using Pydantic:
```python
from src.config.settings import settings
```

### 4. Factory Pattern

Tool factories provide easy tool set management:
```python
def get_calculator_tools() -> List:
    return [add, subtract, multiply, divide, power, square_root]
```

## LangGraph Integration

### Graph Structure

All agents follow a standard LangGraph pattern:

```
START → Assistant → Tools (conditional) → Assistant → END
                 ↘ END (if no tools needed)
```

### State Management

- Uses `MessagesState` for conversation flow
- Supports custom state schemas for complex agents
- Automatic checkpointing for memory persistence

### Tool Integration

- Uses `ToolNode` for tool execution
- `tools_condition` for conditional tool routing
- Automatic tool binding to LLMs

## Memory Architecture

### Three-Layer Memory System

1. **Checkpointing Layer**: LangGraph's built-in conversation persistence
2. **Summary Layer**: Conversation summaries for long-term context
3. **Metadata Layer**: User preferences and conversation metadata

### Database Schema

```sql
-- LangGraph checkpoints (automatic)
checkpoints
checkpoint_blobs
checkpoint_writes

-- Custom memory tables
conversation_summaries
user_preferences
conversation_metadata
```

## Testing Strategy

### Unit Tests
- Tool functionality testing
- Agent initialization testing
- Configuration validation

### Integration Tests
- End-to-end agent workflows
- Memory persistence testing
- Tool integration testing

### Mocking Strategy
- LLM responses mocked for deterministic testing
- Environment variable mocking
- Database mocking for isolated tests

## Extensibility

### Adding New Agents

1. Inherit from `BaseAgent`
2. Implement `_build_graph()` method
3. Add to studio configuration
4. Write tests

Example:
```python
class MyCustomAgent(BaseAgent):
    def _build_graph(self):
        # Build your custom graph
        pass
```

### Adding New Tools

1. Create tool functions with `@tool` decorator
2. Add to appropriate tool module
3. Create factory function
4. Update agent configurations

### Adding New Memory Features

1. Extend `MemoryStore` class
2. Add new database tables in `_init_custom_tables()`
3. Implement new methods
4. Update agent base class if needed

## Deployment Considerations

### Environment Variables
- All sensitive data in environment variables
- `.env` file for development
- Production secrets management

### Database
- SQLite for development and small deployments
- PostgreSQL adapter available for production
- Automatic schema migration

### Scaling
- Stateless agent design
- Shared memory store
- Horizontal scaling ready

### Monitoring
- LangSmith integration for tracing
- Structured logging
- Performance metrics collection

## Security

### API Key Management
- Environment variable storage
- No hardcoded secrets
- Optional key rotation support

### Input Validation
- Pydantic models for configuration
- Tool input validation
- SQL injection prevention

### Memory Isolation
- Thread-based conversation isolation
- User-based preference isolation
- Configurable data retention

## Performance

### Optimization Strategies
- Lazy loading of components
- Connection pooling for database
- Efficient memory usage patterns
- Streaming for long responses

### Monitoring
- Response time tracking
- Memory usage monitoring
- Error rate tracking
- Tool usage analytics 