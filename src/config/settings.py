"""
Settings configuration using Pydantic for environment variable management.

Based on LangChain Academy module patterns for configuration management.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    default_model: str = Field(default="gpt-4o-mini")
    default_temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4000)
    
    # LangSmith Configuration
    langchain_api_key: Optional[str] = Field(default=None)
    langchain_tracing_v2: bool = Field(default=True)
    langchain_project: str = Field(default="my-agents-project")
    langchain_endpoint: Optional[str] = Field(default='https://api.smith.langchain.com')
    physics_agent_temperature: Optional[float] = Field(default=0.1)
    physics_agent_model: Optional[str] = Field(default='gpt-4o-mini')
    
    # Tool API Keys
    tavily_api_key: Optional[str] = Field(default=None)
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./data/agents.db")
    memory_db_path: str = Field(default="./data/memory.db")
    
    # Vector Store Configuration
    chroma_persist_directory: str = Field(default="./data/chroma")
    embedding_model: str = Field(default="text-embedding-3-small")
    
    # Development Settings
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # Web Interface
    streamlit_port: int = Field(default=8501)
    gradio_port: int = Field(default=7860)
    
    # Production Settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    workers: int = Field(default=1)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Allow extra fields without error
    }
        
    def setup_environment(self):
        """Set up environment variables for LangChain/LangSmith."""
        if self.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
        if self.langchain_tracing_v2:
            os.environ["LANGCHAIN_TRACING_V2"] = str(self.langchain_tracing_v2).lower()
        if self.langchain_project:
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            
    def get_llm_config(self) -> dict:
        """Get LLM configuration dictionary."""
        return {
            "model": self.default_model,
            "temperature": self.default_temperature,
            "max_tokens": self.max_tokens,
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.setup_environment()
    return _settings


# Convenience function for backward compatibility
settings = get_settings() 