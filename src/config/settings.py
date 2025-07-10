"""
Settings configuration using Pydantic for environment variable management.

Based on LangChain Academy module patterns for configuration management.
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings as PydanticBaseSettings


class Settings(PydanticBaseSettings):
    """Application settings with environment variable support."""
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    default_model: str = Field(default="gpt-4o-mini", env="DEFAULT_MODEL")
    default_temperature: float = Field(default=0.7, env="DEFAULT_TEMPERATURE")
    max_tokens: int = Field(default=4000, env="MAX_TOKENS")
    
    # LangSmith Configuration
    langchain_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="my-agents-project", env="LANGCHAIN_PROJECT")
    
    # Tool API Keys
    tavily_api_key: Optional[str] = Field(default=None, env="TAVILY_API_KEY")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./data/agents.db", env="DATABASE_URL")
    memory_db_path: str = Field(default="./data/memory.db", env="MEMORY_DB_PATH")
    
    # Vector Store Configuration
    chroma_persist_directory: str = Field(default="./data/chroma", env="CHROMA_PERSIST_DIRECTORY")
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    
    # Development Settings
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Web Interface
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    gradio_port: int = Field(default=7860, env="GRADIO_PORT")
    
    # Production Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=1, env="WORKERS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
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