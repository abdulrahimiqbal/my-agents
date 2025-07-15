"""
Helper utilities for the agents project.
"""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        ensure_directory(Path(log_file).parent)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def ensure_directory(path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
    """
    path.mkdir(parents=True, exist_ok=True)


def load_env_file(env_file: str = ".env") -> None:
    """
    Load environment variables from a file.
    
    Args:
        env_file: Path to environment file
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        logging.warning("python-dotenv not installed. Environment file not loaded.")


def format_conversation_for_display(history: list) -> str:
    """
    Format conversation history for display.
    
    Args:
        history: List of conversation records from MemoryStore
        
    Returns:
        Formatted conversation string
    """
    formatted = []
    
    for item in history:
        if item["type"] == "summary":
            formatted.append(f"📝 **Summary** ({item['message_count']} messages): {item['content']}\n")
        elif item["type"] == "conversation":
            formatted.append(f"👤 **User**: {item['user_message']}")
            formatted.append(f"🤖 **Agent**: {item['agent_response']}\n")
    
    return "\n".join(formatted)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def validate_api_key(api_key: Optional[str], service_name: str) -> bool:
    """
    Validate that an API key is present and non-empty.
    
    Args:
        api_key: API key to validate
        service_name: Name of the service for error messages
        
    Returns:
        True if valid, False otherwise
    """
    if not api_key or not api_key.strip():
        logging.warning(f"{service_name} API key is not configured")
        return False
    return True 