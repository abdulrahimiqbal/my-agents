"""
Semantic Search Configuration
"""

import os
from typing import Dict, Any

# Default semantic search configuration
SEMANTIC_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",  # Lightweight, fast model
    "fallback_model": "all-mpnet-base-v2",  # Higher quality fallback
    "enable_gpu": False,  # Set to True if GPU available
    "cache_embeddings": True,
    "similarity_threshold": 0.7,
    "max_results": 10,
    "enable_query_expansion": True,
    "enable_concept_matching": True
}

def get_semantic_config() -> Dict[str, Any]:
    """Get semantic search configuration with environment overrides."""
    config = SEMANTIC_CONFIG.copy()
    
    # Override with environment variables if available
    if os.getenv("SEMANTIC_MODEL"):
        config["embedding_model"] = os.getenv("SEMANTIC_MODEL")
    
    if os.getenv("ENABLE_GPU", "").lower() == "true":
        config["enable_gpu"] = True
    
    if os.getenv("SIMILARITY_THRESHOLD"):
        try:
            config["similarity_threshold"] = float(os.getenv("SIMILARITY_THRESHOLD"))
        except ValueError:
            pass
    
    return config