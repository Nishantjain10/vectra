"""
Configuration settings for the Vector Database Backend
"""
import os
from typing import List

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Cohere API Configuration for embeddings
COHERE_API_KEYS: List[str] = [
    "",
    ""
]

# Use environment variable if provided, otherwise fallback to provided keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY", COHERE_API_KEYS[0])

# Vector Database Configuration
DEFAULT_EMBEDDING_MODEL = "embed-english-v3.0"
DEFAULT_EMBEDDING_DIMENSION = 1024

