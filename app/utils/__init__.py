"""
Utility modules for the Vector Database Backend.
"""

from .concurrency import ReadWriteLock, thread_safe_read, thread_safe_write, ThreadSafeCounter, ThreadSafeSingleton
from .exceptions import (
    VectorDBException,
    EntityNotFoundError,
    EntityAlreadyExistsError,
    ValidationError,
    StorageError,
    IndexingError,
    SearchError,
    EmbeddingError,
    ConcurrencyError
)

__all__ = [
    # Concurrency utilities
    "ReadWriteLock",
    "thread_safe_read",
    "thread_safe_write", 
    "ThreadSafeCounter",
    "ThreadSafeSingleton",
    # Exception classes
    "VectorDBException",
    "EntityNotFoundError",
    "EntityAlreadyExistsError",
    "ValidationError",
    "StorageError",
    "IndexingError",
    "SearchError",
    "EmbeddingError",
    "ConcurrencyError",
]
