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
from .similarity import (
    cosine_similarity,
    cosine_distance,
    euclidean_distance,
    manhattan_distance,
    dot_product,
    vector_magnitude,
    normalize_vector,
    SimilarityMetrics
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
    # Similarity functions
    "cosine_similarity",
    "cosine_distance",
    "euclidean_distance",
    "manhattan_distance",
    "dot_product",
    "vector_magnitude",
    "normalize_vector",
    "SimilarityMetrics",
]
