"""
Similarity functions for vector operations.

This module implements various distance and similarity metrics used in vector
databases for nearest neighbor search and clustering operations.
"""
import math
from typing import List, Tuple
import numpy as np

from .exceptions import ValidationError


def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Cosine similarity measures the cosine of the angle between two vectors.
    Range: [-1, 1] where 1 means identical direction, 0 means orthogonal, -1 means opposite.
    
    Formula: cos(θ) = (A · B) / (||A|| * ||B||)
    
    Time Complexity: O(d) where d is the dimension
    Space Complexity: O(1)
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Cosine similarity score
        
    Raises:
        ValidationError: If vectors have different dimensions or are empty
    """
    if len(vector1) != len(vector2):
        raise ValidationError(
            "vector_dimensions", 
            (len(vector1), len(vector2)),
            "Vectors must have the same dimensions"
        )
    
    if len(vector1) == 0:
        raise ValidationError("vector_length", 0, "Vectors cannot be empty")
    
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vector1))
    magnitude2 = math.sqrt(sum(b * b for b in vector2))
    
    # Handle zero vectors
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def cosine_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate cosine distance between two vectors.
    
    Cosine distance = 1 - cosine_similarity
    Range: [0, 2] where 0 means identical, 2 means opposite direction.
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Cosine distance
    """
    return 1.0 - cosine_similarity(vector1, vector2)


def euclidean_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Euclidean distance is the straight-line distance between two points.
    Formula: sqrt(Σ(ai - bi)²)
    
    Time Complexity: O(d) where d is the dimension
    Space Complexity: O(1)
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Euclidean distance
        
    Raises:
        ValidationError: If vectors have different dimensions or are empty
    """
    if len(vector1) != len(vector2):
        raise ValidationError(
            "vector_dimensions",
            (len(vector1), len(vector2)),
            "Vectors must have the same dimensions"
        )
    
    if len(vector1) == 0:
        raise ValidationError("vector_length", 0, "Vectors cannot be empty")
    
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vector1, vector2)))


def manhattan_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate Manhattan (L1) distance between two vectors.
    
    Manhattan distance is the sum of absolute differences.
    Formula: Σ|ai - bi|
    
    Time Complexity: O(d) where d is the dimension
    Space Complexity: O(1)
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Manhattan distance
        
    Raises:
        ValidationError: If vectors have different dimensions or are empty
    """
    if len(vector1) != len(vector2):
        raise ValidationError(
            "vector_dimensions",
            (len(vector1), len(vector2)), 
            "Vectors must have the same dimensions"
        )
    
    if len(vector1) == 0:
        raise ValidationError("vector_length", 0, "Vectors cannot be empty")
    
    return sum(abs(a - b) for a, b in zip(vector1, vector2))


def dot_product(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate dot product between two vectors.
    
    Formula: Σ(ai * bi)
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Dot product
    """
    if len(vector1) != len(vector2):
        raise ValidationError(
            "vector_dimensions",
            (len(vector1), len(vector2)),
            "Vectors must have the same dimensions"
        )
    
    return sum(a * b for a, b in zip(vector1, vector2))


def vector_magnitude(vector: List[float]) -> float:
    """
    Calculate the magnitude (L2 norm) of a vector.
    
    Formula: sqrt(Σ(ai²))
    
    Args:
        vector: Input vector
        
    Returns:
        Vector magnitude
    """
    return math.sqrt(sum(x * x for x in vector))


def normalize_vector(vector: List[float]) -> List[float]:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector
        
    Raises:
        ValidationError: If vector is zero vector
    """
    magnitude = vector_magnitude(vector)
    
    if magnitude == 0:
        raise ValidationError("zero_vector", vector, "Cannot normalize zero vector")
    
    return [x / magnitude for x in vector]


class SimilarityMetrics:
    """
    Factory class for similarity metrics with caching and optimization.
    
    This class provides a unified interface for different similarity metrics
    and can cache results for performance optimization.
    """
    
    # Available similarity functions
    COSINE = "cosine"
    EUCLIDEAN = "euclidean" 
    MANHATTAN = "manhattan"
    DOT_PRODUCT = "dot_product"
    
    _functions = {
        COSINE: cosine_similarity,
        EUCLIDEAN: lambda v1, v2: -euclidean_distance(v1, v2),  # Negative for similarity
        MANHATTAN: lambda v1, v2: -manhattan_distance(v1, v2),  # Negative for similarity
        DOT_PRODUCT: dot_product,
    }
    
    _distance_functions = {
        COSINE: cosine_distance,
        EUCLIDEAN: euclidean_distance,
        MANHATTAN: manhattan_distance,
    }
    
    @classmethod
    def similarity(cls, vector1: List[float], vector2: List[float], metric: str = COSINE) -> float:
        """
        Calculate similarity using specified metric.
        
        Args:
            vector1: First vector
            vector2: Second vector
            metric: Similarity metric to use
            
        Returns:
            Similarity score
            
        Raises:
            ValidationError: If metric is unknown
        """
        if metric not in cls._functions:
            raise ValidationError(
                "similarity_metric",
                metric,
                f"Unknown metric. Available: {list(cls._functions.keys())}"
            )
        
        return cls._functions[metric](vector1, vector2)
    
    @classmethod
    def distance(cls, vector1: List[float], vector2: List[float], metric: str = COSINE) -> float:
        """
        Calculate distance using specified metric.
        
        Args:
            vector1: First vector
            vector2: Second vector
            metric: Distance metric to use
            
        Returns:
            Distance score
            
        Raises:
            ValidationError: If metric is unknown
        """
        if metric not in cls._distance_functions:
            raise ValidationError(
                "distance_metric",
                metric,
                f"Unknown metric. Available: {list(cls._distance_functions.keys())}"
            )
        
        return cls._distance_functions[metric](vector1, vector2)
    
    @classmethod
    def batch_similarity(cls, query_vector: List[float], vectors: List[List[float]], 
                        metric: str = COSINE) -> List[Tuple[int, float]]:
        """
        Calculate similarity between query vector and multiple vectors.
        
        Args:
            query_vector: Query vector
            vectors: List of vectors to compare against
            metric: Similarity metric to use
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        for i, vector in enumerate(vectors):
            try:
                similarity = cls.similarity(query_vector, vector, metric)
                similarities.append((i, similarity))
            except ValidationError:
                # Skip vectors with dimension mismatch
                continue
        
        return similarities
    
    @classmethod
    def top_k_similar(cls, query_vector: List[float], vectors: List[List[float]], 
                     k: int, metric: str = COSINE) -> List[Tuple[int, float]]:
        """
        Find top-k most similar vectors.
        
        Args:
            query_vector: Query vector
            vectors: List of vectors to compare against
            k: Number of top results to return
            metric: Similarity metric to use
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity descending
        """
        similarities = cls.batch_similarity(query_vector, vectors, metric)
        
        # Sort by similarity score (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
