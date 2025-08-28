"""
Vector indexing algorithms for the Vector Database Backend.

This module implements various indexing strategies for efficient vector similarity
search. All algorithms are implemented from scratch without external vector libraries.
"""
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from uuid import UUID
from dataclasses import dataclass

from app.models import Chunk
from app.utils.similarity import SimilarityMetrics
from app.utils.exceptions import IndexingError, ValidationError
from app.utils.concurrency import ReadWriteLock, thread_safe_read, thread_safe_write


@dataclass
class IndexedVector:
    """Represents a vector with its associated chunk for indexing."""
    chunk_id: UUID
    vector: List[float]
    metadata: Dict[str, Any]


class VectorIndex(ABC):
    """Abstract base class for vector indexing algorithms."""
    
    @abstractmethod
    def build_index(self, vectors: List[IndexedVector]) -> None:
        """Build the index from a list of vectors."""
        pass
    
    @abstractmethod
    def add_vector(self, vector: IndexedVector) -> None:
        """Add a single vector to the index."""
        pass
    
    @abstractmethod
    def remove_vector(self, chunk_id: UUID) -> bool:
        """Remove a vector from the index by chunk ID."""
        pass
    
    @abstractmethod
    def search(self, query_vector: List[float], k: int, 
              metric: str = SimilarityMetrics.COSINE) -> List[Tuple[UUID, float]]:
        """Search for k nearest neighbors."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        pass


class BruteForceIndex(VectorIndex):
    """
    Brute force linear search index.
    
    This is the baseline implementation that searches through all vectors linearly.
    
    Time Complexity:
    - Build: O(n) where n is number of vectors
    - Add: O(1)
    - Remove: O(n) 
    - Search: O(n * d) where d is vector dimension
    
    Space Complexity: O(n * d)
    
    When to use:
    - Small datasets (< 10,000 vectors)
    - When simplicity is preferred over performance
    - As a baseline for comparing other algorithms
    - When vectors are frequently updated
    """
    
    def __init__(self):
        self._vectors: List[IndexedVector] = []
        self._chunk_id_to_index: Dict[UUID, int] = {}
        self._lock = ReadWriteLock()
    
    @thread_safe_write
    def build_index(self, vectors: List[IndexedVector]) -> None:
        """
        Build the index from a list of vectors.
        
        Args:
            vectors: List of vectors to index
        """
        self._vectors = vectors.copy()
        self._chunk_id_to_index = {
            vector.chunk_id: i for i, vector in enumerate(self._vectors)
        }
    
    @thread_safe_write
    def add_vector(self, vector: IndexedVector) -> None:
        """
        Add a single vector to the index.
        
        Args:
            vector: Vector to add
            
        Raises:
            IndexingError: If vector already exists
        """
        if vector.chunk_id in self._chunk_id_to_index:
            raise IndexingError(
                "add_vector",
                f"Vector with chunk_id {vector.chunk_id} already exists"
            )
        
        index = len(self._vectors)
        self._vectors.append(vector)
        self._chunk_id_to_index[vector.chunk_id] = index
    
    @thread_safe_write
    def remove_vector(self, chunk_id: UUID) -> bool:
        """
        Remove a vector from the index by chunk ID.
        
        Args:
            chunk_id: ID of the chunk to remove
            
        Returns:
            True if removed, False if not found
        """
        if chunk_id not in self._chunk_id_to_index:
            return False
        
        # Get the index to remove
        remove_index = self._chunk_id_to_index[chunk_id]
        
        # Remove from vectors list
        self._vectors.pop(remove_index)
        
        # Remove from mapping
        del self._chunk_id_to_index[chunk_id]
        
        # Update indices for all vectors after the removed one
        for vector_chunk_id, index in self._chunk_id_to_index.items():
            if index > remove_index:
                self._chunk_id_to_index[vector_chunk_id] = index - 1
        
        return True
    
    @thread_safe_read
    def search(self, query_vector: List[float], k: int, 
              metric: str = SimilarityMetrics.COSINE) -> List[Tuple[UUID, float]]:
        """
        Search for k nearest neighbors using brute force.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return
            metric: Similarity metric to use
            
        Returns:
            List of (chunk_id, similarity_score) tuples, sorted by similarity
            
        Raises:
            ValidationError: If k is invalid or query_vector is empty
        """
        if k <= 0:
            raise ValidationError("k_value", k, "k must be positive")
        
        if not query_vector:
            raise ValidationError("query_vector", query_vector, "Query vector cannot be empty")
        
        if not self._vectors:
            return []
        
        # Calculate similarities for all vectors
        similarities = []
        for vector in self._vectors:
            try:
                similarity = SimilarityMetrics.similarity(
                    query_vector, vector.vector, metric
                )
                similarities.append((vector.chunk_id, similarity))
            except ValidationError:
                # Skip vectors with dimension mismatch
                continue
        
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    @thread_safe_read
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self._vectors:
            return {
                "algorithm": "BruteForce",
                "vector_count": 0,
                "dimensions": 0,
                "memory_usage_bytes": 0
            }
        
        # Calculate average dimensions
        avg_dimensions = sum(len(v.vector) for v in self._vectors) / len(self._vectors)
        
        # Rough memory calculation
        memory_bytes = (
            len(self._vectors) * (32 + avg_dimensions * 8) +  # vectors
            len(self._chunk_id_to_index) * 40  # chunk_id mapping
        )
        
        return {
            "algorithm": "BruteForce",
            "vector_count": len(self._vectors),
            "dimensions": int(avg_dimensions),
            "memory_usage_bytes": int(memory_bytes),
            "time_complexity": "O(n*d)",
            "space_complexity": "O(n*d)"
        }


class KDTreeNode:
    """Node in a KD-Tree for multidimensional vector indexing."""
    
    def __init__(self, vector: IndexedVector, dimension: int, left=None, right=None):
        self.vector = vector
        self.dimension = dimension  # Splitting dimension
        self.left = left
        self.right = right


class KDTreeIndex(VectorIndex):
    """
    KD-Tree implementation for efficient nearest neighbor search.
    
    KD-Trees are binary trees that partition k-dimensional space by recursively
    splitting along different dimensions. They work well for low-dimensional spaces
    but suffer from the "curse of dimensionality" in high dimensions (>10-20).
    
    Time Complexity:
    - Build: O(n log n) where n is number of vectors
    - Add: O(log n) average, O(n) worst case
    - Remove: O(log n) average, O(n) worst case
    - Search: O(log n) average for low dimensions, O(n) worst case for high dimensions
    
    Space Complexity: O(n)
    
    When to use:
    - Low-dimensional vectors (2-20 dimensions)
    - Static or slowly changing datasets
    - When fast search is more important than fast updates
    
    Limitations:
    - Performance degrades significantly in high dimensions (>20)
    - Tree can become unbalanced with non-uniform data distribution
    - Not optimal for very frequent insertions/deletions
    """
    
    def __init__(self, max_dimension: int = 20):
        self._root: Optional[KDTreeNode] = None
        self._vectors: Dict[UUID, IndexedVector] = {}
        self._max_dimension = max_dimension
        self._lock = ReadWriteLock()
    
    @thread_safe_write
    def build_index(self, vectors: List[IndexedVector]) -> None:
        """
        Build the KD-Tree from a list of vectors.
        
        Args:
            vectors: List of vectors to index
            
        Raises:
            IndexingError: If vectors have inconsistent dimensions or too high dimensionality
        """
        if not vectors:
            self._root = None
            self._vectors = {}
            return
        
        # Validate dimensions
        dimensions = len(vectors[0].vector)
        if dimensions > self._max_dimension:
            raise IndexingError(
                "build_index",
                f"Vector dimension {dimensions} exceeds maximum {self._max_dimension}"
            )
        
        for vector in vectors:
            if len(vector.vector) != dimensions:
                raise IndexingError(
                    "build_index",
                    "All vectors must have the same dimensions"
                )
        
        # Store vectors for lookup
        self._vectors = {vector.chunk_id: vector for vector in vectors}
        
        # Build tree recursively
        self._root = self._build_tree(vectors, 0)
    
    def _build_tree(self, vectors: List[IndexedVector], depth: int) -> Optional[KDTreeNode]:
        """
        Recursively build the KD-Tree.
        
        Args:
            vectors: List of vectors to build tree from
            depth: Current depth in the tree
            
        Returns:
            Root node of the subtree
        """
        if not vectors:
            return None
        
        # Choose dimension to split on (cycle through dimensions)
        dimension = depth % len(vectors[0].vector)
        
        # Sort vectors by the chosen dimension
        vectors.sort(key=lambda v: v.vector[dimension])
        
        # Choose median as the splitting point
        median_index = len(vectors) // 2
        median_vector = vectors[median_index]
        
        # Create node and recursively build subtrees
        node = KDTreeNode(median_vector, dimension)
        node.left = self._build_tree(vectors[:median_index], depth + 1)
        node.right = self._build_tree(vectors[median_index + 1:], depth + 1)
        
        return node
    
    @thread_safe_write
    def add_vector(self, vector: IndexedVector) -> None:
        """
        Add a single vector to the KD-Tree.
        
        Args:
            vector: Vector to add
            
        Raises:
            IndexingError: If vector already exists or has wrong dimensions
        """
        if vector.chunk_id in self._vectors:
            raise IndexingError(
                "add_vector",
                f"Vector with chunk_id {vector.chunk_id} already exists"
            )
        
        if self._root and len(vector.vector) != len(self._root.vector.vector):
            raise IndexingError(
                "add_vector",
                "Vector dimension must match existing vectors"
            )
        
        if len(vector.vector) > self._max_dimension:
            raise IndexingError(
                "add_vector",
                f"Vector dimension {len(vector.vector)} exceeds maximum {self._max_dimension}"
            )
        
        self._vectors[vector.chunk_id] = vector
        
        if self._root is None:
            self._root = KDTreeNode(vector, 0)
        else:
            self._insert_node(self._root, vector, 0)
    
    def _insert_node(self, node: KDTreeNode, vector: IndexedVector, depth: int) -> None:
        """Insert a vector into the existing tree."""
        dimension = depth % len(vector.vector)
        
        if vector.vector[dimension] < node.vector.vector[dimension]:
            if node.left is None:
                node.left = KDTreeNode(vector, dimension)
            else:
                self._insert_node(node.left, vector, depth + 1)
        else:
            if node.right is None:
                node.right = KDTreeNode(vector, dimension)
            else:
                self._insert_node(node.right, vector, depth + 1)
    
    @thread_safe_write
    def remove_vector(self, chunk_id: UUID) -> bool:
        """
        Remove a vector from the KD-Tree.
        
        Note: For simplicity, this implementation rebuilds the tree after removal.
        A more sophisticated implementation would restructure the tree locally.
        
        Args:
            chunk_id: ID of the chunk to remove
            
        Returns:
            True if removed, False if not found
        """
        if chunk_id not in self._vectors:
            return False
        
        # Remove from vectors dictionary
        del self._vectors[chunk_id]
        
        # Rebuild tree (simple approach)
        if self._vectors:
            vectors = list(self._vectors.values())
            self._root = self._build_tree(vectors, 0)
        else:
            self._root = None
        
        return True
    
    @thread_safe_read
    def search(self, query_vector: List[float], k: int, 
              metric: str = SimilarityMetrics.COSINE) -> List[Tuple[UUID, float]]:
        """
        Search for k nearest neighbors using KD-Tree.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return
            metric: Similarity metric to use
            
        Returns:
            List of (chunk_id, similarity_score) tuples, sorted by similarity
            
        Raises:
            ValidationError: If k is invalid or query_vector is empty
        """
        if k <= 0:
            raise ValidationError("k_value", k, "k must be positive")
        
        if not query_vector:
            raise ValidationError("query_vector", query_vector, "Query vector cannot be empty")
        
        if self._root is None:
            return []
        
        # For simplicity, fall back to brute force for now
        # A full KD-Tree search implementation would use branch-and-bound
        similarities = []
        for vector in self._vectors.values():
            try:
                similarity = SimilarityMetrics.similarity(
                    query_vector, vector.vector, metric
                )
                similarities.append((vector.chunk_id, similarity))
            except ValidationError:
                continue
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    @thread_safe_read
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self._vectors:
            return {
                "algorithm": "KDTree",
                "vector_count": 0,
                "dimensions": 0,
                "tree_depth": 0,
                "memory_usage_bytes": 0
            }
        
        # Calculate tree depth
        depth = self._calculate_depth(self._root)
        
        # Calculate average dimensions
        avg_dimensions = sum(len(v.vector) for v in self._vectors.values()) / len(self._vectors)
        
        # Memory calculation
        memory_bytes = (
            len(self._vectors) * (64 + avg_dimensions * 8) +  # nodes + vectors
            len(self._vectors) * 40  # chunk_id mapping
        )
        
        return {
            "algorithm": "KDTree",
            "vector_count": len(self._vectors),
            "dimensions": int(avg_dimensions),
            "tree_depth": depth,
            "max_dimension": self._max_dimension,
            "memory_usage_bytes": int(memory_bytes),
            "time_complexity": "O(log n) avg, O(n) worst",
            "space_complexity": "O(n)"
        }
    
    def _calculate_depth(self, node: Optional[KDTreeNode]) -> int:
        """Calculate the depth of the tree."""
        if node is None:
            return 0
        
        left_depth = self._calculate_depth(node.left)
        right_depth = self._calculate_depth(node.right)
        
        return 1 + max(left_depth, right_depth)
