"""
Vector Index Manager for coordinating different indexing algorithms.

This module provides a unified interface for managing vector indexes and
automatically selecting the best algorithm based on data characteristics.
"""
from typing import List, Tuple, Optional, Dict, Any, Type
from uuid import UUID
from enum import Enum

from app.models import Chunk
from app.utils.exceptions import IndexingError, ValidationError
from app.utils.concurrency import ReadWriteLock, thread_safe_read, thread_safe_write
from .indexing import VectorIndex, BruteForceIndex, KDTreeIndex, IndexedVector


class IndexType(Enum):
    """Available index types."""
    BRUTE_FORCE = "brute_force"
    KD_TREE = "kd_tree"
    AUTO = "auto"


class VectorIndexManager:
    """
    Manages vector indexes and provides automatic algorithm selection.
    
    This class provides a unified interface for vector indexing while
    automatically selecting the best algorithm based on data characteristics
    such as vector dimensions, dataset size, and update frequency.
    
    Design Principles:
    1. Automatic algorithm selection based on data characteristics
    2. Thread-safe operations for concurrent access
    3. Lazy index building for performance
    4. Fallback mechanisms for robustness
    5. Comprehensive statistics and monitoring
    """
    
    def __init__(self, index_type: IndexType = IndexType.AUTO):
        self._index_type = index_type
        self._index: Optional[VectorIndex] = None
        self._chunks: Dict[UUID, Chunk] = {}
        self._is_built = False
        self._lock = ReadWriteLock()
        
        # Configuration thresholds for auto-selection
        self._brute_force_threshold = 10000  # Max vectors for brute force
        self._high_dimension_threshold = 20   # Max dimensions for KD-Tree
        
    @thread_safe_write
    def add_chunk(self, chunk: Chunk) -> None:
        """
        Add a chunk with its vector to the index.
        
        Args:
            chunk: Chunk with embedding to add
            
        Raises:
            ValidationError: If chunk has no embedding
            IndexingError: If chunk already exists
        """
        if chunk.embedding is None:
            raise ValidationError(
                "embedding", 
                None, 
                "Chunk must have an embedding to be indexed"
            )
        
        if chunk.id in self._chunks:
            raise IndexingError(
                "add_chunk",
                f"Chunk {chunk.id} already exists in index"
            )
        
        # Store chunk
        self._chunks[chunk.id] = chunk
        
        # Add to index if it exists
        if self._index is not None:
            indexed_vector = IndexedVector(
                chunk_id=chunk.id,
                vector=chunk.embedding,
                metadata=chunk.metadata
            )
            try:
                self._index.add_vector(indexed_vector)
            except Exception as e:
                # If adding fails, remove from chunks and re-raise
                del self._chunks[chunk.id]
                raise IndexingError("add_chunk", f"Failed to add to index: {e}")
        else:
            # Mark as not built since we have new data
            self._is_built = False
    
    @thread_safe_write
    def remove_chunk(self, chunk_id: UUID) -> bool:
        """
        Remove a chunk from the index.
        
        Args:
            chunk_id: ID of the chunk to remove
            
        Returns:
            True if removed, False if not found
        """
        if chunk_id not in self._chunks:
            return False
        
        # Remove from chunks
        del self._chunks[chunk_id]
        
        # Remove from index if it exists
        if self._index is not None:
            self._index.remove_vector(chunk_id)
        
        return True
    
    @thread_safe_write
    def update_chunk(self, chunk: Chunk) -> None:
        """
        Update a chunk's embedding in the index.
        
        Args:
            chunk: Updated chunk with new embedding
        """
        if chunk.id in self._chunks:
            # Remove old version
            self.remove_chunk(chunk.id)
        
        # Add new version
        self.add_chunk(chunk)
    
    @thread_safe_write
    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Build the vector index from stored chunks.
        
        Args:
            force_rebuild: Force rebuild even if index is already built
            
        Raises:
            IndexingError: If no chunks with embeddings are available
        """
        if self._is_built and not force_rebuild:
            return
        
        # Get chunks with embeddings
        chunks_with_embeddings = [
            chunk for chunk in self._chunks.values() 
            if chunk.embedding is not None
        ]
        
        if not chunks_with_embeddings:
            raise IndexingError(
                "build_index",
                "No chunks with embeddings available for indexing"
            )
        
        # Select appropriate index algorithm
        index_class = self._select_index_algorithm(chunks_with_embeddings)
        
        # Create and build index
        self._index = index_class()
        
        indexed_vectors = [
            IndexedVector(
                chunk_id=chunk.id,
                vector=chunk.embedding,
                metadata=chunk.metadata
            )
            for chunk in chunks_with_embeddings
        ]
        
        try:
            self._index.build_index(indexed_vectors)
            self._is_built = True
        except Exception as e:
            self._index = None
            self._is_built = False
            raise IndexingError("build_index", f"Failed to build index: {e}")
    
    def _select_index_algorithm(self, chunks: List[Chunk]) -> Type[VectorIndex]:
        """
        Select the best index algorithm based on data characteristics.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Index class to use
        """
        if self._index_type == IndexType.BRUTE_FORCE:
            return BruteForceIndex
        elif self._index_type == IndexType.KD_TREE:
            return KDTreeIndex
        elif self._index_type == IndexType.AUTO:
            return self._auto_select_algorithm(chunks)
        else:
            raise IndexingError(
                "algorithm_selection",
                f"Unknown index type: {self._index_type}"
            )
    
    def _auto_select_algorithm(self, chunks: List[Chunk]) -> Type[VectorIndex]:
        """
        Automatically select the best algorithm based on data characteristics.
        
        Selection criteria:
        1. Small datasets (< 10k vectors) → Brute Force
        2. High dimensions (> 20) → Brute Force  
        3. Low-medium dimensions + medium dataset → KD-Tree
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Index class to use
        """
        num_vectors = len(chunks)
        
        # Get average vector dimension
        dimensions = len(chunks[0].embedding) if chunks else 0
        
        # Selection logic
        if num_vectors <= self._brute_force_threshold:
            # Small dataset: brute force is fine and simple
            return BruteForceIndex
        
        if dimensions > self._high_dimension_threshold:
            # High dimensions: KD-Tree suffers from curse of dimensionality
            return BruteForceIndex
        
        # Medium dataset, low-medium dimensions: KD-Tree should be better
        return KDTreeIndex
    
    @thread_safe_read
    def search(self, query_vector: List[float], k: int, 
              metric: str = "cosine") -> List[Tuple[Chunk, float]]:
        """
        Search for k nearest neighbor chunks.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors
            metric: Similarity metric to use
            
        Returns:
            List of (chunk, similarity_score) tuples
            
        Raises:
            IndexingError: If index is not built
        """
        if not self._is_built or self._index is None:
            # Auto-build if not built
            if self._chunks:
                self.build_index()
            else:
                return []
        
        # Search using the index
        results = self._index.search(query_vector, k, metric)
        
        # Convert chunk IDs to chunk objects
        chunk_results = []
        for chunk_id, similarity in results:
            if chunk_id in self._chunks:
                chunk_results.append((self._chunks[chunk_id], similarity))
        
        return chunk_results
    
    @thread_safe_read
    def search_with_filter(self, query_vector: List[float], k: int,
                          metadata_filter: Optional[Dict[str, Any]] = None,
                          metric: str = "cosine") -> List[Tuple[Chunk, float]]:
        """
        Search for k nearest neighbors with metadata filtering.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors
            metadata_filter: Dictionary of metadata key-value pairs to filter by
            metric: Similarity metric to use
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        # Get all results first
        all_results = self.search(query_vector, len(self._chunks), metric)
        
        # Apply metadata filtering if specified
        if metadata_filter:
            filtered_results = []
            for chunk, similarity in all_results:
                # Check if chunk matches all filter criteria
                matches = True
                for key, value in metadata_filter.items():
                    if chunk.metadata.get(key) != value:
                        matches = False
                        break
                
                if matches:
                    filtered_results.append((chunk, similarity))
            
            return filtered_results[:k]
        
        return all_results[:k]
    
    @thread_safe_read
    def get_chunk_count(self) -> int:
        """Get the number of chunks in the index."""
        return len(self._chunks)
    
    @thread_safe_read
    def get_indexed_count(self) -> int:
        """Get the number of chunks with embeddings."""
        return len([c for c in self._chunks.values() if c.embedding is not None])
    
    @thread_safe_read
    def is_built(self) -> bool:
        """Check if the index is built."""
        return self._is_built
    
    @thread_safe_read
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        base_stats = {
            "total_chunks": len(self._chunks),
            "indexed_chunks": self.get_indexed_count(),
            "is_built": self._is_built,
            "index_type": self._index_type.value,
        }
        
        if self._index is not None:
            index_stats = self._index.get_stats()
            base_stats.update(index_stats)
        else:
            base_stats.update({
                "algorithm": "None",
                "vector_count": 0,
                "dimensions": 0,
                "memory_usage_bytes": 0
            })
        
        return base_stats
    
    @thread_safe_write
    def clear(self) -> None:
        """Clear all chunks and rebuild the index."""
        self._chunks.clear()
        self._index = None
        self._is_built = False
    
    @thread_safe_write
    def set_configuration(self, brute_force_threshold: Optional[int] = None,
                         high_dimension_threshold: Optional[int] = None) -> None:
        """
        Update configuration thresholds for algorithm selection.
        
        Args:
            brute_force_threshold: Max vectors for brute force algorithm
            high_dimension_threshold: Max dimensions for KD-Tree algorithm
        """
        if brute_force_threshold is not None:
            self._brute_force_threshold = brute_force_threshold
        
        if high_dimension_threshold is not None:
            self._high_dimension_threshold = high_dimension_threshold
