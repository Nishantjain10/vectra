"""
Thread-safe in-memory storage for the Vector Database Backend.

This module provides the core storage layer with CRUD operations for libraries,
documents, and chunks. All operations are thread-safe using read-write locks
to prevent data races while allowing concurrent reads.
"""
from typing import Dict, List, Optional, Set
from uuid import UUID
from datetime import datetime
from copy import deepcopy

from app.models import Library, Document, Chunk, LibraryCreate, DocumentCreate, ChunkCreate
from app.utils.concurrency import ReadWriteLock, thread_safe_read, thread_safe_write, ThreadSafeSingleton
from app.utils.exceptions import EntityNotFoundError, EntityAlreadyExistsError, StorageError


class VectorStorage(ThreadSafeSingleton):
    """
    Thread-safe in-memory storage for vector database entities.
    
    This class implements a singleton pattern to ensure there's only one
    storage instance per application. Uses read-write locks for thread safety.
    
    Design Choices:
    1. **Read-Write Locks**: Allow multiple concurrent reads but exclusive writes
    2. **Singleton Pattern**: Ensures single source of truth for data
    3. **Deep Copy on Read**: Prevents external modifications to stored data
    4. **UUID Indexing**: Fast O(1) lookups using hash maps
    
    Time Complexity:
    - Create: O(1) average case
    - Read: O(1) for single entity, O(n) for list operations
    - Update: O(1) average case
    - Delete: O(1) average case
    
    Space Complexity: O(n) where n is the total number of entities
    """
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._lock = ReadWriteLock()
            
            # Primary storage - UUID -> Entity mappings
            self._libraries: Dict[UUID, Library] = {}
            self._documents: Dict[UUID, Document] = {}
            self._chunks: Dict[UUID, Chunk] = {}
            
            # Relationship indexes for fast lookups
            self._library_documents: Dict[UUID, Set[UUID]] = {}  # library_id -> document_ids
            self._document_chunks: Dict[UUID, Set[UUID]] = {}    # document_id -> chunk_ids
            self._chunk_libraries: Dict[UUID, UUID] = {}         # chunk_id -> library_id
            
            self._initialized = True
    
    # ================== LIBRARY OPERATIONS ==================
    
    @thread_safe_write
    def create_library(self, library_data: LibraryCreate) -> Library:
        """Create a new library."""
        library = Library(**library_data.dict())
        
        if library.id in self._libraries:
            raise EntityAlreadyExistsError("Library", str(library.id))
        
        self._libraries[library.id] = library
        self._library_documents[library.id] = set()
        
        return deepcopy(library)
    
    @thread_safe_read
    def get_library(self, library_id: UUID) -> Library:
        """Get a library by ID."""
        if library_id not in self._libraries:
            raise EntityNotFoundError("Library", str(library_id))
        
        library = deepcopy(self._libraries[library_id])
        
        # Populate documents and chunks
        document_ids = self._library_documents.get(library_id, set())
        library.documents = [
            self._get_document_with_chunks(doc_id) 
            for doc_id in document_ids 
            if doc_id in self._documents
        ]
        
        return library
    
    @thread_safe_read
    def list_libraries(self) -> List[Library]:
        """Get all libraries."""
        libraries = []
        for library_id in self._libraries:
            try:
                library = self.get_library(library_id)
                libraries.append(library)
            except EntityNotFoundError:
                # Skip if library was deleted between listing and retrieval
                continue
        return libraries
    
    @thread_safe_write
    def update_library(self, library_id: UUID, **updates) -> Library:
        """Update a library."""
        if library_id not in self._libraries:
            raise EntityNotFoundError("Library", str(library_id))
        
        library = self._libraries[library_id]
        
        # Update allowed fields
        for field, value in updates.items():
            if field in ['name', 'description', 'metadata']:
                setattr(library, field, value)
        
        library.updated_at = datetime.utcnow()
        return deepcopy(library)
    
    @thread_safe_write
    def delete_library(self, library_id: UUID) -> bool:
        """Delete a library and all its documents and chunks."""
        if library_id not in self._libraries:
            raise EntityNotFoundError("Library", str(library_id))
        
        # Delete all documents in the library (which will delete their chunks)
        document_ids = list(self._library_documents.get(library_id, set()))
        for doc_id in document_ids:
            self._delete_document_internal(doc_id)
        
        # Clean up library
        del self._libraries[library_id]
        del self._library_documents[library_id]
        
        return True
    
    # ================== DOCUMENT OPERATIONS ==================
    
    @thread_safe_write
    def create_document(self, library_id: UUID, document_data: DocumentCreate) -> Document:
        """Create a new document in a library."""
        if library_id not in self._libraries:
            raise EntityNotFoundError("Library", str(library_id))
        
        document = Document(**document_data.dict())
        
        if document.id in self._documents:
            raise EntityAlreadyExistsError("Document", str(document.id))
        
        # Store document
        self._documents[document.id] = document
        self._document_chunks[document.id] = set()
        
        # Add to library
        self._library_documents[library_id].add(document.id)
        
        # Update library timestamp
        self._libraries[library_id].updated_at = datetime.utcnow()
        
        return deepcopy(document)
    
    @thread_safe_read
    def get_document(self, library_id: UUID, document_id: UUID) -> Document:
        """Get a document by ID."""
        if library_id not in self._libraries:
            raise EntityNotFoundError("Library", str(library_id))
        
        if document_id not in self._documents:
            raise EntityNotFoundError("Document", str(document_id))
        
        # Verify document belongs to library
        if document_id not in self._library_documents[library_id]:
            raise EntityNotFoundError("Document", str(document_id))
        
        return self._get_document_with_chunks(document_id)
    
    @thread_safe_read
    def list_documents(self, library_id: UUID) -> List[Document]:
        """Get all documents in a library."""
        if library_id not in self._libraries:
            raise EntityNotFoundError("Library", str(library_id))
        
        document_ids = self._library_documents.get(library_id, set())
        documents = []
        
        for doc_id in document_ids:
            if doc_id in self._documents:
                documents.append(self._get_document_with_chunks(doc_id))
        
        return documents
    
    @thread_safe_write
    def update_document(self, library_id: UUID, document_id: UUID, **updates) -> Document:
        """Update a document."""
        if library_id not in self._libraries:
            raise EntityNotFoundError("Library", str(library_id))
        
        if document_id not in self._documents:
            raise EntityNotFoundError("Document", str(document_id))
        
        # Verify document belongs to library
        if document_id not in self._library_documents[library_id]:
            raise EntityNotFoundError("Document", str(document_id))
        
        document = self._documents[document_id]
        
        # Update allowed fields
        for field, value in updates.items():
            if field in ['title', 'description', 'metadata']:
                setattr(document, field, value)
        
        document.updated_at = datetime.utcnow()
        self._libraries[library_id].updated_at = datetime.utcnow()
        
        return self._get_document_with_chunks(document_id)
    
    @thread_safe_write
    def delete_document(self, library_id: UUID, document_id: UUID) -> bool:
        """Delete a document and all its chunks."""
        if library_id not in self._libraries:
            raise EntityNotFoundError("Library", str(library_id))
        
        if document_id not in self._documents:
            raise EntityNotFoundError("Document", str(document_id))
        
        # Verify document belongs to library
        if document_id not in self._library_documents[library_id]:
            raise EntityNotFoundError("Document", str(document_id))
        
        self._delete_document_internal(document_id)
        
        # Remove from library
        self._library_documents[library_id].discard(document_id)
        self._libraries[library_id].updated_at = datetime.utcnow()
        
        return True
    
    # ================== CHUNK OPERATIONS ==================
    
    @thread_safe_write
    def create_chunk(self, library_id: UUID, chunk_data: ChunkCreate) -> Chunk:
        """Create a new chunk in a library."""
        if library_id not in self._libraries:
            raise EntityNotFoundError("Library", str(library_id))
        
        chunk = Chunk(**chunk_data.dict())
        
        if chunk.id in self._chunks:
            raise EntityAlreadyExistsError("Chunk", str(chunk.id))
        
        # Store chunk
        self._chunks[chunk.id] = chunk
        self._chunk_libraries[chunk.id] = library_id
        
        # Update library timestamp
        self._libraries[library_id].updated_at = datetime.utcnow()
        
        return deepcopy(chunk)
    
    @thread_safe_read
    def get_chunk(self, library_id: UUID, chunk_id: UUID) -> Chunk:
        """Get a chunk by ID."""
        if library_id not in self._libraries:
            raise EntityNotFoundError("Library", str(library_id))
        
        if chunk_id not in self._chunks:
            raise EntityNotFoundError("Chunk", str(chunk_id))
        
        # Verify chunk belongs to library
        if self._chunk_libraries.get(chunk_id) != library_id:
            raise EntityNotFoundError("Chunk", str(chunk_id))
        
        return deepcopy(self._chunks[chunk_id])
    
    @thread_safe_read
    def list_chunks(self, library_id: UUID) -> List[Chunk]:
        """Get all chunks in a library."""
        if library_id not in self._libraries:
            raise EntityNotFoundError("Library", str(library_id))
        
        chunks = []
        for chunk_id, chunk_lib_id in self._chunk_libraries.items():
            if chunk_lib_id == library_id and chunk_id in self._chunks:
                chunks.append(deepcopy(self._chunks[chunk_id]))
        
        return chunks
    
    @thread_safe_write
    def update_chunk(self, library_id: UUID, chunk_id: UUID, **updates) -> Chunk:
        """Update a chunk."""
        if library_id not in self._libraries:
            raise EntityNotFoundError("Library", str(library_id))
        
        if chunk_id not in self._chunks:
            raise EntityNotFoundError("Chunk", str(chunk_id))
        
        # Verify chunk belongs to library
        if self._chunk_libraries.get(chunk_id) != library_id:
            raise EntityNotFoundError("Chunk", str(chunk_id))
        
        chunk = self._chunks[chunk_id]
        
        # Update allowed fields
        for field, value in updates.items():
            if field in ['text', 'embedding', 'metadata']:
                setattr(chunk, field, value)
        
        chunk.updated_at = datetime.utcnow()
        self._libraries[library_id].updated_at = datetime.utcnow()
        
        return deepcopy(chunk)
    
    @thread_safe_write
    def delete_chunk(self, library_id: UUID, chunk_id: UUID) -> bool:
        """Delete a chunk."""
        if library_id not in self._libraries:
            raise EntityNotFoundError("Library", str(library_id))
        
        if chunk_id not in self._chunks:
            raise EntityNotFoundError("Chunk", str(chunk_id))
        
        # Verify chunk belongs to library
        if self._chunk_libraries.get(chunk_id) != library_id:
            raise EntityNotFoundError("Chunk", str(chunk_id))
        
        # Remove from document if it's part of one
        for doc_id, chunk_ids in self._document_chunks.items():
            if chunk_id in chunk_ids:
                chunk_ids.remove(chunk_id)
                if doc_id in self._documents:
                    self._documents[doc_id].updated_at = datetime.utcnow()
                break
        
        # Delete chunk
        del self._chunks[chunk_id]
        del self._chunk_libraries[chunk_id]
        
        self._libraries[library_id].updated_at = datetime.utcnow()
        
        return True
    
    # ================== HELPER METHODS ==================
    
    def _get_document_with_chunks(self, document_id: UUID) -> Document:
        """Get a document with its chunks populated (internal use)."""
        document = deepcopy(self._documents[document_id])
        
        # Populate chunks
        chunk_ids = self._document_chunks.get(document_id, set())
        document.chunks = [
            deepcopy(self._chunks[chunk_id]) 
            for chunk_id in chunk_ids 
            if chunk_id in self._chunks
        ]
        
        return document
    
    def _delete_document_internal(self, document_id: UUID) -> None:
        """Delete a document and its chunks (internal use)."""
        # Delete all chunks in the document
        chunk_ids = list(self._document_chunks.get(document_id, set()))
        for chunk_id in chunk_ids:
            if chunk_id in self._chunks:
                del self._chunks[chunk_id]
            if chunk_id in self._chunk_libraries:
                del self._chunk_libraries[chunk_id]
        
        # Clean up document
        del self._documents[document_id]
        del self._document_chunks[document_id]
    
    # ================== UTILITY METHODS ==================
    
    @thread_safe_read
    def get_statistics(self) -> Dict[str, int]:
        """Get storage statistics."""
        return {
            "total_libraries": len(self._libraries),
            "total_documents": len(self._documents),
            "total_chunks": len(self._chunks),
        }
    
    @thread_safe_read
    def get_chunks_with_embeddings(self, library_id: Optional[UUID] = None) -> List[Chunk]:
        """Get all chunks that have embeddings, optionally filtered by library."""
        chunks_with_embeddings = []
        
        for chunk_id, chunk in self._chunks.items():
            if chunk.embedding is not None:
                # Filter by library if specified
                if library_id is None or self._chunk_libraries.get(chunk_id) == library_id:
                    chunks_with_embeddings.append(deepcopy(chunk))
        
        return chunks_with_embeddings
    
    @thread_safe_write
    def clear_all(self) -> None:
        """Clear all data (useful for testing)."""
        self._libraries.clear()
        self._documents.clear()
        self._chunks.clear()
        self._library_documents.clear()
        self._document_chunks.clear()
        self._chunk_libraries.clear()
