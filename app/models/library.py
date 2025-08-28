"""
Library model for vector database.

A library contains multiple documents and metadata.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from .document import Document, DocumentResponse
from .chunk import Chunk


class LibraryCreate(BaseModel):
    """Schema for creating a new library."""
    name: str = Field(..., min_length=1, max_length=200, description="Name of the library")
    description: Optional[str] = Field(None, max_length=1000, description="Description of the library")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('name')
    def validate_name(cls, v):
        """Ensure name is not empty or just whitespace."""
        if not v.strip():
            raise ValueError('Name cannot be empty or just whitespace')
        return v.strip()
    
    @validator('description')
    def validate_description(cls, v):
        """Ensure description is not just whitespace if provided."""
        if v is not None and v.strip() == "":
            return None
        return v.strip() if v else v


class LibraryUpdate(BaseModel):
    """Schema for updating an existing library."""
    name: Optional[str] = Field(None, min_length=1, max_length=200, description="Name of the library")
    description: Optional[str] = Field(None, max_length=1000, description="Description of the library")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('name')
    def validate_name(cls, v):
        """Ensure name is not empty or just whitespace if provided."""
        if v is not None and not v.strip():
            raise ValueError('Name cannot be empty or just whitespace')
        return v.strip() if v else v
    
    @validator('description')
    def validate_description(cls, v):
        """Ensure description is not just whitespace if provided."""
        if v is not None and v.strip() == "":
            return None
        return v.strip() if v else v


class Library(BaseModel):
    """Complete library model with all fields."""
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the library")
    name: str = Field(..., description="Name of the library")
    description: Optional[str] = Field(None, description="Description of the library")
    documents: List[Document] = Field(default_factory=list, description="List of documents in this library")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    @property
    def document_count(self) -> int:
        """Number of documents in this library."""
        return len(self.documents)
    
    @property
    def total_chunk_count(self) -> int:
        """Total number of chunks across all documents."""
        return sum(doc.chunk_count for doc in self.documents)
    
    @property
    def total_text_length(self) -> int:
        """Total character count of all chunks in all documents."""
        return sum(doc.total_text_length for doc in self.documents)
    
    def add_document(self, document: Document) -> None:
        """Add a document to this library."""
        self.documents.append(document)
        self.updated_at = datetime.utcnow()
    
    def remove_document(self, document_id: UUID) -> bool:
        """Remove a document by ID. Returns True if removed, False if not found."""
        initial_count = len(self.documents)
        self.documents = [doc for doc in self.documents if doc.id != document_id]
        if len(self.documents) < initial_count:
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def get_document(self, document_id: UUID) -> Optional[Document]:
        """Get a document by ID."""
        for document in self.documents:
            if document.id == document_id:
                return document
        return None
    
    def get_all_chunks(self) -> List[Chunk]:
        """Get all chunks from all documents in this library."""
        chunks = []
        for document in self.documents:
            chunks.extend(document.chunks)
        return chunks
    
    def get_chunk(self, chunk_id: UUID) -> Optional[Chunk]:
        """Get a chunk by ID from any document in this library."""
        for document in self.documents:
            chunk = document.get_chunk(chunk_id)
            if chunk:
                return chunk
        return None
    
    def get_chunks_with_embeddings(self) -> List[Chunk]:
        """Get all chunks that have embeddings."""
        chunks_with_embeddings = []
        for document in self.documents:
            for chunk in document.chunks:
                if chunk.embedding is not None:
                    chunks_with_embeddings.append(chunk)
        return chunks_with_embeddings
    
    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "Research Papers Library",
                "description": "A collection of AI and ML research papers.",
                "documents": [],
                "metadata": {
                    "owner": "research_team",
                    "domain": "artificial_intelligence",
                    "access_level": "public"
                },
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }


class LibraryResponse(BaseModel):
    """Response schema for library operations."""
    id: UUID
    name: str
    description: Optional[str]
    document_count: int
    total_chunk_count: int
    total_text_length: int
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    documents: Optional[List[DocumentResponse]] = Field(None, description="Documents (only included in detailed views)")
    
    @classmethod
    def from_library(cls, library: Library, include_documents: bool = False) -> "LibraryResponse":
        """Create a response from a library instance."""
        documents = None
        if include_documents:
            documents = [DocumentResponse.from_document(doc) for doc in library.documents]
        
        return cls(
            id=library.id,
            name=library.name,
            description=library.description,
            document_count=library.document_count,
            total_chunk_count=library.total_chunk_count,
            total_text_length=library.total_text_length,
            metadata=library.metadata,
            created_at=library.created_at,
            updated_at=library.updated_at,
            documents=documents
        )
