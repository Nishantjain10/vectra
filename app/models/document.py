"""
Document model for vector database.

A document contains multiple chunks and metadata.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

from .chunk import Chunk, ChunkResponse


class DocumentCreate(BaseModel):
    """Schema for creating a new document."""
    title: str = Field(..., min_length=1, max_length=500, description="Title of the document")
    description: Optional[str] = Field(None, max_length=2000, description="Description of the document")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('title')
    def validate_title(cls, v):
        """Ensure title is not empty or just whitespace."""
        if not v.strip():
            raise ValueError('Title cannot be empty or just whitespace')
        return v.strip()
    
    @validator('description')
    def validate_description(cls, v):
        """Ensure description is not just whitespace if provided."""
        if v is not None and v.strip() == "":
            return None
        return v.strip() if v else v


class DocumentUpdate(BaseModel):
    """Schema for updating an existing document."""
    title: Optional[str] = Field(None, min_length=1, max_length=500, description="Title of the document")
    description: Optional[str] = Field(None, max_length=2000, description="Description of the document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('title')
    def validate_title(cls, v):
        """Ensure title is not empty or just whitespace if provided."""
        if v is not None and not v.strip():
            raise ValueError('Title cannot be empty or just whitespace')
        return v.strip() if v else v
    
    @validator('description')
    def validate_description(cls, v):
        """Ensure description is not just whitespace if provided."""
        if v is not None and v.strip() == "":
            return None
        return v.strip() if v else v


class Document(BaseModel):
    """Complete document model with all fields."""
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the document")
    title: str = Field(..., description="Title of the document")
    description: Optional[str] = Field(None, description="Description of the document")
    chunks: List[Chunk] = Field(default_factory=list, description="List of chunks in this document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    @property
    def chunk_count(self) -> int:
        """Number of chunks in this document."""
        return len(self.chunks)
    
    @property
    def total_text_length(self) -> int:
        """Total character count of all chunks."""
        return sum(len(chunk.text) for chunk in self.chunks)
    
    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to this document."""
        self.chunks.append(chunk)
        self.updated_at = datetime.utcnow()
    
    def remove_chunk(self, chunk_id: UUID) -> bool:
        """Remove a chunk by ID. Returns True if removed, False if not found."""
        initial_count = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk.id != chunk_id]
        if len(self.chunks) < initial_count:
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def get_chunk(self, chunk_id: UUID) -> Optional[Chunk]:
        """Get a chunk by ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None
    
    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "title": "Sample Document",
                "description": "This is a sample document for testing.",
                "chunks": [],
                "metadata": {
                    "author": "John Doe",
                    "category": "research",
                    "tags": ["ai", "machine learning"]
                },
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }


class DocumentResponse(BaseModel):
    """Response schema for document operations."""
    id: UUID
    title: str
    description: Optional[str]
    chunk_count: int
    total_text_length: int
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    chunks: Optional[List[ChunkResponse]] = Field(None, description="Chunks (only included in detailed views)")
    
    @classmethod
    def from_document(cls, document: Document, include_chunks: bool = False) -> "DocumentResponse":
        """Create a response from a document instance."""
        chunks = None
        if include_chunks:
            chunks = [ChunkResponse.from_chunk(chunk) for chunk in document.chunks]
        
        return cls(
            id=document.id,
            title=document.title,
            description=document.description,
            chunk_count=document.chunk_count,
            total_text_length=document.total_text_length,
            metadata=document.metadata,
            created_at=document.created_at,
            updated_at=document.updated_at,
            chunks=chunks
        )
