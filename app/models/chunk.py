"""
Chunk model for vector database.

A chunk represents a piece of text with an associated embedding and metadata.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class ChunkCreate(BaseModel):
    """Schema for creating a new chunk."""
    text: str = Field(..., min_length=1, max_length=10000, description="Text content of the chunk")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('text')
    def validate_text(cls, v):
        """Ensure text is not empty or just whitespace."""
        if not v.strip():
            raise ValueError('Text cannot be empty or just whitespace')
        return v.strip()


class ChunkUpdate(BaseModel):
    """Schema for updating an existing chunk."""
    text: Optional[str] = Field(None, min_length=1, max_length=10000, description="Text content of the chunk")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('text')
    def validate_text(cls, v):
        """Ensure text is not empty or just whitespace if provided."""
        if v is not None and not v.strip():
            raise ValueError('Text cannot be empty or just whitespace')
        return v.strip() if v else v


class Chunk(BaseModel):
    """Complete chunk model with all fields."""
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the chunk")
    text: str = Field(..., description="Text content of the chunk")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "text": "This is a sample chunk of text for indexing.",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "metadata": {
                    "source": "document_1",
                    "page": 1,
                    "section": "introduction"
                },
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }


class ChunkResponse(BaseModel):
    """Response schema for chunk operations."""
    id: UUID
    text: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    has_embedding: bool = Field(..., description="Whether this chunk has an embedding")
    
    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "ChunkResponse":
        """Create a response from a chunk instance."""
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata,
            created_at=chunk.created_at,
            updated_at=chunk.updated_at,
            has_embedding=chunk.embedding is not None
        )
