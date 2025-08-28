"""
Custom exceptions for the Vector Database Backend.

These exceptions provide clear error messages and proper HTTP status codes
for different types of errors that can occur in the vector database.
"""
from typing import Optional, Any


class VectorDBException(Exception):
    """Base exception for all vector database errors."""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class EntityNotFoundError(VectorDBException):
    """Raised when a requested entity (library, document, chunk) is not found."""
    
    def __init__(self, entity_type: str, entity_id: str, details: Optional[Any] = None):
        self.entity_type = entity_type
        self.entity_id = entity_id
        message = f"{entity_type} with ID '{entity_id}' not found"
        super().__init__(message, details)


class EntityAlreadyExistsError(VectorDBException):
    """Raised when trying to create an entity that already exists."""
    
    def __init__(self, entity_type: str, entity_id: str, details: Optional[Any] = None):
        self.entity_type = entity_type
        self.entity_id = entity_id
        message = f"{entity_type} with ID '{entity_id}' already exists"
        super().__init__(message, details)


class ValidationError(VectorDBException):
    """Raised when data validation fails."""
    
    def __init__(self, field: str, value: Any, reason: str, details: Optional[Any] = None):
        self.field = field
        self.value = value
        self.reason = reason
        message = f"Validation failed for field '{field}': {reason}"
        super().__init__(message, details)


class StorageError(VectorDBException):
    """Raised when storage operations fail."""
    
    def __init__(self, operation: str, reason: str, details: Optional[Any] = None):
        self.operation = operation
        self.reason = reason
        message = f"Storage operation '{operation}' failed: {reason}"
        super().__init__(message, details)


class IndexingError(VectorDBException):
    """Raised when indexing operations fail."""
    
    def __init__(self, operation: str, reason: str, details: Optional[Any] = None):
        self.operation = operation
        self.reason = reason
        message = f"Indexing operation '{operation}' failed: {reason}"
        super().__init__(message, details)


class SearchError(VectorDBException):
    """Raised when search operations fail."""
    
    def __init__(self, query_type: str, reason: str, details: Optional[Any] = None):
        self.query_type = query_type
        self.reason = reason
        message = f"Search operation '{query_type}' failed: {reason}"
        super().__init__(message, details)


class EmbeddingError(VectorDBException):
    """Raised when embedding operations fail."""
    
    def __init__(self, operation: str, reason: str, details: Optional[Any] = None):
        self.operation = operation
        self.reason = reason
        message = f"Embedding operation '{operation}' failed: {reason}"
        super().__init__(message, details)


class ConcurrencyError(VectorDBException):
    """Raised when concurrency-related operations fail."""
    
    def __init__(self, operation: str, reason: str, details: Optional[Any] = None):
        self.operation = operation
        self.reason = reason
        message = f"Concurrency issue in operation '{operation}': {reason}"
        super().__init__(message, details)
