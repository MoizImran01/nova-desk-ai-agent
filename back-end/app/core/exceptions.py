from typing import Any, Optional


class NovaDeskError(Exception):
    """
    Base exception for all custom errors in the Nova Desk application.
    Provides a standardized HTTP status_code, machine-readable error_code,
    human-readable message, and optional structured details dictionary.
    """

    status_code: int = 500
    error_code: str = "INTERNAL_SERVER_ERROR"

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# ---------------------------------------------------------------------------
# 4xx Domain & Client Errors
# ---------------------------------------------------------------------------

class ResourceNotFoundError(NovaDeskError):
    """Raised when a requested entity (Conversation, Appointment, User, Service) does not exist."""

    status_code = 404
    error_code = "RESOURCE_NOT_FOUND"


class AuthenticationError(NovaDeskError):
    """Raised when an admin or protected endpoint is called with missing/invalid credentials."""

    status_code = 401
    error_code = "UNAUTHORIZED"


class DocumentIngestionError(NovaDeskError):
    """Raised when an uploaded FAQ document is invalid, empty, or fails parsing."""

    status_code = 422
    error_code = "DOCUMENT_INGESTION_ERROR"


# ---------------------------------------------------------------------------
# 5xx External Provider & Infrastructure Errors
# ---------------------------------------------------------------------------

class LLMProviderError(NovaDeskError):
    """Raised when an upstream LLM provider (Groq, Gemini) fails, times out, or hits rate limits."""

    status_code = 503
    error_code = "LLM_PROVIDER_UNAVAILABLE"


class VectorStoreError(NovaDeskError):
    """Raised when Pinecone or the embedding service fails during retrieval or upsert."""

    status_code = 503
    error_code = "VECTOR_STORE_UNAVAILABLE"


class DatabaseOperationError(NovaDeskError):
    """Raised when a database transaction fails unexpectedly."""

    status_code = 500
    error_code = "DATABASE_OPERATION_ERROR"


class ServiceUnavailableError(NovaDeskError):
    """Raised when an endpoint or feature is disabled due to missing configuration."""

    status_code = 503
    error_code = "SERVICE_UNAVAILABLE"
