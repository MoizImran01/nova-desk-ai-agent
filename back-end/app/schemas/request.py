from typing import Optional
import uuid

from pydantic import field_validator
from sqlmodel import Field, SQLModel


class ChatRequest(SQLModel):
    conversation_id: Optional[uuid.UUID] = None
    message: str = Field(min_length=1, max_length=2000, description="User chat message (1-2000 characters)")

    @field_validator("message")
    @classmethod
    def validate_non_blank_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Message cannot be empty or whitespace only.")
        return stripped
