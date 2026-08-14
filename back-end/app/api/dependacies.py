from fastapi import Depends, HTTPException, status
from sqlmodel import Session
from typing import Generator
import uuid

from app.core.database import get_session
from app.models.domain import Conversation

def get_db() -> Generator[Session, None, None]:
    yield from get_session()

async def get_active_conversation(conversation_id: uuid.UUID, db: Session = Depends(get_db)) -> Conversation:

    conversation = db.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation session with ID {conversation_id} not found."
        )
    return conversation