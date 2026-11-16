import uuid

from sqlmodel import Session, update

from app.models.domain import Conversation, Message, get_utc_now
from app.repositories.base import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):
    """Data access layer strictly for `Conversation` and `Message` persistence."""

    def __init__(self, session: Session) -> None:
        super().__init__(session, Conversation)

    def create_conversation(self, user_id: uuid.UUID) -> Conversation:
        """Create a new `Conversation` linked to an existing `user_id`."""
        conversation = Conversation(user_id=user_id)
        return self.add(conversation)

    def append_turn(self, conversation_id: uuid.UUID, user_message: str, assistant_message: str) -> None:
        """Persist both messages and advance the conversation timestamp in the caller's transaction."""
        self.session.add_all([
            Message(conversation_id=conversation_id, role="user", content=user_message),
            Message(conversation_id=conversation_id, role="assistant", content=assistant_message),
        ])
        self.session.exec(
            update(Conversation).where(Conversation.id == conversation_id).values(updated_at=get_utc_now())
        )
