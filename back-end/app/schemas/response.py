from sqlmodel import SQLModel
from typing import Optional, List
import uuid
from datetime import datetime, timezone
from app.models.domain import UserBase, ServiceCategoryBase, ServiceBase, ConversationBase, MessageBase, AppointmentBase, ReminderBase

class AgentStateResponse(SQLModel):
    intent: Optional[dict] = None
    appointment_details: Optional[dict] = None
    reschedule_details: Optional[dict] = None
    modification_details: Optional[dict] = None
    executed_tools: Optional[List[str]] = None

class ChatResponse(SQLModel):
    message: str
    conversation_id: uuid.UUID
    human_override: bool
    agent_state: Optional[AgentStateResponse] = None
    latency_ms: Optional[int] = None

class MessageRead(MessageBase):
    id: uuid.UUID
    created_at: datetime

class ConversationRead(ConversationBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime

class ConversationWithMessages(ConversationRead):
    messages: List[MessageRead] = []

class UserRead(UserBase):
    id: uuid.UUID
    created_at: datetime

class AppointmentRead(AppointmentBase):
    id: uuid.UUID
    created_at: datetime

class ServiceRead(ServiceBase):
    id: uuid.UUID

