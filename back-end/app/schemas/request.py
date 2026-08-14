from sqlmodel import Field, SQLModel
from typing import Optional
import uuid
from datetime import datetime, timezone
from app.models.domain import UserBase, ServiceCategoryBase, ServiceBase

class UserCreate(UserBase):
    pass

class UserUpdate(UserBase):
    pass

class ServiceCategoryCreate(ServiceCategoryBase):
    pass

class ServiceCreate(ServiceBase):
    pass

class AppointmentCreate(SQLModel):
    pass

class ChatRequest(SQLModel):
    conversation_id: Optional[uuid.UUID] = None
    message: str

class ConversationUpdate(SQLModel):
    human_override: Optional[bool] = None
    status: Optional[str] = None

