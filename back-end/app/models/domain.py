from sqlmodel import Field, SQLModel, Relationship
from sqlalchemy import Text
from typing import Optional, List
import uuid
from datetime import datetime, timezone, date, time

def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)

class UserBase(SQLModel):
    email: Optional[str] = Field(default=None, unique=True, index=True)
    full_name: Optional[str] = Field(default=None)
    phone: Optional[str] = Field(default=None)

class User(UserBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    is_admin: bool = Field(default=False)
    created_at: datetime = Field(default_factory=get_utc_now)
    updated_at: datetime = Field(default_factory=get_utc_now)
    appointments: List["Appointment"] = Relationship(back_populates='user')
    conversations: List["Conversation"] = Relationship(back_populates='user')


class ServiceCategoryBase(SQLModel):
    name: str = Field(unique=True, index=True)
    is_active: bool = Field(default=True)

class ServiceCategory(ServiceCategoryBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=get_utc_now)
    services: List["Service"] = Relationship(back_populates='category')


class ServiceBase(SQLModel):
    name: str = Field(unique=True, index=True)
    description: str = Field(sa_type=Text)
    price: float = Field(default=0.0)
    duration_minutes: int = Field(default=30)
    category_id: Optional[uuid.UUID] = Field(default=None, foreign_key='servicecategory.id')
    

class Service(ServiceBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    category: ServiceCategory = Relationship(back_populates='services')
    appointments: List["Appointment"] = Relationship(back_populates='service')

class ConversationBase(SQLModel):
    user_id: uuid.UUID = Field(foreign_key='user.id')
    human_override: bool = Field(default=False)
    status: str = Field(default="active")

class Conversation(ConversationBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=get_utc_now)
    updated_at: datetime = Field(default_factory=get_utc_now)
    user: User = Relationship(back_populates='conversations')
    messages: List["Message"] = Relationship(back_populates='conversation')

class MessageBase(SQLModel):
    conversation_id: uuid.UUID = Field(foreign_key='conversation.id')
    content: str = Field(sa_type=Text)
    role: str = Field(index=True)

class Message(MessageBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=get_utc_now)
    conversation: Conversation = Relationship(back_populates='messages')

class AppointmentBase(SQLModel):
    user_id: uuid.UUID = Field(foreign_key='user.id')
    service_id: Optional[uuid.UUID] = Field(foreign_key='service.id')
    service_name: Optional[str] = Field(index=True)
    appointment_date: Optional[date] = Field(index=True)
    appointment_time: Optional[time] = Field(index=True)
    status: str = Field(default="in-progress", index=True)

class Appointment(AppointmentBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=get_utc_now)
    updated_at: datetime = Field(default_factory=get_utc_now)

    user: User = Relationship(back_populates='appointments')
    service: Service = Relationship(back_populates='appointments')
    reminders: List["Reminder"] = Relationship(back_populates='appointment')

class ReminderBase(SQLModel):
    appointment_id: uuid.UUID = Field(foreign_key='appointment.id')
    reminder_time: datetime = Field(index=True)
    reminder_sent: bool = Field(default=False, index=True)

class Reminder(ReminderBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=get_utc_now)
    appointment: Appointment = Relationship(back_populates='reminders')
