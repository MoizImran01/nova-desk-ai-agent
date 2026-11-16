from datetime import date, datetime, time, timezone
from typing import List, Optional
import uuid

from sqlalchemy import CheckConstraint, Index, Text, text
from sqlmodel import Field, Relationship, SQLModel


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
    appointments: List["Appointment"] = Relationship(back_populates="user")
    conversations: List["Conversation"] = Relationship(back_populates="user")


class ServiceCategoryBase(SQLModel):
    name: str = Field(unique=True, index=True)
    is_active: bool = Field(default=True)


class ServiceCategory(ServiceCategoryBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=get_utc_now)
    services: List["Service"] = Relationship(back_populates="category")


class ServiceBase(SQLModel):
    name: str = Field(unique=True, index=True)
    description: str = Field(sa_type=Text)
    price: float = Field(default=0.0)
    duration_minutes: int = Field(default=30)
    category_id: Optional[uuid.UUID] = Field(default=None, foreign_key="servicecategory.id")


class Service(ServiceBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    category: ServiceCategory = Relationship(back_populates="services")
    appointments: List["Appointment"] = Relationship(back_populates="service")


class ConversationBase(SQLModel):
    user_id: uuid.UUID = Field(foreign_key="user.id")
    human_override: bool = Field(default=False)
    status: str = Field(default="active")


class Conversation(ConversationBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=get_utc_now)
    updated_at: datetime = Field(default_factory=get_utc_now)
    user: User = Relationship(back_populates="conversations")
    messages: List["Message"] = Relationship(back_populates="conversation")


class MessageBase(SQLModel):
    conversation_id: uuid.UUID = Field(foreign_key="conversation.id")
    content: str = Field(sa_type=Text)
    role: str = Field(index=True)


class Message(MessageBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=get_utc_now)
    conversation: Conversation = Relationship(back_populates="messages")


class AppointmentBase(SQLModel):
    user_id: uuid.UUID = Field(foreign_key="user.id")
    service_id: Optional[uuid.UUID] = Field(foreign_key="service.id")
    service_name: Optional[str] = Field(index=True)
    appointment_date: Optional[date] = Field(index=True)
    appointment_time: Optional[time] = Field(index=True)
    status: str = Field(default="in-progress", index=True)


class Appointment(AppointmentBase, table=True):
    __table_args__ = (
        # Guarantees at the PostgreSQL engine level that two "scheduled" appointments
        # can NEVER occupy the same (appointment_date, appointment_time) slot,
        # while still allowing cancelled/historical rows on that same slot.
        Index(
            "ix_unique_scheduled_slot",
            "appointment_date",
            "appointment_time",
            unique=True,
            postgresql_where=text("status = 'scheduled'"),
            sqlite_where=text("status = 'scheduled'"),
        ),
        CheckConstraint(
            "status IN ('in-progress', 'scheduled', 'cancelled', 'completed')",
            name="ck_appointment_valid_status",
        ),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=get_utc_now)
    updated_at: datetime = Field(default_factory=get_utc_now)

    user: User = Relationship(back_populates="appointments")
    service: Service = Relationship(back_populates="appointments")
    reminders: List["Reminder"] = Relationship(back_populates="appointment")


class ReminderBase(SQLModel):
    appointment_id: uuid.UUID = Field(foreign_key="appointment.id")
    reminder_time: datetime = Field(index=True)
    reminder_sent: bool = Field(default=False, index=True)


class Reminder(ReminderBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=get_utc_now)
    appointment: Appointment = Relationship(back_populates="reminders")
