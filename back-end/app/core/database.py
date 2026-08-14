from sqlmodel import SQLModel, create_engine
from app.core.config import settings
from sqlmodel import Session
from app.models.domain import User, ServiceCategory, Service, Conversation, Message, Appointment, Reminder

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session