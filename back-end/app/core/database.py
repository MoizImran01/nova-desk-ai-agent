from contextlib import contextmanager
import logging
from typing import Generator

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import SQLModel, Session, create_engine

from app.core.config import settings
from app.core.exceptions import DatabaseOperationError
from app.models.domain import (
    Appointment,
    Conversation,
    Message,
    Reminder,
    Service,
    ServiceCategory,
    User,
)

logger = logging.getLogger(__name__)

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=settings.DB_POOL_RECYCLE_SECONDS,
)


def create_db_and_tables() -> None:
    """
    Ensure all SQLModel tables and critical ACID indexes exist in the database.
    Because we are not using Alembic migrations right now, we also explicitly
    run `CREATE UNIQUE INDEX IF NOT EXISTS` so existing databases automatically
    gain the double-booking protection on startup.
    """
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ix_unique_scheduled_slot
                ON appointment (appointment_date, appointment_time)
                WHERE status = 'scheduled';
                """
            )
        )


def create_session() -> Session:
    return Session(engine)


@contextmanager
def atomic_transaction(session: Session) -> Generator[Session, None, None]:
    """
    Unit-of-Work helper that wraps multiple database writes in a single atomic
    transaction. Commits once at the end of the block, or rolls back everything
    if an exception occurs.
    """
    try:
        yield session
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.error("Database transaction rolled back due to SQLAlchemyError: %s", exc, exc_info=True)
        raise DatabaseOperationError("A database error occurred while processing your request.") from exc
    except Exception:
        session.rollback()
        raise
