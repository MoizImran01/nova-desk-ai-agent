from typing import Optional

from sqlmodel import Session, select

from app.models.domain import User, get_utc_now
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """Data access layer strictly for `User` entities."""

    def __init__(self, session: Session) -> None:
        super().__init__(session, User)

    def get_by_email(self, email: str) -> Optional[User]:
        """Fetch a user by normalized email address."""
        normalized_email = email.strip().lower()
        statement = select(User).where(User.email == normalized_email)
        return self.session.exec(statement).first()

    def create_user(
        self,
        *,
        full_name: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
    ) -> User:
        """
        Create a new `User` row (supports both anonymous users when called with no
        arguments and identified users when called with name/email/phone).
        """
        normalized_email = email.strip().lower() if email else None
        new_user = User(
            full_name=full_name,
            email=normalized_email,
            phone=phone,
        )
        return self.add(new_user)

    def update_contact_info(
        self,
        user: User,
        *,
        full_name: Optional[str] = None,
        phone: Optional[str] = None,
    ) -> User:
        """Update an existing user's contact details if new values are provided."""
        updated = False
        if full_name and user.full_name != full_name:
            user.full_name = full_name
            updated = True
        if phone and user.phone != phone:
            user.phone = phone
            updated = True

        if updated:
            user.updated_at = get_utc_now()
            self.session.add(user)
            self.session.flush()
        return user
