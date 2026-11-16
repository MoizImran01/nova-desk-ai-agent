from collections import defaultdict
from datetime import date, time
from typing import Dict, List, Optional, Set
import uuid

from sqlalchemy import text
from sqlmodel import Session, select

from app.models.domain import Appointment, get_utc_now
from app.repositories.base import BaseRepository


class AppointmentRepository(BaseRepository[Appointment]):
    """
    Data access layer for `Appointment` entities, including ACID slot locking,
    pessimistic row locking (`FOR UPDATE`), and single-query multi-day availability lookups.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, Appointment)

    def acquire_slot_advisory_lock(
        self,
        appointment_date: date,
        appointment_time: time,
    ) -> None:
        """
        Acquire a transaction-level PostgreSQL advisory lock for a specific
        `(appointment_date, appointment_time)` slot.

        Why this is needed:
        When inserting a brand-new appointment, no row exists yet to lock via
        `SELECT ... FOR UPDATE`. Calling `pg_advisory_xact_lock(hashtext(:slot_key))`
        serializes concurrent attempts to book the exact same date+time slot, and
        automatically releases the lock when the transaction commits or rolls back.
        """
        dialect_name = self.session.bind.dialect.name if self.session.bind else ""
        if dialect_name == "postgresql":
            slot_key = f"slot:{appointment_date.isoformat()}:{appointment_time.strftime('%H:%M')}"
            self.session.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:slot_key))"),
                {"slot_key": slot_key},
            )

    def get_by_id_for_update(self, appointment_id: uuid.UUID | str) -> Optional[Appointment]:
        """
        Fetch an `Appointment` by ID and acquire a row-level `FOR UPDATE` lock
        so no other transaction can modify it until the current transaction finishes.
        """
        try:
            parsed_id = (
                uuid.UUID(str(appointment_id))
                if not isinstance(appointment_id, uuid.UUID)
                else appointment_id
            )
        except (ValueError, TypeError):
            return None

        statement = (
            select(Appointment)
            .where(Appointment.id == parsed_id)
            .with_for_update()
        )
        return self.session.exec(statement).first()

    def is_slot_taken(
        self,
        appointment_date: date,
        appointment_time: time,
        exclude_appointment_id: Optional[uuid.UUID] = None,
    ) -> bool:
        """
        Check whether a `(appointment_date, appointment_time)` slot already has an
        active `'scheduled'` appointment. Optionally excludes a specific appointment ID
        when a user is modifying their own existing appointment.
        """
        statement = select(Appointment.id).where(
            Appointment.appointment_date == appointment_date,
            Appointment.appointment_time == appointment_time,
            Appointment.status == "scheduled",
        )
        if exclude_appointment_id is not None:
            statement = statement.where(Appointment.id != exclude_appointment_id)

        return self.session.exec(statement).first() is not None

    def get_booked_times_for_date_range(
        self,
        start_date: date,
        end_date: date,
        *,
        exclude_appointment_id: Optional[uuid.UUID] = None,
    ) -> Dict[date, Set[time]]:
        """
        Fetch all `'scheduled'` appointment times between `start_date` and `end_date`
        in one query, omitting the appointment being rescheduled when supplied.
        """
        statement = select(Appointment.appointment_date, Appointment.appointment_time).where(
            Appointment.appointment_date >= start_date,
            Appointment.appointment_date <= end_date,
            Appointment.status == "scheduled",
        )
        if exclude_appointment_id is not None:
            statement = statement.where(Appointment.id != exclude_appointment_id)
        rows = self.session.exec(statement).all()

        booked_by_date: Dict[date, Set[time]] = defaultdict(set)
        for appt_date, appt_time in rows:
            if appt_date and appt_time:
                booked_by_date[appt_date].add(appt_time)

        return booked_by_date

    def get_upcoming_scheduled_by_user(
        self,
        user_id: uuid.UUID,
        from_date: date,
    ) -> List[Appointment]:
        """Return all upcoming `'scheduled'` appointments for a user ordered chronologically."""
        statement = (
            select(Appointment)
            .where(
                Appointment.user_id == user_id,
                Appointment.appointment_date >= from_date,
                Appointment.status == "scheduled",
            )
            .order_by(Appointment.appointment_date, Appointment.appointment_time)
        )
        return list(self.session.exec(statement).all())

    def create_scheduled_appointment(
        self,
        *,
        user_id: uuid.UUID,
        service_id: uuid.UUID,
        service_name: str,
        appointment_date: date,
        appointment_time: time,
    ) -> Appointment:
        appointment = Appointment(
            user_id=user_id,
            service_id=service_id,
            service_name=service_name,
            appointment_date=appointment_date,
            appointment_time=appointment_time,
            status="scheduled",
        )
        return self.add(appointment)

    def update_appointment(
        self,
        appointment: Appointment,
        *,
        new_date: Optional[date] = None,
        new_time: Optional[time] = None,
        new_service_id: Optional[uuid.UUID] = None,
        new_service_name: Optional[str] = None,
    ) -> Appointment:
        if new_date is not None:
            appointment.appointment_date = new_date
        if new_time is not None:
            appointment.appointment_time = new_time
        if new_service_id is not None and new_service_name is not None:
            appointment.service_id = new_service_id
            appointment.service_name = new_service_name

        appointment.updated_at = get_utc_now()
        self.session.add(appointment)
        self.session.flush()
        return appointment
