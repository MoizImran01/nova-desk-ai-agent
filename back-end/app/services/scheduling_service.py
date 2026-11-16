from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, time, timedelta
import logging
import sqlite3
from typing import Optional
import uuid

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from app.core.config import settings
from app.core.database import atomic_transaction
from app.core.exceptions import DatabaseOperationError
from app.models.domain import Appointment
from app.repositories import (
    AppointmentRepository,
    ServiceRepository,
    UserRepository,
)
from app.schemas.scheduling import (
    AppointmentLookup,
    AppointmentSummary,
    Availability,
    DaySlots,
    FullDay,
    InvalidDate,
    SchedulingFailure,
    SchedulingResult,
    SlotConflict,
)
from app.utils.date_utils import (
    booking_horizon_end,
    bookable_slots_for,
    format_date_display,
    format_time_display,
    now_local,
    parse_iso_date,
    parse_time,
    today_local,
    validate_date,
    validate_slot_time,
)

logger = logging.getLogger(__name__)


def _is_slot_conflict(exc: BaseException) -> bool:
    error = exc if isinstance(exc, IntegrityError) else exc.__cause__
    if not isinstance(error, IntegrityError):
        return False
    cause = error.orig
    if isinstance(cause, sqlite3.IntegrityError):
        return str(cause) == "UNIQUE constraint failed: appointment.appointment_date, appointment.appointment_time"
    return getattr(getattr(cause, "diag", None), "constraint_name", None) == "ix_unique_scheduled_slot"


def _appointment_summary(appointment: Appointment) -> AppointmentSummary:
    day = appointment.appointment_date
    start = appointment.appointment_time
    return {
        "appointment_id": str(appointment.id),
        "appointment_date": day.isoformat() if day else "",
        "date_display": format_date_display(day),
        "appointment_time": format_time_display(start) if start else "TBD",
        "service_name": appointment.service_name,
    }


@dataclass(frozen=True)
class SchedulingPreview:
    service_name: str | None
    availability: Availability | None
    valid_service_names: tuple[str, ...] = ()


class SchedulingService:
    """
    Application service encapsulating all appointment scheduling, availability
    calculation, fuzzy service matching, and ACID-compliant booking/modification.
    """

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def preview(
        self,
        service_name: str | None,
        appointment_date: str | None,
        *,
        exclude_appointment_id: str | None = None,
    ) -> SchedulingPreview:
        """Resolve the requested treatment and check its target date in one read scope."""
        with self._session_factory() as session:
            if service_name is not None:
                match, choices = ServiceRepository(session).match_service(service_name)
                if match is None:
                    return SchedulingPreview(None, None, choices)
                service_name = match.name

            availability = None
            if appointment_date is not None:
                availability = self._available_slots(session, appointment_date, exclude_appointment_id)
            return SchedulingPreview(service_name, availability)

    def check_available_slots(
        self,
        appointment_date: str,
        *,
        exclude_appointment_id: str | uuid.UUID | None = None,
    ) -> Availability:
        with self._session_factory() as session:
            return self._available_slots(session, appointment_date, exclude_appointment_id)

    @staticmethod
    def _available_slots(
        session: Session,
        appointment_date: str,
        exclude_appointment_id: str | uuid.UUID | None = None,
    ) -> Availability:
        """
        Check available appointment slots on `appointment_date` ('YYYY-MM-DD').
        If fully booked, look ahead up to `LOOKAHEAD_DAYS_FOR_ALTERNATIVES` days
        (capped at the booking horizon) using a SINGLE database query.

        A reschedule can retain the selected appointment's own start time.
        Returned slots exclude times that have already passed today.
        """
        now = now_local()
        today = now.date()
        check = validate_date(appointment_date, today)
        if check.error or check.value is None:
            return InvalidDate(check.error or "I couldn't pin down that date.")
        requested_date = check.value

        horizon_end = booking_horizon_end(today)
        end_date = min(
            requested_date + timedelta(days=settings.LOOKAHEAD_DAYS_FOR_ALTERNATIVES),
            horizon_end,
        )

        if isinstance(exclude_appointment_id, str):
            exclude_appointment_id = uuid.UUID(exclude_appointment_id.strip())
        booked_by_date = AppointmentRepository(session).get_booked_times_for_date_range(
            start_date=requested_date,
            end_date=end_date,
            exclude_appointment_id=exclude_appointment_id,
        )

        def open_slots(day: date) -> DaySlots:
            occupied = booked_by_date.get(day, set())
            return DaySlots(day, tuple(t for t in bookable_slots_for(day, now) if t not in occupied))

        requested = open_slots(requested_date)
        if requested.times:
            return requested

        alternatives = []
        for offset in range(1, (end_date - requested_date).days + 1):
            candidate = open_slots(requested_date + timedelta(days=offset))
            if candidate.times:
                alternatives.append(candidate)
                if len(alternatives) == settings.MAX_ALTERNATIVE_DAYS_RETURNED:
                    break
        return FullDay(requested_date, tuple(alternatives), end_date, end_date == horizon_end)

    def _build_slot_conflict_result(
        self,
        session: Session,
        target_date: date,
        target_time: time,
        *,
        exclude_appointment_id: Optional[uuid.UUID] = None,
    ) -> SlotConflict:
        availability = self._available_slots(session, target_date.isoformat(), exclude_appointment_id)
        return {
            "status": "error",
            "error_code": "slot_unavailable" if isinstance(availability, DaySlots) else "slot_unavailable_day_full",
            "attempted_date": target_date,
            "attempted_time": target_time,
            "availability": availability,
        }

    def _validate_date_and_time(
        self,
        target_date: Optional[date],
        target_time: Optional[time],
        *,
        date_prompt: str,
        time_prompt: str,
    ) -> SchedulingFailure | None:
        """Server-side defense in depth: horizon, past-date, slot-grid and past-time checks."""
        today = today_local()
        if target_date is not None:
            date_check = validate_date(target_date, today)
            if date_check.error:
                return {
                    "status": "error",
                    "error_code": "invalid_date",
                    "message": f"{date_check.error} {date_prompt}",
                }
        if target_time is not None:
            time_check = validate_slot_time(target_time, on=target_date)
            if time_check.error:
                return {
                    "status": "error",
                    "error_code": "invalid_time",
                    "message": f"{time_check.error} {time_prompt}",
                }
        return None

    def book_appointment(
        self,
        *,
        user_name: str,
        appointment_date: str,
        appointment_time: str,
        service_name: str,
        user_email: Optional[str] = None,
        user_phone: Optional[str] = None,
    ) -> SchedulingResult:
        """
        Atomically validate, lock the target `(date, time)` slot, upsert the user,
        and create a `'scheduled'` appointment. Returns a structured dict with
        `status` ("success" | "error") and `error_code`.
        """
        parsed_date = parse_iso_date(appointment_date)
        if not parsed_date:
            return {
                "status": "error",
                "error_code": "invalid_date",
                "message": "I couldn't recognize that date. What date would you like to come in? ✨",
            }

        parsed_time = parse_time(appointment_time)
        if not parsed_time:
            return {
                "status": "error",
                "error_code": "invalid_time",
                "message": "I couldn't recognize that time. Which available time slot would you prefer? ✨",
            }

        validation_error = self._validate_date_and_time(
            parsed_date,
            parsed_time,
            date_prompt="What date would you like to book? ✨",
            time_prompt="Which available time slot would you prefer? ✨",
        )
        if validation_error:
            return validation_error

        with self._session_factory() as session:
            appointments = AppointmentRepository(session)
            services = ServiceRepository(session)
            users = UserRepository(session)
            try:
                with atomic_transaction(session):
                    # 1. Fuzzy-match service in catalog
                    service, valid_names = services.match_service(service_name)
                    if not service:
                        available_str = f": **{', '.join(valid_names)}**" if valid_names else ""
                        return {
                            "status": "error",
                            "error_code": "invalid_service",
                            "message": (
                                f"I couldn't find **{service_name}** on our treatment menu. "
                                f"Here are our available services{available_str}. "
                                "Which treatment would you like to book? 🌸"
                            ),
                        }

                    # 2. Acquire transaction-scoped advisory lock for (date, time)
                    appointments.acquire_slot_advisory_lock(parsed_date, parsed_time)

                    # 3. Verify slot is not already booked by another guest
                    if appointments.is_slot_taken(parsed_date, parsed_time):
                        return self._build_slot_conflict_result(session, parsed_date, parsed_time)

                    # 4. Resolve existing user by email or create a new user via UserRepository
                    user = users.get_by_email(user_email) if user_email else None
                    if user:
                        user = users.update_contact_info(
                            user,
                            full_name=user_name,
                            phone=user_phone,
                        )
                    else:
                        user = users.create_user(
                            full_name=user_name,
                            email=user_email,
                            phone=user_phone,
                        )

                    # 5. Create the scheduled appointment
                    appointment = appointments.create_scheduled_appointment(
                        user_id=user.id,
                        service_id=service.id,
                        service_name=service.name,
                        appointment_date=parsed_date,
                        appointment_time=parsed_time,
                    )

                    summary = _appointment_summary(appointment)
                    summary.update(user_name=user_name, user_email=user_email)
                    return {
                        "status": "success",
                        "error_code": None,
                        "appointment": summary,
                        "message": (
                            f"Your appointment is officially confirmed, **{user_name}**! ✨\n\n"
                            f"- **Service:** {service.name}\n"
                            f"- **Date:** {summary['date_display']}\n"
                            f"- **Time:** {summary['appointment_time']}\n\n"
                            "Thank you for booking with Nova Wellness & Aesthetics! We look forward to welcoming you. 🌸"
                        ),
                    }
            except (IntegrityError, DatabaseOperationError) as exc:
                if not _is_slot_conflict(exc):
                    raise
                logger.warning(
                    "IntegrityError prevented concurrent double-booking on %s at %s",
                    parsed_date,
                    parsed_time,
                )
                return self._build_slot_conflict_result(session, parsed_date, parsed_time)

    def get_appointments_by_email(self, user_email: str) -> AppointmentLookup:
        """Look up upcoming scheduled appointments for a user by email address."""
        if not user_email or not user_email.strip():
            return {"status": "not_found", "appointments": []}

        with self._session_factory() as session:
            today = today_local()
            user = UserRepository(session).get_by_email(user_email)
            if not user:
                return {"status": "not_found", "appointments": []}

            appointments = AppointmentRepository(session).get_upcoming_scheduled_by_user(
                user_id=user.id,
                from_date=today,
            )
            if not appointments:
                return {
                    "status": "no_appointments",
                    "appointments": [],
                }

            return {
                "status": "found",
                "appointments": [_appointment_summary(appt) for appt in appointments],
            }

    def modify_appointment(
        self,
        *,
        appointment_id: str,
        new_appointment_date: Optional[str] = None,
        new_appointment_time: Optional[str] = None,
        new_service_name: Optional[str] = None,
    ) -> SchedulingResult:
        """
        Atomically lock an existing appointment (`FOR UPDATE`), validate any new
        date/time/service, lock and verify the target slot is free, and update
        both `service_id` and `service_name` together.
        """
        if not any([new_appointment_date, new_appointment_time, new_service_name]):
            return {
                "status": "error",
                "error_code": "no_changes",
                "message": "What changes would you like to make to your appointment (the date, the time, or the service)? ✨",
            }

        try:
            parsed_uuid = uuid.UUID(str(appointment_id).strip())
        except (ValueError, TypeError, AttributeError):
            return {
                "status": "error",
                "error_code": "not_found",
                "message": "I couldn't locate that appointment. Could you confirm which appointment you'd like to modify? 🌸",
            }

        parsed_new_date: Optional[date] = None
        if new_appointment_date:
            parsed_new_date = parse_iso_date(new_appointment_date)
            if not parsed_new_date:
                return {
                    "status": "error",
                    "error_code": "invalid_date",
                    "message": "I couldn't recognize that date. What new date would you like to move your appointment to? ✨",
                }

        parsed_new_time: Optional[time] = None
        if new_appointment_time:
            parsed_new_time = parse_time(new_appointment_time)
            if not parsed_new_time:
                return {
                    "status": "error",
                    "error_code": "invalid_time",
                    "message": "I couldn't recognize that time. Which available time would you prefer? ✨",
                }

        target_date: Optional[date] = parsed_new_date
        target_time: Optional[time] = parsed_new_time
        with self._session_factory() as session:
            appointments = AppointmentRepository(session)
            services = ServiceRepository(session)
            try:
                with atomic_transaction(session):
                    # 1. Lock the existing appointment row with SELECT ... FOR UPDATE
                    appointment = appointments.get_by_id_for_update(parsed_uuid)
                    if not appointment:
                        return {
                            "status": "error",
                            "error_code": "not_found",
                            "message": "I couldn't find that appointment in our system. Could you share the email address used for booking? 🌸",
                        }

                    # 2. If date or time is changing, validate and lock the target slot
                    target_date = parsed_new_date or appointment.appointment_date
                    target_time = parsed_new_time or appointment.appointment_time

                    if parsed_new_date or parsed_new_time:
                        validation_error = self._validate_date_and_time(
                            target_date,
                            target_time,
                            date_prompt="What new date would you like to move your appointment to? ✨",
                            time_prompt="Which available time would you prefer? ✨",
                        )
                        if validation_error:
                            return validation_error

                        if target_date and target_time:
                            appointments.acquire_slot_advisory_lock(target_date, target_time)
                            if appointments.is_slot_taken(
                                target_date,
                                target_time,
                                exclude_appointment_id=appointment.id,
                            ):
                                return self._build_slot_conflict_result(
                                    session, target_date, target_time, exclude_appointment_id=parsed_uuid
                                )

                    # 3. If service is changing, resolve both service_id and service_name
                    matched_service_id: Optional[uuid.UUID] = None
                    matched_service_name: Optional[str] = None
                    if new_service_name:
                        service, valid_names = services.match_service(new_service_name)
                        if not service:
                            available_str = f": **{', '.join(valid_names)}**" if valid_names else ""
                            return {
                                "status": "error",
                                "error_code": "invalid_service",
                                "message": (
                                    f"I couldn't find **{new_service_name}** on our treatment menu. "
                                    f"Available treatments{available_str}. Which service would you like to switch to? 🌸"
                                ),
                            }
                        matched_service_id = service.id
                        matched_service_name = service.name

                    # 4. Persist updates atomically
                    updated = appointments.update_appointment(
                        appointment,
                        new_date=parsed_new_date,
                        new_time=parsed_new_time,
                        new_service_id=matched_service_id,
                        new_service_name=matched_service_name,
                    )

                    summary = _appointment_summary(updated)
                    return {
                        "status": "success",
                        "error_code": None,
                        "appointment": summary,
                        "message": (
                            f"Your appointment has been updated! ✨ Here are your new details:\n\n"
                            f"- **Service:** {updated.service_name}\n"
                            f"- **Date:** {summary['date_display']}\n"
                            f"- **Time:** {summary['appointment_time']}\n\n"
                            "We look forward to seeing you at Nova Wellness & Aesthetics! 🌸"
                        ),
                    }
            except (IntegrityError, DatabaseOperationError) as exc:
                if not _is_slot_conflict(exc):
                    raise
                logger.warning(
                    "IntegrityError prevented concurrent double-booking during modify_appointment for %s",
                    appointment_id,
                )
                if target_date and target_time:
                    return self._build_slot_conflict_result(
                        session, target_date, target_time, exclude_appointment_id=parsed_uuid
                    )
                raise
