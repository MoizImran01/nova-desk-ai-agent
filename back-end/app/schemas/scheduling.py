"""Scheduling facts shared by the service and conversational flows."""

from dataclasses import dataclass
from datetime import date, time
from typing import Literal

from typing_extensions import NotRequired, TypedDict


@dataclass(frozen=True)
class DaySlots:
    date: date
    times: tuple[time, ...]


@dataclass(frozen=True)
class FullDay:
    date: date
    alternatives: tuple[DaySlots, ...]
    window_end: date
    at_booking_horizon: bool


@dataclass(frozen=True)
class InvalidDate:
    message: str


Availability = DaySlots | FullDay | InvalidDate


class AppointmentSummary(TypedDict):
    appointment_id: str
    appointment_date: str
    date_display: str
    appointment_time: str
    service_name: str | None
    user_name: NotRequired[str]
    user_email: NotRequired[str | None]


class Scheduled(TypedDict):
    status: Literal["success"]
    error_code: None
    appointment: AppointmentSummary
    message: str


class SchedulingFailure(TypedDict):
    status: Literal["error"]
    error_code: Literal["invalid_date", "invalid_time", "invalid_service", "no_changes", "not_found"]
    message: str


class SlotConflict(TypedDict):
    status: Literal["error"]
    error_code: Literal["slot_unavailable", "slot_unavailable_day_full"]
    availability: Availability
    attempted_date: date
    attempted_time: time


SchedulingResult = Scheduled | SchedulingFailure | SlotConflict


class AppointmentLookup(TypedDict):
    status: Literal["found", "not_found", "no_appointments"]
    appointments: list[AppointmentSummary]
