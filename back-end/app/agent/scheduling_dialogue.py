import logging
from typing import Any

from app.agent.chat_model import clean_str
from app.agent.prompts import DATE_UNCLEAR_MESSAGE
from app.core.config import settings
from app.schemas.scheduling import Availability, DaySlots, InvalidDate, SchedulingResult
from app.utils.date_utils import (
    format_date_display,
    format_time_display,
    parse_iso_date,
    parse_time,
    validate_date,
    validate_slot_time,
)

logger = logging.getLogger(__name__)


def _norm(value: Any) -> str:
    """Normalized comparison key so casing/whitespace echoes are not treated as changes."""
    return str(value).strip().lower() if value is not None else ""


def merge_date_time(
    *,
    extracted: dict[str, Any],
    current: dict[str, Any],
    date_key: str,
    time_key: str,
    today,
    time_reference_date=None,
) -> tuple[dict[str, Any], bool]:
    """
    Validate the LLM's resolved date/time and return `(state_updates, time_rejected)`.

    - Invalid / unclear / inconsistent dates clear the date and set a user-facing `notice`.
    - A changed date clears any previously chosen time (it belonged to the old date).
    - Times must be on the slot grid and not in the past when the date is today.
    Nothing is ever dropped silently.
    """
    updates: dict[str, Any] = {}
    raw_date = clean_str(extracted.get(date_key))

    if extracted.get("date_unclear") and not raw_date:
        return {
            date_key: None,
            time_key: None,
            "notice": DATE_UNCLEAR_MESSAGE.format(horizon_days=settings.BOOKING_HORIZON_DAYS),
        }, False

    if raw_date:
        check = validate_date(raw_date, today)
        if check.error or check.value is None:
            return {
                date_key: None,
                time_key: None,
                "notice": f"{check.error} Which date would you like instead? ✨",
            }, False

        weekday = _norm(extracted.get("date_weekday"))
        if weekday and check.value.strftime("%a").lower() != weekday[:3]:
            logger.warning(
                "LLM date/weekday mismatch: %s is a %s, LLM said %s",
                check.value,
                check.value.strftime("%A"),
                weekday,
            )
            return {
                date_key: None,
                time_key: None,
                "notice": DATE_UNCLEAR_MESSAGE.format(horizon_days=settings.BOOKING_HORIZON_DAYS),
            }, False

        new_iso = check.value.isoformat()
        if new_iso != current.get(date_key):
            updates[date_key] = new_iso
            updates[time_key] = None
            updates["offered_times"] = []

    raw_time = clean_str(extracted.get(time_key))
    if raw_time:
        target_date = parse_iso_date(updates.get(date_key) or current.get(date_key)) or time_reference_date
        time_check = validate_slot_time(raw_time, on=target_date)
        if time_check.error or time_check.value is None:
            updates[time_key] = None
            updates["notice"] = time_check.error
            return updates, True
        updates[time_key] = time_check.value.strftime("%H:%M")

    return updates, False


def summary_reply(draft: dict[str, Any], reply: str, decline_message: str) -> dict[str, Any] | None:
    """Only a standalone answer to a pending summary can bypass extraction."""
    if not draft.get("awaiting_confirmation"):
        return None
    answer = reply.strip().lower().rstrip(".!")
    if answer not in ("yes", "no"):
        return None
    return {
        "awaiting_confirmation": False,
        "confirmed": answer == "yes",
        "notice": decline_message if answer == "no" else None,
    }


def resolve_confirmation(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    tracked_fields: tuple[str, ...],
    extracted: dict[str, Any],
    decline_message: str,
) -> dict[str, Any]:
    """
    Final-summary confirmation state machine:
    - Any change to a tracked field (or a validation notice) invalidates a pending summary.
    - 'yes' to a pending summary marks the flow confirmed; 'no' asks what to change.
    """
    changed = any(_norm(before.get(f)) != _norm(after.get(f)) for f in tracked_fields)
    if changed or after.get("notice"):
        return {"awaiting_confirmation": False, "confirmed": False}

    return summary_reply(before, extracted.get("confirmation_response") or "", decline_message) or {"confirmed": False}


def _format_times(slots: DaySlots) -> str:
    return ", ".join(format_time_display(t) for t in slots.times)


def _slot_offer(availability: Availability) -> str:
    if isinstance(availability, InvalidDate):
        return f"{availability.message} Which date would you like instead? ✨"

    day = format_date_display(availability.date)
    if isinstance(availability, DaySlots):
        return (
            f"Here are the available appointment times on **{day}**:\n\n"
            f"**{_format_times(availability)}**\n\nWhich time works best for you? ✨"
        )

    if availability.alternatives:
        choices = "\n".join(
            f"- **{format_date_display(alt.date)}**: {_format_times(alt)}"
            for alt in availability.alternatives
        )
        return (
            f"**{day}** is completely booked, but here are our nearest available dates and times:\n\n"
            f"{choices}\n\nWhich of these dates and times would you prefer? 🌸"
        )
    end = format_date_display(availability.window_end)
    if availability.at_booking_horizon:
        return (
            f"We're completely booked from **{day}** through **{end}**, "
            f"the end of our {settings.BOOKING_HORIZON_DAYS}-day booking window. "
            "Is there an earlier date that works for you? 🌸"
        )
    return f"We're completely booked from **{day}** through **{end}**. Could you share a date after **{end}** that works for you? 🌸"


def offer_slots(
    draft: dict[str, Any], availability: Availability, *, date_key: str, time_key: str,
    notice: str | None = None,
) -> str:
    """Save precisely the options that the guest will see on this turn."""
    if isinstance(availability, DaySlots):
        times = [t.strftime("%H:%M") for t in availability.times]
        draft.update({date_key: availability.date.isoformat(), time_key: None, "offered_times": times})
    else:
        draft.update({date_key: None, time_key: None, "offered_times": []})
    message = _slot_offer(availability)
    return f"{notice}\n\n{message}" if notice else message


def check_selected_slot(
    draft: dict[str, Any], availability: Availability, *, date_key: str, time_key: str,
) -> str | None:
    notice = None
    if isinstance(availability, DaySlots):
        selected = draft.get(time_key)
        if parse_time(selected) in availability.times:
            return None
        if selected:
            notice = (
                f"**{format_time_display(selected)}** on **{format_date_display(availability.date)}** "
                "is no longer available."
            )
    return offer_slots(draft, availability, date_key=date_key, time_key=time_key, notice=notice)


def finish_scheduling(
    draft: dict[str, Any], result: SchedulingResult, *, date_key: str, time_key: str, service_key: str,
) -> tuple[str, dict[str, Any]]:
    if result["status"] == "success":
        return result["message"], {}

    draft.update(awaiting_confirmation=False, confirmed=False, notice=None)
    if "availability" in result:
        availability = result["availability"]
        day = format_date_display(result["attempted_date"])
        time = format_time_display(result["attempted_time"])
        notice = f"The **{time}** slot on **{day}** is already booked!"
        message = offer_slots(draft, availability, date_key=date_key, time_key=time_key, notice=notice)
        return message, draft

    if result["error_code"] == "invalid_time":
        draft[time_key] = None
    elif result["error_code"] == "invalid_date":
        draft.update({date_key: None, time_key: None, "offered_times": []})
    elif result["error_code"] == "invalid_service":
        draft[service_key] = None
    return result["message"], draft
