"""
Deterministic date resolution for the booking agent.

The LLM's ONLY job is to look at the user's message and answer two questions:
  1. What KIND of date expression did they use? ("weekday", "relative", "explicit")
  2. What was the raw value? ("friday", "tomorrow", "2026-07-04")

Everything else -- turning "friday" into an actual calendar date, and deciding
whether that's ambiguous enough to need confirmation -- happens here, in plain
Python, with no LLM guessing involved.
"""
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from dataclasses import dataclass

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def get_today(tz: str = "Asia/Karachi") -> date:
    return datetime.now(ZoneInfo(tz)).date()


@dataclass
class ResolvedDate:
    date_obj: date
    date_str: str          # "2026-06-26"
    needs_confirmation: bool
    confirmation_question: str | None = None


def resolve_weekday(
    weekday_name: str,
    today: date,
    qualifier: str | None = None,
) -> ResolvedDate:
    """
    weekday_name: 'friday', 'monday', etc.
    qualifier: None, 'next', or 'this' -- comes from how the user phrased it.
               - None   -> user said just "friday"  -> AMBIGUOUS, ask
               - 'this' -> user said "this friday"  -> nearest one, no ask
               - 'next' -> user said "next friday"  -> the one AFTER nearest, no ask
    """
    weekday_name = weekday_name.strip().lower()
    if weekday_name not in WEEKDAYS:
        raise ValueError(f"'{weekday_name}' is not a recognized weekday name")

    target_idx = WEEKDAYS.index(weekday_name)
    days_ahead = (target_idx - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7  # if today IS that weekday, the nearest one is next week
    nearest = today + timedelta(days=days_ahead)

    if qualifier == "this" or qualifier == "coming":
        return ResolvedDate(nearest, nearest.strftime("%Y-%m-%d"), needs_confirmation=False)

    if qualifier == "next":
        later = nearest + timedelta(days=7)
        return ResolvedDate(later, later.strftime("%Y-%m-%d"), needs_confirmation=False)

    # qualifier is None -- the ambiguous case. Default to nearest, but flag it.
    question = (
        f"Just to confirm — when you say {weekday_name.capitalize()}, do you mean this coming "
        f"{nearest.strftime('%A, %B %d')}, or one further out?"
    )
    return ResolvedDate(nearest, nearest.strftime("%Y-%m-%d"), needs_confirmation=True,
                         confirmation_question=question)


def resolve_relative_term(term: str, today: date) -> ResolvedDate:
    term = term.strip().lower()
    if term == "today":
        d = today
    elif term == "tomorrow":
        d = today + timedelta(days=1)
    else:
        raise ValueError(f"Unrecognized relative term: {term}")
    return ResolvedDate(d, d.strftime("%Y-%m-%d"), needs_confirmation=False)


def resolve_explicit(date_str: str, today: date) -> ResolvedDate:
    """User gave an actual date already, e.g. '2026-07-04'. Just validate it."""
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    return ResolvedDate(d, d.strftime("%Y-%m-%d"), needs_confirmation=False)