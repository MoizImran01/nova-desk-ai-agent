"""
Single source of truth for "now", the bookable calendar, and date/time validation.

Strategy:
  - The LLM resolves natural-language dates by LOOKING THEM UP in a calendar table
    built here (`build_calendar_context`). It never does date arithmetic.
  - Python strictly validates every value the LLM returns (`validate_date`,
    `validate_slot_time`). Anything invalid is surfaced to the user, never dropped.
  - The resolved date is always read back to the user (slot list + final summary),
    which acts as the human confirmation step.
"""
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from app.core.config import settings

ISO_DATE = "%Y-%m-%d"
_WEEK_LABELS = ("this week", "next week", "week after next")


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

def now_local() -> datetime:
    return datetime.now(ZoneInfo(settings.SPA_TIMEZONE))


def today_local() -> date:
    return now_local().date()


def booking_horizon_end(today: date | None = None) -> date:
    """Last bookable date (inclusive)."""
    return (today or today_local()) + timedelta(days=settings.BOOKING_HORIZON_DAYS)


# ---------------------------------------------------------------------------
# Parsing & formatting
# ---------------------------------------------------------------------------

def parse_iso_date(value: str | date | None) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value).strip(), ISO_DATE).date()
    except (ValueError, TypeError):
        return None


def parse_time(value: str | time | None) -> time | None:
    """Accept '14:30', '14:30:00', '02:30 PM', '2:30pm', '2 PM'."""
    if isinstance(value, time):
        return value
    if not value or not isinstance(value, str):
        return None
    cleaned = value.strip().upper()
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p", "%I %p", "%I%p"):
        try:
            return datetime.strptime(cleaned, fmt).time()
        except ValueError:
            continue
    return None


def format_date_display(value: str | date | None) -> str:
    """'Friday, November 20, 2026'. Falls back to the raw value."""
    parsed = parse_iso_date(value)
    return parsed.strftime("%A, %B %d, %Y") if parsed else str(value or "")


def format_time_display(value: str | time | None) -> str:
    """'02:30 PM'. Falls back to the raw value."""
    parsed = parse_time(value)
    return parsed.strftime("%I:%M %p") if parsed else str(value or "")


# ---------------------------------------------------------------------------
# Slot grid
# ---------------------------------------------------------------------------

def daily_slot_grid() -> list[time]:
    """All appointment start times within business hours."""
    step = timedelta(minutes=settings.DEFAULT_SLOT_DURATION_MINUTES)
    ref = date(2000, 1, 1)
    current = datetime.combine(ref, settings.business_open_time)
    end = datetime.combine(ref, settings.business_close_time)
    slots: list[time] = []
    while current < end:
        slots.append(current.time())
        current += step
    return slots


def bookable_slots_for(day: date, now: datetime | None = None) -> list[time]:
    """Slot grid for `day`, excluding times that already passed if `day` is today."""
    now = now or now_local()
    grid = daily_slot_grid()
    if day == now.date():
        current = now.time().replace(tzinfo=None)
        return [slot for slot in grid if slot > current]
    return grid


# ---------------------------------------------------------------------------
# LLM calendar context
# ---------------------------------------------------------------------------

def build_calendar_context(today: date | None = None) -> str:
    """
    Lookup table of every bookable date, labelled with weekday and calendar week
    (Mon–Sun), so the LLM resolves "Friday", "next Friday", "this weekend" by lookup.
    """
    today = today or today_local()
    week_start = today - timedelta(days=today.weekday())
    rows: list[str] = []
    for offset in range(settings.BOOKING_HORIZON_DAYS + 1):
        day = today + timedelta(days=offset)
        week_idx = (day - week_start).days // 7
        week_label = _WEEK_LABELS[week_idx] if week_idx < len(_WEEK_LABELS) else f"in {week_idx} weeks"
        tag = " (today)" if offset == 0 else " (tomorrow)" if offset == 1 else ""
        rows.append(f"{day:%Y-%m-%d} | {day:%A}{tag} | {week_label}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DateCheck:
    value: date | None
    error: str | None = None


@dataclass(frozen=True)
class TimeCheck:
    value: time | None
    error: str | None = None


def validate_date(raw: str | date | None, today: date | None = None) -> DateCheck:
    today = today or today_local()
    parsed = parse_iso_date(raw)
    if parsed is None:
        return DateCheck(None, "I couldn't pin down that date.")
    if parsed < today:
        return DateCheck(None, f"**{format_date_display(parsed)}** has already passed.")
    horizon_end = booking_horizon_end(today)
    if parsed > horizon_end:
        return DateCheck(
            None,
            f"We only take bookings up to {settings.BOOKING_HORIZON_DAYS} days ahead "
            f"(through **{format_date_display(horizon_end)}**).",
        )
    return DateCheck(parsed)


def validate_slot_time(
    raw: str | time | None,
    on: date | None = None,
    now: datetime | None = None,
) -> TimeCheck:
    """
    Validate that `raw` is a real appointment start time on the slot grid and,
    when `on` is today, that it hasn't already passed.
    """
    parsed = parse_time(raw)
    if parsed is None or parsed not in daily_slot_grid():
        open_disp = settings.business_open_time.strftime("%I:%M %p")
        close_disp = settings.business_close_time.strftime("%I:%M %p")
        return TimeCheck(
            None,
            f"That isn't one of our appointment times (we book every "
            f"{settings.DEFAULT_SLOT_DURATION_MINUTES} minutes between {open_disp} and {close_disp}).",
        )
    if on is not None and parsed not in bookable_slots_for(on, now):
        return TimeCheck(None, f"**{format_time_display(parsed)}** has already passed today.")
    return TimeCheck(parsed)
