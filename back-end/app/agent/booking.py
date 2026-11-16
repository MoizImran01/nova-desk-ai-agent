import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage

from app.agent.chat_model import LLMRegistry, clean_str, invoke_structured
from app.agent.prompts import (
    BOOKING_SUMMARY_TEMPLATE,
    COLLECT_APPOINTMENT_DETAILS_PROMPT,
    DECLINED_SUMMARY_MESSAGE,
    MISSING_BOOKING_FIELD_PROMPTS,
    build_date_rules,
    llm_view,
)
from app.agent.scheduling_dialogue import (
    check_selected_slot,
    finish_scheduling,
    offer_slots,
    merge_date_time,
    resolve_confirmation,
    summary_reply,
)
from app.agent.state import (
    AgentState,
    BookingState,
    get_last_ai_message,
    get_latest_human_message,
    record_tool,
)
from app.schemas.agent import BookingExtraction
from app.services.scheduling_service import SchedulingService
from app.utils.date_utils import format_date_display, format_time_display, today_local

logger = logging.getLogger(__name__)

_BOOKING_REQUIRED_FIELDS = ("service_name", "appointment_date", "appointment_time", "user_name", "user_email")
_BOOKING_TEXT_FIELDS = ("service_name", "user_name", "user_email")


def _first_missing(details: dict[str, Any]) -> str | None:
    return next((field for field in _BOOKING_REQUIRED_FIELDS if not details.get(field)), None)


def collect_appointment_details(state: AgentState) -> dict:
    """
    Extract booking details with a calendar-grounded LLM call, then validate every
    value in Python and advance the final-summary confirmation state machine.
    """
    existing: dict[str, Any] = dict(state.get("appointment_details", {}))
    message = get_latest_human_message(state)
    reply = summary_reply(existing, message, DECLINED_SUMMARY_MESSAGE)
    if reply is not None:
        return {"appointment_details": {**existing, **reply}}

    structured_chat_model = LLMRegistry.get_structured_llm(
        "collect_appointment_details",
        BookingExtraction,
    )
    today = today_local()

    prompt = COLLECT_APPOINTMENT_DETAILS_PROMPT.format(
        current_appointment_details=llm_view(existing),
        last_ai_message=get_last_ai_message(state),
        latest_human_message=message,
        date_rules=build_date_rules(
            today,
            existing.get("appointment_date"),
            existing.get("offered_times"),
        ),
    )
    extracted = invoke_structured(structured_chat_model, prompt, BookingExtraction)

    updated: dict[str, Any] = {**existing, "notice": None}
    for key in _BOOKING_TEXT_FIELDS:
        value = clean_str(extracted.get(key))
        if value:
            updated[key] = value

    date_updates, time_rejected = merge_date_time(
        extracted=extracted,
        current=existing,
        date_key="appointment_date",
        time_key="appointment_time",
        today=today,
    )
    updated.update(date_updates)
    if time_rejected:
        first_missing = _first_missing(updated)
        if first_missing and first_missing != "appointment_time":
            updated["notice"] += "\n\n" + MISSING_BOOKING_FIELD_PROMPTS[first_missing]
    updated.update(resolve_confirmation(
        before=existing,
        after=updated,
        tracked_fields=_BOOKING_REQUIRED_FIELDS,
        extracted=extracted,
        decline_message=DECLINED_SUMMARY_MESSAGE,
    ))

    booking: BookingState = updated  # type: ignore[assignment]
    logger.debug("Updated BookingState: %s", booking)
    return {"appointment_details": booking}


def route_after_collecting_details(state: AgentState) -> Literal[
    "ask_booking_clarification",
    "handle_appointment_booking",
]:
    details = state.get("appointment_details", {})
    if details.get("notice") or _first_missing(details) or not details.get("confirmed"):
        return "ask_booking_clarification"
    return "handle_appointment_booking"


def ask_booking_clarification(state: AgentState, *, scheduler: SchedulingService) -> dict:
    """
    Deterministic clarification node for the booking flow (0 LLM calls):
    - Case 1: `appointment_time` is the next missing field -> show open slots for the date
      (prefixed by any validation notice).
    - Case 2: Another required field is missing -> show the notice, or ask for that field.
    - Case 3: All fields present but a notice exists (user declined the summary) -> show it.
    - Case 4: All fields present -> re-verify the slot & service, then show the final summary.
    """
    details: dict[str, Any] = dict(state.get("appointment_details", {}))
    notice = details.pop("notice", None)
    first_missing = _first_missing(details)

    # Case 1: Show available slots for the chosen date
    if first_missing == "appointment_time" and details.get("appointment_date"):
        date = details["appointment_date"]
        availability = scheduler.check_available_slots(date)
        message = offer_slots(details, availability, date_key="appointment_date", time_key="appointment_time", notice=notice)
        return {
            "messages": [AIMessage(content=message)],
            "appointment_details": details,
            "executed_tools": record_tool(state, "check_available_appointment_slots"),
        }

    # Case 2 & 3: Ask for the next missing field, or surface the notice
    if first_missing or notice:
        question = notice or MISSING_BOOKING_FIELD_PROMPTS[first_missing]
        return {"messages": [AIMessage(content=question)], "appointment_details": details}

    # Case 4: Final summary — re-verify the service and that the slot is still open
    requested_service = details["service_name"]
    preview = scheduler.preview(requested_service, details["appointment_date"])
    if preview.availability is None:
        valid_services = ", ".join(preview.valid_service_names)
        details["service_name"] = None
        message = (
            f"I couldn't find **{requested_service}** on our treatment menu. "
            f"Here are our available services: **{valid_services}**. Which treatment would you like to book? 🌸"
        )
        return {"messages": [AIMessage(content=message)], "appointment_details": details}
    details["service_name"] = preview.service_name

    clarification = check_selected_slot(
        details, preview.availability, date_key="appointment_date", time_key="appointment_time"
    )
    tools = record_tool(state, "check_available_appointment_slots")
    if clarification:
        return {
            "messages": [AIMessage(content=clarification)],
            "appointment_details": details,
            "executed_tools": tools,
        }

    summary = BOOKING_SUMMARY_TEMPLATE.format(
        service_name=details["service_name"],
        date_display=format_date_display(details["appointment_date"]),
        time_display=format_time_display(details["appointment_time"]),
        user_name=details["user_name"],
        user_email=details["user_email"],
    )
    details["awaiting_confirmation"] = True
    return {"messages": [AIMessage(content=summary)], "appointment_details": details, "executed_tools": tools}


def handle_appointment_booking(state: AgentState, *, scheduler: SchedulingService) -> dict:
    """
    Executes the final atomic booking step directly in Python via `SchedulingService.book_appointment`
    once all required fields are present AND the user confirmed the summary (0 LLM calls).
    """
    details = dict(state.get("appointment_details", {}))
    tools = record_tool(state, "book_appointment")

    result = scheduler.book_appointment(
        user_name=details.get("user_name") or "",
        appointment_date=details.get("appointment_date") or "",
        appointment_time=details.get("appointment_time") or "",
        service_name=details.get("service_name") or "",
        user_email=details.get("user_email"),
    )

    message, updated = finish_scheduling(
        details, result,
        date_key="appointment_date", time_key="appointment_time", service_key="service_name",
    )
    return {
        "messages": [AIMessage(content=message)],
        "appointment_details": updated,
        "executed_tools": tools,
    }
