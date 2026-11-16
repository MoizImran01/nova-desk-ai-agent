import json
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage

from app.agent.chat_model import LLMRegistry, clean_str, invoke_structured
from app.agent.prompts import (
    ASK_FOR_EMAIL_MESSAGE,
    COLLECT_RESCHEDULE_DETAILS_PROMPT,
    DECLINED_RESCHEDULE_MESSAGE,
    RESCHEDULE_SUMMARY_TEMPLATE,
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
    RescheduleState,
    get_last_ai_message,
    get_latest_human_message,
    record_tool,
)
from app.schemas.agent import RescheduleExtraction
from app.services.scheduling_service import SchedulingService
from app.utils.date_utils import (
    format_date_display,
    format_time_display,
    parse_iso_date,
    today_local,
)

logger = logging.getLogger(__name__)

_RESCHEDULE_CHANGE_FIELDS = ("new_appointment_date", "new_appointment_time", "new_service_name")


def _selected_appointment(details: dict[str, Any]) -> dict[str, Any] | None:
    appointment_id = details.get("appointment_id")
    return next(
        (a for a in details.get("existing_appointments", []) if a.get("appointment_id") == appointment_id),
        None,
    )


def collect_reschedule_details(state: AgentState, *, scheduler: SchedulingService) -> dict:
    """
    Single unified node for the rescheduling flow:
    1. Extracts email, appointment selection, and requested changes (calendar-grounded dates).
    2. Looks up the guest's appointments in Python whenever a new email is provided.
    3. Accepts `appointment_id` only if it matches a looked-up appointment; auto-selects
       when there is exactly one.
    4. Validates the new date/time and advances the final-summary confirmation state machine.
    """
    existing: dict[str, Any] = dict(state.get("reschedule_details", {}))
    message = get_latest_human_message(state)
    reply = summary_reply(existing, message, DECLINED_RESCHEDULE_MESSAGE)
    if reply is not None:
        return {"reschedule_details": {**existing, **reply}}

    structured_chat_model = LLMRegistry.get_structured_llm(
        "collect_reschedule_details",
        RescheduleExtraction,
    )
    today = today_local()
    selected_before = _selected_appointment(existing)
    current_date = existing.get("new_appointment_date") or (
        selected_before.get("appointment_date") if selected_before else None
    )

    prompt = COLLECT_RESCHEDULE_DETAILS_PROMPT.format(
        existing_appointments=json.dumps(existing.get("existing_appointments", []), indent=2),
        current_reschedule=llm_view(existing),
        last_ai_message=get_last_ai_message(state),
        latest_human_message=message,
        date_rules=build_date_rules(today, current_date, existing.get("offered_times")),
    )
    extracted = invoke_structured(structured_chat_model, prompt, RescheduleExtraction)

    merged: dict[str, Any] = {**existing, "notice": None}
    tools = list(state.get("executed_tools", []))

    new_service = clean_str(extracted.get("new_service_name"))
    if new_service:
        merged["new_service_name"] = new_service

    # Step 2: Direct Python DB lookup whenever a new email is provided
    extracted_email = clean_str(extracted.get("user_email"))
    if extracted_email:
        merged["user_email"] = extracted_email
    current_email = merged.get("user_email")
    email_changed = bool(current_email and current_email != existing.get("user_email"))

    if current_email and (not merged.get("lookup_done") or email_changed):
        lookup_result = scheduler.get_appointments_by_email(current_email)
        tools = record_tool(state, "get_appointments_by_email")

        status = lookup_result["status"]
        merged["lookup_status"] = status
        merged["appointment_id"] = None
        if status == "found":
            merged["lookup_done"] = True
            merged["existing_appointments"] = lookup_result["appointments"]
        else:
            merged["lookup_done"] = False
            merged["existing_appointments"] = []

    # Step 3: Accept only appointment IDs that actually belong to this guest
    valid_ids = {a["appointment_id"] for a in merged.get("existing_appointments", [])}
    extracted_id = clean_str(extracted.get("appointment_id"))
    if extracted_id in valid_ids:
        merged["appointment_id"] = extracted_id
    elif extracted_id:
        logger.warning("Ignoring unknown appointment_id from LLM: %s", extracted_id)
    if not merged.get("appointment_id") and len(valid_ids) == 1:
        merged["appointment_id"] = next(iter(valid_ids))

    # Step 4: Validate requested date/time against the selected appointment
    selected = _selected_appointment(merged)
    reference_date = parse_iso_date(selected.get("appointment_date")) if selected else None
    date_updates, time_rejected = merge_date_time(
        extracted=extracted,
        current=merged,
        date_key="new_appointment_date",
        time_key="new_appointment_time",
        today=today,
        time_reference_date=reference_date,
    )
    merged.update(date_updates)
    # A rejected time-only change: show the slot list for the appointment's current date
    if time_rejected and not merged.get("new_appointment_date") and reference_date:
        merged["new_appointment_date"] = reference_date.isoformat()

    merged.update(resolve_confirmation(
        before=existing,
        after=merged,
        tracked_fields=("appointment_id", *_RESCHEDULE_CHANGE_FIELDS),
        extracted=extracted,
        decline_message=DECLINED_RESCHEDULE_MESSAGE,
    ))

    reschedule: RescheduleState = merged  # type: ignore[assignment]
    logger.debug("Updated RescheduleState: %s", reschedule)
    return {
        "reschedule_details": reschedule,
        "executed_tools": tools,
    }


def _next_action(details: dict[str, Any], selected: dict[str, Any] | None) -> Literal[
    "email", "lookup_miss", "selection", "slots", "notice", "changes", "preview", "apply"
]:
    if not details.get("user_email"):
        return "email"
    if not details.get("lookup_done") and details.get("lookup_status") in ("not_found", "no_appointments"):
        return "lookup_miss"
    if selected is None:
        return "selection"
    if details.get("new_appointment_date") and not details.get("new_appointment_time"):
        return "slots"
    if details.get("notice"):
        return "notice"
    if not any(details.get(field) for field in _RESCHEDULE_CHANGE_FIELDS):
        return "changes"
    return "apply" if details.get("confirmed") and details.get("lookup_done") else "preview"


def route_after_collecting_reschedule(state: AgentState) -> Literal[
    "ask_reschedule_clarification", "handle_apply_modification"
]:
    details = state.get("reschedule_details", {})
    if _next_action(details, _selected_appointment(details)) == "apply":
        return "handle_apply_modification"
    return "ask_reschedule_clarification"


def ask_reschedule_clarification(state: AgentState, *, scheduler: SchedulingService) -> dict:
    """
    Deterministic clarification node for the rescheduling flow (0 LLM calls):
    - Case 1: No email yet -> Ask for email.
    - Case 2: Email looked up, but 0 appointments found -> Inform user & reset email.
    - Case 3: Multiple appointments found, none selected yet -> List them & ask which one.
    - Case 4: New date without a time -> Show open slots (prefixed by any notice).
    - Case 5: Validation notice / declined summary -> Show it.
    - Case 6: Appointment selected, but no change specified yet -> Ask what to change.
    - Case 7: Change complete -> Re-verify slot & service, then show the final change summary.
    """
    reschedule: dict[str, Any] = dict(state.get("reschedule_details", {}))
    selected = _selected_appointment(reschedule)
    action = _next_action(reschedule, selected)
    notice = reschedule.pop("notice", None)
    user_email = reschedule.get("user_email")
    existing_appointments = reschedule.get("existing_appointments", [])

    # Case 1: Missing email
    if action == "email":
        return {"messages": [AIMessage(content=ASK_FOR_EMAIL_MESSAGE)], "reschedule_details": reschedule}

    # Case 2: Email looked up, but no upcoming appointments found
    if action == "lookup_miss":
        reset_state = {
            **reschedule,
            "user_email": None,
            "lookup_done": False,
            "lookup_status": None,
            "existing_appointments": [],
        }
        message = (
            f"I couldn't find any upcoming scheduled appointments under **{user_email}**. "
            "Could you double-check the email address, or share another email you might have booked with? 🌸"
        )
        return {"messages": [AIMessage(content=message)], "reschedule_details": reset_state}

    # Case 3: Multiple appointments found, user hasn't picked which one yet
    if action == "selection":
        lines = [
            f"{idx}. **{appt.get('service_name')}** — {appt.get('date_display')} at {appt.get('appointment_time')}"
            for idx, appt in enumerate(existing_appointments, start=1)
        ]
        message = (
            f"I found {len(existing_appointments)} upcoming appointments under **{user_email}**:\n\n"
            + "\n".join(lines)
            + "\n\nWhich appointment would you like to modify, and what changes would you like to make? ✨"
        )
        return {"messages": [AIMessage(content=message)], "reschedule_details": reschedule}

    # Case 4: New date chosen but no time yet -> show open slots
    assert selected is not None
    new_date = reschedule.get("new_appointment_date")
    if action == "slots":
        availability = scheduler.check_available_slots(new_date, exclude_appointment_id=selected["appointment_id"])
        message = offer_slots(reschedule, availability, date_key="new_appointment_date", time_key="new_appointment_time", notice=notice)
        return {
            "messages": [AIMessage(content=message)],
            "reschedule_details": reschedule,
            "executed_tools": record_tool(state, "check_available_appointment_slots"),
        }

    # Case 5: Validation notice or declined summary
    if action == "notice":
        return {"messages": [AIMessage(content=notice)], "reschedule_details": reschedule}

    # Case 6: Appointment selected, but no change specified yet
    if action == "changes":
        message = (
            f"I found your **{selected.get('service_name')}** appointment on "
            f"**{selected.get('date_display')}** at **{selected.get('appointment_time')}**! ✨ "
            "What would you like to change (the date, the time, or the service)?"
        )
        return {"messages": [AIMessage(content=message)], "reschedule_details": reschedule}

    # Case 7: Final change summary — re-verify service and target slot first
    tools = list(state.get("executed_tools", []))
    target_time = reschedule.get("new_appointment_time")
    target_date = new_date or (selected["appointment_date"] if target_time else None)
    requested_service = reschedule.get("new_service_name")
    preview = scheduler.preview(requested_service, target_date, exclude_appointment_id=selected["appointment_id"])
    if requested_service:
        if preview.service_name is None:
            reschedule["new_service_name"] = None
            message = (
                f"I couldn't find **{requested_service}** on our treatment menu. Here are our available services: "
                f"**{', '.join(preview.valid_service_names)}**. Which service would you like to switch to? 🌸"
            )
            return {"messages": [AIMessage(content=message)], "reschedule_details": reschedule}
        reschedule["new_service_name"] = preview.service_name

    if target_date is not None and preview.availability is not None:
        clarification = check_selected_slot(
            reschedule, preview.availability, date_key="new_appointment_date", time_key="new_appointment_time"
        )
        tools = record_tool(state, "check_available_appointment_slots")
        if clarification:
            return {
                "messages": [AIMessage(content=clarification)],
                "reschedule_details": reschedule,
                "executed_tools": tools,
            }

    old_time = selected.get("appointment_time") or "TBD"
    new_time = reschedule.get("new_appointment_time")
    summary = RESCHEDULE_SUMMARY_TEMPLATE.format(
        old_service=selected.get("service_name"),
        old_date=selected.get("date_display"),
        old_time=old_time,
        new_service=reschedule.get("new_service_name") or selected.get("service_name"),
        new_date=format_date_display(new_date) if new_date else selected.get("date_display"),
        new_time=format_time_display(new_time) if new_time else old_time,
    )
    reschedule["awaiting_confirmation"] = True
    return {"messages": [AIMessage(content=summary)], "reschedule_details": reschedule, "executed_tools": tools}


def handle_apply_modification(state: AgentState, *, scheduler: SchedulingService) -> dict:
    """
    Executes the final atomic modification step directly in Python via
    `SchedulingService.modify_appointment` once the user approved the change summary
    (0 LLM calls, 0 tool loops).
    """
    reschedule = dict(state.get("reschedule_details", {}))
    tools = record_tool(state, "modify_appointment")

    result = scheduler.modify_appointment(
        appointment_id=reschedule.get("appointment_id") or "",
        new_appointment_date=reschedule.get("new_appointment_date"),
        new_appointment_time=reschedule.get("new_appointment_time"),
        new_service_name=reschedule.get("new_service_name"),
    )

    message, updated = finish_scheduling(
        reschedule, result,
        date_key="new_appointment_date", time_key="new_appointment_time", service_key="new_service_name",
    )
    return {
        "messages": [AIMessage(content=message)],
        "reschedule_details": updated,
        "executed_tools": tools,
    }
