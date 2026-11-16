from typing import Annotated, List, Literal, Optional, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.schemas.scheduling import AppointmentSummary


class IntentState(TypedDict, total=False):
    """Persisted state for the latest intent classification."""

    user_intent: Literal[
        "faq",
        "appointment",
        "appointment_reschedule",
        "human_escalation",
        "out_of_scope",
    ]
    confidence_score: float


class SchedulingFlowState(TypedDict, total=False):
    """Fields shared by the booking and rescheduling flows."""

    offered_times: List[str]                 # HH:MM slots last shown to the user (LLM hint)
    awaiting_confirmation: bool              # Final summary shown, waiting for yes/no
    confirmed: bool                          # User approved the final summary
    notice: Optional[str]                    # Transient user-facing problem for this turn


class BookingState(SchedulingFlowState, total=False):
    """Persisted state for an active new-appointment booking flow."""

    user_name: Optional[str]
    user_email: Optional[str]
    service_name: Optional[str]
    appointment_date: Optional[str]          # YYYY-MM-DD (LLM calendar lookup, Python-validated)
    appointment_time: Optional[str]          # HH:MM 24-hour (validated against the slot grid)


class RescheduleState(SchedulingFlowState, total=False):
    """
    Unified persisted state for an active appointment rescheduling/modification flow.
    """

    user_email: Optional[str]
    lookup_done: bool
    lookup_status: Optional[Literal["found", "not_found", "no_appointments"]]
    existing_appointments: List[AppointmentSummary]
    appointment_id: Optional[str]
    new_appointment_date: Optional[str]      # YYYY-MM-DD (LLM calendar lookup, Python-validated)
    new_appointment_time: Optional[str]      # HH:MM 24-hour (validated against the slot grid)
    new_service_name: Optional[str]


class AgentState(TypedDict):
    """Root state schema persisted by LangGraph's checkpointer across conversation turns."""

    messages: Annotated[List[BaseMessage], add_messages]
    intent: IntentState
    appointment_details: BookingState
    reschedule_details: RescheduleState
    retrieved_contexts: List[str]            # Raw FAQ chunks retrieved during the turn (for DeepEval)
    executed_tools: List[str]                # Service operations executed during the current turn


def get_latest_human_message(state: AgentState) -> str:
    """Safely extract the text content of the most recent HumanMessage in state."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) and msg.content:
            return str(msg.content)
    return ""


def get_last_ai_message(state: AgentState) -> str:
    """Text of the assistant message that preceded the latest human message (if any)."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return ""


def record_tool(state: AgentState, tool_name: str) -> list[str]:
    """Append `tool_name` to the current turn's `executed_tools` list without duplicates."""
    existing = list(state.get("executed_tools", []))
    if tool_name not in existing:
        existing.append(tool_name)
    return existing
