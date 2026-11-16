from typing import Literal, Optional
from pydantic import BaseModel, Field


class IntentClassification(BaseModel):
    """Structured output schema for the intent routing node."""

    user_intent: Literal[
        "faq",
        "appointment",
        "appointment_reschedule",
        "human_escalation",
        "out_of_scope",
    ] = Field(description="The primary intent of the user's latest message")
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="A score from 0.0 to 1.0 indicating confidence in this intent classification",
    )


class SchedulingExtractionMixin(BaseModel):
    """
    Shared LLM extraction fields for dates, ambiguity, and final-summary confirmation.
    The LLM resolves dates by LOOKING THEM UP in the injected CALENDAR; Python validates.
    """

    date_unclear: bool = Field(
        default=False,
        description=(
            "true ONLY if the latest message refers to a date that does not map to exactly one "
            "calendar day (e.g. 'sometime next week', 'early December', 'whenever')."
        ),
    )
    date_weekday: Optional[str] = Field(
        default=None,
        description="English weekday name of the date you returned (copied from the CALENDAR row), else null.",
    )
    confirmation_response: Literal["yes", "no", "none"] = Field(
        default="none",
        description=(
            "Only when the assistant just showed a summary and asked the user to confirm it: "
            "'yes' if the user approves, 'no' if they decline or want something changed. Otherwise 'none'."
        ),
    )


class BookingExtraction(SchedulingExtractionMixin):
    """Structured output schema for extracting new appointment booking details from a user turn."""

    service_name: Optional[str] = Field(
        default=None,
        description="Spa service the user asked for in their LATEST message, else null.",
    )
    user_name: Optional[str] = Field(
        default=None,
        description="Full name the user gave in their LATEST message, else null.",
    )
    user_email: Optional[str] = Field(
        default=None,
        description="Email address the user gave in their LATEST message, else null.",
    )
    appointment_date: Optional[str] = Field(
        default=None,
        description=(
            "Date the user asked for in their LATEST message as YYYY-MM-DD, looked up from the CALENDAR. "
            "null if the latest message mentions no date."
        ),
    )
    appointment_time: Optional[str] = Field(
        default=None,
        description=(
            "Time the user picked in their LATEST message, HH:MM 24-hour (e.g. '14:30'). "
            "Pick from OFFERED TIMES when the user refers to one. null if no time mentioned."
        ),
    )


class RescheduleExtraction(SchedulingExtractionMixin):
    """
    Unified structured output schema for the rescheduling flow.
    Extracts the user's email, target appointment selection, and requested modifications
    in a single pass so multi-part messages ('My email is X, move my Botox to tomorrow at 2pm')
    are captured completely on the first turn.
    """

    user_email: Optional[str] = Field(
        default=None,
        description="Email address provided by the user to look up their appointments",
    )
    appointment_id: Optional[str] = Field(
        default=None,
        description=(
            "UUID of the existing appointment to modify, copied exactly from EXISTING APPOINTMENTS "
            "if the user specifies which one (e.g. 'the Botox one', 'Monday's appointment', '#1')"
        ),
    )
    new_appointment_date: Optional[str] = Field(
        default=None,
        description=(
            "NEW date the user wants to move to, as YYYY-MM-DD looked up from the CALENDAR. "
            "null if the latest message requests no date change."
        ),
    )
    new_appointment_time: Optional[str] = Field(
        default=None,
        description=(
            "NEW time the user wants, HH:MM 24-hour (e.g. '15:00'). Pick from OFFERED TIMES when the "
            "user refers to one. null if no time change requested."
        ),
    )
    new_service_name: Optional[str] = Field(
        default=None,
        description="Replacement treatment as written in the latest message. Null when a treatment only identifies the original appointment.",
    )
