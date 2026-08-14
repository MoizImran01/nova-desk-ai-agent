from calendar import c
from pydantic import BaseModel, Field
from typing import Literal, Optional
class IntentClassification(BaseModel):
    user_intent: Literal["faq", "appointment",  "appointment_reschedule", "human_escalation"] = Field(description="The primary intent of the user's latest message")
    confidence_score: str = Field(description="A score from 0.0 to 1.0 indicating how confident you are in this intent classification")

class AppointmentDetails(BaseModel):
    service_name: Optional[str] = Field(default=None)
    user_name: Optional[str] = Field(default=None)
    user_email: Optional[str] = Field(default=None)

    date_expression_type: Optional[Literal["weekday", "relative", "explicit", "none"]] = Field(
        default=None,
        description="'weekday' if user named a day (friday, monday...), "
                    "'relative' if they said today/tomorrow, "
                    "'explicit' if they gave a full date, 'none' if not mentioned this turn"
    )
    date_expression_value: Optional[str] = Field(
        default=None,
        description="lowercase weekday name ('friday'), or 'today'/'tomorrow', or YYYY-MM-DD"
    )
    date_qualifier: Optional[Literal["this", "next", "none"]] = Field(
        default="none",
        description="Only relevant when date_expression_type='weekday'. "
                    "'this' if user said 'this friday', 'next' if 'next friday', "
                    "'none' if they just said 'friday' with no qualifier"
    )
    date_confirmed: Optional[Literal["true", "false"]] = Field(
        default="false",
        description="'true' only once the user has explicitly confirmed which exact date they meant"
    )

    appointment_date: Optional[str] = Field(default=None, description="Resolved by code only — never set by you")
    appointment_time: Optional[str] = Field(default=None, description="HH:MM 24-hour format")

class RescheduleIntent(BaseModel):
    user_email: Optional[str] = Field(default=None, description="The email of the user to modify the appointment for")


class AppointmentReschedulingDetails(BaseModel):
    appointment_id: Optional[str] = Field(
        default=None,
        description="The ID of the appointment to modify — extracted if user specifies which one"
    )
    # What the user wants to change — same extraction pattern as AppointmentDetails
    date_expression_type: Optional[Literal["weekday", "relative", "explicit", "none"]] = Field(default=None)
    date_expression_value: Optional[str] = Field(default=None)
    date_qualifier: Optional[Literal["this", "next", "none"]] = Field(default="none")
    date_confirmed: Literal["true", "false"] = Field(
        default="false",
        description="'true' only once the user has confirmed the new date"
    )
    new_appointment_date: Optional[str] = Field(default=None, description="Resolved by code only")
    new_appointment_time: Optional[str] = Field(default=None, description="HH:MM 24-hour format")
    new_service_name: Optional[str] = Field(default=None)
