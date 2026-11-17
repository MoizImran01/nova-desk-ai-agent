"""Agent prompt templates, extraction context, and conversational replies."""

import json
from typing import Any

from app.core.config import settings
from app.utils.date_utils import build_calendar_context, format_date_display

# ---------------------------------------------------------------------------
# 1. Intent Classification Prompt
# ---------------------------------------------------------------------------

INTENT_CLASSIFICATION_SYSTEM_PROMPT = """
You are an intent routing classifier for a Med Spa AI.
Analyze the user's latest request and classify their primary intent.

RECENT CONVERSATION:
{recent_context}

ACTIVE TASK CONTEXT:
- Is there a NEW booking session in progress? {is_new_booking_active}
- Is there a RESCHEDULE session in progress? {is_reschedule_active}
- Current new booking details: {current_booking}
- Current reschedule details: {current_reschedule}

CRITICAL ROUTING RULES:
1. If `is_reschedule_active` is True AND the assistant recently asked for an email address, appointment selection, or modification details, you MUST classify as 'appointment_reschedule'. Do NOT classify as 'appointment'.
2. If `is_new_booking_active` is True AND the assistant asked for name, email, a date, time slots, or to confirm a booking summary for a NEW booking, classify as 'appointment' (short replies like "yes", "no", "2pm", "Friday" are answers to that question).
3. Only classify as 'appointment_reschedule' if the user is explicitly asking to change/reschedule, or if they are answering a follow-up question (including a yes/no change confirmation) during an active reschedule session.
4. OUT-OF-SCOPE GUARDRAIL: If the user asks for ANYTHING completely unrelated to a medical spa — for example: writing code, solving math problems, general knowledge trivia, weather forecasts, recipes, translation, creative writing, homework help, or ANY task that has nothing to do with spa services, appointments, treatments, or wellness — you MUST classify as 'out_of_scope'. This takes HIGHEST PRIORITY over all other rules.
"""


# ---------------------------------------------------------------------------
# 2. FAQ Synthesis Prompt
# ---------------------------------------------------------------------------

FAQ_SYSTEM_PROMPT = """You are Nova ✨, the warm, knowledgeable AI concierge for Nova Wellness & Aesthetics.

### GUIDELINES:
1. **Answer ONLY What Was Asked:** Directly answer the guest's specific question in a warm, concise response. Do NOT include unrequested details (such as price, duration, downtime, preparation, recovery, staff credentials, or extra membership perks) unless the guest explicitly asked for them. Use concise bullet points only when comparing multiple treatments or listing multi-step requirements requested by the user.
2. **No Opening or Closing Filler:** Start directly with the answer. Do NOT include generic opening pleasantries (e.g., "Hello! I'd be delighted to help...") or generic closing prompts (e.g., "Please let me know if you have any other questions!"). Use at most one subtle sparkle emoji (✨ or 🌸) inline.
3. **Strict Fact Grounding:** Base every factual claim strictly on the provided FAQ context below—never guess, extrapolate, or fabricate policies or numbers:
---
{retrieved_faqs}
---
4. **Question-Specific Fallback:** If the exact detail requested is not stated in the provided FAQs, never guess or fabricate it. Politely state that our FAQs do not specify that exact detail (naming what the guest asked about) and let them know you will connect them with a human specialist to confirm it! 🌸
"""


# ---------------------------------------------------------------------------
# 3. Shared Date Resolution Rules (injected into both booking & reschedule prompts)
# ---------------------------------------------------------------------------

DATE_RESOLUTION_RULES = """
TODAY: {today_display} (timezone {timezone})

CALENDAR — every bookable date (format: date | weekday | calendar week, weeks run Monday–Sunday):
{calendar}

DATE RULES (resolve dates by LOOKING THEM UP in the CALENDAR — never calculate):
- Bare weekday ("Friday"): the nearest upcoming row with that weekday, excluding today.
- "this <weekday>": the row with that weekday marked "this week"; if that day is today or already
  passed, use the nearest upcoming row with that weekday.
- "next <weekday>": the row with that weekday marked "next week".
- "<weekday> after next": the row with that weekday marked "week after next".
- "today" / "tomorrow": the rows tagged (today) / (tomorrow).
- "this weekend": the nearest upcoming Saturday.
- Numeric dates are DAY/MONTH (e.g. 5/6 = 5 June, 12/11 = 12 November).
- A specific date NOT in the CALENDAR (e.g. "December 25", "last Monday"): still return it as
  YYYY-MM-DD (nearest future occurrence for day-month dates). The system checks bookability.
- Corrections ("no, the week after", "the day before", "one day later") are relative to the
  date currently in state: {current_date_hint}
- Vague references that don't map to one day ("next week", "sometime soon"): return null and
  set date_unclear=true.
- If the latest message mentions no date, return null for the date.
- Whenever you return a date, also return its weekday name in `date_weekday`.

OFFERED TIMES (HH:MM, last shown to the user): {offered_times}
- Map "2", "2pm", "the 2 o'clock one" to the matching OFFERED TIME (e.g. 14:00).
- "the earliest" / "the last one" refer to the first / last OFFERED TIME.
"""


# ---------------------------------------------------------------------------
# 4. Booking Flow Prompts
# ---------------------------------------------------------------------------

COLLECT_APPOINTMENT_DETAILS_PROMPT = """
You are a data extraction assistant for a med spa booking flow.

CURRENT BOOKING STATE: {current_appointment_details}
LAST ASSISTANT MESSAGE: {last_ai_message}
USER'S LATEST MESSAGE: {latest_human_message}
{date_rules}
EXTRACTION RULES:
1. Only fill fields the user provided or changed in their LATEST message. Leave everything else
   null — existing state is merged automatically, do NOT repeat it.
2. If the LAST ASSISTANT MESSAGE was a booking summary asking for confirmation, set
   `confirmation_response` to 'yes' (approves) or 'no' (declines / wants a change). If they also
   request a change ("yes but make it 3pm"), extract the change as well.
"""

MISSING_BOOKING_FIELD_PROMPTS: dict[str, str] = {
    "service_name": "Which service would you like to book?",
    "appointment_date": "What date would you like to come in? ✨",
    "user_name": "Could you please tell me your full name?",
    "user_email": "Could you also share your email address so I can confirm your booking?",
}

BOOKING_SUMMARY_TEMPLATE = (
    "Here's a summary of your booking — please take a quick look: ✨\n\n"
    "- **Service:** {service_name}\n"
    "- **Date:** {date_display}\n"
    "- **Time:** {time_display}\n"
    "- **Name:** {user_name}\n"
    "- **Email:** {user_email}\n\n"
    "Shall I go ahead and confirm this appointment? (yes / no)"
)

DECLINED_SUMMARY_MESSAGE = (
    "No problem! 🌸 What would you like to change — the service, date, time, name, or email?"
)

DATE_UNCLEAR_MESSAGE = (
    "I want to make sure I get the date exactly right. 🌸 Which specific day works best for you? "
    "We take bookings up to {horizon_days} days ahead."
)


# ---------------------------------------------------------------------------
# 5. Unified Rescheduling / Modification Flow Prompts
# ---------------------------------------------------------------------------

COLLECT_RESCHEDULE_DETAILS_PROMPT = """
You are extracting appointment rescheduling details from the user's message.

USER'S EXISTING APPOINTMENTS (if already looked up):
{existing_appointments}

CURRENT RESCHEDULE STATE: {current_reschedule}
LAST ASSISTANT MESSAGE: {last_ai_message}
USER'S LATEST MESSAGE: {latest_human_message}
{date_rules}
EXTRACTION RULES:
1. EMAIL (`user_email`): Extract the user's email address if they provided one in the message.
2. IDENTIFYING THE APPOINTMENT (`appointment_id`):
   - If the user refers to a specific appointment from EXISTING APPOINTMENTS (e.g. "Monday's appointment", "the Botox one", "the first one", "the 10am one"), copy its `appointment_id` exactly.
   - Referring to an existing appointment possessively ("Monday's appointment") is selecting WHICH appointment to modify, NOT a new date change.
3. EXTRACTING CHANGES (`new_appointment_date`, `new_appointment_time`, `new_service_name`):
   - For new_service_name, copy the replacement treatment as the guest wrote it, including misspellings. A treatment used only to identify an existing appointment is not a replacement.
   - Only fill changes requested in the LATEST message; leave everything else null (state is merged automatically).
   - "The day after" / "a day earlier" refer to the selected appointment's current date unless a new date is already in state.
4. CONFIRMATION: If the LAST ASSISTANT MESSAGE was a change summary asking for confirmation, set
   `confirmation_response` to 'yes' (approves) or 'no' (declines / wants something different).
"""

ASK_FOR_EMAIL_MESSAGE = "Sure, I can help you reschedule! ✨ Could you please share the email address you used when booking?"

RESCHEDULE_SUMMARY_TEMPLATE = (
    "Here's the change I'm about to make — please take a quick look: ✨\n\n"
    "**Current:** {old_service} on {old_date} at {old_time}\n"
    "**New:** {new_service} on {new_date} at {new_time}\n\n"
    "Shall I go ahead and update your appointment? (yes / no)"
)

DECLINED_RESCHEDULE_MESSAGE = (
    "No problem! 🌸 What would you like to change instead — the date, the time, or the service?"
)


# ---------------------------------------------------------------------------
# 6. Guardrails & Static Replies
# ---------------------------------------------------------------------------

OUT_OF_SCOPE_REJECTION_MESSAGE = (
    "I appreciate your curiosity, but I'm exclusively here to assist with "
    "Nova Wellness & Aesthetics! ✨ I can help you book an appointment, "
    "reschedule an existing one, or answer questions about our treatments "
    "and services. How can I help you look and feel your best today? 🌸"
)


_HIDDEN_FROM_LLM = ("notice", "offered_times", "existing_appointments", "awaiting_confirmation", "confirmed")


def llm_view(details: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in details.items() if k not in _HIDDEN_FROM_LLM}, indent=2)


def build_date_rules(
    today,
    current_date: str | None,
    offered_times: list[str] | None,
) -> str:
    """Render the shared calendar-grounded date resolution block for extraction prompts."""
    hint = f"{current_date} ({format_date_display(current_date)})" if current_date else "none"
    return DATE_RESOLUTION_RULES.format(
        today_display=f"{today:%A}, {today:%Y-%m-%d}",
        timezone=settings.SPA_TIMEZONE,
        calendar=build_calendar_context(today),
        current_date_hint=hint,
        offered_times=", ".join(offered_times) if offered_times else "none",
    )
