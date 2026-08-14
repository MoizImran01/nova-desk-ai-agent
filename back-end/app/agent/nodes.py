from app.agent.state import AgentState
from app.schemas.agent import IntentClassification
from app.agent.tools import retreive_faqs, check_available_appointment_slots, book_appointment, get_appointments_by_email, modify_appointment
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import Literal
from app.agent.chat_model import get_intent_classification_model, get_collect_appointment_details_model, get_handle_appointment_booking_model, get_handle_retreive_faqs_model, get_collect_reschedule_intent_model, get_handle_appointment_lookup_model, get_collect_modification_details_model, get_handle_apply_modification_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from app.schemas.agent import IntentClassification, AppointmentDetails, RescheduleIntent, AppointmentReschedulingDetails
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from app.utils.date_utils import resolve_weekday, resolve_relative_term, resolve_explicit, get_today
from app.core.database import engine
from app.models.domain import Service
from sqlmodel import Session, select
import ast

def get_valid_service_names() -> list[str]:
    with Session(engine) as session:
        services = session.exec(select(Service)).all()
    return [s.name for s in services]

def _modification_complete(modification: dict) -> bool:
    has_id = modification.get("appointment_id") is not None
    has_change = any([
        modification.get("new_appointment_date"),
        modification.get("new_appointment_time"),
        modification.get("new_service_name"),
    ])
    date_ok = (
        not modification.get("new_appointment_date")
        or modification.get("date_confirmed") is True
    )
    # If they're moving to a new date, a new time must accompany it —
    # otherwise we don't actually have enough to apply the change.
    time_ok = (
        not modification.get("new_appointment_date")
        or modification.get("new_appointment_time")
    )
    return has_id and has_change and date_ok and time_ok

def classify_intent(state: AgentState) -> dict:
    chat_model = get_intent_classification_model()

    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    latest_human_message = human_messages[-1].content

    current_booking = state.get("appointment_details", {})
    current_reschedule = state.get("reschedule_details", {})
    
    # --- THE FIX: Separate the active states so the LLM knows exactly which flow is active ---
    is_new_booking_active = bool(current_booking)
    is_reschedule_active = bool(current_reschedule)

    recent_context = state.get("messages", [])[-4:]
    recent_context_str = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in recent_context
        if hasattr(m, "content") and m.content
    ])

    # --- THE FIX: Updated, hyper-strict routing rules ---
    system_prompt = """
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
    1. If `is_reschedule_active` is True AND the assistant recently asked for a different email address to look up an appointment, you MUST classify as 'appointment_reschedule'. Do NOT classify as 'appointment'.
    2. If `is_new_booking_active` is True AND the assistant asked for name, email, or time slots for a NEW booking, classify as 'appointment'.
    3. Only classify as 'appointment_reschedule' if the user is explicitly asking to change/reschedule, or if they are answering a follow-up question during an active reschedule lookup.
    """

    structured_chat_model = chat_model.with_structured_output(IntentClassification)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{latest_human_message}"),
    ])
    chain = prompt | structured_chat_model
    response = chain.invoke({
        "is_new_booking_active": is_new_booking_active,
        "is_reschedule_active": is_reschedule_active,
        "current_booking": json.dumps(current_booking),
        "current_reschedule": json.dumps(current_reschedule),
        "recent_context": recent_context_str,
        "latest_human_message": latest_human_message,
    })
    
    response_intent = response.model_dump()
    state_updates = {"intent": response_intent}

    # Wipe competing flow data when switching contexts
    if response_intent["user_intent"] == "appointment":
        state_updates["reschedule_details"] = {}
        state_updates["modification_details"] = {}
        
    elif response_intent["user_intent"] == "appointment_reschedule":
        state_updates["appointment_details"] = {}

    print(f"IntentClassification: {response_intent}")
    return state_updates

def route_intent(state: AgentState) -> Literal["handle_retreive_faqs", "collect_appointment_details", "collect_reschedule_intent", "collect_modification_details", "handle_apply_modification", "handle_human_escalation"]:
    intent = state["intent"]["user_intent"]

    if intent == "faq":
        return "handle_retreive_faqs"

    elif intent == "appointment":
        return "collect_appointment_details"

    elif intent == "appointment_reschedule":
        reschedule = state.get("reschedule_details", {})
        modification = state.get("modification_details", {})

        # Phase 1: no email yet — go collect it
        if not reschedule.get("user_email"):
            return "collect_reschedule_intent"

        # Phase 2: have email but haven't shown the appointment list yet
        if not reschedule.get("lookup_done"):
            return "collect_reschedule_intent"  # routes onward to handle_appointment_lookup

        # Phase 3: list was shown, collecting what the user wants to change
        if not _modification_complete(modification):
            return "collect_modification_details"

        # Phase 4: everything collected — apply the change
        return "handle_apply_modification"

    elif intent == "human_escalation":
        return "handle_human_escalation"

    else:
        return "handle_retreive_faqs"


def handle_retreive_faqs(state: AgentState) -> dict:
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    query = human_messages[-1].content
    retrieved_faqs = retreive_faqs(query)
    print(f"retrieved_faqs inside handle_retreive_faqs: {retrieved_faqs}")
    chat_model = get_handle_retreive_faqs_model()
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant that can answer questions about the following FAQs: {retrieved_faqs}. Always use the information from the FAQs to answer the question. If the you dont know the answer based on the FAQs, just politely say hmmm... let me connect you with a human agent.'),
        ('human', '{query}')
    ])
    chain = prompt | chat_model | StrOutputParser()
    faq_response = chain.invoke({"retrieved_faqs": retrieved_faqs, "query": query})
    current_booking = state.get('appointment_details', {})
    service = current_booking.get('service_name', 'your appointment')
    if current_booking:
        is_booking_active = True
    else:
        is_booking_active = False
    if is_booking_active:
        handle_in_progress_appointment_prompt =f"""
            The user just asked an off-topic question. You have already generated this answer: "{faq_response}".
            Now, look at their current partial booking data: {json.dumps(current_booking, indent=2)}.
            Write a single closing sentence that smoothly transitions back and asks if they would like to 
            proceed with finishing their booking for {service}.
            """
        handle_in_progress_appointment_response = chat_model.invoke(handle_in_progress_appointment_prompt).content
        final_response = f"{faq_response}\n\n{handle_in_progress_appointment_response}"
    else:
        final_response = faq_response
    return {"messages": [AIMessage(content=final_response)]}




def collect_appointment_details(state: AgentState) -> dict:
    chat_model = get_collect_appointment_details_model()
    structured_chat_model = chat_model.with_structured_output(AppointmentDetails)
    today = get_today()
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    latest_human_message = human_messages[-1].content
    appointment_details = state.get('appointment_details', {})

    prompt = f"""
    You are a data extraction assistant. Extract booking details from the user's message.

    CURRENT APPOINTMENT DETAILS: {json.dumps(appointment_details, indent=2)}
    USER'S LATEST MESSAGE: {latest_human_message}

    For dates: DO NOT calculate or convert any date yourself. Only classify the TYPE
    of expression and extract the RAW value (see field descriptions). Code resolves
    the actual calendar date — that is not your job.

    If the user is answering a yes/no confirmation about which date was meant
    (e.g. "yes that one", "no, the week after"), set date_confirmed accordingly and
    adjust date_qualifier to 'this' or 'next' based on their answer.

    Retain all existing details not explicitly changed in the latest message.
    """
    response = structured_chat_model.invoke(prompt).model_dump()
    response["date_confirmed"] = str(response.get("date_confirmed", "false")).lower() == "true"

    expr_type = response.get("date_expression_type")
    expr_value = response.get("date_expression_value")
    qualifier = response.get("date_qualifier") or "none"

    if expr_type == "weekday" and expr_value:
        resolved = resolve_weekday(expr_value, today, qualifier=None if qualifier == "none" else qualifier)
        response["appointment_date"] = resolved.date_str
        response["date_confirmed"] = not resolved.needs_confirmation
        response["_pending_confirmation_question"] = resolved.confirmation_question
    elif expr_type == "relative" and expr_value:
        resolved = resolve_relative_term(expr_value, today)
        response["appointment_date"] = resolved.date_str
        response["date_confirmed"] = True
    elif expr_type == "explicit" and expr_value:
        try:
            resolved = resolve_explicit(expr_value, today)
            response["appointment_date"] = resolved.date_str
            response["date_confirmed"] = True
        except ValueError:
            response["appointment_date"] = appointment_details.get("appointment_date")
    else:
        # nothing new said about date this turn — keep whatever was already there
        response["appointment_date"] = appointment_details.get("appointment_date")
        response["date_confirmed"] = appointment_details.get("date_confirmed", False)

    print(f"🚨 EXTRACTED DETAILS: {json.dumps(response, indent=2, default=str)}")
    return {"appointment_details": response}

def route_after_collecting_details(state: AgentState) -> Literal["ask_date_confirmation", "ask_for_missing_fields", "handle_appointment_booking"]:
    details = state.get("appointment_details", {})
    if details.get("appointment_date") and not details.get("date_confirmed", False):
        return "ask_date_confirmation"
    required_fields = ["appointment_date", "appointment_time", "service_name", "user_name", "user_email"]
    missing_fields = [field for field in required_fields if not details.get(field)]
    blocking_fields = [field for field in missing_fields if field!="appointment_time"]
    if blocking_fields:
        return "ask_for_missing_field"
    return "handle_appointment_booking"


def ask_date_confirmation(state: AgentState) -> dict:
    booking = state.get("appointment_details", {})
    modification = state.get("modification_details", {})
    question = (
        booking.get("_pending_confirmation_question")
        or modification.get("_pending_confirmation_question")
        or "Which exact date did you mean?"
    )
    return {"messages": [AIMessage(content=question)]}

def ask_for_missing_field(state: AgentState) -> dict:
    details = state.get("appointment_details", {})

    field_prompts = {
        "user_name": "Could you please tell me your full name?",
        "user_email": "Could you also share your email address so I can confirm your booking?",
        "service_name": "Which service would you like to book?",
        "appointment_date": "What date would you like to come in?",
    }

    required_fields = ["service_name", "appointment_date", "user_name", "user_email"]
    missing_fields = [field for field in required_fields if not details.get(field)]

    question = field_prompts.get(missing_fields[0], "Could you provide a bit more detail so I can finish booking this?")

    return {"messages": [AIMessage(content=question)]}

def handle_appointment_booking(state: AgentState) -> dict:
    chat_model = get_handle_appointment_booking_model()

    current_appointment_details = state.get('appointment_details', {})
    required_fields = ["appointment_date", "appointment_time", "service_name", "user_name", "user_email"]
    has_all_required = all(current_appointment_details.get(field) for field in required_fields)

    if has_all_required:
        tools_to_bind = [check_available_appointment_slots, book_appointment]
        booking_instruction = """
    3. All required details are present and confirmed. Call 'book_appointment' now
       to finalize this booking. Do not ask the user any further questions first.
       Once you receive the tool result for 'book_appointment':
    - Warmly confirm the booking with the user.
    - Summarize their appointment details (date, time, and service).
    - End with a polite, welcoming closing statement like "Thank you for booking with us! We look forward to seeing you."
    """
    else:
        tools_to_bind = [check_available_appointment_slots]
        missing = [field for field in required_fields if not current_appointment_details.get(field)]
        booking_instruction = f"""
    3. You do not have permission to finalize this booking yet. The following
       required details are still missing: {missing}. Do not attempt to book
       anything. Simply continue the conversation to collect what's missing,
       or present available slots if you're still waiting on a time.
    """

    chat_model_with_tools = chat_model.bind_tools(tools_to_bind)
    recent_user_context = state.get('messages', [])[-4:]
    current_time = datetime.now(ZoneInfo("Asia/Karachi")).strftime("%A, %B %d, %Y")

    system_prompt = """
    You are the booking assistant for a Med Spa.
    Today's exact date is: {current_time}.

    Details collected so far:
    {current_appointment_details}

    INSTRUCTIONS:
    1. If `appointment_date` is set but `appointment_time` is null, call
    'check_available_appointment_slots' with that exact date string.

    Once you receive the tool result:
    - If status is "available": present the slots to the user and ask them to pick one.
    - If status is "fully_booked": inform the user that date is unavailable, then
        immediately present the nearest_alternatives (each with its date and slots)
        in the same reply so the user can choose a different day without any back-and-forth.
        Example: "Friday the 3rd is fully booked — but here are the nearest available days: ..."

    2. If `user_name` is missing, ask for it conversationally. Do not guess or invent it.
    """ + booking_instruction

    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder(variable_name="recent_user_context"),
    ])

    chain = prompt | chat_model_with_tools

    try:
        response = chain.invoke({
            "recent_user_context": recent_user_context,
            "current_time": current_time,
            "current_appointment_details": json.dumps(current_appointment_details, indent=2)
        })
    except Exception as e:
        # Groq occasionally hallucinates a tool call for a tool that wasn't bound
        # (tool_use_failed / "not in request.tools"). Fall back to a plain,
        # non-tool-bound response instead of letting this 503 the whole endpoint.
        error_str = str(e)
        if "tool_use_failed" in error_str or "was not in request.tools" in error_str:
            print(f"⚠️ Tool call hallucination caught in handle_appointment_booking: {error_str}")
            fallback_chain = prompt | chat_model | StrOutputParser()
            fallback_text = fallback_chain.invoke({
                "recent_user_context": recent_user_context,
                "current_time": current_time,
                "current_appointment_details": json.dumps(current_appointment_details, indent=2)
            })
            return {"messages": [AIMessage(content=fallback_text)]}
        raise
   
    if recent_user_context and isinstance(recent_user_context[-1], ToolMessage) and recent_user_context[-1].name == "book_appointment":
        if "successfully booked" in recent_user_context[-1].content.lower():
            return {
                "messages": [response],
                "appointment_details": {}
            }
    return {"messages": [response]}

def collect_reschedule_intent(state: AgentState) -> dict:
    chat_model = get_collect_reschedule_intent_model()
    structured_chat_model = chat_model.with_structured_output(RescheduleIntent)
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    latest_human_message = human_messages[-1].content
    existing_reschedule_details = state.get("reschedule_details", {})

    prompt = f"""
    You are extracting an email address for an appointment modification request.
    CURRENT KNOWN DETAILS: {json.dumps(existing_reschedule_details, indent=2)}
    USER'S LATEST MESSAGE: {latest_human_message}
    Extract the user's email address if they provided one. If not present, leave as null.
    """
    extracted = structured_chat_model.invoke(prompt).model_dump()
    merged = {**existing_reschedule_details}
    if extracted.get("user_email"):
        merged["user_email"] = extracted["user_email"]

    return {"reschedule_details": merged}

def route_after_collecting_reschedule_intent(state: AgentState) -> Literal["ask_for_email","handle_appointment_lookup"]:
    details = state.get("reschedule_details", {})
    if not details.get("user_email"):
        return "ask_for_email"
    return "handle_appointment_lookup"

def ask_for_email(state: AgentState) -> dict:
    return {"messages": [AIMessage(content="Sure, I can help you with that. Can i get your email address?")]}

def handle_appointment_lookup(state: AgentState) -> dict:
    chat_model = get_handle_appointment_lookup_model()
    chat_model_with_tools = chat_model.bind_tools([get_appointments_by_email])
    rescheduling_details = state.get("reschedule_details", {})
    user_email = rescheduling_details.get("user_email")
    recent_user_context = state.get("messages", [])[-6:]

    # Pass 1 vs Pass 2 detection
    has_tool_result = False
    tool_status = None
    
    for m in recent_user_context:
        if isinstance(m, ToolMessage) and (m.name or "") == "get_appointments_by_email":
            has_tool_result = True
            
            # --- ROBUST PARSING LOGIC ---
            try:
                data = json.loads(m.content)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(m.content)
                except Exception as e:
                    print(f"Error parsing tool message in lookup: {e}")
                    data = {}
            except Exception as e:
                print(f"Unexpected error parsing tool message: {e}")
                data = {}
                
            tool_status = data.get("status")
            break

    system_prompt = """
    You are the appointment modification assistant.
    The user wants to modify an existing appointment.
    Their email is: {user_email}

    STEP 1: Call 'get_appointments_by_email' with this email immediately.

    Once you have the result:
    - If status is "not_found" or "no_appointments": Warmly inform the user that you couldn't find any upcoming appointments under that email, and ask if they might have booked under a different email address.
    - If status is "found" and there is ONE appointment: confirm its details and ask
      what they would like to change (date, time, or service).
    - If status is "found" and there are MULTIPLE appointments: list all of them clearly
      (numbered, with date/time/service) and ask which one they want to modify.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="recent_user_context"),
    ])
    chain = prompt | chat_model_with_tools
    response = chain.invoke({
        "recent_user_context": recent_user_context,
        "user_email": user_email,
    })

    # --- THE STATE ROUTING FIX ---
    if has_tool_result:
        if tool_status in ["not_found", "no_appointments"]:
            # Wipe the state completely to force the router to ask for email again
            updated_reschedule = {"user_email": None, "lookup_done": False}
        else:
            # Success! Mark lookup as done
            updated_reschedule = {**rescheduling_details, "lookup_done": True}
    else:
        # Tool hasn't run yet, preserve state
        updated_reschedule = {**rescheduling_details}

    return {
        "messages": [response],
        "reschedule_details": updated_reschedule,
    }

def collect_modification_details(state: AgentState) -> dict:
    chat_model = get_collect_modification_details_model()
    structured_chat_model = chat_model.with_structured_output(AppointmentReschedulingDetails)
    today = get_today()

    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    latest_human_message = human_messages[-1].content
    existing = state.get("modification_details", {})
    valid_services = get_valid_service_names()

    # Pull appointment list from tool message history
    existing_appointments = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and (msg.name or "") == "get_appointments_by_email":
            try:
                data = json.loads(msg.content)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(msg.content)
                except Exception as e:
                    print(f"Error parsing tool message: {e}")
                    data = {}
            if data.get("status") == "found":
                existing_appointments = data.get("appointments", [])
            break

    # Build weekday lookup for the identification guard
    appointments_by_weekday = {}
    for appt in existing_appointments:
        try:
            appt_date = datetime.strptime(appt["appointment_date"], "%Y-%m-%d").date()
            weekday = appt_date.strftime("%A").lower()
            appointments_by_weekday[weekday] = appt
        except Exception:
            pass

    prompt = f"""
    You are extracting the changes a user wants to make to their appointment.

    USER'S EXISTING APPOINTMENTS:
    {json.dumps(existing_appointments, indent=2)}

    VALID SERVICES: {', '.join(valid_services)}
    CURRENT KNOWN MODIFICATION DETAILS: {json.dumps(existing, indent=2)}
    USER'S LATEST MESSAGE: {latest_human_message}

    CRITICAL EXTRACTION RULES:

    1. IDENTIFYING THE APPOINTMENT (appointment_id):
       - If the user has ONLY ONE existing appointment, ALWAYS extract its appointment_id.
       - If the user refers to an appointment possessively (e.g. "Monday's appointment",
         "the Botox one", "the 10am one"), find the match in EXISTING APPOINTMENTS and
         extract that appointment_id. Do NOT treat this as a date change.

    2. EXTRACTING DATE CHANGES — read carefully:
       - A date change means the user wants the appointment on a DIFFERENT day.
       - Signals of a date change: words like "move to", "reschedule to", "change to",
         "next [day]", "this [day]", or any qualifier like "next" or "this".
       - If the user says "move it to next Monday" or "reschedule to this Friday",
         that IS a date change — extract date_expression_type, date_expression_value,
         and date_qualifier.
       - A possessive like "Monday's time" or "the Monday one" is NOT a date change.
         It is identifying which appointment. Only extract new_appointment_time in that case.
       - Key rule: if date_qualifier is "next" or "this", it is ALWAYS a date change,
         even if that weekday matches an existing appointment.

    3. For date changes: classify type/value/qualifier only — never calculate the date yourself.
    """

    extracted = structured_chat_model.invoke(prompt).model_dump()
    merged = {**existing}

    for key in ("appointment_id", "new_appointment_time", "new_service_name"):
        if extracted.get(key) is not None:
            merged[key] = extracted[key]

    expr_type = extracted.get("date_expression_type")
    expr_value = (extracted.get("date_expression_value") or "").lower()
    qualifier = extracted.get("date_qualifier") or "none"
    date_confirmed_raw = extracted.get("date_confirmed", "false")

    # THE FIXED GUARD
    # Old logic: block date change if weekday matches any existing appointment.
    # New logic: only block if weekday matches AND there's no qualifier indicating
    # the user wants to move (next/this always means moving, never identifying).
    # Also check the raw message for directional verbs as a secondary signal.
    directional_verbs = ["move to", "reschedule to", "change to", "switch to", "shift to"]
    message_lower = latest_human_message.lower()
    has_directional_verb = any(verb in message_lower for verb in directional_verbs)

    is_identifying_not_moving = (
        expr_type == "weekday"
        and expr_value in appointments_by_weekday   # weekday matches an existing appointment
        and qualifier == "none"                      # no "next" or "this" qualifier
        and not has_directional_verb                 # no "move to / reschedule to" in message
    )

    if is_identifying_not_moving:
        # User is pointing at an existing appointment by its day, not moving it.
        # Extract the appointment_id from the matching appointment if not already known.
        if not merged.get("appointment_id"):
            matched_appt = appointments_by_weekday[expr_value]
            merged["appointment_id"] = matched_appt.get("appointment_id")
        # Do not touch new_appointment_date
        merged["date_confirmed"] = existing.get("date_confirmed", False)

    else:
        # Normal date resolution
        if expr_type == "weekday" and expr_value:
            resolved = resolve_weekday(
                expr_value, today,
                qualifier=None if qualifier == "none" else qualifier
            )
            merged["new_appointment_date"] = resolved.date_str
            merged["date_confirmed"] = not resolved.needs_confirmation
            merged["_pending_confirmation_question"] = resolved.confirmation_question

        elif expr_type == "relative" and expr_value:
            resolved = resolve_relative_term(expr_value, today)
            merged["new_appointment_date"] = resolved.date_str
            merged["date_confirmed"] = True

        elif expr_type == "explicit" and expr_value:
            try:
                resolved = resolve_explicit(expr_value, today)
                merged["new_appointment_date"] = resolved.date_str
                merged["date_confirmed"] = True
            except ValueError:
                pass

        elif str(date_confirmed_raw).lower() == "true":
            merged["date_confirmed"] = True

        else:
            merged["date_confirmed"] = existing.get("date_confirmed", False)

    print(f"🔧 MODIFICATION DETAILS: {json.dumps(merged, indent=2, default=str)}")
    return {"modification_details": merged}

def route_after_collecting_modification(state: AgentState) -> Literal["ask_date_confirmation", "handle_apply_modification"]:
    details = state.get("modification_details", {})
    if details.get("new_appointment_date") and not details.get("date_confirmed", False):
        return "ask_date_confirmation"   # reuse the same node — it reads _pending_confirmation_question
    return "handle_apply_modification"

def handle_apply_modification(state: AgentState) -> dict:
    chat_model = get_handle_apply_modification_model()
    modification = state.get("modification_details", {})

    has_appointment_id = modification.get("appointment_id") is not None
    has_new_date = bool(modification.get("new_appointment_date"))
    has_new_time = bool(modification.get("new_appointment_time"))
    has_new_service = bool(modification.get("new_service_name"))

    has_something_to_change = has_new_date or has_new_time or has_new_service

    # --- THE STRICT GUARD ---
    is_ready_to_modify = False
    if has_appointment_id and has_something_to_change:
        is_ready_to_modify = True

        # If they are moving to a new date, they MUST specify a new time
        # before the AI is allowed to touch the database.
        if has_new_date and not has_new_time:
            is_ready_to_modify = False

    # Bind tools based on readiness
    if is_ready_to_modify:
        tools_to_bind = [check_available_appointment_slots, modify_appointment]
        modify_instruction = """
    3. All required details are present and confirmed. Call 'modify_appointment' now.
       CRITICAL: You MUST pass EVERY field that has a value in the modification_details 
       dictionary (new_appointment_date, new_appointment_time, new_service_name) into 
       the tool call. Do not leave any of them out. Do not ask the user any further 
       questions first.
    """
    else:
        tools_to_bind = [check_available_appointment_slots]

        if not has_appointment_id:
            missing_reason = "the appointment_id is missing — ask the user which appointment they mean."
        elif has_new_date and not has_new_time:
            missing_reason = (
                "a new date was requested but no new_appointment_time has been chosen yet. "
                "If slots for the new date haven't been shown yet, call "
                "'check_available_appointment_slots' with the new_appointment_date. "
                "If slots were already shown, read the user's latest message for their "
                "chosen time and ask them to confirm it if it's ambiguous."
            )
        elif not has_something_to_change:
            missing_reason = "no changes have been specified yet — ask the user what they'd like to change."
        else:
            missing_reason = "required details are still incomplete — continue collecting them."

        modify_instruction = f"""
    3. You do not have permission to finalize this modification yet, because
       {missing_reason}
       Do not attempt to call 'modify_appointment' under any circumstances right now.
    """

    chat_model_with_tools = chat_model.bind_tools(tools_to_bind)
    recent_context = state.get("messages", [])[-6:]

    system_prompt = """
    You are the appointment modification assistant.
    Modification details collected so far: {modification_details}

    INSTRUCTIONS:
    1. If the user wants a new date but new_appointment_time is missing,
       call 'check_available_appointment_slots' with the new_appointment_date first.
       - If status is "available": present slots and ask which one they want.
       - If status is "fully_booked": present the nearest_alternatives immediately.
    2. If appointment_id is missing, ask the user which appointment they meant.
    """ + modify_instruction

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="recent_context"),
    ])

    chain = prompt | chat_model_with_tools

    try:
        response = chain.invoke({
            "recent_context": recent_context,
            "modification_details": json.dumps(modification, indent=2)
        })
    except Exception as e:
        error_str = str(e)
        if "tool_use_failed" in error_str or "was not in request.tools" in error_str:
            print(f"⚠️ Tool call hallucination caught in handle_apply_modification: {error_str}")
            fallback_chain = prompt | chat_model | StrOutputParser()
            fallback_text = fallback_chain.invoke({
                "recent_context": recent_context,
                "modification_details": json.dumps(modification, indent=2)
            })
            return {"messages": [AIMessage(content=fallback_text)]}
        raise
    if recent_context and isinstance(recent_context[-1], ToolMessage) and recent_context[-1].name == "modify_appointment":
        if "success" in recent_context[-1].content.lower():

            return {
                "messages": [response],
                "reschedule_details": {},
                "modification_details": {}
            }
    return {"messages": [response]}