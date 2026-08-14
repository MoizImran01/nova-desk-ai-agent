
from app.core.vectordb import get_vector_store
from app.utils.utils import format_faqs, return_closest_match
from langchain_core.tools import tool
from sqlmodel import Session, select
from app.core.database import engine
from app.models.domain import User, Service, Appointment, Conversation
from datetime import time, timedelta
from typing import List, Optional
from datetime import datetime
import difflib
from zoneinfo import ZoneInfo

vector_store = get_vector_store()

def retreive_faqs(query: str) -> str:
    """
    Retrieve FAQs from the knowledge base
    """
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query)
    formatted_faqs = format_faqs(retrieved_docs)
    return formatted_faqs

def get_user_appointments(conversation_id: str) -> list[Appointment]:
    """
    Get the user's appointments from the database
    """
    with Session(engine) as session:
        user_id = session.exec(select(Conversation).where(Conversation.id == conversation_id)).first()
        if not user_id:
            return []
        appointments = session.exec(select(Appointment).where(Appointment.user_id == user_id.user_id)).all()
        if not appointments:
            return []
        return appointments

@tool
def check_available_appointment_slots(appointment_date: str) -> dict:
    """
    Checks the database for available 30-minute time slots on the requested date.
    If that date is fully booked, automatically checks the next several days and
    returns the nearest alternatives instead.
    CRITICAL: appointment_date MUST be a string formatted exactly as 'YYYY-MM-DD'.
    """
    try:
        requested_date = datetime.strptime(appointment_date, "%Y-%m-%d").date()
    except ValueError:
        return {"status": "error", "message": "Invalid date format. Please format as YYYY-MM-DD."}

    today = datetime.now(ZoneInfo("Asia/Karachi")).date()
    if requested_date < today:
        return {"status": "error", "message": f"{appointment_date} is in the past. Please ask the user for a future date."}

    def _slots_for(d):
        open_time = time(9, 0)
        close_time = time(17, 0)
        slot_duration = timedelta(minutes=30)
        all_slots = []
        current_dt = datetime.combine(d, open_time)
        end_dt = datetime.combine(d, close_time)
        while current_dt < end_dt:
            all_slots.append(current_dt.time())
            current_dt += slot_duration
        with Session(engine) as session:
            booked = session.exec(select(Appointment).where(Appointment.appointment_date == d)).all()
        booked_times = [a.appointment_time for a in booked if a.appointment_time]
        return [s.strftime("%I:%M %p") for s in all_slots if s not in booked_times]

    requested_slots = _slots_for(requested_date)
    if requested_slots:
        return {
            "status": "available",
            "date": appointment_date,
            "slots": requested_slots,
        }

    # Requested date is fully booked -- look ahead for the nearest alternatives
    alternatives = []
    for offset in range(1, 8):  # look up to a week ahead
        candidate = requested_date + timedelta(days=offset)
        slots = _slots_for(candidate)
        if slots:
            alternatives.append({"date": candidate.strftime("%Y-%m-%d"), "slots": slots})
            if len(alternatives) >= 3:
                break

    return {
        "status": "fully_booked",
        "requested_date": appointment_date,
        "message": f"No slots available on {appointment_date}.",
        "nearest_alternatives": alternatives,
    }

@tool
def book_appointment(
    user_name: str, 
    appointment_date: str, 
    appointment_time: str, 
    service_name: str, 
    user_email: Optional[str] = None, 
    user_phone: Optional[str] = None
) -> str:
    """
    Book the appointment in the database
    """
    with Session(engine) as session:
        all_services = session.exec(select(Service)).all()
        valid_service_names = [service.name for service in all_services]
        closest_matches = return_closest_match(service_name, valid_service_names)

        if not closest_matches:
            return f"Error: '{service_name}' is not recognized. Please ask the user to clarify which service they want."
            
        matched_clean_name = valid_service_names[[name.lower() for name in valid_service_names].index(closest_matches[0])]
        service = session.exec(select(Service).where(Service.name == matched_clean_name)).first()
        user = session.exec(select(User).where(User.email == user_email)).first()
        if not user:
            user = User(full_name=user_name, email=user_email, phone=user_phone)
            session.add(user)
            session.commit()
            session.refresh(user)
        start_datetime = datetime.strptime(f"{appointment_date} {appointment_time}", "%Y-%m-%d %H:%M")
        appointment = Appointment(
            user_id=user.id,
            service_id=service.id,
            service_name=service.name,
            status="scheduled",
            appointment_date=start_datetime.date(),
            appointment_time=start_datetime.time()
        )
        session.add(appointment)
        session.commit()
        return f"Successfully booked the appointment for {user_name} on {appointment_date} at {appointment_time} for {service_name}"
    
@tool
def get_appointments_by_email(user_email: str) -> dict:
    """
    Get the user's appointments from the database by email address
    """
    today = datetime.now(ZoneInfo("Asia/Karachi")).date()
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == user_email)).first()
        if not user:
            return {"status": "not_found", "message": "No user found with this email address."}
        appointments = session.exec(select(Appointment).where(Appointment.user_id == user.id, Appointment.appointment_date>=today, Appointment.status=="scheduled")).all()
        if not appointments:
            return {"status": "no_appointments", "message": f"No appointments found for user {user_email}."}
        return {"status": "found",  "appointments": [{"appointment_id": str(appointment.id), "appointment_date": appointment.appointment_date.strftime("%Y-%m-%d"),"date_display": appointment.appointment_date.strftime("%A, %B %d, %Y"), "appointment_time": appointment.appointment_time.strftime("%I:%M %p"), "service_name": appointment.service_name} for appointment in sorted(appointments, key=lambda x: x.appointment_date)]} 

@tool
def modify_appointment(
    appointment_id: str,
    new_appointment_date: Optional[str] = None,
    new_appointment_time: Optional[str] = None,
    new_service_name: Optional[str] = None,
) -> dict:
    """
    Modify an existing appointment. Only pass the fields that are actually changing.
    appointment_id is required. At least one of the other fields must be provided.
    new_appointment_date must be YYYY-MM-DD, new_appointment_time must be HH:MM.
    """
    if not any([new_appointment_date, new_appointment_time, new_service_name]):
        return {"status": "error", "message": "No changes specified. Please provide at least one field to modify."}

    with Session(engine) as session:
        appointment = session.get(Appointment, appointment_id)
        if not appointment:
            return {"status": "error", "message": f"Appointment ID {appointment_id} not found."}

        if new_appointment_date:
            try:
                appointment.appointment_date = datetime.strptime(new_appointment_date, "%Y-%m-%d").date()
            except ValueError:
                return {"status": "error", "message": "Invalid date format. Use YYYY-MM-DD."}

        if new_appointment_time:
            try:
                appointment.appointment_time = datetime.strptime(new_appointment_time, "%H:%M").time()
            except ValueError:
                return {"status": "error", "message": "Invalid time format. Use HH:MM (24-hour)."}

        if new_service_name:
            all_services = session.exec(select(Service)).all()
            valid_names = [s.name for s in all_services]
            match = return_closest_match(new_service_name, valid_names)
            if not match:
                return {
                    "status": "error",
                    "message": f"'{new_service_name}' is not a valid service. Available: {', '.join(valid_names)}"
                }
            matched = valid_names[[v.lower() for v in valid_names].index(match[0])]
            appointment.service_name = matched

        session.add(appointment)
        session.commit()

        return {
            "status": "success",
            "message": (
                f"Appointment updated successfully. "
                f"New details: {appointment.service_name} on "
                f"{appointment.appointment_date.strftime('%A, %B %d, %Y')} "
                f"at {appointment.appointment_time.strftime('%I:%M %p') if appointment.appointment_time else 'TBD'}."
            )
        }