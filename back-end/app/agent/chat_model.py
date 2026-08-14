from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

def get_intent_classification_model():
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)

def get_collect_appointment_details_model():
    return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=settings.GOOGLE_GEMINI_API_KEY, temperature=0.0)

def get_handle_appointment_booking_model():
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)

def get_handle_retreive_faqs_model():
    return ChatGroq(model="llama-3.1-8b-instant", api_key=settings.GROQ_API_KEY)

def get_collect_modification_details_model():
    return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=settings.GOOGLE_GEMINI_API_KEY, temperature=0.0)

def get_collect_reschedule_intent_model():
    return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=settings.GOOGLE_GEMINI_API_KEY, temperature=0.0)

def get_handle_appointment_lookup_model():
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)

def get_handle_apply_modification_model():
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)