import json
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.agent.chat_model import LLMRegistry, invoke_structured
from app.agent.prompts import (
    FAQ_SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_SYSTEM_PROMPT,
    OUT_OF_SCOPE_REJECTION_MESSAGE,
)
from app.agent.state import AgentState, get_latest_human_message, record_tool
from app.schemas.agent import IntentClassification
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)


def classify_intent(state: AgentState) -> dict:
    latest_human_message = get_latest_human_message(state)

    current_booking = state.get("appointment_details", {})
    current_reschedule = state.get("reschedule_details", {})

    is_new_booking_active = bool(current_booking)
    is_reschedule_active = bool(current_reschedule)

    recent_context = state.get("messages", [])[-4:]
    recent_context_str = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in recent_context
        if hasattr(m, "content") and m.content
    ])

    structured_chat_model = LLMRegistry.get_structured_llm(
        "intent_classification",
        IntentClassification,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_CLASSIFICATION_SYSTEM_PROMPT),
        ("human", "{latest_human_message}"),
    ])
    chain = prompt | structured_chat_model

    response_intent = invoke_structured(
        chain,
        {
            "is_new_booking_active": is_new_booking_active,
            "is_reschedule_active": is_reschedule_active,
            "current_booking": json.dumps(current_booking),
            "current_reschedule": json.dumps(current_reschedule),
            "recent_context": recent_context_str,
            "latest_human_message": latest_human_message,
        },
        IntentClassification,
        fallback_dict={"user_intent": "faq", "confidence_score": 0.5},
    )

    state_updates: dict[str, Any] = {
        "intent": response_intent,
        "executed_tools": [],
    }
    if response_intent["user_intent"] == "appointment":
        state_updates["reschedule_details"] = {}
    elif response_intent["user_intent"] == "appointment_reschedule":
        state_updates["appointment_details"] = {}

    logger.info("Classified Intent: %s", response_intent)
    return state_updates


def route_intent(state: AgentState) -> Literal[
    "handle_retreive_faqs",
    "collect_appointment_details",
    "collect_reschedule_details",
    "handle_human_escalation",
    "handle_out_of_scope",
]:
    intent = state["intent"]["user_intent"]

    if intent == "out_of_scope":
        return "handle_out_of_scope"
    if intent == "faq":
        return "handle_retreive_faqs"
    if intent == "appointment":
        return "collect_appointment_details"
    if intent == "appointment_reschedule":
        return "collect_reschedule_details"
    if intent == "human_escalation":
        return "handle_human_escalation"
    return "handle_retreive_faqs"


def handle_retreive_faqs(state: AgentState) -> dict:
    query = get_latest_human_message(state)

    prompt = ChatPromptTemplate.from_messages([
        ("system", FAQ_SYSTEM_PROMPT),
        ("human", "{query}"),
    ])

    def _generate_faq_answer(retrieved_faqs: str) -> str:
        chat_model = LLMRegistry.get_text_llm("handle_retreive_faqs")
        chain = prompt | chat_model | StrOutputParser()
        return chain.invoke({"retrieved_faqs": retrieved_faqs, "query": query})

    rag_service = RAGService()
    faq_result = rag_service.answer_faq(query=query, generate_fn=_generate_faq_answer)
    faq_response = faq_result.answer

    current_booking = state.get("appointment_details", {})
    if bool(current_booking):
        service = current_booking.get("service_name") or "your appointment"
        transition = (
            f"Whenever you're ready, would you like to continue with finishing "
            f"your booking for {service}? ✨"
        )
        final_response = f"{faq_response}\n\n{transition}"
    else:
        final_response = faq_response

    return {
        "messages": [AIMessage(content=final_response)],
        "retrieved_contexts": faq_result.raw_contexts,
        "executed_tools": record_tool(state, "retrieve_faqs"),
    }


def handle_out_of_scope(state: AgentState) -> dict:
    """Deterministic guardrail for out-of-bounds requests."""
    return {"messages": [AIMessage(content=OUT_OF_SCOPE_REJECTION_MESSAGE)]}
