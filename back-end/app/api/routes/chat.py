import logging
import time
import uuid

from fastapi import APIRouter
from langchain_core.messages import BaseMessage, HumanMessage

from app.agent.graph import get_workflow
from app.core.database import atomic_transaction, create_session
from app.core.exceptions import LLMProviderError, NovaDeskError, ResourceNotFoundError
from app.repositories import ConversationRepository, UserRepository
from app.schemas.request import ChatRequest
from app.schemas.response import AgentStateResponse, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Phase 1: Short-lived DB session (~2ms) to resolve or create the conversation.
    # Releases the pooled connection BEFORE running slow LLM/RAG network calls.
    with create_session() as db:
        user_repo = UserRepository(db)
        conversation_repo = ConversationRepository(db)

        if request.conversation_id:
            conversation = conversation_repo.get_by_id(request.conversation_id)
            if not conversation:
                raise ResourceNotFoundError(
                    "Conversation not found.",
                    details={"conversation_id": str(request.conversation_id)},
                )
            conversation_id = conversation.id
            human_override = conversation.human_override
        else:
            with atomic_transaction(db):
                anonymous_user = user_repo.create_user()
                conversation = conversation_repo.create_conversation(user_id=anonymous_user.id)
                conversation_id = conversation.id
                human_override = conversation.human_override

    if human_override:
        return ChatResponse(
            message="Human Agent typing...",
            conversation_id=conversation_id,
            human_override=True,
        )

    # Phase 2: Execute LangGraph workflow (0 SQLAlchemy connections held idle during LLM calls).
    # Pass ONLY the new HumanMessage for this turn into LangGraph.
    turn_human_msg_id = str(uuid.uuid4())
    current_human_message = HumanMessage(content=request.message, id=turn_human_msg_id)

    try:
        config = {"configurable": {"thread_id": str(conversation_id)}}
        start_time = time.perf_counter()
        response = get_workflow().invoke({"messages": [current_human_message]}, config=config)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
    except NovaDeskError:
        raise
    except Exception as exc:
        logger.error(
            "Workflow invocation failed for conversation %s: %s",
            conversation_id,
            exc,
            exc_info=True,
        )
        raise LLMProviderError(
            "The AI service is currently unavailable. Please try again in a moment.",
            details={"conversation_id": str(conversation_id)},
        ) from exc

    all_messages: list[BaseMessage] = response.get("messages", [])
    ai_message = str(all_messages[-1].content) if all_messages else ""

    # Phase 3: Short-lived DB session (~2ms) to persist both messages atomically.
    with create_session() as db:
        conversation_repo = ConversationRepository(db)
        with atomic_transaction(db):
            conversation_repo.append_turn(conversation_id, request.message, ai_message)

    agent_state_data = AgentStateResponse(
        intent=response.get("intent"),
        appointment_details=response.get("appointment_details"),
        reschedule_details=response.get("reschedule_details"),
        executed_tools=response.get("executed_tools", []),
    )

    return ChatResponse(
        message=ai_message,
        conversation_id=conversation_id,
        human_override=False,
        agent_state=agent_state_data,
        latency_ms=latency_ms,
    )
