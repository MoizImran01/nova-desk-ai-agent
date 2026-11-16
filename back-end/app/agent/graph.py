from functools import partial
import logging
from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg_pool import ConnectionPool

from app.agent.booking import (
    ask_booking_clarification,
    collect_appointment_details,
    handle_appointment_booking,
    route_after_collecting_details,
)
from app.agent.nodes import (
    classify_intent,
    handle_out_of_scope,
    handle_retreive_faqs,
    route_intent,
)
from app.agent.rescheduling import (
    ask_reschedule_clarification,
    collect_reschedule_details,
    handle_apply_modification,
    route_after_collecting_reschedule,
)
from app.agent.state import AgentState
from app.core.config import settings
from app.core.database import create_session
from app.services.scheduling_service import SchedulingService

logger = logging.getLogger(__name__)

_pool: Optional[ConnectionPool] = None
_workflow: Optional[CompiledStateGraph] = None


def build_workflow(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    *,
    scheduler: SchedulingService | None = None,
) -> CompiledStateGraph:
    """
    Pure factory that constructs and compiles the Nova Desk LangGraph workflow.
    Has zero import-time or database side effects:
    - Production passes `PostgresSaver(pool)`.
    - Tests & evaluation suites can pass `InMemorySaver()` or `None`.
    """
    if scheduler is None:
        scheduler = SchedulingService(create_session)
    graph = StateGraph(AgentState)

    # ---- shared nodes ----
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_retreive_faqs", handle_retreive_faqs)
    graph.add_node("handle_out_of_scope", handle_out_of_scope)

    # ---- booking flow nodes ----
    graph.add_node("collect_appointment_details", collect_appointment_details)
    graph.add_node("ask_booking_clarification", partial(ask_booking_clarification, scheduler=scheduler))
    graph.add_node("handle_appointment_booking", partial(handle_appointment_booking, scheduler=scheduler))

    # ---- unified reschedule flow nodes ----
    graph.add_node("collect_reschedule_details", partial(collect_reschedule_details, scheduler=scheduler))
    graph.add_node("ask_reschedule_clarification", partial(ask_reschedule_clarification, scheduler=scheduler))
    graph.add_node("handle_apply_modification", partial(handle_apply_modification, scheduler=scheduler))

    # ---- Entry & Intent Routing ----
    graph.add_edge(START, "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "handle_retreive_faqs": "handle_retreive_faqs",
            "handle_out_of_scope": "handle_out_of_scope",
            "collect_appointment_details": "collect_appointment_details",
            "collect_reschedule_details": "collect_reschedule_details",
        },
    )

    graph.add_edge("handle_retreive_faqs", END)
    graph.add_edge("handle_out_of_scope", END)

    # ---- Booking Flow Edges ----
    graph.add_conditional_edges(
        "collect_appointment_details",
        route_after_collecting_details,
        {
            "ask_booking_clarification": "ask_booking_clarification",
            "handle_appointment_booking": "handle_appointment_booking",
        },
    )

    graph.add_edge("ask_booking_clarification", END)
    graph.add_edge("handle_appointment_booking", END)

    # ---- Unified Reschedule Flow Edges ----
    graph.add_conditional_edges(
        "collect_reschedule_details",
        route_after_collecting_reschedule,
        {
            "ask_reschedule_clarification": "ask_reschedule_clarification",
            "handle_apply_modification": "handle_apply_modification",
        },
    )

    graph.add_edge("ask_reschedule_clarification", END)
    graph.add_edge("handle_apply_modification", END)

    return graph.compile(checkpointer=checkpointer)


def init_workflow() -> CompiledStateGraph:
    """
    Initialize the bounded PostgreSQL checkpointer pool, run `checkpointer.setup()`,
    and compile the singleton production workflow during FastAPI lifespan startup.
    """
    global _pool, _workflow
    if _workflow is not None:
        return _workflow

    _pool = ConnectionPool(
        conninfo=settings.DATABASE_URL,
        min_size=1,
        max_size=settings.DB_POOL_SIZE,
        kwargs={"autocommit": True},
    )
    checkpointer = PostgresSaver(_pool)
    checkpointer.setup()
    _workflow = build_workflow(checkpointer=checkpointer)
    return _workflow


def get_workflow() -> CompiledStateGraph:
    """Return the compiled production workflow (initializing lazily if needed)."""
    if _workflow is None:
        return init_workflow()
    return _workflow


def close_workflow() -> None:
    """Cleanly close the PostgreSQL checkpointer connection pool on shutdown."""
    global _pool, _workflow
    if _pool is not None:
        try:
            _pool.close()
        except Exception as exc:
            logger.warning("Error while closing checkpointer pool: %s", exc)
        finally:
            _pool = None
            _workflow = None
