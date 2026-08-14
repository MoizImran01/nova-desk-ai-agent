from langgraph.graph import StateGraph, START, END
from app.agent.nodes import (
    classify_intent,
    route_intent,
    handle_retreive_faqs,
    handle_out_of_scope,
    collect_appointment_details,
    route_after_collecting_details,
    ask_date_confirmation,
    ask_for_missing_field,
    handle_appointment_booking,
    collect_reschedule_intent,
    route_after_collecting_reschedule_intent,
    ask_for_email,
    handle_appointment_lookup,
    collect_modification_details,
    route_after_collecting_modification,
    handle_apply_modification,
)
from app.agent.state import AgentState
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from app.core.config import settings
from langgraph.prebuilt import ToolNode, tools_condition
from app.agent.tools import (
    check_available_appointment_slots,
    book_appointment,
    get_appointments_by_email,
    modify_appointment,
)
from langchain_core.messages import ToolMessage
from typing import Literal
# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------

pool = ConnectionPool(conninfo=settings.DATABASE_URL, kwargs={"autocommit": True})
checkpointer = PostgresSaver(pool)
checkpointer.setup()

# ---------------------------------------------------------------------------
# Two separate ToolNodes — one per flow — so each can route back to its own
# calling node without ambiguity. A single shared ToolNode cannot have two
# different "return to" edges in LangGraph.
# ---------------------------------------------------------------------------

booking_tool_node = ToolNode([check_available_appointment_slots, book_appointment])

reschedule_tool_node = ToolNode([
    get_appointments_by_email,
    check_available_appointment_slots,
    modify_appointment,
])

# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

graph = StateGraph(AgentState)

# ---- shared nodes ----
graph.add_node("classify_intent", classify_intent)
graph.add_node("handle_retreive_faqs", handle_retreive_faqs)
graph.add_node("handle_out_of_scope", handle_out_of_scope)
graph.add_node("ask_date_confirmation", ask_date_confirmation)  # shared by both flows
graph.add_node("ask_for_missing_field", ask_for_missing_field)
# ---- booking flow nodes ----
graph.add_node("collect_appointment_details", collect_appointment_details)
graph.add_node("handle_appointment_booking", handle_appointment_booking)
graph.add_node("booking_tools", booking_tool_node)

# ---- reschedule flow nodes ----
graph.add_node("collect_reschedule_intent", collect_reschedule_intent)
graph.add_node("ask_for_email", ask_for_email)
graph.add_node("handle_appointment_lookup", handle_appointment_lookup)
graph.add_node("collect_modification_details", collect_modification_details)
graph.add_node("handle_apply_modification", handle_apply_modification)
graph.add_node("reschedule_tools", reschedule_tool_node)

# ---------------------------------------------------------------------------
# Edges — entry point
# ---------------------------------------------------------------------------

graph.add_edge(START, "classify_intent")

# classify_intent → route_intent decides which flow to enter, and which
# phase within the reschedule flow to resume at.
graph.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "handle_retreive_faqs":       "handle_retreive_faqs",
        "handle_out_of_scope":        "handle_out_of_scope",
        "collect_appointment_details": "collect_appointment_details",
        "collect_reschedule_intent":   "collect_reschedule_intent",
        "collect_modification_details":"collect_modification_details",
        "handle_apply_modification":   "handle_apply_modification",
    },
)

# ---------------------------------------------------------------------------
# Booking flow edges
# ---------------------------------------------------------------------------

graph.add_conditional_edges(
    "collect_appointment_details",
    route_after_collecting_details,
    {
        "ask_date_confirmation":    "ask_date_confirmation",
        "ask_for_missing_field": "ask_for_missing_field",
        "handle_appointment_booking":"handle_appointment_booking",
    },
)

# ask_date_confirmation is shared — both flows end their turn here and wait
# for the user to confirm which date they meant.
graph.add_edge("ask_date_confirmation", END)
graph.add_edge("ask_for_missing_field", END)

graph.add_conditional_edges(
    "handle_appointment_booking",
    tools_condition,
    {
        "tools": "booking_tools",
        END: END,
    },
)
graph.add_edge("booking_tools", "handle_appointment_booking")

graph.add_edge("handle_retreive_faqs", END)
graph.add_edge("handle_out_of_scope", END)

# ---------------------------------------------------------------------------
# Reschedule flow edges
# ---------------------------------------------------------------------------

# Phase 1 / 2: collect email → look up appointments
graph.add_conditional_edges(
    "collect_reschedule_intent",
    route_after_collecting_reschedule_intent,
    {
        "ask_for_email": "ask_for_email",
        "handle_appointment_lookup":"handle_appointment_lookup",
    },
)
graph.add_edge("ask_for_email", END)

# handle_appointment_lookup calls get_appointments_by_email via the tool node,
# then loops back to itself to present the result conversationally.
graph.add_conditional_edges(
    "handle_appointment_lookup",
    tools_condition,
    {
        "tools": "reschedule_tools",
        END: END,
    },
)


# Phase 3: collect what to change → confirm date if ambiguous → apply
graph.add_conditional_edges(
    "collect_modification_details",
    route_after_collecting_modification,
    {
        "ask_date_confirmation":   "ask_date_confirmation",
        "handle_apply_modification":"handle_apply_modification",
    },
)

# handle_apply_modification calls check_available_appointment_slots and/or
# modify_appointment, then loops back to itself to present results.
graph.add_conditional_edges(
    "handle_apply_modification",
    tools_condition,
    {
        "tools": "reschedule_tools",
        END: END,
    },
)
# reschedule_tools already has its return edge above (→ handle_appointment_lookup).
# We need it to also return to handle_apply_modification — but LangGraph doesn't
# allow a single node to have two different outgoing fixed edges.
#
# SOLUTION: replace the fixed edge with a conditional one that checks which phase
# we're in and routes to the right node.
# Remove the fixed edge above and replace with:

# graph.add_edge("reschedule_tools", "handle_appointment_lookup")  ← REMOVED

# The conditional below replaces BOTH return edges from reschedule_tools:

def route_after_reschedule_tools(state: AgentState) -> Literal[
    "handle_appointment_lookup", "handle_apply_modification"
]:
    # Find the most recent ToolMessage to see what just ran
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            # If the lookup tool just ran, go back to lookup node to present results
            if "get_appointments_by_email" in msg.name:
                return "handle_appointment_lookup"
            # Any other tool (check_slots, modify_appointment) → back to modification node
            return "handle_apply_modification"
    return "handle_appointment_lookup"  # fallback


# Replace the fixed reschedule_tools edge with this conditional one:
graph.add_conditional_edges(
    "reschedule_tools",
    route_after_reschedule_tools,
    {
        "handle_appointment_lookup": "handle_appointment_lookup",
        "handle_apply_modification": "handle_apply_modification",
    },
)

# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

workflow = graph.compile(checkpointer=checkpointer)
