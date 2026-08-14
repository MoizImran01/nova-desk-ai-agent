from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import uuid
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    intent: dict
    appointment_details: dict
    reschedule_details: dict        # reschedule flow: email + lookup_done flag
    modification_details: dict      # reschedule flow: what the user wants to change
    booking_status: str             # "in_progress" | "completed" | ""