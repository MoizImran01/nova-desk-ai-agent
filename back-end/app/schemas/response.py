from typing import List, Optional
import uuid

from sqlmodel import SQLModel


class AgentStateResponse(SQLModel):
    intent: Optional[dict] = None
    appointment_details: Optional[dict] = None
    reschedule_details: Optional[dict] = None
    executed_tools: Optional[List[str]] = None


class ChatResponse(SQLModel):
    message: str
    conversation_id: uuid.UUID
    human_override: bool
    agent_state: Optional[AgentStateResponse] = None
    latency_ms: Optional[int] = None
