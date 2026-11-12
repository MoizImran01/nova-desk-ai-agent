"""Deterministic agent regression: an out-of-scope chat turn.

Drives the real FastAPI route, the compiled LangGraph and PostgreSQL (SQL
transcript plus checkpointer). The only double is the intent classifier's
model call, which returns a scripted result.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda
from sqlmodel import Session, func, select

from app.models.domain import Appointment, Conversation, Message, Reminder
from app.schemas.agent import IntentClassification

PATIENT_TEXT = "Write me a Python script that sorts a list."


class ScriptedIntentModel:
    """Stands in for the hosted classifier model; records what it was asked."""

    def __init__(self, result: IntentClassification) -> None:
        self.result = result
        self.inputs: list[Any] = []

    def with_structured_output(self, schema: Any) -> RunnableLambda:
        assert schema is IntentClassification

        def respond(prompt_value: Any) -> IntentClassification:
            self.inputs.append(prompt_value)
            return self.result

        return RunnableLambda(respond)


@pytest.fixture
def scripted_out_of_scope(monkeypatch: pytest.MonkeyPatch) -> ScriptedIntentModel:
    model = ScriptedIntentModel(
        IntentClassification(user_intent="out_of_scope", confidence_score="0.99")
    )
    monkeypatch.setattr(
        "app.agent.nodes.get_intent_classification_model", lambda: model
    )
    return model


def _count(session: Session, table: Any) -> int:
    return session.exec(select(func.count()).select_from(table)).one()


@pytest.mark.agent
def test_out_of_scope_turn_refuses_and_persists_transcript(
    client: TestClient, scripted_out_of_scope: ScriptedIntentModel
) -> None:
    from app.agent.nodes import handle_out_of_scope
    from app.core.database import engine

    response = client.post("/chat", json={"message": PATIENT_TEXT})

    assert response.status_code == 200
    body = response.json()
    empty_state: Any = {"messages": []}
    refusal = handle_out_of_scope(empty_state)["messages"][0].content
    assert body["message"] == refusal
    assert "exclusively here to assist with Nova Wellness" in refusal
    assert body["human_override"] is False
    assert body["agent_state"]["intent"]["user_intent"] == "out_of_scope"
    assert body["agent_state"]["executed_tools"] == []
    conversation_id = uuid.UUID(body["conversation_id"])

    # The scripted classifier was reached through the normal request path.
    assert len(scripted_out_of_scope.inputs) == 1
    assert PATIENT_TEXT in str(scripted_out_of_scope.inputs[0])

    with Session(engine) as session:
        conversation = session.get(Conversation, conversation_id)
        assert conversation is not None
        transcript = session.exec(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)  # type: ignore[arg-type]
        ).all()
        assert [(m.role, m.content) for m in transcript] == [
            ("user", PATIENT_TEXT),
            ("assistant", refusal),
        ]
        # No tool side effects.
        assert _count(session, Appointment) == 0
        assert _count(session, Reminder) == 0
