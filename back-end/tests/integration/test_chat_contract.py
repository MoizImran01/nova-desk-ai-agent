"""HTTP contract of POST /chat for requests that never reach the graph."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, func, select

from app.models.domain import Conversation, Message


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"conversation_id": None},
        {"message": 42},
        {"message": "hi", "conversation_id": "not-a-uuid"},
    ],
)
def test_invalid_input_returns_422(client: TestClient, payload: dict) -> None:
    response = client.post("/chat", json=payload)

    assert response.status_code == 422


def test_missing_conversation_returns_404_and_writes_nothing(
    client: TestClient,
) -> None:
    from app.core.database import engine

    response = client.post(
        "/chat", json={"message": "hello", "conversation_id": str(uuid.uuid4())}
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found"}
    with Session(engine) as session:
        assert session.exec(select(func.count()).select_from(Conversation)).one() == 0
        assert session.exec(select(func.count()).select_from(Message)).one() == 0
