from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage


def test_out_of_scope_intent_routes_to_refusal() -> None:
    # Imported lazily: the app needs the disposable DATABASE_URL set by conftest.
    from app.agent.nodes import route_intent

    state: Any = {"intent": {"user_intent": "out_of_scope"}, "messages": []}

    assert route_intent(state) == "handle_out_of_scope"


def test_refusal_is_a_single_fixed_reply_without_tools() -> None:
    from app.agent.nodes import handle_out_of_scope

    state: Any = {"messages": []}
    update = handle_out_of_scope(state)

    assert len(update["messages"]) == 1
    message = update["messages"][0]
    assert isinstance(message, AIMessage)
    assert not message.tool_calls
    assert "exclusively here to assist with Nova Wellness" in message.content
