from fastapi import APIRouter, Depends, HTTPException
from app.agent.graph import workflow
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse, AgentStateResponse
from app.models.domain import Conversation, Message, User
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from sqlmodel import Session
from app.api.dependacies import get_db
from datetime import datetime, timezone
import time

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    if request.conversation_id:
        conversation = db.get(Conversation, request.conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        new_user = User()
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        conversation = Conversation(user_id=new_user.id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        
    if conversation.human_override:
        return ChatResponse(
            message="Human Agent typing...", 
            conversation_id=conversation.id,
            human_override=conversation.human_override
        )
    new_human_message = Message(conversation_id=conversation.id, content=request.message, role="user")
    db.add(new_human_message)
    db.commit()
    db.refresh(new_human_message)
    db.refresh(conversation)
    lang_chain_messages = []
    past_messages = sorted(conversation.messages, key=lambda x: x.created_at)
    for message in past_messages:
        if message.role == "user":
            lang_chain_messages.append(HumanMessage(content=message.content))
            print(f"HumanMessage: {message.content}")
        elif message.role == "assistant":
            lang_chain_messages.append(AIMessage(content=message.content))
            print(f"AIMessage: {message.content}")
    try:
        config = {"configurable": {"thread_id": str(conversation.id)}}
        start_time = time.time()
        response = workflow.invoke({"messages": lang_chain_messages}, config=config)
        latency_ms = int((time.time() - start_time) * 1000)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=503, detail="The AI service is currently unavailable. Please try again in a moment.")
        
    ai_message = response["messages"][-1].content
    new_ai_message = Message(conversation_id=conversation.id, content=ai_message, role="assistant")

    db.add(new_ai_message)
    conversation.updated_at = datetime.now(timezone.utc)
    db.add(conversation)
    db.commit()

    # Extract tool names from the message history for the inspector
    executed_tools = list(set(
        msg.name for msg in response.get("messages", [])
        if isinstance(msg, ToolMessage) and msg.name
    ))

    agent_state_data = AgentStateResponse(
        intent=response.get("intent"),
        appointment_details=response.get("appointment_details"),
        reschedule_details=response.get("reschedule_details"),
        modification_details=response.get("modification_details"),
        executed_tools=executed_tools,
    )

    return ChatResponse(
        message=ai_message,
        conversation_id=conversation.id,
        human_override=False,
        agent_state=agent_state_data,
        latency_ms=latency_ms,
    )