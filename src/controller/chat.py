from typing import Dict, List

from fastapi import APIRouter
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.supervisor import master_agent_graph, members
from src.helpers.chat import convert_to_langchain_messages, format_ai_response
from src.models.chat import ChatRequest, ChatResponse

router = APIRouter()


# --- In-Memory Conversation State Management ---
# WARNING: This is for demonstration only. It will lose all data if the server restarts.
# For production, use a database like Redis, a file-based session store, or a managed service.
# Consider moving this to a dedicated `state_management` module if it grows.
conversation_histories: Dict[str, List] = {}


@router.post("/api/chats", response_model=ChatResponse)
async def master_chat(request: ChatRequest):
    """
    This is the single endpoint for all chat interactions with the Master AI Agent.
    It takes the user's question and the entire conversation history.
    """

    # 1. Convert the incoming chat history from the request to Langchain's format
    langchain_history = convert_to_langchain_messages(request.chat_history)

    # 2. Build the user's new question with any additional info
    user_message_content = request.question
    suffixes = []

    if request.info and request.info.get("uploaded_file_path"):
        suffixes.append(
            f"(Note: I have just uploaded a file available at path: {request.info['uploaded_file_path']})"
        )

    if request.agent:
        suffixes.append(f"(User has specified intent for agent: {request.agent.value})")

    if suffixes:
        user_message_content += " ".join(suffixes)

    langchain_history.append(HumanMessage(content=user_message_content))

    # Prepare the input for the LangGraph
    graph_input = {"messages": langchain_history}

    # 3. Stream the graph execution
    ai_response_content = ""
    responding_agent_name = "supervisor"

    async for event in master_agent_graph.astream(graph_input, {"recursion_limit": 25}):
        for agent_name, agent_output in event.items():
            if isinstance(agent_output, dict) and "messages" in agent_output:
                last_message = agent_output["messages"][-1]
                if isinstance(last_message, AIMessage) and last_message.content:
                    if (
                        last_message.content not in members
                        and last_message.content != "supervisor"
                    ):
                        ai_response_content = last_message.content
                        responding_agent_name = last_message.name or agent_name

    if not ai_response_content:
        ai_response_content = "I'm sorry, I encountered an issue and cannot respond right now. Please try rephrasing."

    # 4. Delegate response formatting to helper function
    formatted_response = format_ai_response(ai_response_content, responding_agent_name)

    return ChatResponse(
        content=formatted_response["content"],
        agent=formatted_response["agent"],
        responseType=formatted_response["responseType"],
        response=formatted_response["response"],
    )
