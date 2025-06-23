from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
import os
import shutil
# --- Import your LangGraph setup ---
# It's best practice to put your agent creation logic in a separate file
# (e.g., 'agent_factory.py') and import the final compiled graph.
# For this example, we'll assume it's all in one place for simplicity,
# but we'll import the graph 'app' from a file named 'langgraph_setup.py'.

# Let's assume you've moved all the LangGraph code (tools, agents, supervisor)
# into a file named 'langgraph_setup.py' and it exposes the compiled graph as 'master_agent_graph'
from langgraph_setup import master_agent_graph, members # Assuming 'members' is also exposed
from langchain_core.messages import HumanMessage, AIMessage


UPLOADS_DIR = "user_uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Master AI Agent API",
    description="API for interacting with the LangGraph-based Master AI Agent.",
    version="1.0.0",
)

# --- Pydantic Models for API Request/Response ---
# This defines the expected structure of the JSON data for your API endpoints.

class ChatRequest(BaseModel):
    session_id: str  # To track different user conversations
    message: str
    uploaded_files: Optional[Dict[str, str]] = None # e.g., {"script_file": "/path/on/server/to/script.sql"}

class ChatResponse(BaseModel):
    session_id: str
    ai_response: str
    is_clarification_question: bool # Lets the frontend know if the AI is waiting for an answer
    is_task_complete: bool # Lets the frontend know if a tool finished running

class UploadResponse(BaseModel):
    message: str
    server_path: str
    filename: str


# --- In-Memory Conversation State Management ---
# WARNING: This is for demonstration only. It will lose all data if the server restarts.
# For production, use a database like Redis, a file-based session store, or a managed service.
conversation_histories: Dict[str, List] = {}

# --- API Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Endpoint to send a message to the Master AI Agent and get a response.
    """
    session_id = request.session_id
    user_input = request.message

    # Retrieve or initialize the conversation history for the session
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    current_history = conversation_histories[session_id]
    current_history.append(HumanMessage(content=user_input))

    # In a real app, file uploads would be handled by a separate endpoint.
    # That endpoint would save the file to the server and return its path.
    # The frontend would then include this path in the ChatRequest.
    # For now, we assume paths are provided directly.
    uploaded_files = request.uploaded_files or {}

    # Prepare input for the LangGraph stream
    graph_input = {
        "messages": current_history,
        # In a real system, you'd manage uploaded files per session too.
        # For simplicity, we pass them directly.
    }

    ai_response_content = ""
    is_task_complete = False
    
    # Stream the graph execution
    async for event in master_agent_graph.astream(graph_input, {"recursion_limit": 25}):
        # Look for the supervisor or an agent's response
        for agent_name, agent_output in event.items():
            if isinstance(agent_output, dict) and "messages" in agent_output:
                last_message = agent_output["messages"][-1]
                if isinstance(last_message, AIMessage):
                    # Check if it's a final, user-facing message
                    if last_message.content and last_message.content not in members and last_message.content != "supervisor":
                        ai_response_content = last_message.content
                        # Heuristic: If the message contains "saved to" or "complete", the task is likely done
                        if "saved to" in ai_response_content.lower() or "complete" in ai_response_content.lower():
                            is_task_complete = True

    # Update the conversation history with the AI's final response
    if ai_response_content:
        current_history.append(AIMessage(content=ai_response_content))
    else:
        # If no explicit message was found, it might be an unhandled state.
        ai_response_content = "I'm processing your request. Please wait or provide more details if prompted."

    # Determine if the AI's response is a question (heuristic)
    is_clarification = '?' in ai_response_content

    return ChatResponse(
        session_id=session_id,
        ai_response=ai_response_content,
        is_clarification_question=is_clarification,
        is_task_complete=is_task_complete,
    )

@app.get("/new_session")
async def get_new_session():
    """Endpoint to start a new chat session."""
    session_id = str(uuid.uuid4())
    conversation_histories[session_id] = []
    # Create a dedicated folder for this session's uploads
    os.makedirs(os.path.join(UPLOADS_DIR, session_id), exist_ok=True)
    return {"session_id": session_id}

# --- NEW FILE UPLOAD ENDPOINT ---
@app.post("/upload/{session_id}", response_model=UploadResponse)
async def upload_file(session_id: str, file: UploadFile = File(...)):
    """
    Endpoint to upload a file for a specific chat session.
    The file is saved on the server, and its path is returned.
    """
    if session_id not in conversation_histories:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new session.")

    try:
        session_upload_dir = os.path.join(UPLOADS_DIR, session_id)
        # Security: Sanitize the filename to prevent directory traversal attacks
        safe_filename = os.path.basename(file.filename)
        if not safe_filename:
            raise HTTPException(status_code=400, detail="Invalid filename.")
            
        file_path = os.path.join(session_upload_dir, safe_filename)

        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return UploadResponse(
            message=f"File '{safe_filename}' uploaded successfully.",
            server_path=file_path, # Return the path on the server
            filename=safe_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")

# --- UPDATED CHAT ENDPOINT ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Endpoint to send a message to the Master AI Agent and get a response.
    This message can now contain references to files uploaded via the /upload endpoint.
    """
    session_id = request.session_id
    user_input = request.message

    if session_id not in conversation_histories:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new session.")

    current_history = conversation_histories[session_id]
    current_history.append(HumanMessage(content=user_input))

    graph_input = {"messages": current_history}
    ai_response_content = ""
    is_task_complete = False
    
    # Use the same streaming logic as before
    async for event in master_agent_graph.astream(graph_input, {"recursion_limit": 25}):
        for agent_name, agent_output in event.items():
            if isinstance(agent_output, dict) and "messages" in agent_output:
                last_message = agent_output["messages"][-1]
                if isinstance(last_message, AIMessage):
                    if last_message.content and last_message.content not in members and last_message.content != "supervisor":
                        ai_response_content = last_message.content
                        if "saved to" in ai_response_content.lower() or "complete" in ai_response_content.lower():
                            is_task_complete = True
    
    if ai_response_content:
        current_history.append(AIMessage(content=ai_response_content))
    else:
        ai_response_content = "I am processing your request. There might not be an immediate response."

    is_clarification = '?' in ai_response_content and not is_task_complete

    return ChatResponse(
        session_id=session_id,
        ai_response=ai_response_content,
        is_clarification_question=is_clarification,
        is_task_complete=is_task_complete,
    )