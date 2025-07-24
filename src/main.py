from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from enum import Enum
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
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import regex as re
 
ALLOWED_DIRECTORIES = {
    "define": "C:\EY\ey-genai-datanext-frontend\public",
    "lineage": "C:\EY\ey-genai-datanext-frontend\public",
    "discover": "C:\EY\ey-genai-datanext-frontend\public",
}

UPLOADS_DIR = "user_uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)
 
# --- Initialize FastAPI App ---
app = FastAPI(
    title="Master AI Agent API",
    description="API for interacting with the LangGraph-based Master AI Agent.",
    version="1.0.0",
)

origins = [
    "http://localhost",
    "http://localhost:5173",  # The origin of your React frontend
    # You can add other origins here if needed, e.g., your production frontend URL
]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Specifies the allowed origins
    allow_credentials=True, # Allows cookies to be included in requests
    allow_methods=["*"],    # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],    # Allows all headers
)
# --- Pydantic Models for API Request/Response ---
# This defines the expected structure of the JSON data for your API endpoints.
class AgentName(str, Enum):
    DISCOVER = "DISCOVER"
    DEFINE = "DEFINE"
    LINEAGE = "LINEAGE" # Or LineageExtractor
 
class ResponseType(str, Enum):
    CHAT = "CHAT"
    DATASOURCE_FORM = "DATASOURCE_FORM" # For asking for DB details or file uploads
    LINEAGE_FORM = "LINEAGE_FORM" # For lineage extraction requests
    DISCOVER_FORM = "DISCOVER_FORM" # For discover agent uploads/db connections
    FILE = "FILE" # For returning a file link
 
# --- UPDATED Pydantic Models ---
class ChatRequest(BaseModel):
    question: str
    agent: Optional[AgentName] = None # User can optionally specify an agent
    chat_history: List[Dict[str, str]] # e.g., [{"role": "user", "content": "..."}, {"role": "ai", "content": "..."}]
    info: Optional[Dict[str, Any]] = None # Generic JSON for extra info like file paths
 
class FileResponseData(BaseModel):
    server_path: str
    filename: str

class ChatResponse(BaseModel):
    content: str # The main markdown/text response
    agent: Optional[AgentName] = None # Which agent is responding (or supervisor)
    responseType: ResponseType
    response: Dict[str, Any] = {}
   
class UploadResponse(BaseModel):
    message: str
    server_path: str
    filename: str
 
 
# --- In-Memory Conversation State Management ---
# WARNING: This is for demonstration only. It will lose all data if the server restarts.
# For production, use a database like Redis, a file-based session store, or a managed service.
conversation_histories: Dict[str, List] = {}
 
# --- API Endpoints ---
def convert_to_langchain_messages(history: List[Dict[str, str]]) -> List[BaseMessage]:
    messages = []
    for item in history:
        role = item.get("role", "user").lower()
        content = item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "ai" or role == "assistant": # Accept both "ai" and "assistant" for flexibility
            messages.append(AIMessage(content=content))
    return messages
 
@app.post("/api/master", response_model=ChatResponse)
async def master_chat(request: ChatRequest):
    """
    This is the single endpoint for all chat interactions with the Master AI Agent.
    It takes the user's question and the entire conversation history.
    """
    # 1. Convert the incoming chat history from the request to Langchain's format
    langchain_history = convert_to_langchain_messages(request.chat_history)
   
    # 2. Add the user's new question to the history
    # We build a detailed message to help the supervisor parse inputs easily
    user_message_content = request.question
    if request.info and request.info.get("uploaded_file_path"):
        user_message_content += f" (Note: I have just uploaded a file available at path: {request.info['uploaded_file_path']})"
   
    langchain_history.append(HumanMessage(content=user_message_content))
   
    # Prepare the input for the LangGraph, including the user's explicit agent choice if provided
    graph_input = {"messages": langchain_history}
    if request.agent:
        # We can add this to the last message content to strongly guide the supervisor
        langchain_history[-1].content += f" (User has specified intent for agent: {request.agent.value})"
   
    ai_response_content = ""
    responding_agent_name = "supervisor"
   
    # 3. Stream the graph execution
    async for event in master_agent_graph.astream(graph_input, {"recursion_limit": 25}):
        for agent_name, agent_output in event.items():
            if isinstance(agent_output, dict) and "messages" in agent_output:
                last_message = agent_output["messages"][-1]
                if isinstance(last_message, AIMessage) and last_message.content:
                    if last_message.content not in members and last_message.content != "supervisor":
                        ai_response_content = last_message.content
                        responding_agent_name = last_message.name or agent_name
 
    if not ai_response_content:
        ai_response_content = "I'm sorry, I encountered an issue and cannot respond right now. Please try rephrasing."
 
    # 4. Determine the responseType based on the AI's response content
    final_response_type = ResponseType.CHAT
    final_response_data = {}
 
    if "[upload_required]" in ai_response_content or "provide the file" in ai_response_content.lower() or "connection details" in ai_response_content.lower():
        final_response_type = ResponseType.DATASOURCE_FORM
        ai_response_content = ai_response_content.replace("[upload_required]", "").strip()
    elif "[lineage_file_upload_required]" in ai_response_content.lower() or "lineage form" in ai_response_content.lower():
        final_response_type = ResponseType.LINEAGE_FORM
        ai_response_content = ai_response_content.replace("[lineage_file_upload_required]", "").strip()
    elif "[discover_db_details_required]" in ai_response_content.lower() or "discover form" in ai_response_content.lower():
        final_response_type = ResponseType.DISCOVER_FORM
        ai_response_content = ai_response_content.replace("[discover_db_details_required]", "").strip()
    elif "saved to:" in ai_response_content.lower() or "download link" in ai_response_content.lower():
        final_response_type = ResponseType.FILE
        # Use regex to find a file path in the response string
        path_match = re.search(r"saved to:\s*`?([^`]+?\.(?:xlsx|csv|png|txt))`?", ai_response_content, re.IGNORECASE)
        if path_match:
            server_path = path_match.group(1).strip()
            filename = os.path.basename(server_path)
            final_response_data = FileResponseData(filename=filename, server_path=server_path).dict()
    # 5. Determine which agent finally responded
    final_agent_enum = None
    if "DefineAgent" in responding_agent_name:
        final_agent_enum = AgentName.DEFINE
    elif "LineageExtractorAgent" in responding_agent_name:
        final_agent_enum = AgentName.LINEAGE
    elif "DiscoverAgent" in responding_agent_name:
        final_agent_enum = AgentName.DISCOVER
 
    return ChatResponse(
        content=ai_response_content,
        agent=final_agent_enum,
        responseType=final_response_type,
        response=final_response_data,
    )
@app.get("/api/documents/{filename}")
async def get_document(filename: str):
    found_path = None
    for key, directory in ALLOWED_DIRECTORIES.items():
        potential_path = os.path.join(directory, filename)
        print(potential_path)
        if os.path.exists(potential_path):
            if os.path.abspath(potential_path).startswith(directory):
                found_path = potential_path
                break
    if found_path:
        print(f"Serving file from: {found_path}")
        return FileResponse(path=found_path, filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found or access denied")
   
@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a file. Saves the file to a temporary directory
    on the server and returns its path. This path should then be passed
    in the `info` field of the `/api/master` request.
    """
    try:
        # Using a general upload directory. For multi-user systems,
        # session-specific or user-specific folders are recommended.
        os.makedirs(UPLOADS_DIR, exist_ok=True)
       
        safe_filename = os.path.basename(file.filename)
        if not safe_filename:
            raise HTTPException(status_code=400, detail="Invalid filename.")
       
        # Add a unique prefix to avoid filename collisions
        unique_filename = f"{uuid.uuid4()}_{safe_filename}"
        file_path = os.path.join(UPLOADS_DIR, unique_filename)
 
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
 
        return UploadResponse(
            message=f"File '{safe_filename}' uploaded successfully.",
            server_path=file_path,
            filename=safe_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
