# src/models/shared_models.py
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AgentShortName(str, Enum):
    """
    Enum representing the available agent names.
    """

    DISCOVER = "DISCOVER"
    DEFINE = "DEFINE"
    LINEAGE = "LINEAGE"


class ResponseType(str, Enum):
    """
    Enum representing the type of response returned by the chat endpoint.

    - CHAT: Standard chat message.
    - DATASOURCE_FORM: Used when asking for database details or file uploads.
    - DISCOVER_FORM: Used for discover agent uploads or database connections.
    - LINEAGE_FORM: Used for lineage extraction requests.
    - FILE: Used when returning a file link.
    """

    CHAT = "CHAT"
    DATASOURCE_FORM = "DATASOURCE_FORM"
    DISCOVER_FORM = "DISCOVER_FORM"
    LINEAGE_FORM = "LINEAGE_FORM"
    FILE = "FILE"


class FileResponseData(BaseModel):
    """
    Model representing file response data.

    Attributes:
        server_path: The server path to the file.
        filename: The name of the file.
    """

    server_path: str
    filename: str


class ChatRequest(BaseModel):
    """
    Model representing a chat request to the master agent.

    Attributes:
        question: The user's question or message.
        agent: (Optional) The agent the user wants to interact with.
        chat_history: The conversation history as a list of message dicts,
                      e.g., [{"role": "user", "content": "..."}, {"role": "ai", "content": "..."}]
        info: (Optional) Additional information such as file paths.
    """

    question: str
    agent: Optional[AgentShortName] = None
    chat_history: List[Dict[str, str]]
    info: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """
    Model representing the response from the chat endpoint.

    Attributes:
        content: The main markdown/text response.
        agent: (Optional) The agent responding (or supervisor).
        responseType: The type of response.
        response: Additional response data as a dictionary.
    """

    content: str
    agent: Optional[AgentShortName] = None
    responseType: ResponseType
    response: Dict[str, Any] = {}
