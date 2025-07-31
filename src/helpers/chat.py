# src/api/helpers/response_formatter.py
import os
from typing import Any, Dict, List

import regex as re
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.core.constants import AGENT_NAMES
from src.models.chat import AgentShortName, FileResponseData, ResponseType


def convert_to_langchain_messages(history: List[Dict[str, str]]) -> List[BaseMessage]:
    """Converts a list of dictionary messages into LangChain BaseMessage format."""

    messages = []
    for item in history:
        role = item.get("role", "user").lower()
        content = item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "ai" or role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def format_ai_response(
    ai_response_content: str, responding_agent_name: str
) -> Dict[str, Any]:
    """
    Determines the response type and extracts data based on AI response content.
    """

    final_response_type = ResponseType.CHAT
    final_response_data = {}
    cleaned_content = ai_response_content

    if (
        "[upload_required]" in ai_response_content
        or "provide the file" in ai_response_content.lower()
        or "connection details" in ai_response_content.lower()
    ):
        final_response_type = ResponseType.DATASOURCE_FORM
        cleaned_content = ai_response_content.replace("[upload_required]", "").strip()
    elif (
        "[lineage_file_upload_required]" in ai_response_content.lower()
        or "lineage form" in ai_response_content.lower()
    ):
        final_response_type = ResponseType.LINEAGE_FORM
        cleaned_content = ai_response_content.replace(
            "[lineage_file_upload_required]", ""
        ).strip()
    elif (
        "[discover_db_details_required]" in ai_response_content.lower()
        or "discover form" in ai_response_content.lower()
    ):
        final_response_type = ResponseType.DISCOVER_FORM
        cleaned_content = ai_response_content.replace(
            "[discover_db_details_required]", ""
        ).strip()
    elif (
        "saved to:" in ai_response_content.lower()
        or "download link" in ai_response_content.lower()
    ):
        final_response_type = ResponseType.FILE

        path_match = re.search(
            r"saved to:\s*`?([^`]+?\.(?:xlsx|csv|png|txt))`?",
            ai_response_content,
            re.IGNORECASE,
        )

        if path_match:
            server_path = path_match.group(1).strip()
            filename = os.path.basename(server_path)
            final_response_data = FileResponseData(
                filename=filename, server_path=server_path
            ).dict()

        cleaned_content = (
            ai_response_content  # Keep content as is if it contains path info for user
        )

    final_agent_enum = None
    if AGENT_NAMES.DEFINE in responding_agent_name:
        final_agent_enum = AgentShortName.DEFINE
    elif AGENT_NAMES.LINEAGE in responding_agent_name:
        final_agent_enum = AgentShortName.LINEAGE
    elif AGENT_NAMES.DISCOVER in responding_agent_name:
        final_agent_enum = AgentShortName.DISCOVER

    return {
        "content": cleaned_content,
        "agent": final_agent_enum,
        "responseType": final_response_type,
        "response": final_response_data,
    }
