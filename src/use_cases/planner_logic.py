# src/use_cases/planner_logic.py

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# --- FIX: Import directly from Pydantic to resolve deprecation warning ---
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# --- Pydantic Model for Structured Output ---
class PlannerSuggestion(BaseModel):
    """The structured output for a planner's suggestion."""
    thought: str = Field(
        description="A concise, step-by-step chain of thought explaining the reasoning behind the suggested answer. This should be written from the perspective of an AI business analyst."
    )
    suggestion: str = Field(
        description="A detailed, well-reasoned suggested answer for the user. This should be a high-quality example that the user can adapt."
    )

# --- Reusable LLM and Parser Setup ---
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    openai_api_version="2024-02-15-preview",
    api_key="901cce4a4b4045b39727bf1ce0e36219",
    azure_endpoint="https://aifordata-azureopenai.openai.azure.com/",
    temperature=0,
    max_tokens=2048
)
json_parser = JsonOutputParser(pydantic_object=PlannerSuggestion)

# --- The Core Logic Function ---
def generate_suggestion_and_thought(
    conversation_history: List[Dict[str, Any]],
    current_question: str
) -> Dict[str, str]:
    """
    Generates a "Chain of Thought" and a "Suggested Answer" for a given
    agenda question based on the conversation history.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert AI business analyst and a subject matter expert in multiple domains, including BFSI, Life Sciences, and Retail.
Your task is to help a user by providing a thoughtful, well-reasoned sample answer to a specific question about their business or data project.
Based on the provided conversation history, first, generate a concise "Chain of Thought" (CoT) that explains your reasoning for the sample answer.
Then, create a high-quality, detailed "Suggested Answer" that the user could use as a starting point.
The conversation history is a list of turns between "human" (the user) and "ai" (the assistant).
{format_instructions}
""",
            ),
            ("human", """
## Conversation History
{formatted_history}
## Current Question to Answer
Based on the history, provide a CoT and a Suggested Answer for the following question:
"{current_question}"
"""),
        ]
    )
    def format_history(history: List[Dict[str, Any]]) -> str:
        if not history:
            return "No history yet."
        return "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in history)
    chain = prompt_template | llm | json_parser
    try:
        response = chain.invoke({
            "current_question": current_question,
            "formatted_history": format_history(conversation_history),
            "format_instructions": json_parser.get_format_instructions(),
        })
        return response
    except Exception as e:
        print(f"Error in planner logic: {e}")
        return {
            "thought": "An error occurred while generating the thought process.",
            "suggestion": "Could not generate a suggestion. Please try rephrasing or continuing the conversation."
        }