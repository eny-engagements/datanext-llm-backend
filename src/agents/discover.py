from langgraph.prebuilt import create_react_agent

from src.core.constants import AGENT_NAMES
from src.core.llm import agent_llm
from src.prompts.discover import DISCOVER_AGENT_PROMPT
from src.tools.discover import (
    discover_database_and_generate_glossary,
    discover_databricks_structure,
    discover_purview_tables,
    generate_databricks_glossary_for_schemas,
    generate_purview_glossary_for_tables,
    list_purview_glossaries,
    push_to_purview_glossary,
    update_unity_catalog,
)


def create_discover_agent():
    """
    Creates and returns the DiscoverAgent.
    """

    tools = [
        # RDBMS Tool
        discover_database_and_generate_glossary,
        # Databricks Tools
        discover_databricks_structure,
        generate_databricks_glossary_for_schemas,
        update_unity_catalog,
        # Purview Tools
        discover_purview_tables,
        generate_purview_glossary_for_tables,
        list_purview_glossaries,
        push_to_purview_glossary,
    ]

    return create_react_agent(
        model=agent_llm,
        tools=tools,
        prompt=DISCOVER_AGENT_PROMPT,
        name=AGENT_NAMES.DISCOVER,
    )
