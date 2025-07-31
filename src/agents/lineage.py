from langgraph.prebuilt import create_react_agent

from src.core.constants import AGENT_NAMES
from src.core.llm import agent_llm
from src.prompts.lineage import LINEAGE_AGENT_PROMPT
from src.tools.lineage import extract_etl_lineage_and_documentation


def create_lineage_agent():
    """
    Creates and returns the LineageExtractorAgent.
    """

    return create_react_agent(
        model=agent_llm,
        tools=[extract_etl_lineage_and_documentation],
        prompt=LINEAGE_AGENT_PROMPT,
        name=AGENT_NAMES.LINEAGE,
    )
