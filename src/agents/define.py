from langgraph.prebuilt import create_react_agent

from src.core.constants import AGENT_NAMES
from src.core.llm import agent_llm
from src.prompts.define import DEFINE_AGENT_PROMPT
from src.tools.define import define_data_model_from_excel


def create_define_agent():
    """
    Creates and returns the DefineAgent.
    """

    return create_react_agent(
        model=agent_llm,
        tools=[define_data_model_from_excel],
        prompt=DEFINE_AGENT_PROMPT,
        name=AGENT_NAMES.DEFINE,
    )
