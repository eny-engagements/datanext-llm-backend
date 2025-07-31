from langgraph_supervisor import create_supervisor

from src.agents.define import create_define_agent
from src.agents.discover import create_discover_agent
from src.agents.lineage import create_lineage_agent
from src.core.llm import supervisor_llm
from src.prompts.supervisor import SUPERVISOR_AGENT_PROMPT

# Create instances of each agent
define_agent_instance = create_define_agent()
lineage_extractor_agent_instance = create_lineage_agent()
discover_agent_instance = create_discover_agent()

# Define the members list using the names set in each agent's definition.
# These names must match the 'name' parameter given to create_react_agent.
#
# You could also get the names directly from the agent instances if needed:
# members_names = [agent.name for agent in agents_list_for_supervisor]
members = [
    define_agent_instance.name,
    lineage_extractor_agent_instance.name,
    discover_agent_instance.name,
]

# Compile the master agent graph with the supervisor
master_agent_graph = create_supervisor(
    model=supervisor_llm,
    agents=[
        define_agent_instance,
        lineage_extractor_agent_instance,
        discover_agent_instance,
    ],
    prompt=SUPERVISOR_AGENT_PROMPT,
).compile()
