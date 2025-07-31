class AgentNames:
    """
    Centralized collection of unique names assigned to LangGraph agents.
    These names are used for identification within the LangGraph, for routing,
    and potentially for display logic or API responses.
    """

    DEFINE = "DefineAgent"
    LINEAGE = "LineageExtractorAgent"
    DISCOVER = "DiscoverAgent"


# Instantiate the class for easy import and access
AGENT_NAMES = AgentNames()

# We can add other global, fixed constant groups here if they emerge.
# Example:
# class FileTypeSuffixes:
#     EXCEL = ".xlsx"
#     CSV = ".csv"
# FILE_TYPE_CONSTANTS = FileTypeSuffixes()
