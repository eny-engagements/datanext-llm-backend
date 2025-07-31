from langchain_openai import AzureChatOpenAI

from src.config.settings import settings


def get_agent_llm():
    return AzureChatOpenAI(
        azure_deployment=settings.OPENAI_DEPLOYMENT_NAME,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )


def get_supervisor_llm():
    return AzureChatOpenAI(
        model=settings.OPENAI_DEPLOYMENT_NAME,  # Supervisor might use different model if needed
        temperature=settings.LLM_TEMPERATURE,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    )


agent_llm = get_agent_llm()
supervisor_llm = get_supervisor_llm()
