import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# Why use Pydantic Settings?
# Centralizes configuration: All your app’s settings are in one place.
# Type safety: Ensures that config values are the correct type (e.g., not a string when you expect an int).
# Environment flexibility: Easily switch between development, staging, and production by changing environment variables or .env files.
# Validation: If a required setting is missing or the wrong type, you get a clear error at startup.


class Settings(BaseSettings):
    """
    This class allows us to define configuration variables that can be loaded from
    environment variables or defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore"
    )  # Ignore extra env fields

    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    # TODO: Why are we using this name for the OpenAi model
    OPENAI_DEPLOYMENT_NAME: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 4096

    FRONTEND_REPO_NAME: str = "datanext-frontend"

    # Define output and upload directories relative to the project root
    # Or make them absolute if preferred, but relative is more portable

    # This gets the absolute path of the directory two levels above the current file
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    UPLOADS_DIR: str = os.path.join(BASE_DIR, "..", "output", "user_uploads")

    DISCOVER_OUTPUT_DIR: str = os.path.join(
        BASE_DIR, "..", "output", "business_glossary"
    )  # ASK: Should it be same as define?
    DEFINE_OUTPUT_DIR: str = os.path.join(BASE_DIR, "..", "output", "business_glossary")
    LINEAGE_OUTPUT_DIR: str = os.path.join(BASE_DIR, "..", "output", "lineage_diagrams")

    # For FastAPI CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost", "http://localhost:5173"]

    # This should ideally point to a static directory served by the backend
    # or match the path where the frontend serves its public assets.

    # If both the frontend and backend repos are inside the same parent folder named 'datanext',
    # and the structure is:
    # datanext/
    #   ├── datanext-llm-backend/
    #   └── datanext-frontend/
    # then the public directory is at: ../datanext-frontend/public relative to backend root.

    # Adjust this path based on where your 'datanext-frontend/public' actually resides relative to backend root.
    FRONTEND_PUBLIC_DIR: str = os.path.join(
        BASE_DIR, "..", "..", FRONTEND_REPO_NAME, "public"
    )

    # TODO: Remove these redundant directory
    DISCOVER_AI_OUTPUT_DIR: str = FRONTEND_PUBLIC_DIR
    DEFINE_AI_OUTPUT_DIR: str = FRONTEND_PUBLIC_DIR
    LINEAGE_AI_OUTPUT_DIR: str = FRONTEND_PUBLIC_DIR

    # Directories that are explicitly allowed to be served.
    # These should be subdirectories within a base served directory, or specific absolute paths.
    # Using a list of paths from settings is better than hardcoding.
    ALLOWED_SERVE_DIRECTORIES: list[str] = [
        FRONTEND_REPO_NAME,
        # os.path.join(BASE_DIR, "..", "output"),  # , "discovered_glossary"),
        # os.path.join(BASE_DIR, "..", "output"),  # , "business_glossary"),
        # os.path.join(BASE_DIR, "..", "output"),  # , "lineage_diagrams"),
    ]


settings = Settings()
