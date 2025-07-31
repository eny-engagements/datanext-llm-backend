import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import settings
from src.controller.chat import router as chat_router
from src.controller.file import router as file_router

# Ensure output directories exist at startup
# Moved this to a separate helper function if there are many, or keep here if few.
# For now, it's fine here as it's part of app startup.

os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
os.makedirs(settings.DEFINE_OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.LINEAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.DISCOVER_OUTPUT_DIR, exist_ok=True)

app = FastAPI(
    title="Master AI Agent API",
    description="API for interacting with the LangGraph-based Master AI Agent.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,  # Allows cookies to be included in requests
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


app.include_router(chat_router)
app.include_router(file_router)

# This part should be for local development only.
if __name__ == "__main__":

    print("Running FastAPI app locally...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
