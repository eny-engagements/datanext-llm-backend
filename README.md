Okay, I'll help you transform your `README.md` into the desired format, incorporating the project structure and the API interaction flow you provided.

Here's the revamped `README.md`:

-----

# 🚀 DataNext LLM Backend

## 📌 Project Overview

This project is a **FastAPI-based** microservice that integrates **LangChain** and **LangGraph** to provide advanced AI-powered functionalities. It is designed with a modular architecture to support various use cases, including intelligent data processing, metadata management, and dynamic AI workflows.

## 📂 Project Structure

```bash
datanext-llm-backend/
│── data/                     # Data storage (e.g., uploaded files, processed outputs)
│── logs/                     # Application logs
│── my_env/                   # Python virtual environment (ignored in Git)
│── notebooks/                # Jupyter notebooks for experimentation and analysis
│── src/
│   ├── agents/               # LangChain agents for complex reasoning
│   ├── chains/               # Custom LangChain chains for sequential tasks
│   ├── controllers/          # FastAPI endpoints for different use-cases
│   ├── prompts/              # Prompt templates for Large Language Models (LLMs)
│   ├── tools/                # LangChain tools for interacting with external systems/data
│   ├── utils/                # Utility scripts and helpers
│   ├── use_cases/            # Specific business logic and AI workflows (e.g., define, extract)
│   ├── langgraph_setup.py    # Configuration and setup for LangGraph StateGraph
│   ├── main.py               # FastAPI main application entry point
│── tests/                    # Unit and integration tests for API and logic
│── .env                      # Environment variables (ignored in Git)
│── .gitignore                # Git ignore file
│── README.md                 # Project documentation
│── requirements.txt          # Python dependencies
```

-----

## ⚡ Features

  - **FastAPI Backend**: Provides a robust and asynchronous API interface.
  - **LangChain & LangGraph Integration**: Leverages the power of LangChain for LLM interactions and LangGraph for building stateful, multi-step AI agents.
  - **Modular Architecture**: Easily extensible for new AI-powered modules and use cases.
  - **Dynamic Workflows**: Supports complex conversational flows and data processing pipelines.
  - **File Upload Support**: Handles file uploads for AI analysis and processing.

-----

## 📦 Installation (Running Locally)

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/your-repo/datanext-llm-backend.git # Update with your actual repo URL
cd datanext-llm-backend
```

### 2️⃣ Create a Virtual Environment

```sh
python -m venv my_env

source my_env/bin/activate  # For macOS/Linux
my_env\Scripts\activate     # For Windows
```

### 3️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

-----

## 🔑 Environment Variables

Create a `.env` file in the project root directory and populate it with your API keys and other configurations:

```ini
OPENAI_API_KEY=your-openai-api-key
LANGCHAIN_API_KEY=your-langchain-api-key
LANGCHAIN_TRACING_V2=true # or false

DATANEXT_HOST=datanext-backend-host-base-url # Replace with your actual host if needed
DATANEXT_AGENT_KEY=datanext-agent-key # Replace with your actual agent key if needed
```

-----

## 4️⃣ Running the Application Locally

To start the FastAPI application using `uvicorn`:

```sh
uvicorn src.main:app --reload
```

The application will be accessible at `http://127.0.0.1:8000`. The API documentation (Swagger UI) can be found at `http://127.0.0.1:8000/docs`.

-----

## 💡 How to Interact with the API (Example Workflow)

This example demonstrates how to use the API to create a glossary from an Excel file.

1.  **Access API Documentation**:
    Open your web browser and go to `http://127.0.0.1:8000/docs`.

2.  **Start a New Session**:

      * Expand the **`GET /new_session`** endpoint.
      * Click "Try it out" and then "Execute".
      * Copy the `session_id` from the response. This ID will link your subsequent interactions.

3.  **Initiate Chat for Glossary Creation**:

      * Expand the **`POST /chat`** endpoint.
      * Click "Try it out".
      * In the "Request body" field, enter the following JSON (replace `your-session-id` with the one you copied):
        ```json
        {
          "session_id": "your-session-id",
          "message": "I want to create a glossary from Excel."
        }
        ```
      * Click "Execute". The AI will respond, prompting you to upload the file.

4.  **Upload Your Excel File**:

      * Expand the **`POST /upload/{session_id}`** endpoint.
      * Click "Try it out".
      * Paste your `session_id` into the `session_id` **path parameter** field.
      * An input field for a file will appear. Click **"Choose File"** and select your `schema.xlsx` (or any other Excel file).
      * Click "Execute". The response will provide the server-side path where your file is stored, for example:
        ```json
        {
          "server_path": "user_uploads/your-session-id/schema.xlsx"
        }
        ```
        **Note**: On Windows, the path might use backslashes (`\`) like `user_uploads\your-session-id\schema.xlsx`. Make sure to use the path exactly as returned by the API.

5.  **Continue Chat with File Path**:

      * Go back to the **`POST /chat`** endpoint.
      * Click "Try it out".
      * In the "Request body" field, send your next message, including the `server_path` you received from the upload step (again, replace `your-session-id` and `server_path` with your actual values):
        ```json
        {
          "session_id": "your-session-id",
          "message": "I have uploaded the file. The path is user_uploads/your-session-id/schema.xlsx. The schema is for finance."
        }
        ```
      * Click "Execute". The AI will now process your request using the uploaded file.

-----

## 🚀 Deploying on a Server Using Docker (for the very first time)

To deploy the application on a server, follow these steps:

1.  **Ensure Docker is installed** on the server:
    ```sh
    sudo apt update && sudo apt install docker.io -y
    ```
2.  **Clone the repository**:
    ```sh
    git clone https://github.com/your-repo/datanext-llm-backend.git # Update with your actual repo URL
    cd datanext-llm-backend
    ```
3.  **Build and run the Docker container**:
    ```sh
    docker build -t datanext-llm-app .
    docker run -d --name datanext-llm-module -p 8000:8000 datanext-llm-app
    ```
4.  The application will be accessible at `http://your-server-ip:8000`.

### Redeployment (After Code Updates)

1.  **Pull the latest code**:
    ```sh
    git pull origin main # your main development branch
    ```
2.  **Stop and remove the existing container**:
    ```sh
    docker stop datanext-llm-backend
    docker rm datanext-llm-backend
    ```
3.  **Rebuild the Docker image**:
    ```sh
    docker build --no-cache -t datanext-llm-backend .
    ```
4.  **Run the updated container**:
    ```sh
    docker run -d --name datanext-llm-backend -p 8000:8000 datanext-llm-backend
    ```

-----

## 🛠 API Endpoints

The API provides various endpoints based on the available modules and use cases. Refer to the interactive Swagger UI at `http://127.0.0.1:8000/docs` for detailed API usage, request schemas, and response examples.

-----

## 🧪 Running Tests

To execute the unit and integration tests for the project:

```sh
pytest tests/
```

-----

## 🤝 Contributing

We welcome contributions\! Please follow these steps:

1.  Go to main branch & update it:
    ```sh
    git checkout main
    git pull
    ```
2.  Create a new feature branch:
    ```sh
    git checkout -b feature-branch-name
    ```
3.  Commit your changes:
    ```sh
    git add .
    git commit -m "Brief description of your changes"
    ```
4.  Push to your branch:
    ```sh
    git push origin feature-branch-name
    ```
5.  Open a Pull Request (PR) to the `main` branch.

-----
