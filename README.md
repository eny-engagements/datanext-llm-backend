
Go to Src folder
Run uvicorn main:app --reload.
Go to http://127.0.0.1:8000/docs.
Call GET /new_session and copy the session_id.
Expand POST /chat. Send a message like "I want to create a glossary from Excel."
The AI will respond, asking you to upload the file.
Now, expand POST /upload/{session_id}.
Paste your session_id into the path parameter field.
Click "Try it out."
An input field for a file will appear. Click "Choose File" and select your schema.xlsx.
Click "Execute".
The response will give you the server-side path, e.g., "server_path": "user_uploads\\your-session-id\\schema.xlsx".
Go back to POST /chat. Send your next message, including the path you just received.
Request Body:
Generated json
{
    "session_id": "your-session-id",
    "message": "I have uploaded the file. The path is user_uploads\\your-session-id\\schema.xlsx. The schema is for finance."
}