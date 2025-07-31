from pydantic import BaseModel


class UploadResponse(BaseModel):
    message: str
    server_path: str
    filename: str
