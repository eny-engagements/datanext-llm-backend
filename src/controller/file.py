import os
import shutil
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.config.settings import settings
from src.models.file import UploadResponse

router = APIRouter()


# TODO: This approach is temporary and would be completely revamped later on
@router.get("/api/files/{filename}")
async def get_document(filename: str):
    """
    Serves files from allowed output directories or a public directory.
    Ensures safe path traversal by checking against allowed base directories.
    """

    found_path = None
    for allowed_dir in settings.ALLOWED_SERVE_DIRECTORIES:
        potential_path = os.path.join(allowed_dir, filename)

        if os.path.exists(potential_path):
            if os.path.abspath(potential_path).startswith(allowed_dir):
                found_path = potential_path
                break

    if found_path:
        print(f"Serving file from: {found_path}")
        return FileResponse(path=found_path, filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found or access denied")


@router.post("/api/files", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a file. Saves the file to a temporary directory
    on the server and returns its path. This path should then be passed
    in the `info` field of the `/api/master` request.
    """

    try:
        # Ensure uploads directory exists
        os.makedirs(settings.UPLOADS_DIR, exist_ok=True)

        safe_filename = os.path.basename(file.filename)
        if not safe_filename:
            raise HTTPException(status_code=400, detail="Invalid filename.")

        # Add a unique prefix to avoid filename collisions
        unique_filename = f"{uuid.uuid4()}_{safe_filename}"
        file_path = os.path.join(settings.UPLOADS_DIR, unique_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return UploadResponse(
            message=f"File '{safe_filename}' uploaded successfully.",
            server_path=file_path,
            filename=safe_filename,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
