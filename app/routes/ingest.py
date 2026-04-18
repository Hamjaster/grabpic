import os
import shutil
import tempfile
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.config import get_settings
from app.models import IngestRequest, IngestResult, IngestSingleResult
from app.services.face_service import crawl_and_ingest, process_single_image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post(
    "",
    response_model=IngestResult,
    summary="Crawl & ingest all images in a directory",
    description="Scans a directory for images, detects all faces, assigns grab_ids, "
    "and persists the image-to-face mappings in the database.",
)
def ingest_directory(request: IngestRequest = None):
    settings = get_settings()
    directory = (request.directory if request and request.directory else settings.storage_dir)

    try:
        result = crawl_and_ingest(directory)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return IngestResult(**result)


@router.post(
    "/single",
    response_model=IngestSingleResult,
    summary="Ingest a single uploaded image",
    description="Upload an image file to detect faces, assign grab_ids, "
    "and persist the mappings. The image is saved to the storage directory.",
)
async def ingest_single(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    settings = get_settings()
    os.makedirs(settings.storage_dir, exist_ok=True)
    save_path = os.path.join(settings.storage_dir, file.filename)

    try:
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)

        result = process_single_image(save_path)

        if result["faces_detected"] == 0:
            return IngestSingleResult(
                image_id="",
                file_name=file.filename,
                faces_detected=0,
                faces=[],
            )

        return IngestSingleResult(
            image_id=result["image_id"],
            file_name=result["file_name"],
            faces_detected=result["faces_detected"],
            faces=result["faces"],
        )
    except Exception as e:
        logger.error(f"Single ingest failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
