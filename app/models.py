from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# ── Request Models ──

class IngestRequest(BaseModel):
    directory: Optional[str] = None  # defaults to STORAGE_DIR if not provided


# ── Response Models ──

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "grabpic"


class FaceRecord(BaseModel):
    id: str
    grab_id: str
    created_at: Optional[datetime] = None


class ImageRecord(BaseModel):
    image_id: str
    file_path: str
    file_name: str
    facial_area: Optional[dict] = None
    created_at: Optional[datetime] = None


class IngestResult(BaseModel):
    total_images_processed: int
    total_faces_detected: int
    new_faces_created: int
    existing_faces_matched: int
    errors: list[str] = []


class IngestSingleResult(BaseModel):
    image_id: str
    file_name: str
    faces_detected: int
    faces: list[dict]


class SelfieAuthResponse(BaseModel):
    authenticated: bool
    grab_id: Optional[str] = None
    similarity: Optional[float] = None
    message: str


class GrabIdImages(BaseModel):
    grab_id: str
    total_images: int
    images: list[ImageRecord]
