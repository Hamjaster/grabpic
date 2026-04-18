import os
import uuid
import logging
import numpy as np
from pathlib import Path
from deepface import DeepFace
from app.config import get_settings
from app.database import get_supabase

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _generate_grab_id() -> str:
    return f"GRAB-{uuid.uuid4().hex[:8].upper()}"


def extract_faces(image_path: str) -> list[dict]:
    """
    Detect all faces in an image and return their embeddings + bounding boxes.
    Returns list of {"embedding": list[float], "facial_area": dict}
    """
    settings = get_settings()
    try:
        results = DeepFace.represent(
            img_path=image_path,
            model_name=settings.face_model,
            detector_backend=settings.face_detector,
            enforce_detection=True,
        )
        return [
            {
                "embedding": r["embedding"],
                "facial_area": r["facial_area"],
            }
            for r in results
        ]
    except ValueError as e:
        if "Face could not be detected" in str(e):
            logger.info(f"No faces found in {image_path}")
            return []
        raise


def find_matching_face(embedding: list[float]) -> dict | None:
    """
    Query pgvector via Supabase RPC to find the nearest matching grab_id.
    Returns {"id": ..., "grab_id": ..., "similarity": ...} or None.
    """
    settings = get_settings()
    db = get_supabase()

    result = db.rpc(
        "match_face",
        {
            "query_embedding": embedding,
            "match_threshold": settings.similarity_threshold,
            "match_count": 1,
        },
    ).execute()

    if result.data and len(result.data) > 0:
        return result.data[0]
    return None


def create_face(embedding: list[float]) -> dict:
    """
    Insert a new face with a fresh grab_id. Returns the created record.
    """
    db = get_supabase()
    grab_id = _generate_grab_id()
    embedding_str = f"[{','.join(str(v) for v in embedding)}]"

    result = (
        db.table("faces")
        .insert({"grab_id": grab_id, "embedding": embedding_str})
        .execute()
    )
    return result.data[0]


def create_image(file_path: str, file_name: str) -> dict:
    """Insert an image record. Returns the created record."""
    db = get_supabase()
    result = (
        db.table("images")
        .insert({"file_path": file_path, "file_name": file_name})
        .execute()
    )
    return result.data[0]


def create_image_face_link(image_id: str, face_id: str, facial_area: dict) -> dict:
    """Insert the many-to-many link between image and face."""
    db = get_supabase()
    result = (
        db.table("image_faces")
        .insert({
            "image_id": image_id,
            "face_id": face_id,
            "facial_area": facial_area,
        })
        .execute()
    )
    return result.data[0]


def process_single_image(image_path: str) -> dict:
    """
    Full pipeline for one image:
    1. Detect all faces & embeddings
    2. Match or create grab_ids
    3. Persist image + mappings
    Returns summary dict.
    """
    file_name = os.path.basename(image_path)
    abs_path = str(Path(image_path).resolve())

    faces = extract_faces(abs_path)
    if not faces:
        return {
            "file_name": file_name,
            "faces_detected": 0,
            "faces": [],
            "image_id": None,
        }

    image_record = create_image(file_path=abs_path, file_name=file_name)
    image_id = image_record["id"]

    face_results = []
    for face_data in faces:
        embedding = face_data["embedding"]
        facial_area = face_data["facial_area"]

        match = find_matching_face(embedding)
        if match:
            face_id = match["id"]
            grab_id = match["grab_id"]
            is_new = False
        else:
            face_record = create_face(embedding)
            face_id = face_record["id"]
            grab_id = face_record["grab_id"]
            is_new = True

        create_image_face_link(image_id, face_id, facial_area)

        face_results.append({
            "grab_id": grab_id,
            "is_new_face": is_new,
            "facial_area": facial_area,
        })

    return {
        "file_name": file_name,
        "image_id": image_id,
        "faces_detected": len(faces),
        "faces": face_results,
    }


def crawl_and_ingest(directory: str) -> dict:
    """
    Crawl a directory for images and process each one.
    Returns aggregated stats.
    """
    dir_path = Path(directory).resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    image_files = [
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    total_faces = 0
    new_faces = 0
    matched_faces = 0
    errors = []

    for img_file in image_files:
        try:
            result = process_single_image(str(img_file))
            total_faces += result["faces_detected"]
            for face in result["faces"]:
                if face.get("is_new_face"):
                    new_faces += 1
                else:
                    matched_faces += 1
        except Exception as e:
            logger.error(f"Error processing {img_file.name}: {e}")
            errors.append(f"{img_file.name}: {str(e)}")

    return {
        "total_images_processed": len(image_files),
        "total_faces_detected": total_faces,
        "new_faces_created": new_faces,
        "existing_faces_matched": matched_faces,
        "errors": errors,
    }
