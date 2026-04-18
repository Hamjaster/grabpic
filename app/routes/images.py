import logging
from fastapi import APIRouter, HTTPException
from app.models import GrabIdImages, ImageRecord, FaceRecord
from app.database import get_supabase

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Images & Faces"])


@router.get(
    "/images/{grab_id}",
    response_model=GrabIdImages,
    summary="Get all images for a person",
    description="Given a grab_id, returns all images that contain this person's face.",
)
def get_images_by_grab_id(grab_id: str):
    db = get_supabase()

    face_check = db.table("faces").select("id").eq("grab_id", grab_id).execute()
    if not face_check.data:
        raise HTTPException(status_code=404, detail=f"grab_id '{grab_id}' not found")

    result = db.rpc("get_images_by_grab_id", {"target_grab_id": grab_id}).execute()

    images = [
        ImageRecord(
            image_id=row["image_id"],
            file_path=row["file_path"],
            file_name=row["file_name"],
            facial_area=row.get("facial_area"),
            created_at=row.get("created_at"),
        )
        for row in (result.data or [])
    ]

    return GrabIdImages(
        grab_id=grab_id,
        total_images=len(images),
        images=images,
    )


@router.get(
    "/faces",
    response_model=list[FaceRecord],
    summary="List all known faces / grab_ids",
    description="Returns all unique faces registered in the system with their grab_ids.",
)
def list_faces():
    db = get_supabase()
    result = (
        db.table("faces")
        .select("id, grab_id, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return [
        FaceRecord(
            id=row["id"],
            grab_id=row["grab_id"],
            created_at=row.get("created_at"),
        )
        for row in (result.data or [])
    ]
