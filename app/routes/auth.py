import os
import tempfile
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models import SelfieAuthResponse
from app.services.face_service import extract_faces, find_matching_face

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/selfie",
    response_model=SelfieAuthResponse,
    summary="Authenticate via selfie",
    description="Upload a selfie image. The system extracts the face embedding, "
    "compares it against all known faces, and returns the matching grab_id "
    "if similarity exceeds the threshold.",
)
async def selfie_auth(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename or "selfie.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        faces = extract_faces(tmp_path)

        if not faces:
            return SelfieAuthResponse(
                authenticated=False,
                message="No face detected in the uploaded image. Please upload a clear selfie.",
            )

        if len(faces) > 1:
            return SelfieAuthResponse(
                authenticated=False,
                message="Multiple faces detected. Please upload a selfie with only one face.",
            )

        embedding = faces[0]["embedding"]
        match = find_matching_face(embedding)

        if match:
            return SelfieAuthResponse(
                authenticated=True,
                grab_id=match["grab_id"],
                similarity=round(match["similarity"], 4),
                message=f"Identity verified. Welcome, {match['grab_id']}!",
            )
        else:
            return SelfieAuthResponse(
                authenticated=False,
                message="No matching identity found. Your face is not registered in the system.",
            )

    except Exception as e:
        logger.error(f"Selfie auth failed: {e}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
