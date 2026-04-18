import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Health ──

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "grabpic"


# ── Ingest ──

@patch("app.routes.ingest.crawl_and_ingest")
def test_ingest_directory(mock_crawl):
    mock_crawl.return_value = {
        "total_images_processed": 5,
        "total_faces_detected": 8,
        "new_faces_created": 6,
        "existing_faces_matched": 2,
        "errors": [],
    }
    response = client.post("/ingest", json={"directory": "./storage/raw"})
    assert response.status_code == 200
    data = response.json()
    assert data["total_images_processed"] == 5
    assert data["total_faces_detected"] == 8


@patch("app.routes.ingest.crawl_and_ingest")
def test_ingest_directory_not_found(mock_crawl):
    mock_crawl.side_effect = FileNotFoundError("Directory not found: /nonexistent")
    response = client.post("/ingest", json={"directory": "/nonexistent"})
    assert response.status_code == 404


# ── Selfie Auth ──

@patch("app.routes.auth.find_matching_face")
@patch("app.routes.auth.extract_faces")
def test_selfie_auth_success(mock_extract, mock_match):
    mock_extract.return_value = [
        {"embedding": [0.1] * 512, "facial_area": {"x": 0, "y": 0, "w": 100, "h": 100}}
    ]
    mock_match.return_value = {
        "id": "uuid-123",
        "grab_id": "GRAB-ABCD1234",
        "similarity": 0.89,
    }

    # Create a minimal JPEG-like file
    import io
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/auth/selfie",
        files={"file": ("selfie.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] is True
    assert data["grab_id"] == "GRAB-ABCD1234"


@patch("app.routes.auth.extract_faces")
def test_selfie_auth_no_face(mock_extract):
    mock_extract.return_value = []

    import io
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/auth/selfie",
        files={"file": ("selfie.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] is False
    assert "No face detected" in data["message"]


@patch("app.routes.auth.find_matching_face")
@patch("app.routes.auth.extract_faces")
def test_selfie_auth_no_match(mock_extract, mock_match):
    mock_extract.return_value = [
        {"embedding": [0.1] * 512, "facial_area": {"x": 0, "y": 0, "w": 100, "h": 100}}
    ]
    mock_match.return_value = None

    import io
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/auth/selfie",
        files={"file": ("selfie.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] is False
    assert "not registered" in data["message"]


# ── Images ──

@patch("app.routes.images.get_supabase")
def test_get_images_not_found(mock_db):
    mock_client = MagicMock()
    mock_db.return_value = mock_client
    mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])

    response = client.get("/images/GRAB-NONEXIST")
    assert response.status_code == 404


@patch("app.routes.images.get_supabase")
def test_list_faces(mock_db):
    mock_client = MagicMock()
    mock_db.return_value = mock_client
    mock_client.table.return_value.select.return_value.order.return_value.execute.return_value = MagicMock(
        data=[
            {"id": "uuid-1", "grab_id": "GRAB-AAA", "created_at": "2025-01-01T00:00:00+00:00"},
            {"id": "uuid-2", "grab_id": "GRAB-BBB", "created_at": "2025-01-01T00:00:00+00:00"},
        ]
    )

    response = client.get("/faces")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["grab_id"] == "GRAB-AAA"
