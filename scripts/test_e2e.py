"""
End-to-end test script for Grabpic API.

Runs against a live server at http://localhost:8000.
Tests the full flow: health → ingest → selfie auth → image retrieval.

Usage:
    python scripts/test_e2e.py
    python scripts/test_e2e.py --base-url http://localhost:8000
"""

import os
import sys
import glob
import argparse
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"
PASS = "✓ PASS"
FAIL = "✗ FAIL"


def test_health():
    print("\n── Test: Health Check ──")
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["status"] == "ok"
    print(f"  {PASS} /health → {data}")


def test_ingest_directory(directory: str):
    print("\n── Test: Ingest Directory ──")
    r = requests.post(f"{BASE_URL}/ingest", json={"directory": directory})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    print(f"  Images processed: {data['total_images_processed']}")
    print(f"  Faces detected:   {data['total_faces_detected']}")
    print(f"  New faces:        {data['new_faces_created']}")
    print(f"  Matched faces:    {data['existing_faces_matched']}")
    if data["errors"]:
        print(f"  Errors:           {data['errors']}")
    assert data["total_images_processed"] > 0, "No images processed"
    assert data["total_faces_detected"] > 0, "No faces detected"
    print(f"  {PASS} Ingestion complete")
    return data


def test_ingest_single(image_path: str):
    print(f"\n── Test: Ingest Single ({os.path.basename(image_path)}) ──")
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/ingest/single",
            files={"file": (os.path.basename(image_path), f, "image/jpeg")},
        )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    print(f"  Faces detected: {data['faces_detected']}")
    for face in data["faces"]:
        print(f"    → grab_id={face['grab_id']}, new={face['is_new_face']}")
    print(f"  {PASS} Single ingest complete")
    return data


def test_list_faces():
    print("\n── Test: List All Faces ──")
    r = requests.get(f"{BASE_URL}/faces")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    print(f"  Total faces registered: {len(data)}")
    for face in data[:5]:
        print(f"    → {face['grab_id']}")
    if len(data) > 5:
        print(f"    ... and {len(data) - 5} more")
    assert len(data) > 0, "No faces found"
    print(f"  {PASS} Faces listed")
    return data


def test_selfie_auth(selfie_path: str):
    print(f"\n── Test: Selfie Auth ({os.path.basename(selfie_path)}) ──")
    with open(selfie_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/auth/selfie",
            files={"file": (os.path.basename(selfie_path), f, "image/jpeg")},
        )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    print(f"  Authenticated: {data['authenticated']}")
    print(f"  grab_id:       {data.get('grab_id', 'N/A')}")
    print(f"  Similarity:    {data.get('similarity', 'N/A')}")
    print(f"  Message:       {data['message']}")
    print(f"  {PASS} Selfie auth complete")
    return data


def test_get_images(grab_id: str):
    print(f"\n── Test: Get Images for {grab_id} ──")
    r = requests.get(f"{BASE_URL}/images/{grab_id}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    print(f"  Total images: {data['total_images']}")
    for img in data["images"][:3]:
        print(f"    → {img['file_name']}")
    if data["total_images"] > 3:
        print(f"    ... and {data['total_images'] - 3} more")
    assert data["total_images"] > 0, "No images found"
    print(f"  {PASS} Images retrieved")
    return data


def test_get_images_404():
    print("\n── Test: Get Images (nonexistent grab_id) ──")
    r = requests.get(f"{BASE_URL}/images/GRAB-NONEXIST")
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    print(f"  {PASS} 404 returned correctly")


def main():
    parser = argparse.ArgumentParser(description="E2E tests for Grabpic API")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--storage-dir", default=None, help="Override storage dir for ingest")
    parser.add_argument("--selfie-dir", default=None, help="Override selfie dir for auth")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion (use existing data)")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.base_url

    base_path = Path(__file__).parent.parent
    storage_dir = args.storage_dir or str(base_path / "storage" / "raw")
    selfie_dir = args.selfie_dir or str(base_path / "storage" / "test_selfies")

    passed = 0
    failed = 0

    try:
        # 1. Health
        test_health()
        passed += 1

        # 2. Ingest directory
        if not args.skip_ingest:
            ingest_result = test_ingest_directory(storage_dir)
            passed += 1

        # 3. List faces
        faces = test_list_faces()
        passed += 1

        # 4. Selfie auth (try each selfie in the test_selfies dir)
        selfies = glob.glob(os.path.join(selfie_dir, "*.jpg"))
        if selfies:
            auth_result = test_selfie_auth(selfies[0])
            passed += 1

            # 5. If authenticated, fetch images
            if auth_result.get("authenticated") and auth_result.get("grab_id"):
                test_get_images(auth_result["grab_id"])
                passed += 1

            # Test a few more selfies
            for selfie in selfies[1:3]:
                test_selfie_auth(selfie)
                passed += 1
        else:
            print(f"\n⚠ No selfies found in {selfie_dir}, skipping auth tests")

        # 6. 404 test
        test_get_images_404()
        passed += 1

    except AssertionError as e:
        print(f"\n  {FAIL} {e}")
        failed += 1
    except requests.ConnectionError:
        print(f"\n  {FAIL} Cannot connect to {BASE_URL}. Is the server running?")
        failed += 1
    except Exception as e:
        print(f"\n  {FAIL} Unexpected error: {e}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
