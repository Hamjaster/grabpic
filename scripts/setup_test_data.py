"""
Download face datasets and save as individual image files for testing Grabpic.

Two datasets available:
  - LFW (Labeled Faces in the Wild): Real photos, ~200MB download, best for realistic testing
  - Olivetti: Tiny (4MB), 40 people x 10 images each, grayscale 64x64

Usage:
    python scripts/setup_test_data.py              # LFW (default, recommended)
    python scripts/setup_test_data.py --olivetti    # Olivetti (quick, tiny)
    python scripts/setup_test_data.py --both        # Both datasets

Output structure:
    storage/
    ├── raw/               ← Images for ingestion (POST /ingest)
    │   ├── George_W_Bush_001.jpg
    │   ├── George_W_Bush_002.jpg
    │   ├── Colin_Powell_001.jpg
    │   └── ...
    └── test_selfies/      ← Held-out images for selfie auth testing (POST /auth/selfie)
        ├── George_W_Bush_selfie.jpg
        ├── Colin_Powell_selfie.jpg
        └── ...
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def setup_lfw(storage_dir: str, selfie_dir: str, max_per_person: int = 10, min_faces: int = 30):
    """
    Download LFW dataset, save subset as JPEGs.
    - Puts most images in storage/raw/ for ingestion
    - Holds out 1 image per person in storage/test_selfies/ for auth testing
    """
    from sklearn.datasets import fetch_lfw_people

    print("[LFW] Downloading dataset (first run ~200MB, then cached)...")
    faces = fetch_lfw_people(min_faces_per_person=min_faces, resize=1.0)

    X = faces.images        # (n_samples, h, w)  grayscale float
    y = faces.target        # integer labels
    names = faces.target_names  # person names

    print(f"[LFW] Loaded {len(X)} images of {len(names)} people")
    print(f"[LFW] People: {', '.join(names)}")

    os.makedirs(storage_dir, exist_ok=True)
    os.makedirs(selfie_dir, exist_ok=True)

    person_counts = {}
    selfie_saved = set()
    saved_count = 0

    for i in range(len(X)):
        person_name = names[y[i]].replace(" ", "_")
        person_counts.setdefault(person_name, 0)
        person_counts[person_name] += 1
        idx = person_counts[person_name]

        if idx > max_per_person + 1:
            continue

        img_array = X[i]
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

        if len(img_array.shape) == 2:
            img = Image.fromarray(img_array, mode="L").convert("RGB")
        else:
            img = Image.fromarray(img_array)

        if person_name not in selfie_saved:
            selfie_path = os.path.join(selfie_dir, f"{person_name}_selfie.jpg")
            img.save(selfie_path, "JPEG", quality=95)
            selfie_saved.add(person_name)
            print(f"  [selfie] {selfie_path}")
        else:
            img_path = os.path.join(storage_dir, f"{person_name}_{idx:03d}.jpg")
            img.save(img_path, "JPEG", quality=95)
            saved_count += 1

    print(f"[LFW] Saved {saved_count} images to {storage_dir}")
    print(f"[LFW] Saved {len(selfie_saved)} selfies to {selfie_dir}")


def setup_olivetti(storage_dir: str, selfie_dir: str):
    """
    Download Olivetti faces (40 people, 10 images each, 64x64 grayscale).
    Tiny dataset (~4MB), good for quick smoke tests.
    """
    from sklearn.datasets import fetch_olivetti_faces

    print("[Olivetti] Downloading dataset (~4MB)...")
    faces = fetch_olivetti_faces()

    X = faces.images  # (400, 64, 64) float
    y = faces.target  # 40 people

    print(f"[Olivetti] Loaded {len(X)} images of {len(set(y))} people")

    os.makedirs(storage_dir, exist_ok=True)
    os.makedirs(selfie_dir, exist_ok=True)

    person_counts = {}
    selfie_saved = set()
    saved_count = 0

    for i in range(len(X)):
        person_id = y[i]
        person_name = f"person_{person_id:02d}"
        person_counts.setdefault(person_name, 0)
        person_counts[person_name] += 1
        idx = person_counts[person_name]

        img_array = (X[i] * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L").convert("RGB")
        img = img.resize((250, 250), Image.LANCZOS)

        if person_name not in selfie_saved:
            selfie_path = os.path.join(selfie_dir, f"{person_name}_selfie.jpg")
            img.save(selfie_path, "JPEG", quality=95)
            selfie_saved.add(person_name)
        else:
            img_path = os.path.join(storage_dir, f"{person_name}_{idx:03d}.jpg")
            img.save(img_path, "JPEG", quality=95)
            saved_count += 1

    print(f"[Olivetti] Saved {saved_count} images to {storage_dir}")
    print(f"[Olivetti] Saved {len(selfie_saved)} selfies to {selfie_dir}")


def main():
    parser = argparse.ArgumentParser(description="Setup test face datasets for Grabpic")
    parser.add_argument("--olivetti", action="store_true", help="Use Olivetti dataset (tiny, quick)")
    parser.add_argument("--both", action="store_true", help="Download both LFW and Olivetti")
    parser.add_argument("--max-per-person", type=int, default=10, help="Max images per person for LFW")
    parser.add_argument("--min-faces", type=int, default=30, help="Min faces per person filter for LFW")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    storage_dir = str(base_dir / "storage" / "raw")
    selfie_dir = str(base_dir / "storage" / "test_selfies")

    print(f"Base directory: {base_dir}")
    print(f"Storage (ingest): {storage_dir}")
    print(f"Selfies (auth):   {selfie_dir}")
    print()

    if args.both:
        setup_lfw(storage_dir, selfie_dir, args.max_per_person, args.min_faces)
        olivetti_storage = str(base_dir / "storage" / "raw_olivetti")
        olivetti_selfie = str(base_dir / "storage" / "test_selfies_olivetti")
        setup_olivetti(olivetti_storage, olivetti_selfie)
    elif args.olivetti:
        setup_olivetti(storage_dir, selfie_dir)
    else:
        setup_lfw(storage_dir, selfie_dir, args.max_per_person, args.min_faces)

    print("\n✓ Done! Test data is ready.")
    print("\nNext steps:")
    print("  1. Run migration:  paste migrations/001_init.sql in Supabase SQL Editor")
    print("  2. Start server:   uvicorn app.main:app --reload --port 8000")
    print("  3. Ingest images:  curl -X POST http://localhost:8000/ingest")
    print("  4. Test selfie:    python scripts/test_e2e.py")


if __name__ == "__main__":
    main()
