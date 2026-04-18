# Grabpic — Intelligent Identity & Retrieval Engine

> High-performance image processing backend that uses facial recognition to automatically group images and provide a **Selfie-as-a-Key** retrieval system.
>
> Built for large-scale events — imagine a marathon with 500 runners and 50,000 photos. Instead of manual tagging, Grabpic uses AI to automatically detect, index, and retrieve images by face.

<img width="1751" height="1021" alt="image" src="https://github.com/user-attachments/assets/0848b294-63c0-4b2d-a853-729c6511f132" />

---

## Table of Contents

1. [Tech Stack](#tech-stack)
2. [Architecture & Design](#architecture--design)
3. [Database Schema](#database-schema)
4. [API Endpoints](#api-endpoints)
5. [Requirements Fulfillment](#requirements-fulfillment)
6. [Setup & Run (Local)](#setup--run-local)
7. [Setup & Run (Production — Render)](#setup--run-production--render)
8. [API Usage & cURL Examples](#api-usage--curl-examples)
9. [Running Tests](#running-tests)
10. [Swagger Docs](#swagger-docs)
11. [Project Structure](#project-structure)

---

## Tech Stack

| Layer            | Technology                                    | Why                                                        |
|------------------|-----------------------------------------------|------------------------------------------------------------|
| **Framework**    | Python 3.10 / FastAPI                         | Async-ready, auto Swagger docs, Pydantic validation        |
| **Face AI**      | DeepFace (Facenet512 model + RetinaFace detector) | 512-dim embeddings, 99.65% accuracy, handles crowds    |
| **Database**     | Supabase PostgreSQL + pgvector                | Managed Postgres with native vector similarity search      |
| **Vector Search**| pgvector (`<=>` cosine distance operator)     | Sub-millisecond nearest-neighbor face matching             |
| **API Client**   | supabase-py                                   | Official Supabase Python SDK (REST + RPC)                  |
| **Docs**         | Swagger UI + ReDoc (auto-generated)           | Zero-config interactive API documentation                  |
| **Inference**    | ONNX Runtime                                  | Lightweight ML inference (~30MB vs TensorFlow's ~600MB)    |

---

## Architecture & Design

```
┌──────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                            │
│                                                                   │
│  ┌────────────┐  ┌────────────────┐  ┌─────────────────────────┐ │
│  │ POST       │  │ POST           │  │ GET                     │ │
│  │ /ingest    │  │ /auth/selfie   │  │ /images/{grab_id}       │ │
│  │ /ingest/   │  │                │  │ /faces                  │ │
│  │   single   │  │                │  │                         │ │
│  └─────┬──────┘  └───────┬────────┘  └───────────┬─────────────┘ │
│        │                 │                        │               │
│  ┌─────▼─────────────────▼────────────────────────▼─────────────┐ │
│  │                   Face Service (DeepFace)                     │ │
│  │                                                               │ │
│  │  extract_faces(image)                                         │ │
│  │    → RetinaFace detection (handles multiple faces per image)  │ │
│  │    → Facenet512 embedding (512-dimensional vector per face)   │ │
│  │                                                               │ │
│  │  find_matching_face(embedding)                                │ │
│  │    → pgvector cosine similarity search via Supabase RPC      │ │
│  │    → Returns grab_id if similarity > threshold (0.55)         │ │
│  │                                                               │ │
│  │  create_face(embedding)                                       │ │
│  │    → Generates unique GRAB-XXXXXXXX ID                        │ │
│  │    → Stores embedding as vector(512) in PostgreSQL            │ │
│  └─────┬─────────────────────────────────────────────────────────┘ │
│        │                                                           │
│  ┌─────▼─────────────────────────────────────────────────────────┐ │
│  │              Supabase PostgreSQL + pgvector                    │ │
│  │                                                               │ │
│  │  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐     │ │
│  │  │  images   │    │    faces     │    │   image_faces    │     │ │
│  │  │──────────│    │──────────────│    │──────────────────│     │ │
│  │  │ id (PK)  │◄───│ id (PK)      │◄───│ image_id (FK)   │     │ │
│  │  │ file_path│    │ grab_id (UQ) │    │ face_id (FK)     │     │ │
│  │  │ file_name│    │ embedding    │    │ facial_area      │     │ │
│  │  │ created  │    │   vector(512)│    │ created_at       │     │ │
│  │  └──────────┘    └──────────────┘    └──────────────────┘     │ │
│  │                                                               │ │
│  │  RPC Functions:                                               │ │
│  │    match_face(embedding[], threshold, count) → grab_id        │ │
│  │    get_images_by_grab_id(grab_id) → image[]                   │ │
│  └───────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Facenet512 Model** — 512-dimensional embeddings provide the best accuracy/speed tradeoff for face recognition
2. **RetinaFace Detector** — specifically chosen for crowd/event photos; handles multiple overlapping faces reliably
3. **pgvector for Similarity Search** — cosine distance operator (`<=>`) runs at the database level, avoiding round-trip overhead
4. **Supabase RPC Functions** — `match_face()` executes vector search server-side as a stored procedure for performance
5. **Many-to-Many Schema** — `image_faces` join table correctly models "one image contains many people" and "one person appears in many images"

---

## Database Schema

### Entity Relationship Diagram

```
┌──────────────┐       ┌──────────────────┐       ┌──────────────┐
│   images     │       │   image_faces    │       │    faces     │
│──────────────│       │──────────────────│       │──────────────│
│ id       PK  │◄─────┤ image_id     FK  │  ┌───►│ id       PK  │
│ file_path    │       │ face_id      FK  ├──┘    │ grab_id  UQ  │
│ file_name    │       │ facial_area JSON │       │ embedding    │
│ created_at   │       │ created_at       │       │  vector(512) │
└──────────────┘       └──────────────────┘       │ created_at   │
                                                  └──────────────┘

Relationship: images ←──M:N──► faces  (via image_faces)
```

### Tables

| Table          | Column        | Type             | Description                              |
|----------------|---------------|------------------|------------------------------------------|
| **images**     | `id`          | UUID PK          | Auto-generated unique image identifier   |
|                | `file_path`   | TEXT NOT NULL     | Absolute path to the stored image file   |
|                | `file_name`   | TEXT NOT NULL     | Original filename                        |
|                | `created_at`  | TIMESTAMPTZ      | Ingestion timestamp                      |
| **faces**      | `id`          | UUID PK          | Internal face record ID                  |
|                | `grab_id`     | TEXT UNIQUE       | Public identifier (e.g., `GRAB-A1B2C3D4`)|
|                | `embedding`   | VECTOR(512)       | Facenet512 face embedding                |
|                | `created_at`  | TIMESTAMPTZ      | First discovery timestamp                |
| **image_faces**| `id`          | UUID PK          | Link record ID                           |
|                | `image_id`    | UUID FK → images | Which image                              |
|                | `face_id`     | UUID FK → faces  | Which face/person                        |
|                | `facial_area` | JSONB            | Bounding box `{x, y, w, h}`             |
|                | `created_at`  | TIMESTAMPTZ      | Link creation timestamp                  |

### SQL Functions (RPC)

| Function                 | Input                         | Output                    | Purpose                         |
|--------------------------|-------------------------------|---------------------------|---------------------------------|
| `match_face()`           | embedding[], threshold, count | grab_id, similarity score | Cosine similarity face search   |
| `get_images_by_grab_id()`| grab_id string                | image records             | Retrieve all images for a person|

Full migration: [`migrations/001_init.sql`](migrations/001_init.sql)

---

## API Endpoints

| Method | Endpoint            | Description                                         | Tags           |
|--------|---------------------|-----------------------------------------------------|----------------|
| POST   | `/ingest`           | Crawl a directory, detect all faces, index them      | Ingestion      |
| POST   | `/ingest/single`    | Upload & ingest a single image                       | Ingestion      |
| POST   | `/auth/selfie`      | Upload a selfie → authenticate → get `grab_id`      | Authentication |
| GET    | `/images/{grab_id}` | Retrieve all images containing a specific person     | Images & Faces |
| GET    | `/faces`            | List all registered faces/grab_ids                   | Images & Faces |
| GET    | `/health`           | Service health check                                 | System         |
| GET    | `/docs`             | Swagger UI (interactive API documentation)           | Docs           |
| GET    | `/redoc`            | ReDoc (alternative API documentation)                | Docs           |

---

## Requirements Fulfillment

### Discovery & Transformation

| Requirement                                | How Grabpic Fulfills It                                                                                                         |
|--------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Crawl storage to ingest and index images   | `POST /ingest` scans a configurable directory for all image files (jpg, png, webp, bmp, tiff) and processes each one            |
| Facial recognition → unique `grab_id`      | DeepFace extracts 512-dim embeddings per face; each unique face gets a `GRAB-XXXXXXXX` ID via pgvector similarity matching      |
| Single image → multiple people             | `DeepFace.represent()` with RetinaFace detects **all** faces in one image; `image_faces` join table maps 1 image → N grab_ids  |
| Persist in relational/vector-capable DB    | Supabase PostgreSQL with pgvector extension stores embeddings as `vector(512)` with cosine distance search                      |

### Selfie Authentication

| Requirement                               | How Grabpic Fulfills It                                                                                        |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| Authenticate using an image file          | `POST /auth/selfie` accepts multipart file upload                                                              |
| Compare input face against known grab_ids | Extracts embedding from selfie → calls `match_face()` RPC → pgvector cosine similarity against all stored faces|
| Return grab_id as authorizer              | Returns `{"authenticated": true, "grab_id": "GRAB-...", "similarity": 0.87}` if match exceeds threshold       |

### Data Extraction

| Requirement                    | How Grabpic Fulfills It                                                                          |
|--------------------------------|--------------------------------------------------------------------------------------------------|
| Endpoint for fetching images   | `GET /images/{grab_id}` returns all images containing that person via `get_images_by_grab_id` RPC|

### Nice-to-Have

| Feature                   | Status | Details                                                    |
|---------------------------|--------|------------------------------------------------------------|
| Swagger / Postman Docs    | ✅     | Auto-generated Swagger UI at `/docs`, ReDoc at `/redoc`    |
| Unit Tests                | ✅     | 7 tests in `tests/test_api.py` (mocked, no DB required)   |
| Schema & Architecture     | ✅     | Full ER diagram, architecture diagram, and migration SQL   |

---

## Setup & Run (Local)

### Prerequisites

- **Python 3.10** (TensorFlow/DeepFace require ≤3.12)
- **Supabase project** with pgvector enabled (free tier works)

### Step 1: Clone & Create Virtual Environment

```bash
git clone https://github.com/your-username/grabpic.git
cd grabpic

# Create venv with Python 3.10
py -3.10 -m venv myvenv

# Activate (Windows)
myvenv\Scripts\Activate.ps1

# Activate (macOS/Linux)
# source myvenv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
copy .env.example .env
```

Edit `.env` with your Supabase credentials:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
FACE_MODEL=Facenet512
FACE_DETECTOR=retinaface
SIMILARITY_THRESHOLD=0.55
STORAGE_DIR=./storage/raw
```

### Step 4: Run Database Migration

Copy the contents of [`migrations/001_init.sql`](migrations/001_init.sql) into your **Supabase SQL Editor** and execute. This creates:
- `images`, `faces`, `image_faces` tables with pgvector
- `match_face()` and `get_images_by_grab_id()` RPC functions
- Indexes for performance

### Step 5: Start the Server

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 6: Open Docs

Navigate to **http://localhost:8000/docs** for interactive Swagger UI.

---

## Setup & Run (Production — Render)

| Setting        | Value                                                   |
|----------------|---------------------------------------------------------|
| **Build Cmd**  | `pip install -r requirements.txt`                       |
| **Start Cmd**  | `uvicorn app.main:app --host 0.0.0.0 --port $PORT`     |
| **Env Vars**   | `SUPABASE_URL`, `SUPABASE_KEY`, `FACE_MODEL`, `FACE_DETECTOR`, `SIMILARITY_THRESHOLD`, `STORAGE_DIR` |

Render injects `$PORT` automatically.

---

## API Usage & cURL Examples

### 1. Health Check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "service": "grabpic"}
```

### 2. Ingest All Images from a Directory

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "./storage/raw"}'
```

```json
{
  "total_images_processed": 25,
  "total_faces_detected": 42,
  "new_faces_created": 15,
  "existing_faces_matched": 27,
  "errors": []
}
```

### 3. Ingest a Single Image (Upload)

```bash
curl -X POST http://localhost:8000/ingest/single \
  -F "file=@marathon_photo_001.jpg"
```

```json
{
  "image_id": "550e8400-e29b-41d4-a716-446655440000",
  "file_name": "marathon_photo_001.jpg",
  "faces_detected": 3,
  "faces": [
    {"grab_id": "GRAB-A1B2C3D4", "is_new_face": false, "facial_area": {"x": 120, "y": 80, "w": 95, "h": 110}},
    {"grab_id": "GRAB-E5F6G7H8", "is_new_face": true,  "facial_area": {"x": 340, "y": 90, "w": 88, "h": 105}},
    {"grab_id": "GRAB-I9J0K1L2", "is_new_face": true,  "facial_area": {"x": 560, "y": 75, "w": 92, "h": 108}}
  ]
}
```

### 4. Selfie Authentication (Selfie-as-a-Key)

```bash
curl -X POST http://localhost:8000/auth/selfie \
  -F "file=@my_selfie.jpg"
```

**Success:**
```json
{
  "authenticated": true,
  "grab_id": "GRAB-A1B2C3D4",
  "similarity": 0.8723,
  "message": "Identity verified. Welcome, GRAB-A1B2C3D4!"
}
```

**No match:**
```json
{
  "authenticated": false,
  "grab_id": null,
  "similarity": null,
  "message": "No matching identity found. Your face is not registered in the system."
}
```

### 5. Get All Images for a Person

```bash
curl http://localhost:8000/images/GRAB-A1B2C3D4
```

```json
{
  "grab_id": "GRAB-A1B2C3D4",
  "total_images": 7,
  "images": [
    {
      "image_id": "550e8400-e29b-41d4-a716-446655440000",
      "file_path": "/storage/raw/marathon_001.jpg",
      "file_name": "marathon_001.jpg",
      "facial_area": {"x": 120, "y": 80, "w": 95, "h": 110},
      "created_at": "2025-04-18T06:00:00+00:00"
    }
  ]
}
```

### 6. List All Known Faces

```bash
curl http://localhost:8000/faces
```

```json
[
  {"id": "uuid-1", "grab_id": "GRAB-A1B2C3D4", "created_at": "2025-04-18T06:00:00+00:00"},
  {"id": "uuid-2", "grab_id": "GRAB-E5F6G7H8", "created_at": "2025-04-18T06:01:00+00:00"}
]
```

---

## Running Tests

### Unit Tests (mocked, no DB needed)

```bash
python -m pytest tests/ -v
```

Tests cover:
- Health endpoint
- Directory ingestion (success + 404)
- Selfie auth (match found, no face, no match)
- Image retrieval (success + 404)
- Face listing

### End-to-End Tests (requires running server + DB)

```bash
# Terminal 1: Start server
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2: Download test faces + run E2E
python scripts/setup_test_data.py
python scripts/test_e2e.py
```

---

## Swagger Docs

Interactive API documentation is auto-generated and available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Project Structure

```
grabpic/
├── README.md                  ← This file
├── CLAUDE.md                  ← Internal architecture notes
├── requirements.txt           ← Python dependencies
├── .env.example               ← Environment template
├── .gitignore
│
├── migrations/
│   └── 001_init.sql           ← Database schema + pgvector + RPC functions
│
├── app/
│   ├── __init__.py
│   ├── main.py                ← FastAPI app entry point, CORS, router setup
│   ├── config.py              ← Pydantic settings (reads .env)
│   ├── database.py            ← Supabase client singleton
│   ├── models.py              ← Request/Response Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   └── face_service.py    ← Core logic: detect, embed, match, ingest
│   └── routes/
│       ├── __init__.py
│       ├── ingest.py          ← POST /ingest, POST /ingest/single
│       ├── auth.py            ← POST /auth/selfie
│       └── images.py          ← GET /images/{grab_id}, GET /faces
│
├── scripts/
│   ├── setup_test_data.py     ← Download LFW/Olivetti face datasets
│   └── test_e2e.py            ← End-to-end API test runner
│
├── storage/
│   └── raw/                   ← Drop images here for ingestion
│       └── .gitkeep
│
└── tests/
    ├── __init__.py
    └── test_api.py            ← Unit tests (7 tests, mocked)
```

---

## License

MIT
