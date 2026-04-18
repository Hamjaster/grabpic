# Grabpic — Intelligent Identity & Retrieval Engine

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        FastAPI Server                         │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  /ingest  │  │ /auth/selfie │  │  /images/{grab_id}     │  │
│  └────┬─────┘  └──────┬───────┘  └──────────┬─────────────┘  │
│       │               │                      │                │
│  ┌────▼───────────────▼──────────────────────▼─────────────┐  │
│  │                  Face Service (DeepFace)                 │  │
│  │  - extract_faces()  → detect + embed all faces          │  │
│  │  - match_face()     → cosine similarity via pgvector    │  │
│  │  - Model: Facenet512 (512-dim embeddings)               │  │
│  │  - Detector: RetinaFace (best for crowds)               │  │
│  └────┬────────────────────────────────────────────────────┘  │
│       │                                                       │
│  ┌────▼────────────────────────────────────────────────────┐  │
│  │              Supabase PostgreSQL + pgvector              │  │
│  │  ┌────────┐  ┌────────┐  ┌─────────────┐               │  │
│  │  │ images │  │ faces  │  │ image_faces │               │  │
│  │  │        │◄─┤grab_id │◄─┤ (join table)│               │  │
│  │  │        │  │embed512│  │ img↔face M:N│               │  │
│  │  └────────┘  └────────┘  └─────────────┘               │  │
│  │  RPC: match_face(embedding, threshold, count)           │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| Framework   | Python 3.11+ / FastAPI              |
| Face AI     | DeepFace (Facenet512 + RetinaFace)  |
| Database    | Supabase PostgreSQL + pgvector      |
| ORM/Client  | supabase-py                         |
| Docs        | Swagger UI (auto via FastAPI)       |

## Database Schema

```
images
  id          UUID PK
  file_path   TEXT        -- local path to the original image
  file_name   TEXT        -- original file name
  created_at  TIMESTAMPTZ

faces
  id          UUID PK
  grab_id     TEXT UNIQUE -- e.g. "GRAB-a1b2c3d4"
  embedding   VECTOR(512) -- Facenet512 face embedding
  created_at  TIMESTAMPTZ

image_faces (many-to-many)
  id          UUID PK
  image_id    UUID FK → images
  face_id     UUID FK → faces
  facial_area JSONB       -- bounding box {x, y, w, h}
  created_at  TIMESTAMPTZ
```

## API Endpoints

| Method | Path                | Description                              |
|--------|---------------------|------------------------------------------|
| POST   | /ingest             | Crawl directory, detect faces, index all |
| POST   | /ingest/single      | Ingest a single uploaded image           |
| POST   | /auth/selfie        | Upload selfie → get matching grab_id     |
| GET    | /images/{grab_id}   | Get all images for a person              |
| GET    | /faces              | List all known grab_ids                  |
| GET    | /health             | Health check                             |

## Key Flows

### 1. Ingestion
1. Scan storage directory for image files (jpg/png/webp)
2. For each image → `DeepFace.represent()` → list of face embeddings
3. For each face embedding → `match_face` RPC → check existing faces
4. If match (similarity > threshold) → reuse grab_id
5. If no match → generate new grab_id, INSERT face
6. INSERT image_faces mapping

### 2. Selfie Authentication
1. User uploads selfie via multipart form
2. Extract face embedding from selfie
3. Call `match_face` RPC with embedding
4. Return grab_id if similarity > threshold, else 401

### 3. Image Retrieval
1. Query image_faces JOIN images WHERE face.grab_id = input
2. Return list of image records

## Configuration (.env)

```
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
SUPABASE_DB_URL=postgresql://postgres:xxx@db.xxx.supabase.co:5432/postgres
FACE_MODEL=Facenet512
FACE_DETECTOR=retinaface
SIMILARITY_THRESHOLD=0.55
STORAGE_DIR=./storage/raw
```

## Project Structure

```
grabpic/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── migrations/
│   └── 001_init.sql
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── models.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── face_service.py
│   └── routes/
│       ├── __init__.py
│       ├── ingest.py
│       ├── auth.py
│       └── images.py
├── storage/
│   └── raw/
│       └── .gitkeep
└── tests/
    ├── __init__.py
    └── test_api.py
```
