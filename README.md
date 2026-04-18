# Grabpic — Intelligent Identity & Retrieval Engine

High-performance image processing backend that uses facial recognition to automatically group images and provide a **Selfie-as-a-Key** retrieval system.

Built for large-scale events (e.g., marathons with 50,000+ photos).

## Tech Stack

- **Python 3.11+** / **FastAPI**
- **DeepFace** (Facenet512 + RetinaFace)
- **Supabase** PostgreSQL + pgvector
- **Swagger** auto-generated docs

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A Supabase project with pgvector enabled

### 2. Clone & Install

```bash
git clone <repo-url>
cd vyro
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Supabase credentials
```

### 4. Run Database Migration

Copy the contents of `migrations/001_init.sql` into your **Supabase SQL Editor** and execute it. This creates:
- `images`, `faces`, `image_faces` tables
- pgvector extension + indexes
- `match_face` and `get_images_by_grab_id` RPC functions

### 5. Start the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Open Swagger Docs

Navigate to **http://localhost:8000/docs** for interactive API documentation.

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Ingest Images from Directory
```bash
# Ingest all images from the default storage directory
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "./storage/raw"}'
```

### Ingest Single Image
```bash
curl -X POST http://localhost:8000/ingest/single \
  -F "file=@path/to/marathon_photo.jpg"
```

### Selfie Authentication
```bash
curl -X POST http://localhost:8000/auth/selfie \
  -F "file=@path/to/selfie.jpg"
```

**Response:**
```json
{
  "authenticated": true,
  "grab_id": "GRAB-A1B2C3D4",
  "similarity": 0.8723,
  "message": "Identity verified. Welcome, GRAB-A1B2C3D4!"
}
```

### Get Images by grab_id
```bash
curl http://localhost:8000/images/GRAB-A1B2C3D4
```

### List All Known Faces
```bash
curl http://localhost:8000/faces
```

## Running Tests

```bash
pytest tests/ -v
```

## Architecture

```
POST /ingest ──► Crawl directory ──► DeepFace.represent() ──► Embeddings
                                           │
                                    For each face:
                                    ├─ match_face() RPC (pgvector cosine similarity)
                                    ├─ Match found? → reuse grab_id
                                    └─ No match? → create new grab_id
                                           │
                                    Store in: images + faces + image_faces

POST /auth/selfie ──► Extract face ──► match_face() RPC ──► Return grab_id

GET /images/{grab_id} ──► JOIN query ──► Return all images for person
```

## Schema

| Table | Description |
|-------|-------------|
| `images` | Every ingested image (path, name, timestamp) |
| `faces` | Every unique person (grab_id + 512-dim embedding) |
| `image_faces` | Many-to-many mapping (one image → many faces) |

## License

MIT
