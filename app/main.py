import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models import HealthResponse
from app.routes import ingest, auth, images

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Grabpic",
    description=(
        "**Intelligent Identity & Retrieval Engine**\n\n"
        "High-performance image processing backend that uses facial recognition "
        "to automatically group images and provide a Selfie-as-a-Key retrieval system.\n\n"
        "### Flows\n"
        "1. **Ingest** — Crawl a storage directory, detect faces, assign unique `grab_id`s\n"
        "2. **Selfie Auth** — Upload a selfie to authenticate and get your `grab_id`\n"
        "3. **Retrieve** — Fetch all images containing a specific person by `grab_id`"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(auth.router)
app.include_router(images.router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse()
