-- ============================================================
-- Grabpic Database Schema
-- Run this in Supabase SQL Editor before starting the server
-- ============================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Images table
CREATE TABLE IF NOT EXISTS images (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path   TEXT NOT NULL,
    file_name   TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- 3. Faces table (one row per unique person)
CREATE TABLE IF NOT EXISTS faces (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    grab_id     TEXT UNIQUE NOT NULL,
    embedding   vector(512) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- 4. Many-to-many mapping: image <-> face
CREATE TABLE IF NOT EXISTS image_faces (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id    UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    face_id     UUID NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
    facial_area JSONB,
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE(image_id, face_id)
);

-- 5. Indexes
CREATE INDEX IF NOT EXISTS idx_faces_grab_id ON faces(grab_id);
CREATE INDEX IF NOT EXISTS idx_image_faces_image ON image_faces(image_id);
CREATE INDEX IF NOT EXISTS idx_image_faces_face ON image_faces(face_id);

-- 6. Vector similarity search index (IVFFlat)
--    NOTE: Only create AFTER inserting some initial data (needs rows to build lists).
--    For small datasets, pgvector will do sequential scan which is fine.
-- CREATE INDEX ON faces USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- 7. RPC: Find nearest matching face(s) by cosine similarity
CREATE OR REPLACE FUNCTION match_face(
    query_embedding float8[],
    match_threshold float DEFAULT 0.55,
    match_count int DEFAULT 1
)
RETURNS TABLE (
    id UUID,
    grab_id TEXT,
    similarity float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        faces.id,
        faces.grab_id,
        1 - (faces.embedding <=> query_embedding::vector) AS similarity
    FROM faces
    WHERE 1 - (faces.embedding <=> query_embedding::vector) > match_threshold
    ORDER BY faces.embedding <=> query_embedding::vector
    LIMIT match_count;
$$;

-- 8. RPC: Get all images for a given grab_id
CREATE OR REPLACE FUNCTION get_images_by_grab_id(target_grab_id TEXT)
RETURNS TABLE (
    image_id UUID,
    file_path TEXT,
    file_name TEXT,
    facial_area JSONB,
    created_at TIMESTAMPTZ
)
LANGUAGE sql STABLE
AS $$
    SELECT
        i.id AS image_id,
        i.file_path,
        i.file_name,
        if2.facial_area,
        i.created_at
    FROM images i
    INNER JOIN image_faces if2 ON if2.image_id = i.id
    INNER JOIN faces f ON f.id = if2.face_id
    WHERE f.grab_id = target_grab_id
    ORDER BY i.created_at DESC;
$$;
