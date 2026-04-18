from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    supabase_url: str
    supabase_key: str

    face_model: str = "Facenet512"
    face_detector: str = "retinaface"
    similarity_threshold: float = 0.55

    storage_dir: str = "./storage/raw"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
