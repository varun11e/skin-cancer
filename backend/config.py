import os
from datetime import timedelta

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=int(os.getenv("JWT_ACCESS_TOKEN_HOURS", "24")))
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    DB_NAME = os.getenv("MONGO_DB_NAME", "skin_disease_classifier")
    MODEL_PATH = os.getenv("MODEL_PATH", "models/skin_disease_model.pth")
    MODEL_URL = os.getenv("MODEL_URL", "").strip()
    MODEL_SHA256 = os.getenv("MODEL_SHA256", "").strip().lower()
    CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
    DEBUG = _env_bool("FLASK_DEBUG", False)
    PORT = int(os.getenv("PORT", "5000"))
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "8"))
    DISEASE_CONFIDENCE_THRESHOLD = float(os.getenv("DISEASE_CONFIDENCE_THRESHOLD", "0.55"))
    RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memory://")

    @classmethod
    def validate(cls) -> None:
        if not cls.JWT_SECRET_KEY:
            raise RuntimeError("JWT_SECRET_KEY is required. Set it in the environment or .env file.")
        if len(cls.JWT_SECRET_KEY) < 32:
            raise RuntimeError("JWT_SECRET_KEY must contain at least 32 characters.")
        if not 0.0 < cls.DISEASE_CONFIDENCE_THRESHOLD < 1.0:
            raise RuntimeError("DISEASE_CONFIDENCE_THRESHOLD must be between 0 and 1.")
        if cls.MAX_UPLOAD_MB < 1 or cls.MAX_UPLOAD_MB > 32:
            raise RuntimeError("MAX_UPLOAD_MB must be between 1 and 32.")
        if cls.MODEL_SHA256 and len(cls.MODEL_SHA256) != 64:
            raise RuntimeError("MODEL_SHA256 must be a 64-character SHA-256 digest.")
