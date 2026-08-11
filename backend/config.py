import os
from datetime import timedelta


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=int(os.getenv("JWT_ACCESS_TOKEN_HOURS", "24")))
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    DB_NAME = os.getenv("MONGO_DB_NAME", "skin_disease_classifier")
    MODEL_PATH = os.getenv("MODEL_PATH", "skin_disease_model.pth")
    CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
    DEBUG = _env_bool("FLASK_DEBUG", False)
    PORT = int(os.getenv("PORT", "5000"))
    DISEASE_CONFIDENCE_THRESHOLD = float(os.getenv("DISEASE_CONFIDENCE_THRESHOLD", "0.55"))

    @classmethod
    def validate(cls) -> None:
        if not cls.JWT_SECRET_KEY:
            raise RuntimeError("JWT_SECRET_KEY is required. Set it in the environment or .env file.")
        if len(cls.JWT_SECRET_KEY) < 32:
            raise RuntimeError("JWT_SECRET_KEY must contain at least 32 characters.")
        if not 0.0 < cls.DISEASE_CONFIDENCE_THRESHOLD < 1.0:
            raise RuntimeError("DISEASE_CONFIDENCE_THRESHOLD must be between 0 and 1.")
