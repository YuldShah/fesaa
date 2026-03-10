from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    app_name: str = "IELTS Audio Assessment API"
    environment: str = "development"

    redis_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    whisperx_model_name: str = "large-v3"
    whisperx_compute_type: str = "float16"
    whisperx_batch_size: int = 8
    whisperx_device: str = "cuda"
    whisperx_language: str | None = None
    hf_auth_token: str | None = None

    gemini_api_key: str = Field(default="")
    gemini_model: str = "gemini-2.5-flash"

    api_key: str = Field(default="")

    pause_threshold_seconds: float = 0.40
    low_confidence_threshold: float = 0.60

    enable_vad_pause_refinement: bool = True
    vad_aggressiveness: int = 2
    vad_frame_ms: int = 30
    filler_context_pause_seconds: float = 0.15

    scoring_calibration_scale: float = 1.0
    scoring_calibration_bias: float = 0.0

    upload_dir: Path = Path("uploads")
    reports_dir: Path = Path("reports")
    ffmpeg_path: str = "ffmpeg"
    max_upload_size_mb: int = 25

    # Retention: hours to keep uploaded/preprocessed audio after report is saved.
    # 0 = delete immediately after report, -1 = keep forever.
    audio_retention_hours: int = 24


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
