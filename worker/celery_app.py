from celery import Celery

from core.settings import get_settings

settings = get_settings()
celery_app = Celery(
    "ielts_audio_assessment_v2",
    broker=settings.redis_url,
    backend=settings.celery_result_backend,
    include=["worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "sweep-stale-uploads": {
            "task": "worker.tasks.sweep_stale_uploads",
            "schedule": 3600.0,  # every hour
        },
    },
)
