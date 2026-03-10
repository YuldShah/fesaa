from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from core.settings import get_settings
from models.schemas import SubmitAssessmentResponse, TaskStatusResponse

router = APIRouter()


def _get_celery_app():
    from worker.celery_app import celery_app
    return celery_app


@router.post("", response_model=SubmitAssessmentResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_assessment(audio: UploadFile = File(...)) -> SubmitAssessmentResponse:
    settings = get_settings()
    raw_bytes = await audio.read()

    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio is empty.")

    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(raw_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds {settings.max_upload_size_mb}MB limit.",
        )

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    stored_path = settings.upload_dir / f"{uuid4()}{suffix}"
    stored_path.write_bytes(raw_bytes)

    from worker.tasks import assess_audio_task

    task = assess_audio_task.delay(str(stored_path))
    return SubmitAssessmentResponse(task_id=task.id)


@router.get("/{task_id}", response_model=TaskStatusResponse)
def get_assessment_status(task_id: str) -> TaskStatusResponse:
    from celery.result import AsyncResult

    celery_app = _get_celery_app()
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.successful():
        result_payload = task_result.result if isinstance(task_result.result, dict) else {"value": task_result.result}
        return TaskStatusResponse(task_id=task_id, status="completed", result=result_payload)

    if task_result.failed():
        return TaskStatusResponse(task_id=task_id, status="failed", error=str(task_result.result))

    return TaskStatusResponse(task_id=task_id, status=task_result.state.lower())
