from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from worker.celery_app import celery_app
from core.settings import get_settings
from models.schemas import EvaluateBatchRequest
from pipeline.gemini_service import GeminiEvaluator
from pipeline.orchestrator import AssessmentOrchestrator
from pipeline.whisperx_service import WhisperXProvider

# ── Per-worker singleton for WhisperX ──────────────────────────────────────────
# Model load is expensive (~seconds + multi-GB VRAM).  Reuse across tasks.
_whisperx_provider: WhisperXProvider | None = None
_whisperx_lock = threading.Lock()


def _get_whisperx() -> WhisperXProvider:
    global _whisperx_provider
    with _whisperx_lock:
        if _whisperx_provider is None:
            _whisperx_provider = WhisperXProvider(get_settings())
    return _whisperx_provider


@celery_app.task(name="worker.tasks.assess_audio", bind=True)
def assess_audio_task(self: Any, audio_path: str) -> dict[str, Any]:
    settings = get_settings()
    whisperx_provider = _get_whisperx()
    orchestrator = AssessmentOrchestrator(
        settings=settings,
        asr_provider=whisperx_provider,
        pronunciation_provider=whisperx_provider,
        llm_evaluator=GeminiEvaluator(settings),
    )
    report = orchestrator.run(Path(audio_path))
    return report.model_dump(mode="json")


@celery_app.task(name="worker.tasks.transcribe_audio", bind=True)
def transcribe_audio_task(self: Any, audio_path: str) -> dict[str, Any]:
    settings = get_settings()
    whisperx_provider = _get_whisperx()
    orchestrator = AssessmentOrchestrator(
        settings=settings,
        asr_provider=whisperx_provider,
        pronunciation_provider=whisperx_provider,
        llm_evaluator=GeminiEvaluator(settings), # LLM unused in this method but needed for init
    )
    report = orchestrator.run_transcribe(Path(audio_path))
    return report.model_dump(mode="json")


@celery_app.task(name="worker.tasks.evaluate_batch", bind=True)
def evaluate_batch_task(self: Any, batch_data: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    whisperx_provider = _get_whisperx() # Needed for init, even if unused
    orchestrator = AssessmentOrchestrator(
        settings=settings,
        asr_provider=whisperx_provider,
        pronunciation_provider=whisperx_provider,
        llm_evaluator=GeminiEvaluator(settings),
    )
    req = EvaluateBatchRequest.model_validate(batch_data)
    report = orchestrator.run_evaluate_batch(req)
    return report.model_dump(mode="json")


@celery_app.task(name="worker.tasks.sweep_stale_uploads")
def sweep_stale_uploads_task() -> dict[str, int]:
    from pipeline.cleanup import sweep_stale_uploads
    deleted = sweep_stale_uploads(get_settings())
    return {"deleted": deleted}
