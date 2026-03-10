"""
AA Pipeline Control Server
Instruments every pipeline stage and streams events via SSE.

Start from project root:
    uvicorn frontend.control_server:app --port 8001 --reload

Or standalone:
    python frontend/control_server.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── path: allow importing project modules ──────────────────────────────────────
HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from core.settings import get_settings
from models.schemas import AggregateMetrics, AssessmentReport
from pipeline.audio import preprocess_audio
from pipeline.gemini_service import GeminiEvaluator
from pipeline.metrics import calculate_fluency_metrics, calculate_lexical_metrics
from pipeline.scoring import build_final_evaluation, compute_deterministic_scores, round_half_up
from pipeline.whisperx_service import WhisperXProvider

# ── app ────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Capture the event loop; ensure dirs exist; load persisted jobs; clean up threads on shutdown."""
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    s = get_settings()
    s.upload_dir.mkdir(parents=True, exist_ok=True)
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    _load_jobs()
    yield
    _save_jobs()
    _executor.shutdown(wait=False)


app = FastAPI(title="AA Control Server", docs_url="/api/docs", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── shared state ───────────────────────────────────────────────────────────────
JOBS_FILE = Path("data/jobs.json")
_jobs: dict[str, dict[str, Any]] = {}
_subscribers: dict[str, list[asyncio.Queue]] = {}
_executor = ThreadPoolExecutor(max_workers=4)
_main_loop: asyncio.AbstractEventLoop | None = None


def _load_jobs() -> None:
    """Load persisted jobs from JSON file."""
    global _jobs
    if JOBS_FILE.exists():
        try:
            data = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
            _jobs = {j["id"]: j for j in data}
        except Exception as e:
            print(f"Warning: failed to load jobs from {JOBS_FILE}: {e}")
            _jobs = {}


def _save_jobs() -> None:
    """Persist jobs to JSON file."""
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        JOBS_FILE.write_text(json.dumps(list(_jobs.values()), indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"Warning: failed to save jobs to {JOBS_FILE}: {e}")

# ── WhisperX singleton — model load is expensive, share across jobs ───────────
_whisperx_provider: WhisperXProvider | None = None
_whisperx_lock = threading.Lock()


def _get_whisperx() -> WhisperXProvider:
    global _whisperx_provider
    with _whisperx_lock:
        if _whisperx_provider is None:
            _whisperx_provider = WhisperXProvider(get_settings())
    return _whisperx_provider


# ── event helpers ──────────────────────────────────────────────────────────────
async def _push(job_id: str, event: dict) -> None:
    for q in list(_subscribers.get(job_id, [])):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


def _emit(
    job_id: str,
    stage: str,
    status: str,
    inp: Any = None,
    out: Any = None,
    elapsed_ms: int | None = None,
) -> dict:
    event = {
        "type": "pipeline_event",
        "job_id": job_id,
        "stage": stage,
        "status": status,
        "input": inp,
        "output": out,
        "elapsed_ms": elapsed_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _jobs[job_id]["events"].append(event)
    if _main_loop and _main_loop.is_running():
        asyncio.run_coroutine_threadsafe(_push(job_id, event), _main_loop)
    return event


# ── pipeline runner (thread pool) ─────────────────────────────────────────────
def _run_pipeline(job_id: str, audio_path: str) -> None:
    settings = get_settings()

    def emit(stage: str, status: str, inp: Any = None, out: Any = None, ms: int | None = None) -> None:
        _emit(job_id, stage, status, inp, out, ms)

    def run_stage(name: str, fn, inp_log: dict | None = None):
        """Emit 'started', time fn(), return (result, elapsed_ms). Emit 'failed' and re-raise on error."""
        emit(name, "started", inp=inp_log or {})
        t0 = time.perf_counter()
        try:
            result = fn()
        except Exception as ex:
            emit(name, "failed", inp=inp_log or {}, out={"error": str(ex)})
            raise
        return result, int((time.perf_counter() - t0) * 1000)

    try:
        _jobs[job_id]["status"] = "running"
        emit("job", "started")

        # ── 1. Preprocess audio ────────────────────────────────────────────────
        prepared, ms = run_stage(
            "preprocess_audio",
            lambda: preprocess_audio(Path(audio_path), settings.upload_dir, settings.ffmpeg_path),
            inp_log={"input_path": audio_path},
        )
        emit(
            "preprocess_audio", "completed",
            inp={"input_path": audio_path},
            out={"output_path": str(prepared), "format": prepared.suffix},
            ms=ms,
        )

        # ── 2. ASR transcription ───────────────────────────────────────────────
        whisperx = _get_whisperx()
        runtime = whisperx.runtime_info()
        transcript, ms = run_stage(
            "transcribe_asr",
            lambda: whisperx.transcribe_with_alignment(prepared),
            inp_log={"audio_path": str(prepared), "model": settings.whisperx_model_name, **runtime},
        )
        word_count = sum(len(s.words) for s in transcript.segments)
        emit(
            "transcribe_asr", "completed",
            inp={"audio_path": str(prepared), "model": settings.whisperx_model_name, **runtime},
            out={
                "text": transcript.text[:300] + ("\u2026" if len(transcript.text) > 300 else ""),
                "language": transcript.language,
                "duration_seconds": transcript.duration_seconds,
                "segment_count": len(transcript.segments),
                "word_count": word_count,
                **runtime,
            },
            ms=ms,
        )

        # ── 3. Fluency metrics ─────────────────────────────────────────────────
        fluency, ms = run_stage(
            "fluency_metrics",
            lambda: calculate_fluency_metrics(
                transcript,
                pause_threshold_seconds=settings.pause_threshold_seconds,
                audio_path=prepared,
                enable_vad=settings.enable_vad_pause_refinement,
                vad_aggressiveness=settings.vad_aggressiveness,
                vad_frame_ms=settings.vad_frame_ms,
                filler_context_pause_seconds=settings.filler_context_pause_seconds,
            ),
            inp_log={
                "pause_threshold_seconds": settings.pause_threshold_seconds,
                "enable_vad_pause_refinement": settings.enable_vad_pause_refinement,
                "vad_aggressiveness": settings.vad_aggressiveness,
                "vad_frame_ms": settings.vad_frame_ms,
            },
        )
        emit(
            "fluency_metrics", "completed",
            inp={
                "pause_threshold_seconds": settings.pause_threshold_seconds,
                "enable_vad_pause_refinement": settings.enable_vad_pause_refinement,
                "vad_aggressiveness": settings.vad_aggressiveness,
                "vad_frame_ms": settings.vad_frame_ms,
            },
            out={
                "words_per_minute": fluency.words_per_minute,
                "speech_ratio": fluency.speech_ratio,
                "pause_count": len(fluency.pauses),
                "long_pause_count": fluency.long_pause_count,
                "filler_count": len(fluency.filler_words),
                "pause_rate_per_minute": fluency.pause_rate_per_minute,
                "pause_detection_method": fluency.pause_detection_method,
            },
            ms=ms,
        )

        # ── 4. Lexical metrics ─────────────────────────────────────────────────
        lexical, ms = run_stage(
            "lexical_metrics",
            lambda: calculate_lexical_metrics(transcript),
            inp_log={},
        )
        emit(
            "lexical_metrics", "completed",
            out={
                "total_words": lexical.total_words,
                "unique_words": lexical.unique_words,
                "type_token_ratio": lexical.type_token_ratio,
            },
            ms=ms,
        )

        # ── 5. Pronunciation metrics ───────────────────────────────────────────
        pronunciation, ms = run_stage(
            "pronunciation_metrics",
            lambda: whisperx.extract_pronunciation_metrics(transcript, settings.low_confidence_threshold),
            inp_log={"low_confidence_threshold": settings.low_confidence_threshold},
        )
        emit(
            "pronunciation_metrics", "completed",
            inp={"low_confidence_threshold": settings.low_confidence_threshold},
            out={
                "total_scored_words": pronunciation.total_scored_words,
                "content_scored_words": pronunciation.content_scored_words,
                "low_confidence_ratio": pronunciation.low_confidence_ratio,
                "raw_low_confidence_ratio": pronunciation.raw_low_confidence_ratio,
                "low_confidence_word_count": len(pronunciation.low_confidence_words),
            },
            ms=ms,
        )

        # ── 6. LLM evaluation ─────────────────────────────────────────────────
        metrics = AggregateMetrics(fluency=fluency, lexical=lexical, pronunciation=pronunciation)
        scores, score_metadata = compute_deterministic_scores(
            transcript,
            metrics,
            calibration_scale=settings.scoring_calibration_scale,
            calibration_bias=settings.scoring_calibration_bias,
        )
        evaluator = GeminiEvaluator(settings)
        llm_result, ms = run_stage(
            "llm_evaluate",
            lambda: build_final_evaluation(
                scores,
                evaluator.evaluate(transcript, metrics, fixed_scores=scores),
                score_metadata,
            ),
            inp_log={
                "model": settings.gemini_model,
                "scoring_version": score_metadata.get("scoring_version"),
            },
        )
        emit(
            "llm_evaluate", "completed",
            inp={
                "model": settings.gemini_model,
                "scoring_version": score_metadata.get("scoring_version"),
            },
            out=llm_result,
            ms=ms,
        )

        # ── 7. Save report ─────────────────────────────────────────────────────
        report_id = str(uuid.uuid4())
        report_path = settings.reports_dir / f"{report_id}.json"
        report = AssessmentReport(
            report_id=report_id,
            created_at=datetime.now(timezone.utc),
            report_path=str(report_path),
            transcript=transcript,
            metrics=metrics,
            llm_evaluation=llm_result,
        )
        t0 = time.perf_counter()
        emit("save_report", "started", inp={"report_id": report_id})
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        save_ms = int((time.perf_counter() - t0) * 1000)
        emit(
            "save_report", "completed",
            inp={"report_id": report_id},
            out={"report_id": report_id, "path": str(report_path)},
            ms=save_ms,
        )

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = report.model_dump(mode="json")
        _save_jobs()
        emit("job", "completed", out={"report_id": report_id})

    except Exception as ex:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(ex)
        _save_jobs()
        emit("job", "failed", out={"error": str(ex)})


# ── static HTML pages ──────────────────────────────────────────────────────────
@app.get("/")
async def _root():
    return FileResponse(str(HERE / "index.html"))


@app.get("/job/{job_id}")
async def _job_page(job_id: str):
    return FileResponse(str(HERE / "job.html"))


# ── REST API ───────────────────────────────────────────────────────────────────
@app.get("/api/jobs")
async def list_jobs():
    rows = []
    for j in sorted(_jobs.values(), key=lambda x: x["created_at"], reverse=True):
        band_score = None
        if j.get("result") and j["result"].get("llm_evaluation"):
            sc = j["result"]["llm_evaluation"].get("scores", {})
            if sc:
                vals = [v for v in sc.values() if isinstance(v, (int, float))]
                if vals:
                    avg = sum(vals) / len(vals)
                    band_score = round(round_half_up(avg, 0.5), 1)

        elapsed_seconds = None
        if j["status"] == "completed":
            for ev in j.get("events", []):
                if ev.get("stage") == "job" and ev.get("status") == "completed":
                    completed_ts = ev.get("timestamp")
                    if completed_ts:
                        try:
                            completed_dt = datetime.fromisoformat(completed_ts.replace("Z", "+00:00"))
                            created_dt = datetime.fromisoformat(j["created_at"].replace("Z", "+00:00"))
                            elapsed_seconds = int((completed_dt - created_dt).total_seconds())
                        except Exception:
                            pass
                    break

        rows.append({
            "id": j["id"],
            "status": j["status"],
            "filename": j["filename"],
            "created_at": j["created_at"],
            "event_count": len(j["events"]),
            "error": j.get("error"),
            "band_score": band_score,
            "elapsed_seconds": elapsed_seconds,
        })
    return rows


@app.post("/api/jobs", status_code=202)
async def create_job(audio: UploadFile = File(...)):
    settings = get_settings()
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_upload_size_mb} MB limit")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    job_id = str(uuid.uuid4())
    audio_path = settings.upload_dir / f"{job_id}{suffix}"
    audio_path.write_bytes(raw)

    _jobs[job_id] = {
        "id": job_id,
        "status": "pending",
        "filename": audio.filename or "audio.wav",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audio_path": str(audio_path),
        "events": [],
        "result": None,
        "error": None,
    }
    _subscribers[job_id] = []
    _save_jobs()
    _executor.submit(_run_pipeline, job_id, str(audio_path))
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs/{job_id}/events")
async def stream_events(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")

    async def generator():
        # Replay all buffered events
        for ev in list(_jobs[job_id]["events"]):
            yield f"data: {json.dumps(ev)}\n\n"

        # Already terminal — close immediately
        if _jobs[job_id]["status"] in ("completed", "failed"):
            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
            return

        # Subscribe to live events
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        _subscribers[job_id].append(q)
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield f"data: {json.dumps(ev)}\n\n"
                    if ev.get("stage") == "job" and ev.get("status") in ("completed", "failed"):
                        yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
                        break
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            subs = _subscribers.get(job_id, [])
            if q in subs:
                subs.remove(q)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/jobs/{job_id}/audio")
async def get_audio(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = Path(job["audio_path"])
    if not path.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(str(path))


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs.pop(job_id)
    _subscribers.pop(job_id, None)
    try:
        Path(job["audio_path"]).unlink(missing_ok=True)
    except Exception:
        pass
    _save_jobs()
    return {"deleted": job_id}


# ── entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n  AA Control Server")
    print("  UI  \u2192  http://localhost:8001")
    print("  API \u2192  http://localhost:8001/api/docs\n")
    uvicorn.run("frontend.control_server:app", host="0.0.0.0", port=8001, reload=False)
