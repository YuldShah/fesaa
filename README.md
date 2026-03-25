# IELTS Audio Assessment Backend (V2)

Production-oriented backend for IELTS speaking assessment using **FastAPI + Celery + Redis + WhisperX + Gemini**.

## Architecture

```
Client ──► FastAPI :8000 ──► Celery (Redis) ──► Worker
                                                   │
                                    ┌──────────────┤
                                    ▼              ▼
                              WhisperX (GPU)   Gemini (API)
                              ASR + alignment  IELTS rubric eval
                              pronunciation    structured JSON
```

- **Local GPU only** for transcription, alignment and pronunciation confidence.
- **Deterministic scorer** handles numeric rubric bands from extracted signals.
- **Gemini** handles narrative feedback, evidence, and drill generation.
- Strategy interfaces allow swapping in Azure/NeMo providers without refactor.

## Project Structure

```
api/
  main.py                  # FastAPI entry point (:8000)
  routers/assessments.py   # POST/GET /v1/assessments
core/
  settings.py              # Pydantic settings from .env
models/
  schemas.py               # Pydantic models (WordTiming, LowConfidenceWord, etc.)
pipeline/
  audio.py                 # FFmpeg → 16kHz mono WAV
  interfaces.py            # ABC: ASRProvider, PronunciationProvider + stubs
  whisperx_service.py      # WhisperX ASR + pronunciation (GPU-only)
  metrics.py               # Fluency & lexical metric calculators
  llm_prompts.py           # IELTS rubric XML + Gemini system prompt
  gemini_service.py        # Gemini LLM evaluator
  orchestrator.py          # Full pipeline orchestration
  cleanup.py               # Audio artifact retention / sweep
worker/
  celery_app.py            # Celery config (Redis broker/backend, beat schedule)
  tasks.py                 # Celery tasks (assess_audio, sweep_stale_uploads)
frontend/
  control_server.py        # Standalone UI server (:8001) with SSE (bypasses Celery)
  index.html               # Dashboard
  job.html                 # Job detail view
```

## Setup

### 1. Create isolated virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
```

### 2. Install PyTorch with CUDA

```bash
pip install torch~=2.8.0 torchaudio~=2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify install

```bash
pip check
python -c "import torch; print(torch.cuda.is_available())"
```

### 5. Configure environment

```bash
copy .env.example .env
# Edit .env — fill GEMINI_API_KEY and HF_AUTH_TOKEN
```

## Run

Start Redis first, then in separate terminals:

```bash
# Celery worker (GPU tasks)
celery -A worker.celery_app.celery_app worker --loglevel=info --pool=solo

# Optional: Celery beat (periodic upload sweep)
celery -A worker.celery_app.celery_app beat --loglevel=info

# API server
uvicorn api.main:app --reload
```

Or use the standalone frontend server (no Celery required):

```bash
uvicorn frontend.control_server:app --port 8001 --reload
```

The standalone HTML frontend can also use an explicit backend origin. When `FRONTEND_API_BASE_URL` is set, the browser uses that value instead of hardcoded relative `/api/...` requests. That URL should point at a service exposing the `/api/jobs/...` routes, typically the control server.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/healthz` | Health check |
| `POST` | `/v1/assessments` | Submit audio (multipart `audio` file) → `{"task_id": "..."}` |
| `GET` | `/v1/assessments/{task_id}` | Poll status → `pending` / `started` / `completed` / `failed` |

### Response schema (completed)

```jsonc
{
  "task_id": "...",
  "status": "completed",
  "result": {
    "report_id": "uuid",
    "created_at": "ISO8601",
    "transcript": { "text": "...", "language": "en", "duration_seconds": 42.5, "segments": [...] },
    "metrics": {
      "fluency": { "words_per_minute": 130.5, "speech_ratio": 0.82, "pauses": [...] },
      "lexical": { "type_token_ratio": 0.65, "total_words": 210, "unique_words": 137 },
      "pronunciation": {
        "low_confidence_ratio": 0.08,
        "low_confidence_words": [
          { "word": "particularly", "start": 12.4, "end": 13.1, "confidence": 0.42 }
        ],
        "total_scored_words": 200,
        "low_confidence_threshold": 0.60
      }
    },
    "llm_evaluation": {
      "scores": { "fluency_coherence": 7.0, "lexical_resource": 6.5, ... },
      "summary": "...",
      "domain_feedback": { ... },
      "sample_rewrite": "..."
    }
  }
}
```

## Pipeline Flow

1. Convert input to 16kHz mono WAV (FFmpeg).
2. WhisperX transcription + forced alignment → word timestamps + confidence scores.
3. Compute metrics:
   - **Fluency**: WPM, speech ratio, pauses (alignment + optional VAD), filler detection.
   - **Lexical**: TTR (punctuation-normalized tokens).
   - **Pronunciation**: content-word low-confidence ratio with `(word, start, end, confidence)` evidence.
4. Compute deterministic IELTS criterion scores from transcript + metrics.
5. Send transcript + metrics + fixed scores to Gemini for narrative feedback only.
6. Persist JSON report to `reports/`.
7. Optional: immediate or scheduled cleanup of audio artifacts.

## Configuration

Key `.env` variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPERX_MODEL_NAME` | `large-v3` | Whisper model size |
| `WHISPERX_DEVICE` | `cuda` | GPU only (enforced) |
| `PAUSE_THRESHOLD_SECONDS` | `0.40` | Min gap to count as a pause |
| `ENABLE_VAD_PAUSE_REFINEMENT` | `true` | Enable lightweight WebRTC VAD pause refinement |
| `VAD_AGGRESSIVENESS` | `2` | WebRTC VAD aggressiveness (0-3) |
| `VAD_FRAME_MS` | `30` | VAD frame size in milliseconds (10/20/30) |
| `FILLER_CONTEXT_PAUSE_SECONDS` | `0.15` | Context window used for filler classification |
| `LOW_CONFIDENCE_THRESHOLD` | `0.60` | ASR confidence floor for pronunciation |
| `SCORING_CALIBRATION_SCALE` | `1.0` | Global deterministic score scale |
| `SCORING_CALIBRATION_BIAS` | `0.0` | Global deterministic score bias |
| `AUDIO_RETENTION_HOURS` | `24` | Hours to keep audio (0=immediate delete, -1=forever) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for evaluation |
| `CLOUDFLARE_TUNNEL_TOKEN` | `` | Named Cloudflare Tunnel token for a stable public backend URL |
| `PUBLIC_API_BASE_URL` | `` | Stable public HTTPS URL for the AA API, for example `https://api.example.com` |
| `FRONTEND_API_BASE_URL` | `` | Optional origin used by the standalone HTML frontend for `/api/jobs/...` calls |

## Stable Cloudflare tunnel

If you want the AA backend to stop rotating through `trycloudflare.com`, create a named Cloudflare Tunnel and publish a short hostname in your domain.

Example for `api.yall.uz`:

```bash
python configure_cloudflare_service.py \
  --api-token "$CLOUDFLARE_API_TOKEN" \
  --account-id "$CLOUDFLARE_ACCOUNT_ID" \
  --zone-id "$CLOUDFLARE_ZONE_ID" \
  --zone-name yall.uz \
  --tunnel-name aa-backend \
  --hostname api \
  --service http://127.0.0.1:8000 \
  --env-file .env
```

That writes `CLOUDFLARE_TUNNEL_TOKEN` and `PUBLIC_API_BASE_URL=https://api.yall.uz` into `.env`. After that, `start_aa.ps1` automatically uses named-tunnel mode instead of creating a rotating quick tunnel.
