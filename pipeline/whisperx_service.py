from pathlib import Path
import re
from typing import Any

import torch

from core.settings import Settings
from models.schemas import LowConfidenceWord, PronunciationMetrics, SpeechSegment, TranscriptResult, WordTiming
from pipeline.interfaces import ASRProvider, PronunciationProvider

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")

FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "so",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "too",
    "us",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
}

HESITATION_TOKENS = {
    "uh",
    "um",
    "uhm",
    "er",
    "ah",
    "mm",
    "hmm",
}


def _normalized_token(word: str) -> str | None:
    match = TOKEN_PATTERN.search(word)
    return match.group(0).lower() if match else None


def _is_content_word(word: str) -> bool:
    token = _normalized_token(word)
    if not token:
        return False
    if len(token) <= 2:
        return False
    if token in FUNCTION_WORDS:
        return False
    if token in HESITATION_TOKENS:
        return False
    return True


class WhisperXProvider(ASRProvider, PronunciationProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.requested_device = settings.whisperx_device
        requested = settings.whisperx_device.strip().lower()
        if not requested.startswith("cuda"):
            raise RuntimeError(
                "WhisperX is configured in GPU-only mode. Set WHISPERX_DEVICE to 'cuda' or 'cuda:0'."
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError("WhisperX GPU-only mode requires a detected CUDA device.")

        self.cuda_available = True
        self.device = settings.whisperx_device
        self._asr_model: Any | None = None

    def runtime_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "requested_device": self.requested_device,
            "effective_device": self.device,
            "cuda_available": self.cuda_available,
            "compute_type": self.settings.whisperx_compute_type,
            "model_name": self.settings.whisperx_model_name,
        }
        if self.cuda_available:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        return info

    def _load_whisperx(self) -> Any:
        try:
            import whisperx
        except ImportError as exc:
            raise ImportError("whisperx is required for WhisperXProvider.") from exc
        return whisperx

    def _load_model(self, whisperx_module: Any) -> Any:
        if self._asr_model is None:
            try:
                self._asr_model = whisperx_module.load_model(
                    self.settings.whisperx_model_name,
                    self.device,
                    compute_type=self.settings.whisperx_compute_type,
                    language=self.settings.whisperx_language,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        "WhisperX failed on GPU due to CUDA out of memory. "
                        "Lower WHISPERX_BATCH_SIZE or keep WHISPERX_COMPUTE_TYPE=int8_float16."
                    ) from exc
                raise
        return self._asr_model

    def transcribe_with_alignment(self, audio_path: Path) -> TranscriptResult:
        if not self.device.strip().lower().startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("ASR stage aborted: WhisperX is GPU-only and CUDA is not active.")

        whisperx = self._load_whisperx()

        try:
            audio = whisperx.load_audio(str(audio_path))
            model = self._load_model(whisperx)
            transcription = model.transcribe(
                audio,
                batch_size=self.settings.whisperx_batch_size,
                language=self.settings.whisperx_language,
            )

            language = transcription.get("language") or self.settings.whisperx_language or "en"
            align_model, align_metadata = whisperx.load_align_model(language_code=language, device=self.device)
            aligned = whisperx.align(
                transcription["segments"],
                align_model,
                align_metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )

            segments = [self._to_segment(segment) for segment in aligned.get("segments", [])]
            full_text = " ".join(segment.text.strip() for segment in segments if segment.text).strip()
            if not full_text:
                full_text = transcription.get("text", "").strip()

            duration = max((segment.end for segment in segments), default=0.0)
            return TranscriptResult(
                text=full_text,
                language=language,
                duration_seconds=duration,
                segments=segments,
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def extract_pronunciation_metrics(
        self, transcript: TranscriptResult, low_confidence_threshold: float
    ) -> PronunciationMetrics:
        total_scored_words = 0
        content_scored_words = 0
        raw_low_confidence_count = 0
        low_confidence_words: list[LowConfidenceWord] = []

        for segment in transcript.segments:
            for word in segment.words:
                if word.confidence is None:
                    continue
                total_scored_words += 1

                if word.confidence < low_confidence_threshold:
                    raw_low_confidence_count += 1

                if _is_content_word(word.word):
                    content_scored_words += 1
                    if word.confidence < low_confidence_threshold:
                        low_confidence_words.append(
                            LowConfidenceWord(
                                word=word.word,
                                start=word.start,
                                end=word.end,
                                confidence=round(word.confidence, 3),
                            )
                        )

        low_confidence_ratio = (
            len(low_confidence_words) / content_scored_words if content_scored_words > 0 else 0.0
        )
        raw_low_confidence_ratio = (
            raw_low_confidence_count / total_scored_words if total_scored_words > 0 else 0.0
        )
        return PronunciationMetrics(
            low_confidence_ratio=round(low_confidence_ratio, 3),
            low_confidence_words=low_confidence_words,
            total_scored_words=total_scored_words,
            low_confidence_threshold=low_confidence_threshold,
            content_scored_words=content_scored_words,
            raw_low_confidence_ratio=round(raw_low_confidence_ratio, 3),
            raw_low_confidence_word_count=raw_low_confidence_count,
        )

    def _to_segment(self, raw_segment: dict[str, Any]) -> SpeechSegment:
        segment_start = float(raw_segment.get("start", 0.0) or 0.0)
        segment_end = float(raw_segment.get("end", segment_start) or segment_start)

        words: list[WordTiming] = []
        for raw_word in raw_segment.get("words", []):
            word_text = str(raw_word.get("word", "")).strip()
            word_start = float(raw_word.get("start", segment_start) or segment_start)
            word_end = float(raw_word.get("end", word_start) or word_start)
            raw_confidence = raw_word.get("score", raw_word.get("confidence"))
            confidence = float(raw_confidence) if raw_confidence is not None else None

            words.append(
                WordTiming(
                    word=word_text,
                    start=word_start,
                    end=word_end,
                    confidence=confidence,
                )
            )

        return SpeechSegment(
            start=segment_start,
            end=segment_end,
            text=str(raw_segment.get("text", "")).strip(),
            words=words,
        )
