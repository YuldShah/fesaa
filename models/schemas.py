from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class WordTiming(BaseModel):
    word: str
    start: float
    end: float
    confidence: float | None = None


class SpeechSegment(BaseModel):
    start: float
    end: float
    text: str
    words: list[WordTiming] = Field(default_factory=list)


class TranscriptResult(BaseModel):
    text: str
    language: str | None = None
    duration_seconds: float
    segments: list[SpeechSegment] = Field(default_factory=list)


class Pause(BaseModel):
    start: float
    end: float
    duration: float
    source: str = "alignment"


class FillerWord(BaseModel):
    word: str
    start: float
    end: float


class FluencyMetrics(BaseModel):
    words_per_minute: float
    speech_ratio: float
    pauses: list[Pause] = Field(default_factory=list)
    filler_words: list[FillerWord] = Field(default_factory=list)
    pause_rate_per_minute: float = 0.0
    long_pause_count: int = 0
    filler_rate_per_100_words: float = 0.0
    total_pause_seconds: float = 0.0
    pause_detection_method: str = "alignment"


class LexicalMetrics(BaseModel):
    type_token_ratio: float
    total_words: int
    unique_words: int


class LowConfidenceWord(BaseModel):
    word: str
    start: float
    end: float
    confidence: float


class PronunciationMetrics(BaseModel):
    low_confidence_ratio: float
    low_confidence_words: list[LowConfidenceWord] = Field(default_factory=list)
    total_scored_words: int
    low_confidence_threshold: float
    content_scored_words: int = 0
    raw_low_confidence_ratio: float = 0.0
    raw_low_confidence_word_count: int = 0

    @field_validator("low_confidence_words", mode="before")
    @classmethod
    def _normalize_low_confidence_words(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value

        normalized: list[Any] = []
        for item in value:
            if isinstance(item, str):
                normalized.append(
                    {
                        "word": item,
                        "start": 0.0,
                        "end": 0.0,
                        "confidence": 0.0,
                    }
                )
            else:
                normalized.append(item)
        return normalized


class AggregateMetrics(BaseModel):
    fluency: FluencyMetrics
    lexical: LexicalMetrics
    pronunciation: PronunciationMetrics


class AssessmentReport(BaseModel):
    report_id: str
    created_at: datetime
    report_path: str
    transcript: TranscriptResult
    metrics: AggregateMetrics
    llm_evaluation: dict[str, Any]


class TranscribeReport(BaseModel):
    transcript: TranscriptResult
    metrics: AggregateMetrics


class BatchQuestionPayload(BaseModel):
    question_text: str
    transcript: TranscriptResult
    metrics: AggregateMetrics


class EvaluateBatchRequest(BaseModel):
    batch_id: str
    topic_title: str
    part: int
    questions: list[BatchQuestionPayload]


class SubmitAssessmentResponse(BaseModel):
    task_id: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
