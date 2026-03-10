from abc import ABC, abstractmethod
from pathlib import Path

from models.schemas import PronunciationMetrics, TranscriptResult


class ASRProvider(ABC):
    @abstractmethod
    def transcribe_with_alignment(self, audio_path: Path) -> TranscriptResult:
        """Transcribe audio with word-level alignment and confidence."""


class PronunciationProvider(ABC):
    @abstractmethod
    def extract_pronunciation_metrics(
        self, transcript: TranscriptResult, low_confidence_threshold: float
    ) -> PronunciationMetrics:
        """Produce pronunciation/intelligibility metrics from aligned transcript."""


class AzureASRProvider(ASRProvider):
    def transcribe_with_alignment(self, audio_path: Path) -> TranscriptResult:
        raise NotImplementedError("Azure ASR provider is a V2 extension stub.")


class NemoPronunciationProvider(PronunciationProvider):
    def extract_pronunciation_metrics(
        self, transcript: TranscriptResult, low_confidence_threshold: float
    ) -> PronunciationMetrics:
        raise NotImplementedError("NVIDIA NeMo pronunciation provider is a V2 extension stub.")
