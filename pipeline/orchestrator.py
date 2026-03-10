from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from core.settings import Settings
from models.schemas import AggregateMetrics, AssessmentReport
from pipeline.audio import preprocess_audio
from pipeline.cleanup import cleanup_audio_pair
from pipeline.gemini_service import GeminiEvaluator
from pipeline.interfaces import ASRProvider, PronunciationProvider
from pipeline.metrics import calculate_fluency_metrics, calculate_lexical_metrics
from pipeline.scoring import build_final_evaluation, compute_deterministic_scores


class AssessmentOrchestrator:
    def __init__(
        self,
        settings: Settings,
        asr_provider: ASRProvider,
        pronunciation_provider: PronunciationProvider,
        llm_evaluator: GeminiEvaluator,
    ) -> None:
        self.settings = settings
        self.asr_provider = asr_provider
        self.pronunciation_provider = pronunciation_provider
        self.llm_evaluator = llm_evaluator

    def run(self, source_audio_path: Path) -> AssessmentReport:
        source_audio_path = source_audio_path.resolve()
        prepared_audio = preprocess_audio(
            input_path=source_audio_path,
            output_dir=self.settings.upload_dir,
            ffmpeg_bin=self.settings.ffmpeg_path,
        )

        transcript = self.asr_provider.transcribe_with_alignment(prepared_audio)
        fluency_metrics = calculate_fluency_metrics(
            transcript,
            pause_threshold_seconds=self.settings.pause_threshold_seconds,
            audio_path=prepared_audio,
            enable_vad=self.settings.enable_vad_pause_refinement,
            vad_aggressiveness=self.settings.vad_aggressiveness,
            vad_frame_ms=self.settings.vad_frame_ms,
            filler_context_pause_seconds=self.settings.filler_context_pause_seconds,
        )
        lexical_metrics = calculate_lexical_metrics(transcript)
        pronunciation_metrics = self.pronunciation_provider.extract_pronunciation_metrics(
            transcript, self.settings.low_confidence_threshold
        )

        metrics = AggregateMetrics(
            fluency=fluency_metrics,
            lexical=lexical_metrics,
            pronunciation=pronunciation_metrics,
        )
        scores, score_metadata = compute_deterministic_scores(
            transcript,
            metrics,
            calibration_scale=self.settings.scoring_calibration_scale,
            calibration_bias=self.settings.scoring_calibration_bias,
        )
        llm_narrative = self.llm_evaluator.evaluate(transcript, metrics, fixed_scores=scores)
        llm_evaluation = build_final_evaluation(scores, llm_narrative, score_metadata)

        report_id = str(uuid4())
        report_path = self.settings.reports_dir / f"{report_id}.json"
        report = AssessmentReport(
            report_id=report_id,
            created_at=datetime.now(timezone.utc),
            report_path=str(report_path),
            transcript=transcript,
            metrics=metrics,
            llm_evaluation=llm_evaluation,
        )
        self._save_report(report)

        # Immediate cleanup when retention is 0 (delete right after report save).
        if self.settings.audio_retention_hours == 0:
            cleanup_audio_pair(source_audio_path, self.settings.upload_dir)

        return report

    def _save_report(self, report: AssessmentReport) -> None:
        target = Path(report.report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report.model_dump_json(indent=2), encoding="utf-8")
