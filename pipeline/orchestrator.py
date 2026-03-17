from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from core.settings import Settings
from models.schemas import (
    AggregateMetrics,
    AssessmentReport,
    TranscribeReport,
    EvaluateBatchRequest,
    TranscriptResult,
    SpeechSegment,
    WordTiming,
    FluencyMetrics,
    LexicalMetrics,
    PronunciationMetrics,
)
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
        # Backward compatibility for the single-file pipeline
        report = self.run_transcribe(source_audio_path)
        scores, score_metadata = compute_deterministic_scores(
            report.transcript,
            report.metrics,
            calibration_scale=self.settings.scoring_calibration_scale,
            calibration_bias=self.settings.scoring_calibration_bias,
        )
        llm_narrative = self.llm_evaluator.evaluate(report.transcript, report.metrics, fixed_scores=scores)
        llm_evaluation = build_final_evaluation(scores, llm_narrative, score_metadata)

        report_id = str(uuid4())
        report_path = self.settings.reports_dir / f"{report_id}.json"
        full_report = AssessmentReport(
            report_id=report_id,
            created_at=datetime.now(timezone.utc),
            report_path=str(report_path),
            transcript=report.transcript,
            metrics=report.metrics,
            llm_evaluation=llm_evaluation,
        )
        self._save_report(full_report)
        return full_report

    def run_transcribe(self, source_audio_path: Path) -> TranscribeReport:
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

        if self.settings.audio_retention_hours == 0:
            cleanup_audio_pair(source_audio_path, self.settings.upload_dir)

        return TranscribeReport(
            transcript=transcript,
            metrics=AggregateMetrics(
                fluency=fluency_metrics,
                lexical=lexical_metrics,
                pronunciation=pronunciation_metrics,
            )
        )

    def run_evaluate_batch(self, request: EvaluateBatchRequest) -> AssessmentReport:
        texts = []
        total_duration = 0.0
        total_words = 0
        total_pauses = []
        total_fillers = []
        total_low_conf = []
        total_scored = 0
        total_content_scored = 0
        all_segments = []

        for q in request.questions:
            t = q.transcript
            m = q.metrics

            texts.append(f"Q: {q.question_text}\nA: {t.text}")
            
            # Shift segments by total_duration so they line up sequentially
            for seg in t.segments:
                new_words = []
                for w in seg.words:
                    new_words.append(WordTiming(
                        word=w.word,
                        start=w.start + total_duration,
                        end=w.end + total_duration,
                        confidence=w.confidence
                    ))
                all_segments.append(SpeechSegment(
                    start=seg.start + total_duration,
                    end=seg.end + total_duration,
                    text=seg.text,
                    words=new_words
                ))
            
            # Shift pauses and fillers too!
            import copy
            pauses_shifted = copy.deepcopy(m.fluency.pauses)
            for p in pauses_shifted:
                p.start += total_duration
                p.end += total_duration
            total_pauses.extend(pauses_shifted)
            
            fillers_shifted = copy.deepcopy(m.fluency.filler_words)
            for f in fillers_shifted:
                f.start += total_duration
                f.end += total_duration
            total_fillers.extend(fillers_shifted)

            low_conf_shifted = copy.deepcopy(m.pronunciation.low_confidence_words)
            for lcw in low_conf_shifted:
                lcw.start += total_duration
                lcw.end += total_duration
            total_low_conf.extend(low_conf_shifted)

            total_duration += t.duration_seconds
            total_words += m.lexical.total_words
            total_scored += m.pronunciation.total_scored_words
            total_content_scored += getattr(m.pronunciation, "content_scored_words", 0)

        language = request.questions[0].transcript.language if request.questions else "en"
        threshold = request.questions[0].metrics.pronunciation.low_confidence_threshold if request.questions else 0.6

        agg_transcript = TranscriptResult(
            text="\n\n".join(texts),
            language=language,
            duration_seconds=total_duration,
            segments=all_segments
        )

        agg_lexical = calculate_lexical_metrics(agg_transcript)

        wpm = (total_words / (total_duration / 60)) if total_duration > 0 else 0
        total_pause_secs = sum(p.duration for p in total_pauses)
        speech_ratio = ((total_duration - total_pause_secs) / total_duration) if total_duration > 0 else 1.0
        pause_rpm = (len(total_pauses) / (total_duration / 60)) if total_duration > 0 else 0
        long_pauses = sum(1 for p in total_pauses if p.duration > 1.0)
        filler_rate = (len(total_fillers) / total_words * 100) if total_words > 0 else 0

        agg_fluency = FluencyMetrics(
            words_per_minute=wpm,
            speech_ratio=speech_ratio,
            pauses=total_pauses,
            filler_words=total_fillers,
            pause_rate_per_minute=pause_rpm,
            long_pause_count=long_pauses,
            filler_rate_per_100_words=filler_rate,
            total_pause_seconds=total_pause_secs,
            pause_detection_method="alignment",
        )

        low_conf_ratio = (len(total_low_conf) / total_scored) if total_scored > 0 else 0
        agg_pronunciation = PronunciationMetrics(
            low_confidence_ratio=low_conf_ratio,
            low_confidence_words=total_low_conf,
            total_scored_words=total_scored,
            low_confidence_threshold=threshold,
            content_scored_words=total_content_scored,
            raw_low_confidence_ratio=low_conf_ratio,
            raw_low_confidence_word_count=len(total_low_conf)
        )

        agg_metrics = AggregateMetrics(
            fluency=agg_fluency,
            lexical=agg_lexical,
            pronunciation=agg_pronunciation
        )

        scores, score_metadata = compute_deterministic_scores(
            agg_transcript,
            agg_metrics,
            calibration_scale=self.settings.scoring_calibration_scale,
            calibration_bias=self.settings.scoring_calibration_bias,
        )

        llm_narrative = self.llm_evaluator.evaluate(agg_transcript, agg_metrics, fixed_scores=scores)
        llm_evaluation = build_final_evaluation(scores, llm_narrative, score_metadata)

        report_id = request.batch_id
        report_path = self.settings.reports_dir / f"{report_id}.json"
        report = AssessmentReport(
            report_id=report_id,
            created_at=datetime.now(timezone.utc),
            report_path=str(report_path),
            transcript=agg_transcript,
            metrics=agg_metrics,
            llm_evaluation=llm_evaluation,
        )
        self._save_report(report)
        return report

    def _save_report(self, report: AssessmentReport) -> None:
        target = Path(report.report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report.model_dump_json(indent=2), encoding="utf-8")
