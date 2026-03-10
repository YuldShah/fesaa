from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from models.schemas import AggregateMetrics, TranscriptResult

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")

DOMAIN_KEYS = (
    "fluency_coherence",
    "lexical_resource",
    "grammatical_range_accuracy",
    "pronunciation",
)

DOMAIN_ITEM_KEYS = (
    "strength",
    "strength_evidence",
    "error",
    "error_evidence",
    "rubric_justification",
    "drill",
)

COMPLEX_CONNECTORS = {
    "because",
    "although",
    "though",
    "whereas",
    "however",
    "therefore",
    "meanwhile",
    "instead",
    "despite",
    "unless",
    "while",
    "since",
    "which",
    "that",
    "who",
    "whose",
    "whom",
    "if",
}


def round_half_up(value: float, step: float = 0.5) -> float:
    if step <= 0:
        raise ValueError("step must be greater than zero")

    scaled = value / step
    if scaled >= 0:
        rounded = math.floor(scaled + 0.5)
    else:
        rounded = math.ceil(scaled - 0.5)
    return rounded * step


def _clamp(value: float, lower: float = 0.0, upper: float = 9.0) -> float:
    return max(lower, min(upper, value))


def _normalize_token(raw: str) -> str | None:
    match = TOKEN_PATTERN.search(raw)
    return match.group(0).lower() if match else None


def _collect_tokens(transcript: TranscriptResult) -> list[str]:
    tokens: list[str] = []
    for segment in transcript.segments:
        if segment.words:
            for word in segment.words:
                if not word.word:
                    continue
                token = _normalize_token(word.word)
                if token:
                    tokens.append(token)
        else:
            tokens.extend(token.lower() for token in TOKEN_PATTERN.findall(segment.text))

    if not tokens:
        tokens.extend(token.lower() for token in TOKEN_PATTERN.findall(transcript.text))

    return tokens


def _segment_lengths(transcript: TranscriptResult) -> list[int]:
    lengths: list[int] = []
    for segment in transcript.segments:
        if segment.words:
            count = sum(1 for word in segment.words if _normalize_token(word.word or ""))
        else:
            count = len(TOKEN_PATTERN.findall(segment.text))
        if count > 0:
            lengths.append(count)
    return lengths


def _score_fluency_coherence(metrics: AggregateMetrics, duration_seconds: float) -> float:
    fluency = metrics.fluency
    lexical = metrics.lexical

    duration_minutes = max(duration_seconds / 60.0, 1e-6)
    pause_rate = len(fluency.pauses) / duration_minutes
    long_pause_rate = sum(1 for pause in fluency.pauses if pause.duration >= 1.2) / duration_minutes
    filler_rate = (len(fluency.filler_words) / max(lexical.total_words, 1)) * 100.0

    score = 6.5
    wpm = fluency.words_per_minute
    speech_ratio = fluency.speech_ratio

    if wpm < 75:
        score -= 2.0
    elif wpm < 95:
        score -= 1.2
    elif wpm < 110:
        score -= 0.6
    elif wpm <= 170:
        score += 0.5
    elif wpm <= 205:
        score -= 0.3
    else:
        score -= 0.9

    if speech_ratio < 0.50:
        score -= 1.8
    elif speech_ratio < 0.60:
        score -= 1.0
    elif speech_ratio < 0.68:
        score -= 0.4
    elif speech_ratio <= 0.90:
        score += 0.4
    elif speech_ratio > 0.95:
        score -= 0.4

    if pause_rate > 24:
        score -= 1.1
    elif pause_rate > 18:
        score -= 0.7
    elif pause_rate > 13:
        score -= 0.3
    elif pause_rate < 5:
        score += 0.2

    if long_pause_rate > 5.5:
        score -= 0.9
    elif long_pause_rate > 3.5:
        score -= 0.5
    elif long_pause_rate < 1.2:
        score += 0.1

    if filler_rate > 12:
        score -= 0.8
    elif filler_rate > 8:
        score -= 0.5
    elif filler_rate > 5:
        score -= 0.2
    elif filler_rate < 2.5:
        score += 0.2

    if lexical.total_words < 45:
        score -= 0.3
    elif lexical.total_words > 120 and pause_rate < 10:
        score += 0.2

    return score


def _score_lexical_resource(metrics: AggregateMetrics, tokens: list[str]) -> float:
    lexical = metrics.lexical
    total_words = lexical.total_words
    unique_words = lexical.unique_words
    ttr = lexical.type_token_ratio

    long_word_ratio = sum(1 for token in tokens if len(token) >= 7) / max(len(tokens), 1)
    repetition_ratio = 1.0 - (unique_words / max(total_words, 1))

    score = 5.6

    if total_words < 35:
        score -= 1.1
    elif total_words < 55:
        score -= 0.4
    elif total_words > 110:
        score += 0.2

    if ttr >= 0.62:
        score += 1.4
    elif ttr >= 0.55:
        score += 1.0
    elif ttr >= 0.48:
        score += 0.6
    elif ttr >= 0.41:
        score += 0.2
    elif ttr >= 0.35:
        score -= 0.3
    else:
        score -= 0.8

    if long_word_ratio >= 0.24:
        score += 0.5
    elif long_word_ratio >= 0.18:
        score += 0.3
    elif long_word_ratio < 0.10:
        score -= 0.3

    if repetition_ratio > 0.62:
        score -= 0.8
    elif repetition_ratio > 0.55:
        score -= 0.4
    elif repetition_ratio < 0.42:
        score += 0.2

    if total_words >= 40:
        token_counts = Counter(tokens)
        top_5_density = sum(count for _, count in token_counts.most_common(5)) / total_words
        if top_5_density > 0.46:
            score -= 0.4
        elif top_5_density < 0.30:
            score += 0.2

    return score


def _score_grammatical_range_accuracy(
    metrics: AggregateMetrics,
    transcript: TranscriptResult,
    tokens: list[str],
) -> float:
    total_words = len(tokens)
    if total_words == 0:
        return 0.0

    segment_lengths = _segment_lengths(transcript)
    if not segment_lengths:
        segment_lengths = [total_words]

    average_segment_len = sum(segment_lengths) / len(segment_lengths)
    short_segment_ratio = sum(1 for length in segment_lengths if length <= 2) / len(segment_lengths)
    immediate_repetition_ratio = (
        sum(1 for left, right in zip(tokens, tokens[1:]) if left == right) / total_words
    )

    connector_hits = [token for token in tokens if token in COMPLEX_CONNECTORS]
    connector_ratio = len(connector_hits) / total_words
    connector_variety = len(set(connector_hits))

    score = 5.7

    if total_words < 35:
        score -= 0.8
    elif total_words > 90:
        score += 0.2

    if connector_ratio >= 0.07:
        score += 0.9
    elif connector_ratio >= 0.05:
        score += 0.6
    elif connector_ratio >= 0.03:
        score += 0.3
    elif connector_ratio < 0.015 and total_words > 45:
        score -= 0.6

    if connector_variety >= 6:
        score += 0.3
    elif connector_variety <= 1 and total_words > 40:
        score -= 0.4

    if short_segment_ratio > 0.52:
        score -= 1.0
    elif short_segment_ratio > 0.38:
        score -= 0.6
    elif short_segment_ratio < 0.22:
        score += 0.2

    if immediate_repetition_ratio > 0.05:
        score -= 0.8
    elif immediate_repetition_ratio > 0.03:
        score -= 0.4
    elif immediate_repetition_ratio < 0.015:
        score += 0.1

    if 6 <= average_segment_len <= 18:
        score += 0.2
    elif average_segment_len < 4.5 and total_words > 35:
        score -= 0.5

    if metrics.lexical.type_token_ratio >= 0.56:
        score += 0.2

    return score


def _score_pronunciation(metrics: AggregateMetrics) -> float:
    pronunciation = metrics.pronunciation
    content_ratio = pronunciation.low_confidence_ratio
    raw_ratio = pronunciation.raw_low_confidence_ratio
    content_scored_words = pronunciation.content_scored_words or pronunciation.total_scored_words

    score = 7.2

    if content_ratio <= 0.03:
        score += 0.9
    elif content_ratio <= 0.06:
        score += 0.6
    elif content_ratio <= 0.10:
        score += 0.2
    elif content_ratio <= 0.15:
        score -= 0.3
    elif content_ratio <= 0.22:
        score -= 0.9
    elif content_ratio <= 0.30:
        score -= 1.6
    else:
        score -= 2.4

    if content_scored_words < 25:
        score -= 0.5
    elif content_scored_words > 90 and content_ratio < 0.12:
        score += 0.2

    if raw_ratio - content_ratio > 0.08 and content_ratio < 0.12:
        score += 0.2

    return score


def compute_deterministic_scores(
    transcript: TranscriptResult,
    metrics: AggregateMetrics,
    calibration_scale: float = 1.0,
    calibration_bias: float = 0.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    duration_seconds = max(transcript.duration_seconds, 0.0)
    tokens = _collect_tokens(transcript)

    raw_scores = {
        "fluency_coherence": _score_fluency_coherence(metrics, duration_seconds),
        "lexical_resource": _score_lexical_resource(metrics, tokens),
        "grammatical_range_accuracy": _score_grammatical_range_accuracy(metrics, transcript, tokens),
        "pronunciation": _score_pronunciation(metrics),
    }

    calibrated_scores = {
        key: _clamp((value * calibration_scale) + calibration_bias, 0.0, 9.0)
        for key, value in raw_scores.items()
    }

    rounded_scores = {
        key: round_half_up(value, 0.5)
        for key, value in calibrated_scores.items()
    }

    overall_band = round_half_up(sum(rounded_scores.values()) / len(rounded_scores), 0.5)
    metadata: dict[str, Any] = {
        "scoring_version": "deterministic-v1",
        "overall_band": overall_band,
        "raw_scores": {key: round(value, 3) for key, value in raw_scores.items()},
        "calibrated_scores": {key: round(value, 3) for key, value in calibrated_scores.items()},
        "calibration": {
            "scale": calibration_scale,
            "bias": calibration_bias,
        },
    }
    return rounded_scores, metadata


def _empty_domain_item() -> dict[str, str]:
    return {key: "" for key in DOMAIN_ITEM_KEYS}


def _normalize_domain_feedback(data: Any) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    source = data if isinstance(data, dict) else {}

    for domain in DOMAIN_KEYS:
        item = source.get(domain, {})
        row = _empty_domain_item()
        if isinstance(item, dict):
            for key in DOMAIN_ITEM_KEYS:
                value = item.get(key)
                row[key] = str(value).strip() if value is not None else ""
        normalized[domain] = row
    return normalized


def build_final_evaluation(
    scores: dict[str, float],
    narrative: dict[str, Any],
    scoring_metadata: dict[str, Any],
) -> dict[str, Any]:
    summary = str(narrative.get("summary", "")).strip()
    sample_rewrite_raw = narrative.get("sample_rewrite")
    sample_rewrite = str(sample_rewrite_raw).strip() if sample_rewrite_raw is not None else ""

    return {
        "scores": {domain: float(scores.get(domain, 0.0)) for domain in DOMAIN_KEYS},
        "summary": summary,
        "domain_feedback": _normalize_domain_feedback(narrative.get("domain_feedback")),
        "sample_rewrite": sample_rewrite,
        "scoring_metadata": scoring_metadata,
    }
