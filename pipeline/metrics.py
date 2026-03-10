from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path

from models.schemas import FillerWord, FluencyMetrics, LexicalMetrics, Pause, TranscriptResult, WordTiming

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")

HESITATION_FILLERS = {
    "uh",
    "um",
    "uhm",
    "er",
    "ah",
    "mm",
    "hmm",
}

CONTEXT_FILLERS = {
    "like",
    "so",
    "well",
    "right",
    "okay",
    "ok",
    "actually",
    "basically",
}

PHRASE_FILLERS_STRONG = {
    ("you", "know"),
    ("i", "mean"),
}

PHRASE_FILLERS_CONTEXT = {
    ("kind", "of"),
    ("sort", "of"),
}


@dataclass(frozen=True)
class _WordEvent:
    word: str
    token: str
    start: float
    end: float
    confidence: float | None


def _normalize_token(raw: str) -> str | None:
    match = TOKEN_PATTERN.search(raw)
    return match.group(0) if match else None


def _collect_words(transcript: TranscriptResult) -> list[str]:
    words: list[str] = []
    for segment in transcript.segments:
        if segment.words:
            for timing in segment.words:
                token = _normalize_token(timing.word) if timing.word else None
                if token:
                    words.append(token)
        else:
            words.extend(TOKEN_PATTERN.findall(segment.text))

    if not words:
        words.extend(TOKEN_PATTERN.findall(transcript.text))

    return [word.lower() for word in words]


def _collect_word_timings(transcript: TranscriptResult) -> list[WordTiming]:
    word_timings: list[WordTiming] = []
    for segment in transcript.segments:
        for word in segment.words:
            if not _normalize_token(word.word):
                continue
            word_timings.append(word)
    return sorted(word_timings, key=lambda item: item.start)


def _collect_word_events(transcript: TranscriptResult) -> list[_WordEvent]:
    events: list[_WordEvent] = []
    for segment in sorted(transcript.segments, key=lambda item: item.start):
        for word in sorted(segment.words, key=lambda item: item.start):
            token = _normalize_token(word.word or "")
            if not token:
                continue
            events.append(
                _WordEvent(
                    word=word.word.strip(),
                    token=token.lower(),
                    start=word.start,
                    end=word.end,
                    confidence=word.confidence,
                )
            )
    return events


def _is_hesitation_filler(token: str) -> bool:
    if token in HESITATION_FILLERS:
        return True
    return re.match(r"^(u+h+|u+m+|a+h+|e+r+|h+m+|m+h+)$", token) is not None


def _merge_pauses(pauses: list[Pause]) -> list[Pause]:
    if not pauses:
        return []

    ordered = sorted(pauses, key=lambda pause: pause.start)
    merged: list[Pause] = [ordered[0]]

    for pause in ordered[1:]:
        previous = merged[-1]
        if pause.start <= previous.end + 0.02:
            new_start = min(previous.start, pause.start)
            new_end = max(previous.end, pause.end)
            source = previous.source if previous.source == pause.source else "merged"
            merged[-1] = Pause(start=new_start, end=new_end, duration=round(new_end - new_start, 3), source=source)
        else:
            merged.append(pause)

    return merged


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float, tolerance: float = 0.0) -> bool:
    return (a_start <= (b_end + tolerance)) and (b_start <= (a_end + tolerance))


def _alignment_pauses(transcript: TranscriptResult, pause_threshold_seconds: float) -> list[Pause]:
    word_timings = _collect_word_timings(transcript)
    pauses: list[Pause] = []

    if len(word_timings) > 1:
        for previous, current in zip(word_timings, word_timings[1:]):
            gap = max(0.0, current.start - previous.end)
            if gap >= pause_threshold_seconds:
                pauses.append(
                    Pause(
                        start=previous.end,
                        end=current.start,
                        duration=round(gap, 3),
                        source="alignment",
                    )
                )
        return _merge_pauses(pauses)

    ordered_segments = sorted(transcript.segments, key=lambda item: item.start)
    for previous, current in zip(ordered_segments, ordered_segments[1:]):
        gap = max(0.0, current.start - previous.end)
        if gap >= pause_threshold_seconds:
            pauses.append(
                Pause(
                    start=previous.end,
                    end=current.start,
                    duration=round(gap, 3),
                    source="alignment",
                )
            )

    return _merge_pauses(pauses)


def _bridge_short_gaps(flags: list[bool], max_gap_frames: int) -> list[bool]:
    if max_gap_frames <= 0 or not flags:
        return flags

    bridged = flags[:]
    index = 0
    while index < len(bridged):
        if bridged[index]:
            index += 1
            continue

        gap_start = index
        while index < len(bridged) and not bridged[index]:
            index += 1
        gap_end = index

        gap_length = gap_end - gap_start
        has_left_speech = gap_start > 0 and bridged[gap_start - 1]
        has_right_speech = gap_end < len(bridged) and bridged[gap_end]
        if has_left_speech and has_right_speech and gap_length <= max_gap_frames:
            for pointer in range(gap_start, gap_end):
                bridged[pointer] = True

    return bridged


def _remove_short_speech_runs(flags: list[bool], min_run_frames: int) -> list[bool]:
    if min_run_frames <= 1 or not flags:
        return flags

    filtered = flags[:]
    index = 0
    while index < len(filtered):
        if not filtered[index]:
            index += 1
            continue

        run_start = index
        while index < len(filtered) and filtered[index]:
            index += 1
        run_end = index

        if (run_end - run_start) < min_run_frames:
            for pointer in range(run_start, run_end):
                filtered[pointer] = False

    return filtered


def _detect_vad_pauses(
    audio_path: Path,
    pause_threshold_seconds: float,
    vad_aggressiveness: int,
    vad_frame_ms: int,
) -> tuple[list[Pause], float | None, float | None]:
    try:
        import importlib

        webrtcvad = importlib.import_module("webrtcvad")
    except ImportError:
        return [], None, None

    if vad_frame_ms not in (10, 20, 30):
        return [], None, None

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            pcm_data = wav_file.readframes(frame_count)
    except (FileNotFoundError, wave.Error, OSError):
        return [], None, None

    if channels != 1 or sample_width != 2 or sample_rate not in (8000, 16000, 32000, 48000):
        return [], None, None

    audio_duration_seconds = frame_count / sample_rate if sample_rate > 0 else None
    bytes_per_frame = int(sample_rate * (vad_frame_ms / 1000.0) * sample_width)
    if bytes_per_frame <= 0:
        return [], audio_duration_seconds, None

    vad = webrtcvad.Vad(max(0, min(3, vad_aggressiveness)))
    speech_flags: list[bool] = []
    for start in range(0, len(pcm_data), bytes_per_frame):
        frame = pcm_data[start:start + bytes_per_frame]
        if len(frame) < bytes_per_frame:
            break
        try:
            speech_flags.append(vad.is_speech(frame, sample_rate))
        except ValueError:
            continue

    if not speech_flags:
        return [], audio_duration_seconds, None

    frame_seconds = vad_frame_ms / 1000.0
    max_gap_frames = max(1, int(round(0.18 / frame_seconds)))
    min_run_frames = max(1, int(round(0.09 / frame_seconds)))
    smoothed_flags = _bridge_short_gaps(speech_flags, max_gap_frames=max_gap_frames)
    smoothed_flags = _remove_short_speech_runs(smoothed_flags, min_run_frames=min_run_frames)

    speech_ratio = sum(1 for flag in smoothed_flags if flag) / len(smoothed_flags)
    speech_intervals: list[tuple[float, float]] = []
    index = 0
    while index < len(smoothed_flags):
        if not smoothed_flags[index]:
            index += 1
            continue

        start_index = index
        while index < len(smoothed_flags) and smoothed_flags[index]:
            index += 1
        end_index = index

        interval_start = start_index * frame_seconds
        interval_end = end_index * frame_seconds
        speech_intervals.append((interval_start, interval_end))

    pauses: list[Pause] = []
    for previous, current in zip(speech_intervals, speech_intervals[1:]):
        pause_start = previous[1]
        pause_end = current[0]
        duration = max(0.0, pause_end - pause_start)
        if duration >= pause_threshold_seconds:
            pauses.append(
                Pause(
                    start=round(pause_start, 3),
                    end=round(pause_end, 3),
                    duration=round(duration, 3),
                    source="vad",
                )
            )

    return _merge_pauses(pauses), audio_duration_seconds, speech_ratio


def _combine_pauses(
    alignment_pauses: list[Pause],
    vad_pauses: list[Pause],
    pause_threshold_seconds: float,
) -> tuple[list[Pause], str]:
    if not vad_pauses:
        return _merge_pauses(alignment_pauses), "alignment"

    selected: list[Pause] = []
    for pause in alignment_pauses:
        overlaps_vad = any(_overlaps(pause.start, pause.end, vad.start, vad.end, tolerance=0.06) for vad in vad_pauses)
        if overlaps_vad or pause.duration >= max(1.2, pause_threshold_seconds * 2.0):
            selected.append(Pause(start=pause.start, end=pause.end, duration=pause.duration, source="merged"))

    for vad_pause in vad_pauses:
        overlaps_existing = any(
            _overlaps(vad_pause.start, vad_pause.end, item.start, item.end, tolerance=0.08)
            for item in selected
        )
        if not overlaps_existing:
            selected.append(vad_pause)

    return _merge_pauses(selected), "vad+alignment"


def _detect_filler_words(events: list[_WordEvent], context_pause_seconds: float) -> list[FillerWord]:
    if not events:
        return []

    previous_gaps = [0.0 for _ in events]
    next_gaps = [0.0 for _ in events]
    for index, event in enumerate(events):
        if index > 0:
            previous_gaps[index] = max(0.0, event.start - events[index - 1].end)
        if index < len(events) - 1:
            next_gaps[index] = max(0.0, events[index + 1].start - event.end)

    fillers: list[FillerWord] = []
    seen: set[tuple[str, float, float]] = set()

    def add_filler(word: str, start: float, end: float) -> None:
        key = (word.lower(), round(start, 3), round(end, 3))
        if key in seen:
            return
        seen.add(key)
        fillers.append(FillerWord(word=word, start=start, end=end))

    for index, event in enumerate(events):
        token = event.token
        has_pause_context = (
            previous_gaps[index] >= context_pause_seconds
            or next_gaps[index] >= context_pause_seconds
        )
        repeated_recently = (
            index > 0
            and events[index - 1].token == token
            and (event.start - events[index - 1].start) <= 2.0
        )
        low_confidence = event.confidence is not None and event.confidence < 0.55
        utterance_boundary = index == 0 or previous_gaps[index] >= 0.45

        if _is_hesitation_filler(token):
            add_filler(event.word, event.start, event.end)
            continue

        if token in CONTEXT_FILLERS and (has_pause_context or repeated_recently or low_confidence or utterance_boundary):
            add_filler(event.word, event.start, event.end)

    for index in range(len(events) - 1):
        left = events[index]
        right = events[index + 1]
        pair = (left.token, right.token)
        pair_gap = max(0.0, right.start - left.end)
        if pair_gap > 0.25:
            continue

        surrounding_pause = (
            (index > 0 and previous_gaps[index] >= context_pause_seconds)
            or (index + 2 < len(events) and next_gaps[index + 1] >= context_pause_seconds)
        )
        low_confidence_pair = (
            (left.confidence is not None and left.confidence < 0.55)
            or (right.confidence is not None and right.confidence < 0.55)
        )
        repeated_pair = (
            index > 1
            and (events[index - 2].token, events[index - 1].token) == pair
            and (left.start - events[index - 2].start) <= 3.0
        )

        should_mark = False
        if pair in PHRASE_FILLERS_STRONG:
            should_mark = surrounding_pause or low_confidence_pair or repeated_pair
        elif pair in PHRASE_FILLERS_CONTEXT:
            should_mark = surrounding_pause

        if should_mark:
            combined = f"{left.word} {right.word}".strip()
            add_filler(combined, left.start, right.end)

    fillers.sort(key=lambda item: item.start)
    return fillers


def calculate_fluency_metrics(
    transcript: TranscriptResult,
    pause_threshold_seconds: float,
    audio_path: Path | None = None,
    enable_vad: bool = False,
    vad_aggressiveness: int = 2,
    vad_frame_ms: int = 30,
    filler_context_pause_seconds: float = 0.15,
) -> FluencyMetrics:
    words = _collect_words(transcript)
    alignment_pauses = _alignment_pauses(transcript, pause_threshold_seconds)

    vad_pauses: list[Pause] = []
    vad_duration: float | None = None
    vad_speech_ratio: float | None = None
    if enable_vad and audio_path is not None:
        vad_pauses, vad_duration, vad_speech_ratio = _detect_vad_pauses(
            audio_path=audio_path,
            pause_threshold_seconds=pause_threshold_seconds,
            vad_aggressiveness=vad_aggressiveness,
            vad_frame_ms=vad_frame_ms,
        )

    pauses, pause_detection_method = _combine_pauses(
        alignment_pauses=alignment_pauses,
        vad_pauses=vad_pauses,
        pause_threshold_seconds=pause_threshold_seconds,
    )

    duration = max(transcript.duration_seconds, vad_duration or 0.0, 0.0)
    words_per_minute = (len(words) / duration * 60.0) if duration > 0 else 0.0

    if vad_speech_ratio is not None:
        speech_ratio = vad_speech_ratio
    else:
        speech_seconds = sum(max(segment.end - segment.start, 0.0) for segment in transcript.segments)
        speech_ratio = (speech_seconds / duration) if duration > 0 else 0.0

    filler_words = _detect_filler_words(_collect_word_events(transcript), filler_context_pause_seconds)

    duration_minutes = duration / 60.0 if duration > 0 else 0.0
    pause_rate_per_minute = (len(pauses) / duration_minutes) if duration_minutes > 0 else 0.0
    long_pause_count = sum(1 for pause in pauses if pause.duration >= 1.2)
    filler_rate_per_100_words = (len(filler_words) / max(len(words), 1)) * 100.0
    total_pause_seconds = sum(pause.duration for pause in pauses)

    return FluencyMetrics(
        words_per_minute=round(words_per_minute, 2),
        speech_ratio=round(speech_ratio, 3),
        pauses=pauses,
        filler_words=filler_words,
        pause_rate_per_minute=round(pause_rate_per_minute, 2),
        long_pause_count=long_pause_count,
        filler_rate_per_100_words=round(filler_rate_per_100_words, 2),
        total_pause_seconds=round(total_pause_seconds, 3),
        pause_detection_method=pause_detection_method,
    )


def calculate_lexical_metrics(transcript: TranscriptResult) -> LexicalMetrics:
    words = _collect_words(transcript)
    total_words = len(words)
    unique_words = len(set(words))
    type_token_ratio = (unique_words / total_words) if total_words else 0.0
    return LexicalMetrics(
        type_token_ratio=round(type_token_ratio, 3),
        total_words=total_words,
        unique_words=unique_words,
    )
