"""Audio artifact cleanup utilities.

Provides two modes:
- Immediate cleanup of a specific audio file + its preprocessed derivative.
- Scheduled sweep of stale uploads older than the retention window.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from core.settings import Settings

logger = logging.getLogger(__name__)


def cleanup_audio_pair(source_path: Path, upload_dir: Path) -> None:
    """Delete the original upload and its ``_16k_mono.wav`` derivative.

    Safe to call even if files have already been removed.
    """
    preprocessed = upload_dir / f"{source_path.stem}_16k_mono.wav"
    for p in (source_path, preprocessed):
        try:
            if p.exists():
                p.unlink()
                logger.info("Deleted audio artifact: %s", p)
        except OSError as exc:
            logger.warning("Failed to delete %s: %s", p, exc)


def sweep_stale_uploads(settings: Settings) -> int:
    """Remove upload-dir files older than ``audio_retention_hours``.

    Returns the number of files deleted.  Skips sweep when retention is
    set to ``-1`` (keep forever).
    """
    if settings.audio_retention_hours < 0:
        return 0

    upload_dir = settings.upload_dir
    if not upload_dir.is_dir():
        return 0

    cutoff = time.time() - settings.audio_retention_hours * 3600
    deleted = 0

    for path in upload_dir.iterdir():
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                logger.info("Swept stale upload: %s", path)
                deleted += 1
        except OSError as exc:
            logger.warning("Failed to sweep %s: %s", path, exc)

    return deleted
