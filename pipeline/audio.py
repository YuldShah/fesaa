import subprocess
from pathlib import Path


def preprocess_audio(input_path: Path, output_dir: Path, ffmpeg_bin: str) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_16k_mono.wav"

    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)

    if completed.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {completed.stderr.strip()}")

    return output_path
