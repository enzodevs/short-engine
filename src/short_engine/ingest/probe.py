"""FFprobe adapter."""

import json
from pathlib import Path
from typing import Any

from short_engine.core.errors import MediaError
from short_engine.ingest.models import MediaProbe
from short_engine.system.process import CommandRunner


def _frame_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    numerator, separator, denominator = value.partition("/")
    if separator:
        divisor = float(denominator)
        return float(numerator) / divisor if divisor else None
    return float(value)


class FFprobe:
    def __init__(self, runner: CommandRunner) -> None:
        self.runner = runner

    def inspect(self, path: Path) -> MediaProbe:
        result = self.runner.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
        if result.returncode != 0:
            raise MediaError(f"ffprobe failed for {path}: {result.stderr.strip()}")
        try:
            payload: dict[str, Any] = json.loads(result.stdout)
            streams = payload.get("streams", [])
            video = next(
                (stream for stream in streams if stream.get("codec_type") == "video"), None
            )
            audio = next(
                (stream for stream in streams if stream.get("codec_type") == "audio"), None
            )
            duration = float(payload["format"]["duration"])
            return MediaProbe(
                duration_seconds=duration,
                width=int(video["width"]) if video else None,
                height=int(video["height"]) if video else None,
                frames_per_second=_frame_rate(video.get("avg_frame_rate")) if video else None,
                has_video=video is not None,
                has_audio=audio is not None,
                audio_sample_rate=int(audio["sample_rate"])
                if audio and audio.get("sample_rate")
                else None,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise MediaError(f"ffprobe returned invalid metadata for {path}") from error
