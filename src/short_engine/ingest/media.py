"""FFmpeg-derived analysis assets."""

from pathlib import Path

from short_engine.core.errors import MediaError
from short_engine.system.process import CommandRunner


class FFmpegMediaService:
    def __init__(self, runner: CommandRunner) -> None:
        self.runner = runner

    def extract_analysis_audio(self, source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = self.runner.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(destination),
            ]
        )
        return self._validated_output(result.returncode, result.stderr, destination, "audio")

    def create_proxy(self, source: Path, destination: Path, *, max_width: int = 640) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = self.runner.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-vf",
                f"scale=min({max_width}\\,iw):-2",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-pix_fmt",
                "yuv420p",
                str(destination),
            ]
        )
        return self._validated_output(result.returncode, result.stderr, destination, "proxy")

    @staticmethod
    def _validated_output(
        returncode: int,
        stderr: str,
        destination: Path,
        kind: str,
    ) -> Path:
        if returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
            raise MediaError(f"FFmpeg failed to create {kind}: {stderr.strip()}")
        return destination.resolve()
