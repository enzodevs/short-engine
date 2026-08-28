"""FFmpeg frame evidence sampler."""

from pathlib import Path

from short_engine.candidates.models import Candidate
from short_engine.core.errors import MediaError
from short_engine.system.process import CommandRunner, SubprocessRunner


class FrameSampler:
    def __init__(self, runner: CommandRunner | None = None, frames_per_candidate: int = 4) -> None:
        self.runner = runner or SubprocessRunner()
        self.frames_per_candidate = frames_per_candidate

    def sample(self, source: Path, candidate: Candidate, directory: Path) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for index in range(self.frames_per_candidate):
            fraction = (index + 1) / (self.frames_per_candidate + 1)
            timestamp = (
                candidate.time_range.start_seconds
                + candidate.time_range.duration_seconds * fraction
            )
            path = directory / f"{candidate.id}-{index:02d}.jpg"
            result = self.runner.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    str(source),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=640:-2",
                    str(path),
                ]
            )
            if result.returncode != 0 or not path.is_file():
                raise MediaError(f"Could not sample frame at {timestamp:.3f}s")
            paths.append(path)
        return paths
