"""Safe subprocess execution boundary."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from short_engine.core.errors import DependencyError


def resolve_executable(name: str) -> str:
    """Prefer Homebrew's feature-complete, keg-only FFmpeg on Apple Silicon."""
    candidate = Path("/opt/homebrew/opt/ffmpeg-full/bin") / name
    if name in {"ffmpeg", "ffprobe"} and candidate.is_file():
        return str(candidate)
    return name


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    def run(self, args: list[str]) -> CommandResult: ...


class SubprocessRunner:
    """Execute an argument vector without involving a shell."""

    def run(self, args: list[str]) -> CommandResult:
        resolved_args = [resolve_executable(args[0]), *args[1:]]
        try:
            completed = subprocess.run(resolved_args, capture_output=True, check=False, text=True)
        except FileNotFoundError as error:
            raise DependencyError(f"Required command is not installed: {args[0]}") from error
        return CommandResult(
            args=resolved_args,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
