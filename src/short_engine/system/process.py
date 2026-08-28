"""Safe subprocess execution boundary."""

import subprocess
from dataclasses import dataclass
from typing import Protocol

from short_engine.core.errors import DependencyError


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
        try:
            completed = subprocess.run(args, capture_output=True, check=False, text=True)
        except FileNotFoundError as error:
            raise DependencyError(f"Required command is not installed: {args[0]}") from error
        return CommandResult(
            args=args,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
