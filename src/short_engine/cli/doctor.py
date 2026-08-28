"""Read-only environment diagnostics."""

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass

from short_engine.core.config import Settings
from short_engine.system.process import resolve_executable


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _command_version(name: str) -> Check:
    preferred = resolve_executable(name)
    executable = preferred if preferred != name else shutil.which(name)
    if executable is None:
        return Check(name, False, "not found on PATH")
    result = subprocess.run(
        [executable, "-version"],
        capture_output=True,
        check=False,
        text=True,
    )
    first_line = (result.stdout or result.stderr).splitlines()[0]
    return Check(name, result.returncode == 0, first_line)


def _caption_filter_check() -> Check:
    preferred = resolve_executable("ffmpeg")
    executable = preferred if preferred != "ffmpeg" else shutil.which("ffmpeg")
    if executable is None:
        return Check("FFmpeg captions", False, "ffmpeg not found")
    result = subprocess.run(
        [executable, "-hide_banner", "-filters"],
        capture_output=True,
        check=False,
        text=True,
    )
    has_ass = any(line.split()[1:2] == ["ass"] for line in result.stdout.splitlines())
    detail = "libass filter available" if has_ass else "install brew formula ffmpeg-full"
    return Check("FFmpeg captions", has_ass, detail)


def collect_checks(settings: Settings) -> list[Check]:
    """Collect diagnostics without downloading models or changing the machine."""
    machine = platform.machine()
    disk = shutil.disk_usage(settings.output_root.parent.resolve())
    return [
        Check("Python", sys.version_info >= (3, 12), platform.python_version()),
        Check("Apple Silicon", sys.platform == "darwin" and machine == "arm64", machine),
        _command_version("ffmpeg"),
        _caption_filter_check(),
        _command_version("ffprobe"),
        Check("MLX", importlib.util.find_spec("mlx") is not None, "Python package"),
        Check(
            "Gemini API key",
            settings.has_gemini_key,
            "configured" if settings.has_gemini_key else "missing",
        ),
        Check("Free disk", disk.free >= 5 * 1024**3, f"{disk.free / 1024**3:.1f} GiB"),
    ]
