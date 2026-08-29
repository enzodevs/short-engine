"""Camera motion curves expressed for FFmpeg's evaluator."""

from typing import Protocol


class MotionCurve(Protocol):
    def ffmpeg(self, progress: str) -> str: ...


class SmootherstepCurve:
    """Quintic ease with zero velocity and acceleration at both endpoints."""

    def ffmpeg(self, progress: str) -> str:
        value = f"({progress})"
        return f"{value}*{value}*{value}*(10-15*{value}+6*{value}*{value})"
