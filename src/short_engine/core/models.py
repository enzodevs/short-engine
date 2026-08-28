"""Small validated values shared across domain modules."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AspectRatio(StrEnum):
    """Supported output canvas profiles."""

    VERTICAL = "9:16"
    SQUARE = "1:1"
    LANDSCAPE = "16:9"

    @property
    def ffmpeg_value(self) -> str:
        """Return a division expression accepted by FFmpeg filters."""
        width, height = self.value.split(":", maxsplit=1)
        return f"{width}/{height}"


class TimeRange(BaseModel):
    """A half-open range measured in media seconds."""

    model_config = ConfigDict(frozen=True)

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "TimeRange":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


class RunRequest(BaseModel):
    """Validated user intent at the CLI boundary."""

    model_config = ConfigDict(frozen=True)

    source: Path
    clips: int = Field(default=3, ge=1, le=20)
    aspect: AspectRatio = AspectRatio.VERTICAL
    language: str | None = Field(default=None, min_length=2, max_length=8)
