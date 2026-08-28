"""Normalized media boundary models."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class BoundaryKind(StrEnum):
    MEDIA = "media"
    SCENE = "scene"
    SPEECH = "speech"
    SENTENCE = "sentence"
    SPEAKER = "speaker"


class DetectedBoundary(BaseModel):
    model_config = ConfigDict(frozen=True)

    at_seconds: float = Field(ge=0)
    kind: BoundaryKind
    confidence: float | None = Field(default=None, ge=0, le=1)


class TimelineBoundary(BaseModel):
    model_config = ConfigDict(frozen=True)

    at_seconds: float = Field(ge=0)
    kinds: set[BoundaryKind]


class Timeline(BaseModel):
    model_config = ConfigDict(frozen=True)

    duration_seconds: float = Field(gt=0)
    boundaries: list[TimelineBoundary]
