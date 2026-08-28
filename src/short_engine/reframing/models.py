"""Reframing domain values."""

from pydantic import BaseModel, ConfigDict, Field


class SubjectObservation(BaseModel):
    model_config = ConfigDict(frozen=True)
    time_seconds: float = Field(ge=0)
    center_x: float
    center_y: float
    confidence: float = Field(ge=0, le=1)


class SubjectTrack(BaseModel):
    observations: list[SubjectObservation]


class CropSample(BaseModel):
    time_seconds: float
    x: float
    y: float


class CropPlan(BaseModel):
    crop_width: int
    crop_height: int
    samples: list[CropSample]
    used_fallback: bool
