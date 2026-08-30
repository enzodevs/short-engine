"""Reframing domain values."""

from pydantic import BaseModel, ConfigDict, Field


class SubjectObservation(BaseModel):
    model_config = ConfigDict(frozen=True)
    time_seconds: float = Field(ge=0)
    center_x: float
    center_y: float
    confidence: float = Field(ge=0, le=1)
    scene_id: int = Field(default=0, ge=0)
    left_x: float | None = None
    top_y: float | None = None
    right_x: float | None = None
    bottom_y: float | None = None


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
    hard_cuts_seconds: list[float] = Field(default_factory=list)


class PanelCrop(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ReactionLayout(BaseModel):
    model_config = ConfigDict(frozen=True)

    facecam: PanelCrop
    content: PanelCrop
    facecam_panel_height: int = Field(default=640, gt=0)
