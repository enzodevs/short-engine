"""Typed ingest boundary values."""

from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field


class SourceRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str = Field(min_length=1)
    cookies_from_browser: str | None = None
    download_height: int = Field(default=720, ge=144, le=2160)

    @property
    def is_url(self) -> bool:
        return urlparse(self.source).scheme in {"http", "https"}


class SourceAsset(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    source_fingerprint: str
    original_source: str
    downloaded: bool


class MediaProbe(BaseModel):
    model_config = ConfigDict(frozen=True)

    duration_seconds: float = Field(gt=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    frames_per_second: float | None = Field(default=None, gt=0)
    has_video: bool
    has_audio: bool
    audio_sample_rate: int | None = Field(default=None, gt=0)
