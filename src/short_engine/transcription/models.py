"""Canonical timed transcript models."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ASRConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    language: str | None = None
    word_timestamps: bool = True


class TimedWord(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_order(self) -> "TimedWord":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("word end must be greater than start")
        return self


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str = Field(min_length=1)
    words: list[TimedWord] = Field(default_factory=list)
    speaker_id: str | None = None

    @model_validator(mode="after")
    def validate_order(self) -> "TranscriptSegment":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("segment end must be greater than start")
        return self


class Transcript(BaseModel):
    model_config = ConfigDict(frozen=True)

    language: str
    model: str
    duration_seconds: float = Field(gt=0)
    segments: list[TranscriptSegment] = Field(min_length=1)
