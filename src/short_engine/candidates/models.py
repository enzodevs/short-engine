"""Candidate value objects."""

from pydantic import BaseModel, ConfigDict, Field

from short_engine.core.models import TimeRange


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    time_range: TimeRange
    transcript: str = Field(min_length=1)
    segment_indexes: list[int]
