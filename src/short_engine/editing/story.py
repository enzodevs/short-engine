"""Narrative composition contracts for short-form editing."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from short_engine.core.models import TimeRange


class ContentGenre(StrEnum):
    PODCAST = "podcast"
    INTERVIEW = "interview"
    TUTORIAL = "tutorial"
    TECHNICAL = "technical"
    GAMEPLAY = "gameplay"
    REACTION = "reaction"
    VLOG = "vlog"
    STORY = "story"


class BeatRole(StrEnum):
    HOOK = "hook"
    PREMISE = "premise"
    ESCALATION = "escalation"
    EVIDENCE = "evidence"
    PAYOFF = "payoff"


class StoryBeat(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: BeatRole
    source: TimeRange
    rationale: str = Field(min_length=2, max_length=240)


class RetentionPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_start_seconds: float = Field(ge=0)
    output_end_seconds: float = Field(gt=0)
    attention_reason: str = Field(min_length=2, max_length=240)
    drop_off_risk: int = Field(ge=0, le=100)
    edit_action: str = Field(min_length=2, max_length=240)


class StoryVariant(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=2, max_length=80)
    title: str = Field(min_length=2, max_length=120)
    genre: ContentGenre
    strategy: str = Field(min_length=2, max_length=240)
    hook_text: str = Field(min_length=2, max_length=240)
    beats: list[StoryBeat] = Field(min_length=2, max_length=7)
    retention_map: list[RetentionPoint] = Field(min_length=1, max_length=12)
    predicted_retention_score: int = Field(ge=0, le=100)
    fatal_flaw: str = Field(min_length=2, max_length=240)

    @computed_field
    @property
    def duration_seconds(self) -> float:
        return sum(beat.source.duration_seconds for beat in self.beats)

    @model_validator(mode="after")
    def validate_arc(self) -> "StoryVariant":
        if self.beats[0].role is not BeatRole.HOOK:
            raise ValueError("first beat must be hook")
        if self.beats[-1].role is not BeatRole.PAYOFF:
            raise ValueError("last beat must be payoff")
        if not 15 <= self.duration_seconds <= 60:
            raise ValueError("story duration must be 15-60 seconds")
        return self


class StoryPackage(BaseModel):
    variants: list[StoryVariant] = Field(min_length=3, max_length=20)
    selected_variant_ids: list[str] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_selection(self) -> "StoryPackage":
        identifiers = {variant.id for variant in self.variants}
        if not set(self.selected_variant_ids) <= identifiers:
            raise ValueError("selected variant is missing")
        return self
