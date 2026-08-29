"""Near-live highlight state contracts."""

from enum import StrEnum

from pydantic import BaseModel, Field

from short_engine.core.models import TimeRange


class LiveHighlightState(StrEnum):
    WATCHING = "watching"
    HOOK_DETECTED = "hook_detected"
    ESCALATING = "escalating"
    AWAITING_PAYOFF = "awaiting_payoff"
    READY = "ready"
    EXPIRED = "expired"


class LiveSignal(BaseModel):
    at_seconds: float = Field(ge=0)
    speech_energy: float = Field(ge=0, le=1)
    visual_novelty: float = Field(ge=0, le=1)
    semantic_stakes: float = Field(ge=0, le=1)


class LiveHighlight(BaseModel):
    state: LiveHighlightState
    source: TimeRange | None = None
    confidence: float = Field(default=0, ge=0, le=1)
