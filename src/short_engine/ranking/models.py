"""Typed ranking contracts."""

from pydantic import BaseModel, ConfigDict, Field, computed_field


class RankingScores(BaseModel):
    model_config = ConfigDict(frozen=True)

    hook_immediacy: int = Field(ge=0, le=100)
    curiosity_gap: int = Field(ge=0, le=100)
    narrative_arc: int = Field(ge=0, le=100)
    payoff_strength: int = Field(ge=0, le=100)
    emotional_intensity: int = Field(ge=0, le=100)
    visual_dynamism: int = Field(ge=0, le=100)
    standalone_clarity: int = Field(ge=0, le=100)
    rewatchability: int = Field(ge=0, le=100)
    retention_risk: int = Field(ge=0, le=100)


class CandidateAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    scores: RankingScores
    hook_summary: str = Field(min_length=2, max_length=240)
    body_summary: str = Field(min_length=2, max_length=240)
    payoff_summary: str = Field(min_length=2, max_length=240)
    fatal_flaw: str = Field(min_length=2, max_length=240)
    rationale: str = Field(min_length=2, max_length=1000)

    @computed_field
    @property
    def total_score(self) -> float:
        positive_weights = {
            "hook_immediacy": 0.22,
            "curiosity_gap": 0.14,
            "narrative_arc": 0.15,
            "payoff_strength": 0.16,
            "emotional_intensity": 0.08,
            "visual_dynamism": 0.07,
            "standalone_clarity": 0.08,
            "rewatchability": 0.10,
        }
        positive = sum(
            int(getattr(self.scores, field)) * weight for field, weight in positive_weights.items()
        )
        penalty = self.scores.retention_risk * 0.25
        return round(max(0.0, positive - penalty), 2)


class EditorialQualityGate(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_total_score: float = Field(default=45, ge=0, le=100)
    min_hook_immediacy: int = Field(default=50, ge=0, le=100)
    min_narrative_arc: int = Field(default=50, ge=0, le=100)
    min_payoff_strength: int = Field(default=50, ge=0, le=100)
    min_standalone_clarity: int = Field(default=55, ge=0, le=100)
    max_retention_risk: int = Field(default=70, ge=0, le=100)


class Rejection(BaseModel):
    candidate_id: str
    reason: str
    suppressed_by: str | None = None


class Selection(BaseModel):
    selected: list[CandidateAssessment]
    rejections: list[Rejection]
