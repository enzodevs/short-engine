"""Typed ranking contracts."""

from pydantic import BaseModel, ConfigDict, Field, computed_field


class RankingScores(BaseModel):
    model_config = ConfigDict(frozen=True)

    hook_strength: int = Field(ge=0, le=100)
    completeness: int = Field(ge=0, le=100)
    information_density: int = Field(ge=0, le=100)
    emotional_payoff: int = Field(ge=0, le=100)
    visual_support: int = Field(ge=0, le=100)
    shareability: int = Field(ge=0, le=100)


class CandidateAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    scores: RankingScores
    rationale: str = Field(min_length=2, max_length=500)

    @computed_field
    @property
    def total_score(self) -> float:
        weights = (0.22, 0.20, 0.14, 0.16, 0.12, 0.16)
        values: tuple[int, ...] = tuple(
            int(getattr(self.scores, field)) for field in RankingScores.model_fields
        )
        return round(sum(value * weight for value, weight in zip(values, weights, strict=True)), 2)


class Rejection(BaseModel):
    candidate_id: str
    reason: str
    suppressed_by: str | None = None


class Selection(BaseModel):
    selected: list[CandidateAssessment]
    rejections: list[Rejection]
