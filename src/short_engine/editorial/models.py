"""Typed contracts for global video understanding and safe edit plans."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from short_engine.core.models import TimeRange


class MomentRole(StrEnum):
    HOOK = "hook"
    PREMISE = "premise"
    EVIDENCE = "evidence"
    PAYOFF = "payoff"


class EditStrategy(StrEnum):
    CONTINUOUS = "continuous"
    PAYOFF_TEASER = "payoff_teaser"


class EditorialMoment(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=2, max_length=80)
    subject: str = Field(min_length=2, max_length=160)
    role: MomentRole
    source: TimeRange
    summary: str = Field(min_length=2, max_length=300)


class EditPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=2, max_length=80)
    title: str = Field(min_length=2, max_length=140)
    strategy: EditStrategy
    main_range: TimeRange
    teaser_range: TimeRange | None = None
    hook_range: TimeRange
    payoff_range: TimeRange
    rationale: str = Field(min_length=2, max_length=400)

    @model_validator(mode="after")
    def validate_shape(self) -> "EditPlan":
        if not 10 <= self.main_range.duration_seconds <= 60:
            raise ValueError("main range must be 10-60 seconds")
        for name, interval in (("hook", self.hook_range), ("payoff", self.payoff_range)):
            if not (
                self.main_range.start_seconds <= interval.start_seconds
                and interval.end_seconds <= self.main_range.end_seconds
            ):
                raise ValueError(f"{name} must be contained by the main range")
        if self.hook_range.start_seconds >= self.payoff_range.start_seconds:
            raise ValueError("main story must progress from hook to payoff")
        if self.strategy is EditStrategy.CONTINUOUS and self.teaser_range is not None:
            raise ValueError("continuous plan cannot have a teaser")
        if self.strategy is EditStrategy.PAYOFF_TEASER:
            if self.teaser_range is None:
                raise ValueError("payoff teaser plan requires a teaser")
            if not 0.5 <= self.teaser_range.duration_seconds <= 3:
                raise ValueError("teaser must be 0.5-3 seconds")
            if not (
                self.main_range.start_seconds <= self.teaser_range.start_seconds
                and self.teaser_range.end_seconds <= self.main_range.end_seconds
            ):
                raise ValueError("teaser must be contained by the main range")
            if not (
                self.payoff_range.start_seconds <= self.teaser_range.start_seconds
                and self.teaser_range.end_seconds <= self.payoff_range.end_seconds
            ):
                raise ValueError("teaser must come from the plan payoff")
        return self

    @property
    def source_ranges(self) -> list[TimeRange]:
        return [self.teaser_range, self.main_range] if self.teaser_range else [self.main_range]


class EditorialMap(BaseModel):
    video_summary: str = Field(min_length=2, max_length=800)
    moments: list[EditorialMoment] = Field(min_length=1, max_length=30)
    plans: list[EditPlan] = Field(min_length=1, max_length=8)


class PlanReview(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_id: str
    coherent: bool
    standalone: bool
    payoff_resolves_hook: bool
    no_dangling_reference: bool
    meaning_preserved: bool
    hook_score: int = Field(ge=0, le=100)
    retention_score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=2, max_length=500)

    @property
    def passes(self) -> bool:
        return (
            self.coherent
            and self.standalone
            and self.payoff_resolves_hook
            and self.no_dangling_reference
            and self.meaning_preserved
            and self.hook_score >= 55
            and self.retention_score >= 55
        )


class EditorialDecision(BaseModel):
    editorial_map: EditorialMap
    reviews: list[PlanReview]
    selected_plan_ids: list[str] = Field(max_length=20)

    @model_validator(mode="after")
    def validate_selection(self) -> "EditorialDecision":
        plans = {plan.id for plan in self.editorial_map.plans}
        reviews = {review.plan_id: review for review in self.reviews}
        if not set(self.selected_plan_ids) <= plans:
            raise ValueError("selected plan is missing")
        if any(
            identifier not in reviews or not reviews[identifier].passes
            for identifier in self.selected_plan_ids
        ):
            raise ValueError("selected plan did not pass verification")
        return self
