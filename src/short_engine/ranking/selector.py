"""Deterministic interval suppression."""

from short_engine.candidates.models import Candidate
from short_engine.ranking.models import (
    CandidateAssessment,
    EditorialQualityGate,
    Rejection,
    Selection,
)


class CandidateSelector:
    def __init__(self, quality: EditorialQualityGate, max_overlap_ratio: float = 0.35) -> None:
        self.quality = quality
        self.max_overlap_ratio = max_overlap_ratio

    def select(
        self, candidates: list[Candidate], assessments: list[CandidateAssessment], count: int
    ) -> Selection:
        by_id = {item.id: item for item in candidates}
        selected: list[CandidateAssessment] = []
        rejected: list[Rejection] = []
        for assessment in sorted(assessments, key=lambda item: item.total_score, reverse=True):
            current = by_id[assessment.candidate_id]
            if not self._passes_quality(assessment):
                rejected.append(Rejection(candidate_id=current.id, reason="quality_gate"))
                continue
            conflict = next(
                (
                    kept
                    for kept in selected
                    if self._overlap(current, by_id[kept.candidate_id]) > self.max_overlap_ratio
                ),
                None,
            )
            if conflict:
                rejected.append(
                    Rejection(
                        candidate_id=current.id,
                        reason="overlap",
                        suppressed_by=conflict.candidate_id,
                    )
                )
            elif len(selected) < count:
                selected.append(assessment)
            else:
                rejected.append(Rejection(candidate_id=current.id, reason="below_selection_cutoff"))
        return Selection(selected=selected, rejections=rejected)

    def _passes_quality(self, assessment: CandidateAssessment) -> bool:
        scores = assessment.scores
        return (
            assessment.total_score >= self.quality.min_total_score
            and scores.hook_immediacy >= self.quality.min_hook_immediacy
            and scores.narrative_arc >= self.quality.min_narrative_arc
            and scores.payoff_strength >= self.quality.min_payoff_strength
            and scores.standalone_clarity >= self.quality.min_standalone_clarity
            and scores.retention_risk <= self.quality.max_retention_risk
        )

    @staticmethod
    def _overlap(left: Candidate, right: Candidate) -> float:
        start = max(left.time_range.start_seconds, right.time_range.start_seconds)
        end = min(left.time_range.end_seconds, right.time_range.end_seconds)
        intersection = max(0.0, end - start)
        shortest = min(left.time_range.duration_seconds, right.time_range.duration_seconds)
        return intersection / shortest if shortest else 0.0
