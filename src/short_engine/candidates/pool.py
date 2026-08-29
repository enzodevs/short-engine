"""Deterministic candidate-pool coverage."""

from short_engine.candidates.models import Candidate


class CandidatePoolSampler:
    """Preserve full-video coverage when inference budget is bounded."""

    def select(self, candidates: list[Candidate], limit: int) -> list[Candidate]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if len(candidates) <= limit:
            return candidates
        if limit == 1:
            return [candidates[len(candidates) // 2]]
        indexes = {
            round(position * (len(candidates) - 1) / (limit - 1)) for position in range(limit)
        }
        return [candidates[index] for index in sorted(indexes)]
