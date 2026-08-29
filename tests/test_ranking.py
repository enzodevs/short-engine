import json
from pathlib import Path

import pytest

from short_engine.candidates.models import Candidate
from short_engine.core.errors import ModelOutputError
from short_engine.core.models import TimeRange
from short_engine.ranking.gemini import GeminiRanker
from short_engine.ranking.models import CandidateAssessment, EditorialQualityGate, RankingScores
from short_engine.ranking.selector import CandidateSelector


def candidate(identifier: str, start: float, end: float) -> Candidate:
    return Candidate(
        id=identifier,
        time_range=TimeRange(start_seconds=start, end_seconds=end),
        transcript="text",
        segment_indexes=[0],
    )


class FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


def test_gemini_ranker_validates_structured_response(tmp_path: Path) -> None:
    raw = {
        "candidate_id": "a",
        "scores": {name: 80 for name in RankingScores.model_fields},
        "hook_summary": "Immediate conflict",
        "body_summary": "Escalating proof",
        "payoff_summary": "Clear reveal",
        "fatal_flaw": "Minor jargon",
        "rationale": "Strong complete moment",
    }
    ranker = GeminiRanker(
        "key", "model", generate=lambda **_: FakeResponse(json.dumps(raw)), debug_directory=tmp_path
    )
    result = ranker.rank([candidate("a", 0, 30)], {"a": []})
    assert result[0].total_score == 60


def test_gemini_ranker_fails_and_keeps_debug_response(tmp_path: Path) -> None:
    ranker = GeminiRanker(
        "key",
        "model",
        generate=lambda **_: FakeResponse("not-json"),
        debug_directory=tmp_path,
        attempts=2,
    )
    with pytest.raises(ModelOutputError):
        ranker.rank([candidate("a", 0, 30)], {"a": []})
    assert "not-json" in next(tmp_path.glob("*.txt")).read_text()


def test_selector_suppresses_lower_scoring_overlap() -> None:
    a, b, c = candidate("a", 0, 40), candidate("b", 10, 45), candidate("c", 60, 90)

    def score(item: Candidate, value: int) -> CandidateAssessment:
        values = {name: value for name in RankingScores.model_fields}
        values["retention_risk"] = 20
        return CandidateAssessment(
            candidate_id=item.id,
            scores=RankingScores(**values),
            hook_summary="hook",
            body_summary="body",
            payoff_summary="payoff",
            fatal_flaw="risk",
            rationale="ok",
        )

    selection = CandidateSelector(EditorialQualityGate(), max_overlap_ratio=0.4).select(
        [a, b, c], [score(a, 70), score(b, 95), score(c, 80)], 2
    )
    assert [item.candidate_id for item in selection.selected] == ["b", "c"]
    assert selection.rejections[0].candidate_id == "a"


def test_selector_rejects_best_available_candidate_when_story_quality_is_low() -> None:
    item = candidate("weak", 0, 35)
    assessment = CandidateAssessment(
        candidate_id=item.id,
        scores=RankingScores(
            hook_immediacy=65,
            curiosity_gap=60,
            narrative_arc=40,
            payoff_strength=30,
            emotional_intensity=35,
            visual_dynamism=40,
            standalone_clarity=50,
            rewatchability=20,
            retention_risk=85,
        ),
        hook_summary="Promising bug",
        body_summary="Unrelated explanation",
        payoff_summary="No demonstrated result",
        fatal_flaw="The promise is never resolved",
        rationale="The excerpt is not a complete story",
    )

    selection = CandidateSelector(EditorialQualityGate()).select([item], [assessment], 1)

    assert selection.selected == []
    assert selection.rejections[0].reason == "quality_gate"
