import json

import pytest
from pydantic import ValidationError

from short_engine.editorial.gemini import GeminiEditorialDirector
from short_engine.editorial.models import EditorialMap
from short_engine.transcription.models import Transcript, TranscriptSegment


class FakeResponse:
    def __init__(self, value: dict[str, object]) -> None:
        self.text = json.dumps(value)


def transcript() -> Transcript:
    return Transcript(
        language="en",
        model="test",
        duration_seconds=40,
        segments=[
            TranscriptSegment(start_seconds=0, end_seconds=10, text="Here is the challenge."),
            TranscriptSegment(start_seconds=10, end_seconds=20, text="This is how it works."),
            TranscriptSegment(start_seconds=20, end_seconds=30, text="Now I demonstrate it."),
            TranscriptSegment(start_seconds=30, end_seconds=32, text="Here is the result."),
            TranscriptSegment(start_seconds=32, end_seconds=40, text="The challenge is solved."),
        ],
    )


def editorial_map() -> dict[str, object]:
    return {
        "video_summary": "A complete technical demonstration.",
        "moments": [
            {
                "id": "hook",
                "subject": "challenge",
                "role": "hook",
                "source": {"start_seconds": 0, "end_seconds": 10},
                "summary": "Introduces the challenge.",
            },
            {
                "id": "payoff",
                "subject": "challenge",
                "role": "payoff",
                "source": {"start_seconds": 30, "end_seconds": 40},
                "summary": "Shows the successful result.",
            },
        ],
        "plans": [
            {
                "id": "result-first",
                "title": "Challenge solved",
                "strategy": "payoff_teaser",
                "main_range": {"start_seconds": 0, "end_seconds": 40},
                "teaser_range": {"start_seconds": 30, "end_seconds": 32},
                "hook_range": {"start_seconds": 0, "end_seconds": 10},
                "payoff_range": {"start_seconds": 30, "end_seconds": 40},
                "rationale": "Tease the proven result, then preserve the full explanation.",
            }
        ],
    }


def test_global_director_selects_only_a_verified_plan() -> None:
    responses = iter(
        [
            FakeResponse(editorial_map()),
            FakeResponse(
                {
                    "plan_id": "result-first",
                    "coherent": True,
                    "standalone": True,
                    "payoff_resolves_hook": True,
                    "no_dangling_reference": True,
                    "meaning_preserved": True,
                    "hook_score": 82,
                    "retention_score": 78,
                    "reason": "The repeated result proves the same continuous explanation.",
                }
            ),
        ]
    )
    director = GeminiEditorialDirector(
        "key", "model", generate=lambda **_: next(responses), attempts=1
    )

    decision = director.direct(transcript(), None, 1)

    assert decision.selected_plan_ids == ["result-first"]
    assert decision.editorial_map.plans[0].source_ranges[1].start_seconds == 0


def test_global_director_rejects_a_plan_that_fails_final_coherence() -> None:
    responses = iter(
        [
            FakeResponse(editorial_map()),
            FakeResponse(
                {
                    "plan_id": "result-first",
                    "coherent": False,
                    "standalone": True,
                    "payoff_resolves_hook": True,
                    "no_dangling_reference": False,
                    "meaning_preserved": True,
                    "hook_score": 82,
                    "retention_score": 78,
                    "reason": "The opening begins in the middle of a sentence.",
                }
            ),
        ]
    )
    director = GeminiEditorialDirector(
        "key", "model", generate=lambda **_: next(responses), attempts=1
    )

    decision = director.direct(transcript(), None, 1)

    assert decision.selected_plan_ids == []


def test_editorial_map_rejects_a_teaser_outside_the_plan_payoff() -> None:
    raw = editorial_map()
    plans = raw["plans"]
    assert isinstance(plans, list)
    plans[0]["teaser_range"] = {"start_seconds": 20, "end_seconds": 22}

    with pytest.raises(ValidationError, match="plan payoff"):
        EditorialMap.model_validate(raw)


def test_global_director_retries_invalid_map_with_validation_feedback() -> None:
    invalid = editorial_map()
    plans = invalid["plans"]
    assert isinstance(plans, list)
    plans[0]["main_range"] = {"start_seconds": 0, "end_seconds": 90}
    responses = iter(
        [
            FakeResponse(invalid),
            FakeResponse(editorial_map()),
            FakeResponse(
                {
                    "plan_id": "result-first",
                    "coherent": True,
                    "standalone": True,
                    "payoff_resolves_hook": True,
                    "no_dangling_reference": True,
                    "meaning_preserved": True,
                    "hook_score": 80,
                    "retention_score": 80,
                    "reason": "Valid final edit.",
                }
            ),
        ]
    )
    prompts: list[str] = []

    def generate(**kwargs: object) -> FakeResponse:
        contents = kwargs["contents"]
        assert isinstance(contents, list)
        prompts.append(str(contents[0]))
        return next(responses)

    decision = GeminiEditorialDirector("key", "model", generate=generate).direct(
        transcript(), None, 1
    )

    assert decision.selected_plan_ids == ["result-first"]
    assert "main range must be 10-60 seconds" in prompts[1]
