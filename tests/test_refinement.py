import json

import pytest

from short_engine.candidates.models import Candidate
from short_engine.core.errors import ModelOutputError
from short_engine.core.models import TimeRange
from short_engine.ranking.refiner import GeminiBoundaryRefiner
from short_engine.transcription.models import Transcript, TranscriptSegment


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


def fixture() -> tuple[Candidate, Transcript]:
    segments = [
        TranscriptSegment(start_seconds=10, end_seconds=20, text="pergunta incompleta", words=[]),
        TranscriptSegment(start_seconds=20, end_seconds=32, text="Resposta forte.", words=[]),
        TranscriptSegment(start_seconds=32, end_seconds=45, text="Conclusão completa.", words=[]),
        TranscriptSegment(start_seconds=45, end_seconds=55, text="Outro assunto.", words=[]),
    ]
    transcript = Transcript(language="pt", model="test", duration_seconds=55, segments=segments)
    candidate = Candidate(
        id="a",
        time_range=TimeRange(start_seconds=10, end_seconds=55),
        transcript="text",
        segment_indexes=[0, 1, 2, 3],
    )
    return candidate, transcript


def test_refiner_snaps_to_complete_segment_boundaries() -> None:
    candidate, transcript = fixture()
    response = {"start_seconds": 20, "end_seconds": 45, "rationale": "complete answer"}
    refiner = GeminiBoundaryRefiner(
        "key", "model", generate=lambda **_: FakeResponse(json.dumps(response))
    )

    refined = refiner.refine(candidate, transcript)

    assert refined.time_range == TimeRange(start_seconds=20, end_seconds=45)
    assert refined.transcript == "Resposta forte. Conclusão completa."


def test_refiner_rejects_boundaries_not_present_in_transcript() -> None:
    candidate, transcript = fixture()
    response = {"start_seconds": 21.3, "end_seconds": 44.2, "rationale": "invented"}
    refiner = GeminiBoundaryRefiner(
        "key", "model", generate=lambda **_: FakeResponse(json.dumps(response)), attempts=1
    )

    with pytest.raises(ModelOutputError):
        refiner.refine(candidate, transcript)


def test_refiner_retries_with_validation_feedback() -> None:
    candidate, transcript = fixture()
    responses = iter(
        [
            FakeResponse(
                json.dumps({"start_seconds": 21.3, "end_seconds": 44.2, "rationale": "invented"})
            ),
            FakeResponse(
                json.dumps({"start_seconds": 20, "end_seconds": 45, "rationale": "corrected"})
            ),
        ]
    )
    prompts: list[str] = []

    def generate(**kwargs: object) -> FakeResponse:
        prompts.append(str(kwargs["contents"]))
        return next(responses)

    refined = GeminiBoundaryRefiner("key", "model", generate=generate).refine(candidate, transcript)

    assert refined.time_range == TimeRange(start_seconds=20, end_seconds=45)
    assert "boundary does not match transcript" in prompts[1]
