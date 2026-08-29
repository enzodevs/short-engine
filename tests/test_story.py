import json

from short_engine.candidates.models import Candidate
from short_engine.core.models import TimeRange
from short_engine.editing.gemini import GeminiStoryDirector, JsonValue
from short_engine.editing.story import StoryPackage
from short_engine.transcription.models import Transcript, TranscriptSegment


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


def test_story_director_validates_exact_multibeat_composition() -> None:
    transcript = Transcript(
        language="pt",
        model="test",
        duration_seconds=40,
        segments=[
            TranscriptSegment(start_seconds=0, end_seconds=10, text="Premise"),
            TranscriptSegment(start_seconds=10, end_seconds=20, text="Evidence"),
            TranscriptSegment(start_seconds=20, end_seconds=30, text="Payoff"),
        ],
    )
    candidate = Candidate(
        id="candidate",
        time_range=TimeRange(start_seconds=0, end_seconds=30),
        transcript="Premise Evidence Payoff",
        segment_indexes=[0, 1, 2],
    )
    variant = {
        "id": "payoff-first",
        "title": "Reveal first",
        "genre": "technical",
        "strategy": "Cold open then explain",
        "hook_text": "Payoff",
        "beats": [
            {
                "role": "hook",
                "source": {"start_seconds": 20, "end_seconds": 30},
                "rationale": "Immediate result",
            },
            {
                "role": "premise",
                "source": {"start_seconds": 0, "end_seconds": 10},
                "rationale": "Minimum context",
            },
            {
                "role": "payoff",
                "source": {"start_seconds": 10, "end_seconds": 20},
                "rationale": "Proof",
            },
        ],
        "retention_map": [
            {
                "output_start_seconds": 0,
                "output_end_seconds": 30,
                "attention_reason": "Open loop",
                "drop_off_risk": 20,
                "edit_action": "Keep dense",
            }
        ],
        "predicted_retention_score": 80,
        "fatal_flaw": "Some jargon",
    }
    raw = {
        "variants": [variant, {**variant, "id": "linear"}, {**variant, "id": "short"}],
        "selected_variant_ids": ["payoff-first"],
    }
    director = GeminiStoryDirector(
        "key", "model", generate=lambda **_: FakeResponse(json.dumps(raw))
    )

    package = director.compose([candidate], transcript, 1)

    assert isinstance(package, StoryPackage)
    assert package.variants[0].beats[0].source.start_seconds == 20


def test_story_director_removes_unsupported_gemini_schema_keywords() -> None:
    schema: JsonValue = {
        "type": "string",
        "exclusiveMinimum": 0,
        "minimum": 0,
        "minLength": 1,
        "maxLength": 20,
        "title": "Value",
    }

    cleaned = GeminiStoryDirector._gemini_schema(schema)

    assert cleaned == {"type": "string"}


def test_story_director_preserves_domain_properties_named_like_schema_metadata() -> None:
    schema: JsonValue = {
        "type": "object",
        "properties": {"title": {"type": "string", "title": "Title"}},
        "required": ["title"],
    }

    cleaned = GeminiStoryDirector._gemini_schema(schema)

    assert cleaned == {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }
