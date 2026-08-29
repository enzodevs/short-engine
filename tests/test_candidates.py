from short_engine.candidates.generator import CandidateConfig, TranscriptCandidateGenerator
from short_engine.candidates.models import Candidate
from short_engine.candidates.pool import CandidatePoolSampler
from short_engine.core.models import TimeRange
from short_engine.transcription.models import Transcript, TranscriptSegment


def _transcript() -> Transcript:
    return Transcript(
        language="pt",
        model="test",
        duration_seconds=120,
        segments=[
            TranscriptSegment(
                start_seconds=index * 10,
                end_seconds=(index + 1) * 10,
                text=f"Frase completa {index}.",
            )
            for index in range(12)
        ],
    )


def test_candidates_respect_duration_and_segment_boundaries() -> None:
    candidates = TranscriptCandidateGenerator().generate(
        _transcript(),
        CandidateConfig(
            min_duration_seconds=20, target_duration_seconds=40, max_duration_seconds=60
        ),
    )

    assert candidates
    assert all(20 <= candidate.time_range.duration_seconds <= 60 for candidate in candidates)
    assert candidates[0].time_range.start_seconds == 0
    assert candidates[0].time_range.end_seconds == 40
    assert candidates[0].transcript.endswith(".")


def test_sparse_transcript_does_not_fabricate_a_candidate() -> None:
    transcript = Transcript(
        language="pt",
        model="test",
        duration_seconds=8,
        segments=[TranscriptSegment(start_seconds=0, end_seconds=8, text="Curto.")],
    )

    candidates = TranscriptCandidateGenerator().generate(transcript, CandidateConfig())

    assert candidates == []


def test_candidate_pool_sampler_covers_entire_timeline() -> None:
    candidates = [
        Candidate(
            id=str(index),
            time_range=TimeRange(start_seconds=index * 10, end_seconds=index * 10 + 20),
            transcript=f"candidate {index}",
            segment_indexes=[index],
        )
        for index in range(100)
    ]

    selected = CandidatePoolSampler().select(candidates, 5)

    assert [item.id for item in selected] == ["0", "25", "50", "74", "99"]
