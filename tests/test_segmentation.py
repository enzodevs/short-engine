from short_engine.core.models import TimeRange
from short_engine.segmentation.models import BoundaryKind, DetectedBoundary
from short_engine.segmentation.timeline import TimelineBuilder
from short_engine.transcription.models import Transcript, TranscriptSegment


def test_timeline_merges_clamps_and_deduplicates_multisignal_boundaries() -> None:
    transcript = Transcript(
        language="pt",
        model="test",
        duration_seconds=10,
        segments=[TranscriptSegment(start_seconds=0, end_seconds=10, text="Olá.")],
    )
    boundaries = [
        DetectedBoundary(at_seconds=2.0, kind=BoundaryKind.SCENE),
        DetectedBoundary(at_seconds=2.01, kind=BoundaryKind.SPEECH),
        DetectedBoundary(at_seconds=12.0, kind=BoundaryKind.SCENE),
        DetectedBoundary(at_seconds=5.0, kind=BoundaryKind.SPEAKER),
    ]

    timeline = TimelineBuilder(merge_tolerance_seconds=0.05).build(transcript, boundaries)

    assert [boundary.at_seconds for boundary in timeline.boundaries] == [0.0, 2.0, 5.0, 10.0]
    assert timeline.boundaries[1].kinds == {BoundaryKind.SCENE, BoundaryKind.SPEECH}


def test_absolutize_applies_chunk_offset_exactly_once() -> None:
    local = [TimeRange(start_seconds=10, end_seconds=20)]

    absolute = TimelineBuilder.absolutize(local, offset_seconds=1140, media_duration=1808)

    assert absolute == [TimeRange(start_seconds=1150, end_seconds=1160)]
