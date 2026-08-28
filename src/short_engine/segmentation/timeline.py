"""Merge heterogeneous boundary evidence into one timeline."""

from short_engine.core.models import TimeRange
from short_engine.segmentation.models import (
    BoundaryKind,
    DetectedBoundary,
    Timeline,
    TimelineBoundary,
)
from short_engine.transcription.models import Transcript


class TimelineBuilder:
    def __init__(self, *, merge_tolerance_seconds: float = 0.08) -> None:
        self.merge_tolerance_seconds = merge_tolerance_seconds

    def build(
        self,
        transcript: Transcript,
        detected: list[DetectedBoundary],
    ) -> Timeline:
        values = [
            DetectedBoundary(at_seconds=0, kind=BoundaryKind.MEDIA),
            *[item for item in detected if item.at_seconds <= transcript.duration_seconds],
            DetectedBoundary(at_seconds=transcript.duration_seconds, kind=BoundaryKind.MEDIA),
        ]
        values.sort(key=lambda item: item.at_seconds)
        merged: list[TimelineBoundary] = []
        for item in values:
            if merged and item.at_seconds - merged[-1].at_seconds <= self.merge_tolerance_seconds:
                previous = merged[-1]
                merged[-1] = TimelineBoundary(
                    at_seconds=previous.at_seconds,
                    kinds=previous.kinds | {item.kind},
                )
            else:
                merged.append(TimelineBoundary(at_seconds=item.at_seconds, kinds={item.kind}))
        return Timeline(duration_seconds=transcript.duration_seconds, boundaries=merged)

    @staticmethod
    def absolutize(
        ranges: list[TimeRange],
        *,
        offset_seconds: float,
        media_duration: float,
    ) -> list[TimeRange]:
        absolute: list[TimeRange] = []
        for item in ranges:
            start = max(0.0, item.start_seconds + offset_seconds)
            end = min(media_duration, item.end_seconds + offset_seconds)
            if end > start:
                absolute.append(TimeRange(start_seconds=start, end_seconds=end))
        return absolute
