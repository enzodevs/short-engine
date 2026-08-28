"""Word-timestamp jump-cut planning with natural speech handles."""

from itertools import pairwise

from pydantic import BaseModel, ConfigDict, Field, computed_field

from short_engine.core.models import TimeRange
from short_engine.transcription.models import TimedWord


class JumpCutConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_silence_seconds: float = Field(default=0.5, ge=0.3)
    keep_before_seconds: float = Field(default=0.08, ge=0)
    keep_after_seconds: float = Field(default=0.06, ge=0)
    min_removed_seconds: float = Field(default=0.3, ge=0.1)


class EditSegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: TimeRange
    output: TimeRange


class EditPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    segments: list[EditSegment] = Field(min_length=1)
    original_duration_seconds: float = Field(gt=0)

    @computed_field
    @property
    def output_duration_seconds(self) -> float:
        return sum(segment.output.duration_seconds for segment in self.segments)

    @computed_field
    @property
    def removed_seconds(self) -> float:
        return round(self.original_duration_seconds - self.output_duration_seconds, 3)

    def remap_words(self, words: list[TimedWord]) -> list[TimedWord]:
        remapped: list[TimedWord] = []
        for word in words:
            segment = next(
                (
                    item
                    for item in self.segments
                    if item.source.start_seconds <= word.start_seconds
                    and word.end_seconds <= item.source.end_seconds
                ),
                None,
            )
            if segment is None:
                continue
            offset = segment.output.start_seconds - segment.source.start_seconds
            remapped.append(
                TimedWord(
                    start_seconds=word.start_seconds + offset,
                    end_seconds=word.end_seconds + offset,
                    text=word.text,
                    confidence=word.confidence,
                )
            )
        return remapped


class JumpCutPlanner:
    def __init__(self, config: JumpCutConfig | None = None) -> None:
        self.config = config or JumpCutConfig()

    def plan(self, interval: TimeRange, words: list[TimedWord]) -> EditPlan:
        ordered = sorted(
            (
                word
                for word in words
                if word.end_seconds > interval.start_seconds
                and word.start_seconds < interval.end_seconds
            ),
            key=lambda word: word.start_seconds,
        )
        ranges: list[tuple[float, float]] = []
        current_start = interval.start_seconds
        for left, right in pairwise(ordered):
            gap = right.start_seconds - left.end_seconds
            if gap < self.config.min_silence_seconds:
                continue
            end = min(interval.end_seconds, left.end_seconds + self.config.keep_before_seconds)
            next_start = max(
                interval.start_seconds, right.start_seconds - self.config.keep_after_seconds
            )
            if next_start - end < self.config.min_removed_seconds:
                continue
            ranges.append((current_start, end))
            current_start = next_start
        ranges.append((current_start, interval.end_seconds))
        segments: list[EditSegment] = []
        output_cursor = 0.0
        for start, end in ranges:
            if end <= start:
                continue
            duration = end - start
            segments.append(
                EditSegment(
                    source=TimeRange(start_seconds=start, end_seconds=end),
                    output=TimeRange(
                        start_seconds=output_cursor,
                        end_seconds=output_cursor + duration,
                    ),
                )
            )
            output_cursor += duration
        return EditPlan(
            segments=segments,
            original_duration_seconds=interval.duration_seconds,
        )
