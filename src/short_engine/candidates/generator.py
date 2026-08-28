"""Generate coherent transcript-first candidate windows."""

import hashlib

from pydantic import BaseModel, ConfigDict, Field, model_validator

from short_engine.candidates.models import Candidate
from short_engine.core.models import TimeRange
from short_engine.transcription.models import Transcript, TranscriptSegment


class CandidateConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_duration_seconds: float = Field(default=20, gt=0)
    target_duration_seconds: float = Field(default=50, gt=0)
    max_duration_seconds: float = Field(default=90, gt=0)
    stride_segments: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def validate_durations(self) -> "CandidateConfig":
        if not (
            self.min_duration_seconds <= self.target_duration_seconds <= self.max_duration_seconds
        ):
            raise ValueError("candidate durations must satisfy min <= target <= max")
        return self


class TranscriptCandidateGenerator:
    def generate(self, transcript: Transcript, config: CandidateConfig) -> list[Candidate]:
        candidates: list[Candidate] = []
        segments = transcript.segments
        for start_index in range(0, len(segments), config.stride_segments):
            start = segments[start_index].start_seconds
            end_index = self._choose_end(segments, start_index, start, config)
            if end_index is None:
                continue
            selected = segments[start_index : end_index + 1]
            end = selected[-1].end_seconds
            time_range = TimeRange(start_seconds=start, end_seconds=end)
            text = " ".join(segment.text.strip() for segment in selected).strip()
            identifier = hashlib.sha256(f"{start:.3f}:{end:.3f}:{text}".encode()).hexdigest()[:12]
            candidates.append(
                Candidate(
                    id=identifier,
                    time_range=time_range,
                    transcript=text,
                    segment_indexes=list(range(start_index, end_index + 1)),
                )
            )
        return candidates

    @staticmethod
    def _choose_end(
        segments: list[TranscriptSegment],
        start_index: int,
        start: float,
        config: CandidateConfig,
    ) -> int | None:
        best: int | None = None
        best_distance = float("inf")
        for index in range(start_index, len(segments)):
            duration = segments[index].end_seconds - start
            if duration > config.max_duration_seconds:
                break
            if duration >= config.min_duration_seconds:
                distance = abs(duration - config.target_duration_seconds)
                if distance < best_distance:
                    best = index
                    best_distance = distance
        return best
