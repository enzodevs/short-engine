"""Gemini refinement of editorial clip boundaries."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from short_engine.candidates.models import Candidate
from short_engine.core.errors import ModelOutputError
from short_engine.core.models import TimeRange
from short_engine.transcription.models import Transcript

GenerateFunction = Callable[..., Any]


class RefinedBoundary(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    rationale: str = Field(min_length=2, max_length=300)


class GeminiBoundaryRefiner:
    def __init__(
        self, api_key: str, model: str, generate: GenerateFunction | None = None, attempts: int = 3
    ) -> None:
        self.model = model
        self.attempts = attempts
        self._client: Any | None = None
        if generate is None:
            from google import genai

            self._client = genai.Client(api_key=api_key)
            generate = self._client.models.generate_content
        self.generate = generate

    def refine(self, candidate: Candidate, transcript: Transcript) -> Candidate:
        from google.genai import types

        context = [
            segment
            for segment in transcript.segments
            if segment.end_seconds >= candidate.time_range.start_seconds - 30
            and segment.start_seconds <= candidate.time_range.end_seconds + 30
        ]
        boundaries = {segment.start_seconds for segment in context} | {
            segment.end_seconds for segment in context
        }
        lines = "\n".join(
            f"[{segment.start_seconds:.2f}-{segment.end_seconds:.2f}] {segment.text}"
            for segment in context
        )
        prompt = (
            "Act as a short-form retention editor. Choose a self-contained excerpt using ONLY "
            "exact timestamps shown below. Optimize the first 1-2 seconds for a cold viewer: "
            "begin on conflict, stakes, surprise, a bold claim, or a specific curiosity gap. "
            "Remove greetings, throat-clearing, redundant setup, and context that can be inferred. "
            "The opening sentence must explicitly name its subject and make sense in isolation. "
            "Never begin with a raw number, conjunction, pronoun, or continuation whose referent "
            "appeared in the previous sentence; include the shortest preceding premise needed. "
            "The middle must escalate or provide evidence, and the ending must deliver the "
            "promised "
            "reveal, result, punchline, or insight. Never begin or end mid-sentence. Keep 15-60 "
            "seconds and prefer the shortest complete hook-to-payoff arc.\n"
            f"Initially selected: {candidate.time_range.start_seconds:.2f}-"
            f"{candidate.time_range.end_seconds:.2f}\n{lines}"
        )
        for _ in range(self.attempts):
            response = self.generate(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=RefinedBoundary.model_json_schema(),
                    temperature=0.0,
                ),
            )
            try:
                boundary = RefinedBoundary.model_validate_json(response.text or "")
                start = self._exact(boundary.start_seconds, boundaries)
                end = self._exact(boundary.end_seconds, boundaries)
                if not 15 <= end - start <= 60:
                    raise ValueError("duration outside allowed range")
                selected = [
                    segment
                    for segment in context
                    if segment.start_seconds >= start - 0.01 and segment.end_seconds <= end + 0.01
                ]
                if not selected:
                    raise ValueError("empty refined transcript")
                return Candidate(
                    id=candidate.id,
                    time_range=TimeRange(start_seconds=start, end_seconds=end),
                    transcript=" ".join(segment.text for segment in selected),
                    segment_indexes=[transcript.segments.index(segment) for segment in selected],
                    boundary_signals=[*candidate.boundary_signals, "gemini-semantic-refinement"],
                )
            except (ValidationError, ValueError):
                continue
        raise ModelOutputError(f"Gemini could not refine boundaries for {candidate.id}")

    @staticmethod
    def _exact(value: float, boundaries: set[float]) -> float:
        nearest = min(boundaries, key=lambda boundary: abs(boundary - value))
        if abs(nearest - value) > 0.05:
            raise ValueError("boundary does not match transcript")
        return nearest
