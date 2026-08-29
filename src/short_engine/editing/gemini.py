"""Gemini narrative director adapter."""

import json
from collections.abc import Callable
from typing import Any

from google.genai import types
from pydantic import ValidationError

from short_engine.candidates.models import Candidate
from short_engine.core.errors import ModelOutputError
from short_engine.editing.story import StoryPackage
from short_engine.transcription.models import Transcript

GenerateFunction = Callable[..., Any]


class GeminiStoryDirector:
    def __init__(
        self,
        api_key: str,
        model: str,
        generate: GenerateFunction | None = None,
        attempts: int = 3,
    ) -> None:
        self.model = model
        self.attempts = attempts
        if generate is None:
            from google import genai

            generate = genai.Client(api_key=api_key).models.generate_content
        self.generate = generate

    def compose(
        self, candidates: list[Candidate], transcript: Transcript, selected_count: int
    ) -> StoryPackage:
        indexes = {index for candidate in candidates for index in candidate.segment_indexes}
        expanded = {
            nearby
            for index in indexes
            for nearby in range(max(0, index - 2), min(len(transcript.segments), index + 3))
        }
        lines = "\n".join(
            f"[{transcript.segments[index].start_seconds:.2f}-"
            f"{transcript.segments[index].end_seconds:.2f}] "
            f"{transcript.segments[index].text}"
            for index in sorted(expanded)
        )
        prompt = (
            f"Act as an elite short-form story editor. Build at least {max(3, selected_count)} "
            "meaningfully different edits from "
            "the source transcript. Use only exact timestamp boundaries shown. You may use a later "
            "payoff as a cold-open hook, then return to earlier premise/evidence, but never alter "
            "meaning or fabricate speech. Each variant must be 15-60 seconds and follow hook -> "
            "premise/escalation/evidence -> payoff. Remove setup that a cold viewer does not need. "
            "Create a second-by-second retention map, prescribe cuts for weak spans, and be severe "
            "about context dependency. Variants must represent distinct hook strategies, not minor "
            "boundary changes. Select the strongest IDs for rendering.\n"
            f"Number to select: {selected_count}\nTranscript:\n{lines}"
        )
        boundaries = {
            value
            for index in expanded
            for value in (
                transcript.segments[index].start_seconds,
                transcript.segments[index].end_seconds,
            )
        }
        for _ in range(self.attempts):
            response = self.generate(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=StoryPackage.model_json_schema(),
                    temperature=0.35,
                ),
            )
            try:
                package = StoryPackage.model_validate_json(response.text or "")
                if len(package.selected_variant_ids) != selected_count:
                    raise ValueError("incorrect selected variant count")
                for variant in package.variants:
                    for beat in variant.beats:
                        if not self._exact(beat.source.start_seconds, boundaries):
                            raise ValueError("invented story start boundary")
                        if not self._exact(beat.source.end_seconds, boundaries):
                            raise ValueError("invented story end boundary")
                return package
            except (ValidationError, ValueError, json.JSONDecodeError):
                continue
        raise ModelOutputError("Gemini could not produce a valid story package")

    @staticmethod
    def _exact(value: float, boundaries: set[float]) -> bool:
        return bool(boundaries) and min(abs(item - value) for item in boundaries) <= 0.05
