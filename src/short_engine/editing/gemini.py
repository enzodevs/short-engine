"""Gemini narrative director adapter."""

import json
from collections.abc import Callable
from typing import Any, cast

from google.genai import types
from pydantic import ValidationError

from short_engine.candidates.models import Candidate
from short_engine.core.errors import ModelOutputError
from short_engine.editing.story import StoryPackage
from short_engine.transcription.models import Transcript

GenerateFunction = Callable[..., Any]
type JsonValue = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None


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
        self._client: Any | None = None
        if generate is None:
            from google import genai

            self._client = genai.Client(api_key=api_key)
            generate = self._client.models.generate_content
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
        failures: list[str] = []
        current_prompt = prompt
        for _ in range(self.attempts):
            response = self.generate(
                model=self.model,
                contents=current_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=self._gemini_schema(
                        cast(JsonValue, StoryPackage.model_json_schema())
                    ),
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
            except (ValidationError, ValueError, json.JSONDecodeError) as error:
                failures.append(str(error))
                current_prompt = (
                    f"{prompt}\n\nYour previous response failed validation. Correct every variant, "
                    "return the complete package again, and do not repeat the error below. The "
                    "duration of a variant is the SUM of all its beat durations, not its source "
                    f"timeline span.\nValidation error:\n{error}"
                )
                continue
        detail = failures[-1] if failures else "unknown validation failure"
        raise ModelOutputError(f"Gemini could not produce a valid story package: {detail}")

    @staticmethod
    def _exact(value: float, boundaries: set[float]) -> bool:
        return bool(boundaries) and min(abs(item - value) for item in boundaries) <= 0.05

    @classmethod
    def _gemini_schema(cls, value: JsonValue, preserve_keys: bool = False) -> JsonValue:
        if isinstance(value, dict):
            return {
                key: cls._gemini_schema(item, preserve_keys=key in {"$defs", "properties"})
                for key, item in value.items()
                if preserve_keys
                or key
                not in {
                    "description",
                    "exclusiveMinimum",
                    "exclusiveMaximum",
                    "maximum",
                    "maxItems",
                    "maxLength",
                    "minimum",
                    "minItems",
                    "minLength",
                    "title",
                }
            }
        if isinstance(value, list):
            return [cls._gemini_schema(item) for item in value]
        return value
