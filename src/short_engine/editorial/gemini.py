"""Gemini adapter for whole-video editorial mapping and final-plan verification."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from google.genai import types
from pydantic import BaseModel, ValidationError

from short_engine.core.errors import ModelOutputError
from short_engine.core.models import TimeRange
from short_engine.editorial.models import (
    EditorialDecision,
    EditorialMap,
    EditPlan,
    PlanReview,
)
from short_engine.ranking.video import GeminiVideoEvidence
from short_engine.transcription.models import Transcript

GenerateFunction = Callable[..., Any]
type JsonValue = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None


class GeminiEditorialDirector:
    def __init__(
        self,
        api_key: str,
        model: str,
        generate: GenerateFunction | None = None,
        attempts: int = 3,
        debug_directory: Path | None = None,
    ) -> None:
        self.model = model
        self.attempts = attempts
        self.debug_directory = debug_directory
        self._client: Any | None = None
        if generate is None:
            from google import genai

            self._client = genai.Client(api_key=api_key)
            generate = self._client.models.generate_content
        self.generate = generate

    def direct(
        self, transcript: Transcript, video: Path | None, requested_count: int
    ) -> EditorialDecision:
        video_context: GeminiVideoEvidence | None = None
        try:
            if video is not None:
                if self._client is None:
                    raise ModelOutputError("Video evidence requires a Gemini client")
                video_context = GeminiVideoEvidence(self._client, video)
                video_context.__enter__()
            video_part = (
                video_context.part(
                    TimeRange(start_seconds=0, end_seconds=transcript.duration_seconds), 0.5
                )
                if video_context
                else None
            )
            editorial_map = self._map(transcript, video_part)
            editorial_map = self._normalize_boundaries(editorial_map, transcript)
            reviews = [
                self._review(plan, editorial_map, transcript, video_context)
                for plan in editorial_map.plans
            ]
            plans = {plan.id: plan for plan in editorial_map.plans}
            passing = sorted(
                (review for review in reviews if review.passes),
                key=lambda item: (
                    item.retention_score,
                    item.hook_score,
                    -sum(
                        interval.duration_seconds for interval in plans[item.plan_id].source_ranges
                    ),
                ),
                reverse=True,
            )
            return EditorialDecision(
                editorial_map=editorial_map,
                reviews=reviews,
                selected_plan_ids=[review.plan_id for review in passing[:requested_count]],
            )
        finally:
            if video_context is not None:
                video_context.__exit__(None, None, None)

    def _map(self, transcript: Transcript, video_part: types.Part | None) -> EditorialMap:
        lines = "\n".join(
            f"[{segment.start_seconds:.2f}-{segment.end_seconds:.2f}] {segment.text}"
            for segment in transcript.segments
        )
        prompt = (
            "Understand the entire video as an editor. Identify self-contained moments and the "
            "semantic relations between hook/premise/evidence/payoff. Then propose up to 6 strong "
            "short edits. Do not score arbitrary fixed windows. Every main_range must be one "
            "continuous chronological 10-60 second excerpt containing complete speech and a real "
            "payoff. Each plan must declare hook_range and payoff_range inside its own main_range. "
            "Use continuous unless a 0.5-3 second excerpt from that SAME payoff can be "
            "safely repeated as an opening teaser. A teaser must be contained in main_range and "
            "must not change meaning. Never concatenate unrelated speech. Use only exact timestamp "
            "boundaries below. For each explicit visible result, consider both a continuous plan "
            "and a payoff_teaser plan, but emit the teaser version only when it visibly proves the "
            "same topic or action. Prefer no plan over a weak or unresolved plan.\n\nTRANSCRIPT:\n"
            f"{lines}"
        )
        contents: list[Any] = [prompt]
        if video_part is not None:
            contents.append(video_part)
        return self._generate_model(EditorialMap, contents)

    def _review(
        self,
        plan: EditPlan,
        editorial_map: EditorialMap,
        transcript: Transcript,
        video_context: GeminiVideoEvidence | None,
    ) -> PlanReview:
        ranges = plan.source_ranges
        transcript_parts = [self._text_for(item, transcript) for item in ranges]
        prompt = (
            "Act as an independent final-cut verifier. Judge the exact playback order below, not "
            "the source video generally. Reject dangling sentence fragments, topic changes, fake "
            "payoffs, misleading reordering, or a teaser unrelated to the body. The payoff must "
            "explicitly resolve the opening promise. Score hook_score and retention_score on a "
            "0-100 scale where 50 is average, 70 is strong, and 90 is exceptional. Be severe.\n"
            f"Plan ID: {plan.id}\nTitle: {plan.title}\nStrategy: {plan.strategy}\n"
            f"Rationale: {plan.rationale}\nPlayback transcript:\n"
            + "\n--- CUT ---\n".join(transcript_parts)
        )
        contents: list[Any] = [prompt]
        if video_context is not None:
            contents.extend(video_context.part(item, 2.0) for item in ranges)
        review = self._generate_model(PlanReview, contents)
        if review.plan_id != plan.id:
            raise ModelOutputError("Gemini verifier returned the wrong plan ID")
        return review

    def _generate_model[Model: BaseModel](
        self, model_type: type[Model], contents: list[Any]
    ) -> Model:
        failure = "unknown model output"
        current_contents = list(contents)
        for _ in range(self.attempts):
            response = self.generate(
                model=self.model,
                contents=current_contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=self._schema(
                        cast(JsonValue, model_type.model_json_schema())
                    ),
                    temperature=0.1,
                ),
            )
            if self.debug_directory is not None:
                self.debug_directory.mkdir(parents=True, exist_ok=True)
                (self.debug_directory / f"{model_type.__name__}-last.json").write_text(
                    response.text or ""
                )
            try:
                return model_type.model_validate_json(response.text or "")
            except (ValidationError, json.JSONDecodeError) as error:
                failure = str(error)
                correction = (
                    "Your previous structured response failed validation. Return the complete "
                    "response again and correct every reported issue. Do not relax or reinterpret "
                    f"the constraints.\nValidation error:\n{error}"
                )
                current_contents = [f"{contents[0]}\n\n{correction}", *contents[1:]]
        raise ModelOutputError(f"Gemini returned invalid editorial output: {failure}")

    @staticmethod
    def _text_for(interval: TimeRange, transcript: Transcript) -> str:
        return " ".join(
            segment.text
            for segment in transcript.segments
            if segment.start_seconds >= interval.start_seconds - 0.01
            and segment.end_seconds <= interval.end_seconds + 0.01
        )

    @classmethod
    def _normalize_boundaries(cls, value: EditorialMap, transcript: Transcript) -> EditorialMap:
        boundaries = {
            boundary
            for segment in transcript.segments
            for boundary in (segment.start_seconds, segment.end_seconds)
        }
        normalized_moments = [
            moment.model_copy(update={"source": cls._snap(moment.source, boundaries)})
            for moment in value.moments
        ]
        normalized_plans = [
            EditPlan.model_validate(
                {
                    **plan.model_dump(),
                    "main_range": cls._snap(plan.main_range, boundaries),
                    "hook_range": cls._snap(plan.hook_range, boundaries),
                    "payoff_range": cls._snap(plan.payoff_range, boundaries),
                    "teaser_range": (
                        cls._snap(plan.teaser_range, boundaries)
                        if plan.teaser_range is not None
                        else None
                    ),
                },
            )
            for plan in value.plans
        ]
        return EditorialMap(
            video_summary=value.video_summary,
            moments=normalized_moments,
            plans=normalized_plans,
        )

    @staticmethod
    def _snap(interval: TimeRange, boundaries: set[float]) -> TimeRange:
        values: list[float] = []
        for value in (interval.start_seconds, interval.end_seconds):
            nearest = min(boundaries, key=lambda boundary: abs(boundary - value))
            if abs(nearest - value) > 0.25:
                raise ModelOutputError("Editorial plan invented a transcript boundary")
            values.append(nearest)
        return TimeRange(start_seconds=values[0], end_seconds=values[1])

    @classmethod
    def _schema(cls, value: JsonValue, preserve_keys: bool = False) -> JsonValue:
        unsupported = {
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
        if isinstance(value, dict):
            return {
                key: cls._schema(item, preserve_keys=key in {"$defs", "properties"})
                for key, item in value.items()
                if preserve_keys or key not in unsupported
            }
        if isinstance(value, list):
            return [cls._schema(item) for item in value]
        return value
