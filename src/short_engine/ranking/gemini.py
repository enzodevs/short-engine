"""Gemini structured-output ranker."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from google.genai import types
from pydantic import ValidationError

from short_engine.candidates.models import Candidate
from short_engine.core.errors import ModelOutputError
from short_engine.ranking.models import CandidateAssessment
from short_engine.ranking.video import GeminiVideoEvidence

GenerateFunction = Callable[..., Any]


class GeminiRanker:
    def __init__(
        self,
        api_key: str,
        model: str,
        generate: GenerateFunction | None = None,
        debug_directory: Path | None = None,
        assessment_directory: Path | None = None,
        attempts: int = 3,
    ) -> None:
        self.model = model
        self.debug_directory = debug_directory
        self.assessment_directory = assessment_directory
        self.attempts = attempts
        self._client: Any | None = None
        if generate is None:
            from google import genai

            self._client = genai.Client(api_key=api_key)
            generate = self._client.models.generate_content
        self.generate = generate

    def rank(
        self,
        candidates: list[Candidate],
        evidence: dict[str, list[Path]],
        video: Path | None = None,
    ) -> list[CandidateAssessment]:
        if video is not None:
            if self._client is None:
                raise ModelOutputError("Video evidence requires a Gemini client")
            with GeminiVideoEvidence(self._client, video) as remote:
                return [
                    self._cached_or_rank(
                        candidate,
                        evidence.get(candidate.id, []),
                        remote.part(candidate.time_range),
                    )
                    for candidate in candidates
                ]
        return [
            self._cached_or_rank(candidate, evidence.get(candidate.id, []))
            for candidate in candidates
        ]

    def _cached_or_rank(
        self, candidate: Candidate, frames: list[Path], video_part: types.Part | None = None
    ) -> CandidateAssessment:
        path = (
            self.assessment_directory / f"{candidate.id}.json"
            if self.assessment_directory is not None
            else None
        )
        if path is not None and path.is_file():
            return CandidateAssessment.model_validate_json(path.read_text())
        assessment = self._rank_one(candidate, frames, video_part)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(assessment.model_dump_json(indent=2))
        return assessment

    def _rank_one(
        self, candidate: Candidate, frames: list[Path], video_part: types.Part | None = None
    ) -> CandidateAssessment:
        contents: list[Any] = [self._editorial_prompt(candidate)]
        if video_part is not None:
            contents.append(video_part)
        contents.extend(
            types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg") for path in frames
        )
        last = ""
        for _ in range(self.attempts):
            response = self.generate(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=CandidateAssessment.model_json_schema(),
                    temperature=0.1,
                ),
            )
            last = response.text or ""
            try:
                parsed = CandidateAssessment.model_validate_json(last)
                if parsed.candidate_id != candidate.id:
                    raise ValueError("candidate id mismatch")
                return parsed
            except (ValidationError, ValueError, json.JSONDecodeError):
                continue
        self._write_debug(candidate.id, last)
        raise ModelOutputError(
            f"Gemini returned invalid ranking output for candidate {candidate.id}"
        )

    @staticmethod
    def _editorial_prompt(candidate: Candidate) -> str:
        return (
            "Act as a ruthless short-form retention editor, not a summarizer. Evaluate whether "
            "a cold viewer would stop scrolling and finish this exact clip. Inspect speech, "
            "visual action, pacing, energy, and whether the ending rewards the opening.\n\n"
            "Scoring rules:\n"
            "- hook_immediacy: first 1-2 seconds create conflict, surprise, stakes, a bold claim, "
            "or an irresistible unanswered question; greetings and setup score poorly.\n"
            "- curiosity_gap: opening creates a specific question the viewer needs resolved.\n"
            "- narrative_arc: clear hook -> escalation/evidence -> resolution, not merely "
            "an excerpt.\n"
            "- payoff_strength: ending delivers a concrete reveal, result, punchline, or insight.\n"
            "- emotional_intensity: visible or audible conviction, surprise, tension, humor, "
            "or awe.\n"
            "- visual_dynamism: visuals actively prove or advance the story rather than "
            "decorate it.\n"
            "- standalone_clarity: understandable without title, previous context, or source "
            "video.\n"
            "- rewatchability: density, reveal, or detail makes replay/share plausible.\n"
            "- retention_risk: score HIGH for slow setup, jargon, repetition, dead air, weak "
            "opening, "
            "missing context, or payoff that arrives too late. Be severe.\n\n"
            "Explicitly identify the hook, body, payoff, and single worst retention flaw. Do not "
            "reward production quality when the story itself is weak.\n"
            f"Candidate ID: {candidate.id}\n"
            f"Source interval: {candidate.time_range.start_seconds:.2f}-"
            f"{candidate.time_range.end_seconds:.2f}s\n"
            f"Transcript: {candidate.transcript}"
        )

    def _write_debug(self, candidate_id: str, response: str) -> None:
        if self.debug_directory is None:
            return
        self.debug_directory.mkdir(parents=True, exist_ok=True)
        (self.debug_directory / f"gemini-{candidate_id}.txt").write_text(response[:20_000])
