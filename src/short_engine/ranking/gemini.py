"""Gemini structured-output ranker."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from short_engine.candidates.models import Candidate
from short_engine.core.errors import ModelOutputError
from short_engine.ranking.models import CandidateAssessment

GenerateFunction = Callable[..., Any]


class GeminiRanker:
    def __init__(
        self,
        api_key: str,
        model: str,
        generate: GenerateFunction | None = None,
        debug_directory: Path | None = None,
        attempts: int = 3,
    ) -> None:
        self.model = model
        self.debug_directory = debug_directory
        self.attempts = attempts
        if generate is None:
            from google import genai

            generate = genai.Client(api_key=api_key).models.generate_content
        self.generate = generate

    def rank(
        self, candidates: list[Candidate], evidence: dict[str, list[Path]]
    ) -> list[CandidateAssessment]:
        return [
            self._rank_one(candidate, evidence.get(candidate.id, [])) for candidate in candidates
        ]

    def _rank_one(self, candidate: Candidate, frames: list[Path]) -> CandidateAssessment:
        from google.genai import types

        contents: list[Any] = [
            "Score this short-video candidate. "
            f"ID: {candidate.id}\nTranscript: {candidate.transcript}"
        ]
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

    def _write_debug(self, candidate_id: str, response: str) -> None:
        if self.debug_directory is None:
            return
        self.debug_directory.mkdir(parents=True, exist_ok=True)
        (self.debug_directory / f"gemini-{candidate_id}.txt").write_text(response[:20_000])
