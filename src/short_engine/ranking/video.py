"""Gemini Files API adapter for temporal video evidence."""

import time
from pathlib import Path

from google.genai import Client, types

from short_engine.core.errors import ModelOutputError
from short_engine.core.models import TimeRange


class GeminiVideoEvidence:
    def __init__(
        self,
        client: Client,
        source: Path,
        poll_seconds: float = 2.0,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.client = client
        self.source = source
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.file: types.File | None = None

    def __enter__(self) -> "GeminiVideoEvidence":
        self.file = self.client.files.upload(
            file=self.source,
            config={"mime_type": "video/mp4", "display_name": self.source.name},
        )
        deadline = time.monotonic() + self.timeout_seconds
        while self._state(self.file) == "PROCESSING":
            if time.monotonic() >= deadline:
                raise ModelOutputError("Gemini video processing timed out")
            time.sleep(self.poll_seconds)
            if not self.file.name:
                raise ModelOutputError("Gemini video upload returned no file name")
            self.file = self.client.files.get(name=self.file.name)
        if self._state(self.file) != "ACTIVE":
            raise ModelOutputError(f"Gemini video processing failed: {self._state(self.file)}")
        return self

    def __exit__(self, *_: object) -> None:
        if self.file is not None and self.file.name:
            self.client.files.delete(name=self.file.name)

    def part(self, interval: TimeRange, fps: float = 2.0) -> types.Part:
        if self.file is None or not self.file.uri or not self.file.mime_type:
            raise ModelOutputError("Gemini video evidence is not active")
        return types.Part(
            file_data=types.FileData(file_uri=self.file.uri, mime_type=self.file.mime_type),
            video_metadata=types.VideoMetadata(
                start_offset=f"{interval.start_seconds:.3f}s",
                end_offset=f"{interval.end_seconds:.3f}s",
                fps=fps,
            ),
        )

    @staticmethod
    def _state(file: types.File) -> str:
        state = file.state
        return str(state.value if state is not None else "").upper().split(".")[-1]
