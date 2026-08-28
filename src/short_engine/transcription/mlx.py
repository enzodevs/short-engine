"""MLX Whisper adapter for Apple Silicon."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from short_engine.core.errors import DependencyError, InferenceError
from short_engine.system.process import resolve_executable
from short_engine.transcription.models import ASRConfig, TimedWord, Transcript, TranscriptSegment

TranscribeFunction = Callable[..., dict[str, Any]]


def _default_transcribe(path: str, **options: object) -> dict[str, Any]:
    try:
        import mlx_whisper
    except ImportError as error:
        raise DependencyError("Install the mac extra to use MLX Whisper") from error
    transcribe = cast(TranscribeFunction, mlx_whisper.transcribe)
    original_path = os.environ.get("PATH", "")
    ffmpeg_directory = str(Path(resolve_executable("ffmpeg")).parent)
    os.environ["PATH"] = f"{ffmpeg_directory}{os.pathsep}{original_path}"
    try:
        return transcribe(path, **options)
    finally:
        os.environ["PATH"] = original_path


class MLXWhisperTranscriber:
    def __init__(self, transcribe_fn: TranscribeFunction = _default_transcribe) -> None:
        self.transcribe_fn = transcribe_fn

    def transcribe(self, audio: Path, config: ASRConfig) -> Transcript:
        options: dict[str, object] = {
            "path_or_hf_repo": config.model,
            "word_timestamps": config.word_timestamps,
        }
        if config.language:
            options["language"] = config.language
        try:
            raw = self.transcribe_fn(str(audio), **options)
        except (KeyError, TypeError, ValueError) as error:
            raise InferenceError("MLX Whisper returned malformed transcription data") from error
        segments: list[TranscriptSegment] = []
        for item in raw.get("segments", []):
            try:
                segments.append(self._segment(item))
            except (KeyError, TypeError, ValueError):
                continue
        if not segments:
            raise InferenceError("MLX Whisper detected no speech")
        language = str(raw.get("language") or config.language or "unknown")
        return Transcript(
            language=language,
            model=config.model,
            duration_seconds=max(segment.end_seconds for segment in segments),
            segments=segments,
        )

    @staticmethod
    def _segment(raw: dict[str, Any]) -> TranscriptSegment:
        words: list[TimedWord] = []
        for word in raw.get("words", []):
            try:
                text = str(word.get("word", "")).strip()
                if not text:
                    continue
                words.append(
                    TimedWord(
                        start_seconds=float(word["start"]),
                        end_seconds=float(word["end"]),
                        text=text,
                        confidence=float(word["probability"])
                        if word.get("probability") is not None
                        else None,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return TranscriptSegment(
            start_seconds=float(raw["start"]),
            end_seconds=float(raw["end"]),
            text=str(raw["text"]).strip(),
            words=words,
        )
