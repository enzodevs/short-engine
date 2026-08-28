from pathlib import Path

import pytest

from short_engine.core.errors import InferenceError
from short_engine.transcription.mlx import MLXWhisperTranscriber
from short_engine.transcription.models import ASRConfig


def test_mlx_adapter_normalizes_segments_and_word_timestamps(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")

    def fake_transcribe(path: str, **options: object) -> dict[str, object]:
        assert path == str(audio)
        assert options["word_timestamps"] is True
        return {
            "language": "pt",
            "segments": [
                {
                    "start": 1.0,
                    "end": 3.0,
                    "text": " Olá mundo. ",
                    "words": [
                        {"start": 1.0, "end": 1.5, "word": " Olá", "probability": 0.9},
                        {"start": 1.6, "end": 3.0, "word": " mundo.", "probability": 0.8},
                    ],
                }
            ],
        }

    transcript = MLXWhisperTranscriber(fake_transcribe).transcribe(
        audio,
        ASRConfig(model="test-model", language="pt"),
    )

    assert transcript.language == "pt"
    assert transcript.model == "test-model"
    assert transcript.duration_seconds == 3.0
    assert transcript.segments[0].text == "Olá mundo."
    assert transcript.segments[0].words[1].text == "mundo."


def test_mlx_adapter_rejects_empty_speech(tmp_path: Path) -> None:
    audio = tmp_path / "silence.wav"
    audio.write_bytes(b"wav")

    with pytest.raises(InferenceError, match="no speech"):
        MLXWhisperTranscriber(lambda *_args, **_kwargs: {"segments": []}).transcribe(
            audio,
            ASRConfig(model="test-model"),
        )


def test_mlx_adapter_skips_degenerate_units_but_keeps_valid_speech(tmp_path: Path) -> None:
    raw = {
        "segments": [
            {"start": 0, "end": 0, "text": "invalid", "words": []},
            {
                "start": 1,
                "end": 2,
                "text": "fala válida",
                "words": [
                    {"start": 1, "end": 1, "word": "zero"},
                    {"start": 1.1, "end": 1.8, "word": "válida", "probability": 0.9},
                ],
            },
        ]
    }

    transcript = MLXWhisperTranscriber(lambda *_args, **_kwargs: raw).transcribe(
        tmp_path / "audio.wav", ASRConfig(model="test-model")
    )

    assert [segment.text for segment in transcript.segments] == ["fala válida"]
    assert [word.text for word in transcript.segments[0].words] == ["válida"]
