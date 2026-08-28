"""PySceneDetect and Silero VAD adapters."""

import sys
import wave
from array import array
from pathlib import Path

from short_engine.core.errors import DependencyError, InferenceError
from short_engine.segmentation.models import BoundaryKind, DetectedBoundary


class SceneDetector:
    def detect(self, video: Path) -> list[DetectedBoundary]:
        try:
            from scenedetect import AdaptiveDetector, detect
        except ImportError as error:
            raise DependencyError("Install the mac extra to use scene detection") from error
        scenes = detect(str(video), AdaptiveDetector())
        return [
            DetectedBoundary(at_seconds=float(start.get_seconds()), kind=BoundaryKind.SCENE)
            for start, _end in scenes[1:]
        ]


class SileroSpeechDetector:
    def detect(self, audio: Path) -> list[DetectedBoundary]:
        try:
            import torch
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as error:
            raise DependencyError("Install the mac extra to use Silero VAD") from error
        try:
            model = load_silero_vad(onnx=True)
            with wave.open(str(audio), "rb") as source:
                if source.getnchannels() != 1 or source.getframerate() != 16_000:
                    raise ValueError("Silero input must be mono PCM at 16 kHz")
                samples = array("h", source.readframes(source.getnframes()))
            if sys.byteorder != "little":
                samples.byteswap()
            waveform = torch.tensor(samples, dtype=torch.float32).div_(32768.0)
            regions = get_speech_timestamps(waveform, model, return_seconds=True)
        except Exception as error:
            raise InferenceError("Silero VAD failed") from error
        boundaries: list[DetectedBoundary] = []
        for region in regions:
            boundaries.extend(
                [
                    DetectedBoundary(
                        at_seconds=float(region["start"]),
                        kind=BoundaryKind.SPEECH,
                    ),
                    DetectedBoundary(
                        at_seconds=float(region["end"]),
                        kind=BoundaryKind.SPEECH,
                    ),
                ]
            )
        return boundaries
