"""PySceneDetect and Silero VAD adapters."""

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
            from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
        except ImportError as error:
            raise DependencyError("Install the mac extra to use Silero VAD") from error
        try:
            model = load_silero_vad(onnx=True)
            waveform = read_audio(str(audio))
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
