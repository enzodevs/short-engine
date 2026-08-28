"""Ultralytics MPS subject tracking adapter."""

from pathlib import Path
from typing import Any

from short_engine.core.errors import DependencyError
from short_engine.core.models import TimeRange
from short_engine.reframing.models import SubjectObservation, SubjectTrack


class UltralyticsSubjectTracker:
    def __init__(self, model: str = "yolo26n.pt", confidence: float = 0.3) -> None:
        self.model = model
        self.confidence = confidence

    def track(self, source: Path, interval: TimeRange) -> SubjectTrack:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise DependencyError("Install the mac extra to use subject tracking") from error
        model = YOLO(self.model)
        import cv2

        capture = cv2.VideoCapture(str(source))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        capture.release()
        observations: list[SubjectObservation] = []
        results: Any = model.track(
            source=str(source),
            stream=True,
            device="mps",
            conf=self.confidence,
            classes=[0],
            vid_stride=5,
            verbose=False,
        )
        for index, result in enumerate(results):
            timestamp = index * 5 / fps
            if timestamp > interval.end_seconds or result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            best = max(
                range(len(boxes)),
                key=lambda item: (
                    confidences[item]
                    * (boxes[item][2] - boxes[item][0])
                    * (boxes[item][3] - boxes[item][1])
                ),
            )
            x1, y1, x2, y2 = boxes[best]
            observations.append(
                SubjectObservation(
                    time_seconds=timestamp,
                    center_x=(x1 + x2) / 2,
                    center_y=(y1 + y2) / 2,
                    confidence=float(confidences[best]),
                )
            )
        return SubjectTrack(observations=observations)
