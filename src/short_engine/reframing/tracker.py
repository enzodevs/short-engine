"""Ultralytics MPS subject tracking adapter."""

from pathlib import Path
from typing import Any

from short_engine.core.errors import DependencyError
from short_engine.core.models import TimeRange
from short_engine.reframing.models import SubjectObservation, SubjectTrack


class UltralyticsSubjectTracker:
    def __init__(self, model: str | Path = "yolo26n.pt", confidence: float = 0.3) -> None:
        self.model = model
        self.confidence = confidence

    def track(
        self,
        source: Path,
        interval: TimeRange,
        scene_boundaries: list[float] | None = None,
    ) -> SubjectTrack:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise DependencyError("Install the mac extra to use subject tracking") from error
        model_path = Path(self.model)
        if model_path.is_absolute():
            model_path.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(model_path))
        import cv2

        capture = cv2.VideoCapture(str(source))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        capture.set(cv2.CAP_PROP_POS_MSEC, interval.start_seconds * 1000)
        observations: list[SubjectObservation] = []
        boundaries = iter(sorted(scene_boundaries or []))
        next_boundary = next(boundaries, None)
        frame_index = round(interval.start_seconds * fps)
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break
            timestamp = frame_index / fps
            frame_index += 1
            if timestamp > interval.end_seconds:
                break
            if frame_index % 5:
                continue
            if next_boundary is not None and timestamp >= next_boundary:
                predictor: Any = getattr(model, "predictor", None)
                for tracker in getattr(predictor, "trackers", []):
                    tracker.reset()
                next_boundary = next(boundaries, None)
            result: Any = model.track(
                frame,
                persist=True,
                device="mps",
                conf=self.confidence,
                classes=[0],
                verbose=False,
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
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
        capture.release()
        return SubjectTrack(observations=observations)
