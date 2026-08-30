"""Detect persistent peripheral facecam overlays and plan a stacked layout."""

from statistics import median

from short_engine.reframing.models import PanelCrop, ReactionLayout, SubjectTrack


class ReactionLayoutDetector:
    def __init__(
        self,
        min_observations: int = 10,
        min_corner_presence: float = 0.70,
        max_area_ratio: float = 0.30,
        padding_ratio: float = 0.12,
    ) -> None:
        self.min_observations = min_observations
        self.min_corner_presence = min_corner_presence
        self.max_area_ratio = max_area_ratio
        self.padding_ratio = padding_ratio

    def detect(
        self, track: SubjectTrack, frame_width: int, frame_height: int
    ) -> ReactionLayout | None:
        observations = []
        boxes: dict[int, tuple[float, float, float, float]] = {}
        for item in track.observations:
            if (
                item.left_x is None
                or item.top_y is None
                or item.right_x is None
                or item.bottom_y is None
            ):
                continue
            observations.append(item)
            boxes[id(item)] = (item.left_x, item.top_y, item.right_x, item.bottom_y)
        if len(observations) < self.min_observations:
            return None
        center_x = median(item.center_x for item in observations)
        center_y = median(item.center_y for item in observations)
        left_side = center_x <= frame_width * 0.5
        corner = [
            item
            for item in observations
            if item.center_y >= frame_height * 0.35
            and (
                item.center_x <= frame_width * 0.32
                if left_side
                else item.center_x >= frame_width * 0.68
            )
        ]
        if (
            center_y < frame_height * 0.35
            or len(corner) / len(observations) < self.min_corner_presence
        ):
            return None
        corner_boxes = [boxes[id(item)] for item in corner]
        areas = [
            (box[2] - box[0]) * (box[3] - box[1]) / (frame_width * frame_height)
            for box in corner_boxes
        ]
        if median(areas) > self.max_area_ratio:
            return None
        left = self._percentile([box[0] for box in corner_boxes], 0.05)
        top = self._percentile([box[1] for box in corner_boxes], 0.05)
        right = self._percentile([box[2] for box in corner_boxes], 0.95)
        bottom = self._percentile([box[3] for box in corner_boxes], 0.95)
        facecam = self._aspect_crop(
            left,
            top,
            right,
            bottom,
            frame_width,
            frame_height,
            target_ratio=1080 / 640,
            padding=self.padding_ratio,
        )
        content_width = min(frame_width, round(frame_height * (1080 / 1280)))
        content = PanelCrop(
            x=round((frame_width - content_width) / 2),
            y=0,
            width=content_width,
            height=frame_height,
        )
        return ReactionLayout(facecam=facecam, content=content)

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        return ordered[round((len(ordered) - 1) * fraction)]

    @staticmethod
    def _aspect_crop(
        left: float,
        top: float,
        right: float,
        bottom: float,
        frame_width: int,
        frame_height: int,
        target_ratio: float,
        padding: float,
    ) -> PanelCrop:
        width = (right - left) * (1 + 2 * padding)
        height = (bottom - top) * (1 + 2 * padding)
        if width / height < target_ratio:
            width = height * target_ratio
        else:
            height = width / target_ratio
        width = min(width, frame_width)
        height = min(height, frame_height)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        x = min(max(center_x - width / 2, 0), frame_width - width)
        y = min(max(center_y - height / 2, 0), frame_height - height)
        return PanelCrop(x=round(x), y=round(y), width=round(width), height=round(height))
