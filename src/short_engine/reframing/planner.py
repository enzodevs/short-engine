"""Bounded exponential crop planner."""

from short_engine.core.models import AspectRatio, TimeRange
from short_engine.reframing.models import CropPlan, CropSample, SubjectTrack


class CropPlanner:
    def __init__(self, smoothing: float = 0.82) -> None:
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing

    def plan(
        self,
        interval: TimeRange,
        track: SubjectTrack,
        frame_width: int,
        frame_height: int,
        aspect: AspectRatio,
    ) -> CropPlan:
        target_ratio = {
            AspectRatio.VERTICAL: 9 / 16,
            AspectRatio.SQUARE: 1.0,
            AspectRatio.LANDSCAPE: 16 / 9,
        }[aspect]
        if frame_width / frame_height >= target_ratio:
            crop_height = frame_height
            crop_width = round(frame_height * target_ratio)
        else:
            crop_width = frame_width
            crop_height = round(frame_width / target_ratio)
        observations = [
            item
            for item in track.observations
            if interval.start_seconds <= item.time_seconds <= interval.end_seconds
            and item.confidence >= 0.25
        ]
        if not observations:
            return CropPlan(
                crop_width=crop_width,
                crop_height=crop_height,
                samples=[
                    CropSample(
                        time_seconds=interval.start_seconds,
                        x=(frame_width - crop_width) / 2,
                        y=(frame_height - crop_height) / 2,
                    )
                ],
                used_fallback=True,
            )
        samples: list[CropSample] = []
        x = min(max(observations[0].center_x - crop_width / 2, 0), frame_width - crop_width)
        y = min(max(observations[0].center_y - crop_height / 2, 0), frame_height - crop_height)
        for observation in observations:
            target_x = min(max(observation.center_x - crop_width / 2, 0), frame_width - crop_width)
            target_y = min(
                max(observation.center_y - crop_height / 2, 0), frame_height - crop_height
            )
            x = self.smoothing * x + (1 - self.smoothing) * target_x
            y = self.smoothing * y + (1 - self.smoothing) * target_y
            samples.append(CropSample(time_seconds=observation.time_seconds, x=x, y=y))
        return CropPlan(
            crop_width=crop_width, crop_height=crop_height, samples=samples, used_fallback=False
        )
