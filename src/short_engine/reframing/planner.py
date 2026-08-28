"""Scene-aware comfort camera crop planner."""

from statistics import median

from short_engine.core.models import AspectRatio, TimeRange
from short_engine.reframing.models import CropPlan, CropSample, SubjectTrack


class CropPlanner:
    def __init__(
        self,
        smoothing: float = 0.82,
        stationary_threshold: float = 0.10,
        max_step_ratio: float = 0.018,
        dead_zone_ratio: float = 0.035,
    ) -> None:
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing
        self.stationary_threshold = stationary_threshold
        self.max_step_ratio = max_step_ratio
        self.dead_zone_ratio = dead_zone_ratio

    def plan(
        self,
        interval: TimeRange,
        track: SubjectTrack,
        frame_width: int,
        frame_height: int,
        aspect: AspectRatio,
        takes: list[TimeRange] | None = None,
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
        active_takes = takes or [interval]
        observations = sorted(
            (
                item
                for item in track.observations
                if any(
                    take.start_seconds <= item.time_seconds <= take.end_seconds
                    for take in active_takes
                )
                and item.confidence >= 0.25
            ),
            key=lambda item: item.time_seconds,
        )
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
        for take in active_takes:
            take_observations = [
                item
                for item in observations
                if take.start_seconds <= item.time_seconds <= take.end_seconds
            ]
            scene_ids = list(dict.fromkeys(item.scene_id for item in take_observations))
            for scene_id in scene_ids:
                scene = [item for item in take_observations if item.scene_id == scene_id]
                raw_x = [
                    min(max(item.center_x - crop_width / 2, 0), frame_width - crop_width)
                    for item in scene
                ]
                raw_y = [
                    min(max(item.center_y - crop_height / 2, 0), frame_height - crop_height)
                    for item in scene
                ]
                x_values = self._comfort_path(raw_x, frame_width, crop_width)
                y_values = self._comfort_path(raw_y, frame_height, crop_height)
                samples.extend(
                    CropSample(time_seconds=item.time_seconds, x=x, y=y)
                    for item, x, y in zip(scene, x_values, y_values, strict=True)
                )
        return CropPlan(
            crop_width=crop_width, crop_height=crop_height, samples=samples, used_fallback=False
        )

    def _comfort_path(self, values: list[float], frame_size: int, crop_size: int) -> list[float]:
        if len(values) < 2:
            return values
        available = max(1, frame_size - crop_size)
        if (max(values) - min(values)) / available <= self.stationary_threshold:
            locked = median(values)
            return [locked] * len(values)

        radius = min(5, max(1, len(values) // 4))
        smoothed = self._weighted_average(values, radius)
        smoothed = list(reversed(self._weighted_average(list(reversed(smoothed)), radius)))
        smoothed = [
            self.smoothing * smooth + (1 - self.smoothing) * raw
            for raw, smooth in zip(values, smoothed, strict=True)
        ]
        dead_zone = available * self.dead_zone_ratio
        intentional = [smoothed[0]]
        target = smoothed[0]
        for value in smoothed[1:]:
            if abs(value - target) >= dead_zone:
                target = value
            intentional.append(target)
        max_step = frame_size * self.max_step_ratio
        limited: list[float] = [float(intentional[0])]
        for value in intentional[1:]:
            delta = min(max(value - limited[-1], -max_step), max_step)
            limited.append(limited[-1] + delta)
        return limited

    @staticmethod
    def _weighted_average(values: list[float], radius: int) -> list[float]:
        result: list[float] = []
        for index in range(len(values)):
            start = max(0, index - radius)
            end = min(len(values), index + radius + 1)
            weights = [radius + 1 - abs(position - index) for position in range(start, end)]
            result.append(
                sum(
                    values[position] * weight
                    for position, weight in zip(range(start, end), weights, strict=True)
                )
                / sum(weights)
            )
        return result
