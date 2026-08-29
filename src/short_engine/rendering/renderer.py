"""Atomic FFmpeg clip renderer."""

from itertools import pairwise
from pathlib import Path

from short_engine.core.errors import RenderError
from short_engine.core.models import TimeRange
from short_engine.reframing.curves import MotionCurve, SmootherstepCurve
from short_engine.reframing.models import CropPlan, CropSample
from short_engine.system.process import CommandRunner, SubprocessRunner


class FFmpegRenderer:
    def __init__(
        self,
        runner: CommandRunner | None = None,
        motion_curve: MotionCurve | None = None,
        max_transition_seconds: float = 2.0,
        camera_speed_pixels_per_second: float = 280.0,
    ) -> None:
        self.runner = runner or SubprocessRunner()
        self.motion_curve = motion_curve or SmootherstepCurve()
        self.max_transition_seconds = max_transition_seconds
        self.camera_speed_pixels_per_second = camera_speed_pixels_per_second

    def render(
        self,
        source: Path,
        output: Path,
        interval: TimeRange,
        crop: CropPlan,
        captions: Path | None = None,
        edits: list[TimeRange] | None = None,
    ) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".partial.mp4")
        ratio = crop.crop_width / crop.crop_height
        if ratio < 0.8:
            output_width, output_height = 1080, 1920
        elif ratio < 1.2:
            output_width, output_height = 1080, 1080
        else:
            output_width, output_height = 1920, 1080
        edit_ranges = self._split_at_hard_cuts(edits or [interval], crop.hard_cuts_seconds)
        chains: list[str] = []
        concat_inputs: list[str] = []
        for index, edit in enumerate(edit_ranges):
            video_filters = [
                f"trim=start={edit.start_seconds:.3f}:end={edit.end_seconds:.3f}",
                "setpts=PTS-STARTPTS",
                (
                    f"crop={crop.crop_width}:{crop.crop_height}:"
                    f"x='{self._motion_expression(crop, edit, 'x')}':"
                    f"y='{self._motion_expression(crop, edit, 'y')}'"
                ),
                f"scale={output_width}:{output_height}:force_original_aspect_ratio=decrease",
                f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2",
            ]
            chains.append(f"[0:v]{','.join(video_filters)}[v{index}]")
            chains.append(
                f"[0:a]atrim=start={edit.start_seconds:.3f}:end={edit.end_seconds:.3f},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
            concat_inputs.extend([f"[v{index}]", f"[a{index}]"])
        chains.append(f"{''.join(concat_inputs)}concat=n={len(edit_ranges)}:v=1:a=1[cv][ca]")
        if captions:
            escaped = str(captions).replace("'", r"\'").replace(":", r"\:")
            chains.append(f"[cv]ass='{escaped}'[vout]")
        else:
            chains.append("[cv]null[vout]")
        chains.append("[ca]loudnorm=I=-16:TP=-1.5:LRA=11[aout]")
        args = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            ";".join(chains),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        result = self.runner.run(args)
        if result.returncode != 0:
            raise RenderError(f"FFmpeg render failed: {result.stderr[-500:]}")
        temporary.replace(output)
        return output

    def _motion_expression(self, crop: CropPlan, interval: TimeRange, axis: str) -> str:
        samples = [
            sample
            for sample in crop.samples
            if interval.start_seconds <= sample.time_seconds <= interval.end_seconds
        ]
        if not samples:
            midpoint = (interval.start_seconds + interval.end_seconds) / 2
            samples = [min(crop.samples, key=lambda sample: abs(sample.time_seconds - midpoint))]
        if len(samples) == 1:
            return str(round(getattr(samples[0], axis), 2))
        reduced = self._control_samples(samples, axis)
        if len(reduced) == 1:
            return str(round(getattr(reduced[0], axis), 2))
        expression = str(round(getattr(reduced[-1], axis), 2))
        for left, right in reversed(list(pairwise(reduced))):
            end = max(0.001, right.time_seconds - interval.start_seconds)
            available = max(0.001, right.time_seconds - left.time_seconds)
            distance = abs(getattr(right, axis) - getattr(left, axis))
            natural_duration = max(0.45, distance / self.camera_speed_pixels_per_second)
            duration = min(self.max_transition_seconds, available, natural_duration)
            start = max(0.0, end - duration)
            origin = getattr(left, axis)
            delta = getattr(right, axis) - origin
            progress = f"(t-{start:.3f})/{duration:.3f}"
            eased = self.motion_curve.ffmpeg(progress)
            motion = f"{origin:.2f}+({delta:.2f})*({eased})"
            expression = (
                f"if(lt(t\\,{start:.3f})\\,{origin:.2f}\\,"
                f"if(lt(t\\,{end:.3f})\\,{motion}\\,{expression}))"
            )
        return expression

    @staticmethod
    def _control_samples(samples: list[CropSample], axis: str) -> list[CropSample]:
        if len(samples) < 2:
            return samples
        controls = [samples[0]]
        direction = 0
        previous = samples[0]
        for sample in samples[1:]:
            step = getattr(sample, axis) - getattr(previous, axis)
            step_direction = 1 if step >= 8 else -1 if step <= -8 else 0
            if step_direction and direction and step_direction != direction:
                if abs(getattr(previous, axis) - getattr(controls[-1], axis)) >= 24:
                    controls.append(previous)
                direction = step_direction
            elif step_direction:
                direction = step_direction
            previous = sample
        if abs(getattr(samples[-1], axis) - getattr(controls[-1], axis)) >= 8:
            controls.append(samples[-1])
        return controls

    @staticmethod
    def _split_at_hard_cuts(
        intervals: list[TimeRange], hard_cuts_seconds: list[float]
    ) -> list[TimeRange]:
        result: list[TimeRange] = []
        for interval in intervals:
            points = [
                interval.start_seconds,
                *(
                    cut
                    for cut in hard_cuts_seconds
                    if interval.start_seconds < cut < interval.end_seconds
                ),
                interval.end_seconds,
            ]
            result.extend(
                TimeRange(start_seconds=start, end_seconds=end)
                for start, end in pairwise(sorted(set(points)))
            )
        return result
