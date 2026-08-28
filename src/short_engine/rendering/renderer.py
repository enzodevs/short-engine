"""Atomic FFmpeg clip renderer."""

from itertools import pairwise
from pathlib import Path

from short_engine.core.errors import RenderError
from short_engine.core.models import TimeRange
from short_engine.reframing.models import CropPlan
from short_engine.system.process import CommandRunner, SubprocessRunner


class FFmpegRenderer:
    def __init__(self, runner: CommandRunner | None = None) -> None:
        self.runner = runner or SubprocessRunner()

    def render(
        self,
        source: Path,
        output: Path,
        interval: TimeRange,
        crop: CropPlan,
        captions: Path | None = None,
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
        filters = [
            (
                f"crop={crop.crop_width}:{crop.crop_height}:"
                f"x='{self._motion_expression(crop, interval, 'x')}':"
                f"y='{self._motion_expression(crop, interval, 'y')}'"
            ),
            f"scale={output_width}:{output_height}:force_original_aspect_ratio=decrease",
            f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2",
        ]
        if captions:
            escaped = str(captions).replace("'", r"\'").replace(":", r"\:")
            filters.append(f"ass='{escaped}'")
        args = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{interval.start_seconds:.3f}",
            "-i",
            str(source),
            "-t",
            f"{interval.duration_seconds:.3f}",
            "-vf",
            ",".join(filters),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
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

    @staticmethod
    def _motion_expression(crop: CropPlan, interval: TimeRange, axis: str) -> str:
        samples = crop.samples
        if len(samples) == 1:
            return str(round(getattr(samples[0], axis), 2))
        reduced = [samples[0]]
        for sample in samples[1:-1]:
            if sample.time_seconds - reduced[-1].time_seconds >= 0.75:
                reduced.append(sample)
        reduced.append(samples[-1])
        expression = str(round(getattr(reduced[-1], axis), 2))
        for left, right in reversed(list(pairwise(reduced))):
            start = max(0.0, left.time_seconds - interval.start_seconds)
            end = max(start + 0.001, right.time_seconds - interval.start_seconds)
            origin = getattr(left, axis)
            delta = getattr(right, axis) - origin
            linear = f"{origin:.2f}+({delta:.2f})*(t-{start:.3f})/{end - start:.3f}"
            expression = f"if(lt(t\\,{end:.3f})\\,{linear}\\,{expression})"
        return expression
