import json
from pathlib import Path

import pytest

from short_engine.core.models import TimeRange
from short_engine.reframing.models import CropPlan, CropSample
from short_engine.rendering.captions import AssCaptionWriter
from short_engine.rendering.renderer import FFmpegRenderer
from short_engine.system.process import SubprocessRunner
from short_engine.transcription.models import TimedWord


def test_ass_captions_escape_text_and_shift_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "captions.ass"
    AssCaptionWriter().write(
        path, [TimedWord(start_seconds=12, end_seconds=13, text="Olá {mundo}")], offset_seconds=10
    )
    content = path.read_text()
    assert "0:00:02.00,0:00:03.00" in content
    assert r"OLÁ \{MUNDO\}" in content
    assert "Arial Black" in content
    assert ",-1,0,0,0," in content


def test_ass_captions_highlight_active_word_in_short_chunks(tmp_path: Path) -> None:
    path = tmp_path / "karaoke.ass"
    words = [
        TimedWord(start_seconds=index, end_seconds=index + 0.8, text=text)
        for index, text in enumerate(["Uma", "legenda", "forte", "agora"])
    ]

    AssCaptionWriter(words_per_chunk=4).write(path, words)

    content = path.read_text()
    assert content.count("Dialogue:") == 4
    assert r"{\c&H003BEBFF&\fscx112\fscy112}" in content
    assert "UMA LEGENDA FORTE AGORA" in content.replace(
        r"{\c&H003BEBFF&\fscx112\fscy112}", ""
    ).replace(r"{\r}", "")


def test_ffmpeg_renderer_emits_vertical_h264_aac(tmp_path: Path) -> None:
    runner = SubprocessRunner()
    source = tmp_path / "source.mp4"
    created = runner.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=15:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ]
    )
    assert created.returncode == 0
    plan = CropPlan(
        crop_width=203,
        crop_height=360,
        samples=[
            CropSample(time_seconds=0, x=180, y=0),
            CropSample(time_seconds=0.8, x=220, y=0),
        ],
        used_fallback=False,
    )
    output = FFmpegRenderer(runner).render(
        source,
        tmp_path / "short.mp4",
        TimeRange(start_seconds=0, end_seconds=0.8),
        plan,
        edits=[
            TimeRange(start_seconds=0, end_seconds=0.3),
            TimeRange(start_seconds=0.5, end_seconds=0.8),
        ],
    )
    probe = runner.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_name,width,height",
            "-of",
            "json",
            str(output),
        ]
    )
    assert probe.returncode == 0
    assert '"width": 1080' in probe.stdout
    assert '"height": 1920' in probe.stdout
    assert '"codec_name": "h264"' in probe.stdout
    assert '"codec_name": "aac"' in probe.stdout
    assert float(json.loads(probe.stdout)["format"]["duration"]) == pytest.approx(0.6, abs=0.08)


def test_motion_expression_interpolates_crop_samples() -> None:
    plan = CropPlan(
        crop_width=400,
        crop_height=700,
        samples=[
            CropSample(time_seconds=10, x=100, y=0),
            CropSample(time_seconds=11, x=200, y=0),
            CropSample(time_seconds=12, x=300, y=0),
        ],
        used_fallback=False,
    )

    expression = FFmpegRenderer()._motion_expression(
        plan, TimeRange(start_seconds=10, end_seconds=12), "x"
    )

    assert "t-1.286" in expression
    assert "100.00" in expression
    assert "300.0" in expression
    assert "*(10-15*" in expression


def test_motion_expression_ignores_micro_adjustments() -> None:
    plan = CropPlan(
        crop_width=400,
        crop_height=700,
        samples=[
            CropSample(time_seconds=10, x=100, y=0),
            CropSample(time_seconds=11, x=104, y=0),
            CropSample(time_seconds=12, x=106, y=0),
        ],
        used_fallback=False,
    )

    expression = FFmpegRenderer()._motion_expression(
        plan, TimeRange(start_seconds=10, end_seconds=12), "x"
    )

    assert expression == "100.0"


def test_renderer_splits_camera_at_hard_scene_cuts() -> None:
    segments = FFmpegRenderer._split_at_hard_cuts(
        [TimeRange(start_seconds=10, end_seconds=20)], [12, 17]
    )

    assert [(item.start_seconds, item.end_seconds) for item in segments] == [
        (10, 12),
        (12, 17),
        (17, 20),
    ]


def test_renderer_consolidates_monotonic_tracking_into_one_camera_move() -> None:
    samples = [
        CropSample(time_seconds=index, x=value, y=0)
        for index, value in enumerate([100, 130, 170, 240, 310])
    ]

    controls = FFmpegRenderer._control_samples(samples, "x")

    assert controls == [samples[0], samples[-1]]


def test_renderer_preserves_real_direction_change() -> None:
    samples = [
        CropSample(time_seconds=index, x=value, y=0)
        for index, value in enumerate([100, 180, 260, 160, 80])
    ]

    controls = FFmpegRenderer._control_samples(samples, "x")

    assert controls == [samples[0], samples[2], samples[-1]]
