from pathlib import Path

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
        source, tmp_path / "short.mp4", TimeRange(start_seconds=0, end_seconds=0.8), plan
    )
    probe = runner.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,width,height",
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

    expression = FFmpegRenderer._motion_expression(
        plan, TimeRange(start_seconds=10, end_seconds=12), "x"
    )

    assert "t-0.000" in expression
    assert "100.00" in expression
    assert "300.0" in expression
