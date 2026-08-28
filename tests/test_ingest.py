import json
from pathlib import Path

import pytest

from short_engine.core.errors import InputError
from short_engine.ingest.media import FFmpegMediaService
from short_engine.ingest.models import SourceRequest
from short_engine.ingest.probe import FFprobe
from short_engine.ingest.resolver import SourceResolver
from short_engine.system.process import CommandResult, SubprocessRunner


class RecordingRunner:
    def __init__(self, result: CommandResult | None = None) -> None:
        self.calls: list[list[str]] = []
        self.result = result or CommandResult(args=[], returncode=0, stdout="", stderr="")

    def run(self, args: list[str]) -> CommandResult:
        self.calls.append(args)
        return self.result


def test_local_source_is_fingerprinted_and_kept_in_place(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")

    asset = SourceResolver(RecordingRunner()).resolve(
        SourceRequest(source=str(source)), tmp_path / "run"
    )

    assert asset.path == source.resolve()
    assert len(asset.source_fingerprint) == 64
    assert asset.downloaded is False


def test_missing_local_source_fails_before_running_external_commands(tmp_path: Path) -> None:
    runner = RecordingRunner()

    with pytest.raises(InputError, match="does not exist"):
        SourceResolver(runner).resolve(
            SourceRequest(source=str(tmp_path / "missing.mp4")), tmp_path / "run"
        )

    assert runner.calls == []


def test_url_source_delegates_auth_to_yt_dlp_without_raw_cookies(tmp_path: Path) -> None:
    output = tmp_path / "run" / "source" / "source_abc.mp4"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"downloaded")
    runner = RecordingRunner(CommandResult(args=[], returncode=0, stdout=str(output), stderr=""))

    asset = SourceResolver(runner).resolve(
        SourceRequest(
            source="https://youtu.be/abc",
            cookies_from_browser="chrome:Profile 3",
        ),
        tmp_path / "run",
    )

    command = runner.calls[0]
    assert command[0] == "yt-dlp"
    assert "--cookies-from-browser" in command
    assert "chrome:Profile 3" in command
    assert "--remote-components" in command
    assert asset.path == output
    assert asset.downloaded is True


def test_ffprobe_maps_streams_to_typed_media_probe() -> None:
    payload = {
        "format": {"duration": "12.5"},
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080, "avg_frame_rate": "30/1"},
            {"codec_type": "audio", "sample_rate": "48000"},
        ],
    }
    runner = RecordingRunner(
        CommandResult(args=[], returncode=0, stdout=json.dumps(payload), stderr="")
    )

    probe = FFprobe(runner).inspect(Path("video.mp4"))

    assert probe.duration_seconds == 12.5
    assert probe.width == 1920
    assert probe.height == 1080
    assert probe.frames_per_second == 30.0
    assert probe.has_audio is True


def test_real_ffmpeg_extracts_analysis_audio_and_proxy(tmp_path: Path) -> None:
    runner = SubprocessRunner()
    source = tmp_path / "fixture.mp4"
    generated = runner.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=640x360:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ]
    )
    assert generated.returncode == 0

    service = FFmpegMediaService(runner)
    audio = service.extract_analysis_audio(source, tmp_path / "audio.wav")
    proxy = service.create_proxy(source, tmp_path / "proxy.mp4", max_width=320)
    audio_probe = FFprobe(runner).inspect(audio)
    proxy_probe = FFprobe(runner).inspect(proxy)

    assert audio_probe.has_audio is True
    assert audio_probe.has_video is False
    assert proxy_probe.width == 320
    assert proxy_probe.has_video is True
