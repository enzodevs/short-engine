from pathlib import Path

import pytest
from pydantic import ValidationError

from short_engine.core.models import AspectRatio, RunRequest, TimeRange


def test_time_range_rejects_an_end_before_start() -> None:
    with pytest.raises(ValidationError, match="end_seconds must be greater"):
        TimeRange(start_seconds=4.0, end_seconds=3.0)


def test_run_request_normalizes_paths_without_touching_the_filesystem(tmp_path: Path) -> None:
    request = RunRequest(source=tmp_path / "video.mp4", clips=2, aspect=AspectRatio.VERTICAL)

    assert request.source == tmp_path / "video.mp4"
    assert request.clips == 2
    assert request.aspect.ffmpeg_value == "9/16"


def test_run_request_rejects_non_positive_clip_count() -> None:
    with pytest.raises(ValidationError):
        RunRequest(source=Path("video.mp4"), clips=0)
