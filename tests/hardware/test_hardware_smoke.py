"""Opt-in real model/API smoke; excluded from the offline default run."""

import os
from pathlib import Path

import pytest

from short_engine.pipeline import Engine


@pytest.mark.hardware
@pytest.mark.network
def test_real_apple_pipeline_produces_validated_render() -> None:
    if os.environ.get("SHORT_ENGINE_HARDWARE_SMOKE") != "1":
        pytest.skip("set SHORT_ENGINE_HARDWARE_SMOKE=1 to run real models")
    fixture = os.environ.get("SHORT_ENGINE_HARDWARE_FIXTURE")
    if not fixture:
        pytest.skip("set SHORT_ENGINE_HARDWARE_FIXTURE to a media path")

    result = Engine().run(fixture, clips=1, language="pt")

    assert result.renders
    assert all(Path(item).stat().st_size > 0 for item in result.renders)
