from pathlib import Path

import pytest

from short_engine.run.manifest import Artifact, ManifestStore, RunManifest, StageStatus


def test_stage_reuses_artifacts_when_fingerprint_and_files_match(tmp_path: Path) -> None:
    artifact_path = tmp_path / "analysis.json"
    store = ManifestStore(tmp_path / "manifest.json")
    store.initialize(RunManifest(run_id="run-1", source="video.mp4"))
    calls = 0

    def action() -> list[Artifact]:
        nonlocal calls
        calls += 1
        artifact_path.write_text("ok", encoding="utf-8")
        return [Artifact.from_path(artifact_path, kind="analysis")]

    first = store.execute("analyze", "fingerprint-a", action)
    second = store.execute("analyze", "fingerprint-a", action)

    assert calls == 1
    assert first.reused is False
    assert second.reused is True
    assert second.artifacts[0].sha256 == first.artifacts[0].sha256


def test_stage_reruns_when_material_fingerprint_changes(tmp_path: Path) -> None:
    artifact_path = tmp_path / "analysis.json"
    store = ManifestStore(tmp_path / "manifest.json")
    store.initialize(RunManifest(run_id="run-1", source="video.mp4"))
    calls = 0

    def action() -> list[Artifact]:
        nonlocal calls
        calls += 1
        artifact_path.write_text(str(calls), encoding="utf-8")
        return [Artifact.from_path(artifact_path, kind="analysis")]

    store.execute("analyze", "fingerprint-a", action)
    outcome = store.execute("analyze", "fingerprint-b", action)

    assert calls == 2
    assert outcome.reused is False


def test_failed_stage_is_persisted_and_reraised(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path / "manifest.json")
    store.initialize(RunManifest(run_id="run-1", source="video.mp4"))

    with pytest.raises(RuntimeError, match="boom"):
        store.execute(
            "analyze", "fingerprint-a", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )

    manifest = store.load()
    assert manifest.stages["analyze"].status is StageStatus.FAILED
    assert manifest.stages["analyze"].error == "boom"
