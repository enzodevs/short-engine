"""Durable, resumable state for one engine run."""

import hashlib
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(UTC)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class Artifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    kind: str
    sha256: str
    size_bytes: int = Field(ge=0)

    @classmethod
    def from_path(cls, path: Path, *, kind: str) -> "Artifact":
        resolved = path.resolve()
        return cls(
            path=resolved,
            kind=kind,
            sha256=hash_file(resolved),
            size_bytes=resolved.stat().st_size,
        )

    def is_valid(self) -> bool:
        return self.path.is_file() and hash_file(self.path) == self.sha256


class StageRecord(BaseModel):
    status: StageStatus = StageStatus.PENDING
    fingerprint: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    elapsed_seconds: float | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    error: str | None = None


class RunManifest(BaseModel):
    schema_version: int = 1
    run_id: str
    source: str
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class StageOutcome(BaseModel):
    artifacts: list[Artifact]
    reused: bool


class ManifestStore:
    """Persist stage transitions atomically and reuse valid artifacts."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self, manifest: RunManifest) -> RunManifest:
        if self.path.exists():
            return self.load()
        self._save(manifest)
        return manifest

    def load(self) -> RunManifest:
        return RunManifest.model_validate_json(self.path.read_text(encoding="utf-8"))

    def execute(
        self,
        stage: str,
        fingerprint: str,
        action: Callable[[], list[Artifact]],
    ) -> StageOutcome:
        manifest = self.load()
        previous = manifest.stages.get(stage)
        if (
            previous is not None
            and previous.status is StageStatus.COMPLETED
            and previous.fingerprint == fingerprint
            and previous.artifacts
            and all(artifact.is_valid() for artifact in previous.artifacts)
        ):
            return StageOutcome(artifacts=previous.artifacts, reused=True)

        started = _utc_now()
        manifest.stages[stage] = StageRecord(
            status=StageStatus.RUNNING,
            fingerprint=fingerprint,
            started_at=started,
        )
        self._save(manifest)
        try:
            artifacts = action()
        except BaseException as error:
            finished = _utc_now()
            manifest = self.load()
            manifest.stages[stage] = StageRecord(
                status=StageStatus.INTERRUPTED
                if isinstance(error, KeyboardInterrupt)
                else StageStatus.FAILED,
                fingerprint=fingerprint,
                started_at=started,
                finished_at=finished,
                elapsed_seconds=(finished - started).total_seconds(),
                error=str(error),
            )
            self._save(manifest)
            raise

        finished = _utc_now()
        manifest = self.load()
        manifest.stages[stage] = StageRecord(
            status=StageStatus.COMPLETED,
            fingerprint=fingerprint,
            started_at=started,
            finished_at=finished,
            elapsed_seconds=(finished - started).total_seconds(),
            artifacts=artifacts,
        )
        self._save(manifest)
        return StageOutcome(artifacts=artifacts, reused=False)

    def _save(self, manifest: RunManifest) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        manifest.updated_at = _utc_now()
        payload = manifest.model_dump_json(indent=2)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, self.path)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
