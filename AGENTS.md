# Short Engine — Agent Guide

## Product boundary

Short Engine is a local-first, CLI-first engine that turns long-form video or
audio into ranked, rendered short clips on Apple Silicon. It is not a SaaS and
must not acquire authentication, billing, web UI, multi-user, or publishing
concerns without an explicit spec change.

## Architecture rules

- Keep a modular monolith organized by media domain: `ingest`,
  `transcription`, `segmentation`, `ranking`, `reframing`, `rendering`, and
  `pipeline`.
- Domain code depends on typed `Protocol` contracts, not concrete ML, network,
  CLI, or filesystem implementations.
- Construct dependencies at the composition root. Do not use mutable global
  clients, model singletons, or environment reads inside domain modules.
- A module owns one reason to change. Treat files approaching 300 lines as a
  design review signal; split by responsibility when cohesion is weakening.
- Use Pydantic models at persisted/external boundaries and plain immutable
  dataclasses for internal value objects when validation is unnecessary.
- All model identifiers, thresholds, durations, and output profiles are
  configuration, never scattered constants.
- Each pipeline stage is idempotent and writes an artifact recorded in the run
  manifest. A rerun must resume without repeating valid completed work.
- Wrap FFmpeg/ffprobe, yt-dlp, MLX, VLM, detector, and network APIs behind thin
  adapters. Never duplicate their media or inference functionality.
- Prefer a clear failure with stage context over silent fallback. Any supported
  fallback must be explicit in the manifest.

## Platform and tooling

- Primary target: macOS on Apple Silicon; reference machine is an M5 MacBook
  Pro with 32 GB unified memory.
- Python 3.12 managed by `uv`; dependencies and lockfile live in
  `pyproject.toml` and `uv.lock`.
- CLI: Typer. Validation/serialization: Pydantic v2. Media: FFmpeg/ffprobe.
- Apple-native inference: MLX Whisper and MLX-VLM. Scene detection:
  PySceneDetect. Speech boundaries: Silero VAD through ONNX Runtime.
- Speaker diarization is an optional pyannote.audio adapter; the core timeline
  understands speaker labels but does not require diarization to complete.
- Subject tracking: an adapter selected by benchmark; the initial preferred
  implementation is Ultralytics tracking on MPS, with deterministic center-crop
  fallback.
- Quality gates: Ruff format/lint, `ty check`, pytest, and coverage.

## Canonical commands

These commands become active once the implementation scaffold lands:

```bash
uv sync --all-groups
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
uv run short-engine doctor
```

Run only through `uv`; do not instruct users to maintain or activate a manual
virtualenv.

## Testing discipline

- Acceptance criteria in `SPEC.md` are the test oracle.
- Unit-test candidate generation, timestamp normalization, scoring, crop plans,
  and manifest transitions without loading models.
- Integration-test real FFmpeg/ffprobe seams with small deterministic fixtures.
- Keep ML/network tests opt-in and labeled; the default suite must be offline.
- Observe a failing test before its implementation passes. For high-risk time
  and ranking logic, perform a mutation check before completing the task.
- Do not mock internal modules. Fake only explicit adapter contracts.

## Current phase

The repository is specification-only. Implement tasks from `SPEC.md` in order,
one logical commit per task, with tests shipped in the same commit.
