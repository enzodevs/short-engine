# Short Engine

Turn a long video or podcast into ranked, vertical short clips on your Mac.

Short Engine runs locally from the command line on Apple Silicon. Give it a video file or URL, and it builds a speech and scene timeline, finds the strongest moments, reframes the subject for 9:16, adds karaoke-style captions, and renders the clips with FFmpeg.

## What it does

- Transcribes with MLX Whisper.
- Uses scene changes and speech boundaries to build candidate clips.
- Ranks candidates with Gemini using transcript text and sampled video frames.
- Tracks the primary subject with YOLO on MPS, with a deterministic center crop when tracking is unavailable.
- Renders 9:16 clips with high-contrast, word-by-word captions.
- Stores an atomic, resumable manifest for every run in `output/<run-id>/`.

## Requirements

- macOS on Apple Silicon
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- FFmpeg with `libass`; Homebrew provides this through the keg-only `ffmpeg-full` formula
- A Gemini API key for semantic ranking

## Install

```bash
brew install ffmpeg-full
export PATH="/opt/homebrew/opt/ffmpeg-full/bin:$PATH"
uv sync --all-extras --dev
cp .env.example .env
# Add GEMINI_API_KEY to .env
uv run short-engine doctor
```

The first run downloads the MLX Whisper checkpoint and YOLO model to your user cache. Secrets, model weights, and rendered output are not committed to Git.

## Create clips

Render the best three clips from a local file:

```bash
uv run short-engine run video.mp4 --clips 3 --language pt
```

Use a YouTube URL with cookies from a local Chrome profile when needed:

```bash
uv run short-engine run 'https://youtu.be/ID' \
  --cookies-from-browser 'chrome:Profile 3' --clips 3 --language pt
```

Analyze now and choose what to render later without repeating transcription or ranking:

```bash
uv run short-engine analyze video.mp4 --clips 3 --language pt
uv run short-engine render output/RUN_ID/manifest.json --candidate CANDIDATE_ID
uv run short-engine inspect output/RUN_ID/manifest.json
```

## Development

```bash
make check
```

The check formats the code, runs Ruff and ty, then executes the offline test suite with coverage.

See [SPEC.md](SPEC.md) for the architecture and acceptance criteria.
