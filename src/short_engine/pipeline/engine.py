"""Local-first end-to-end short generation pipeline."""

import hashlib
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from short_engine.candidates.generator import CandidateConfig, TranscriptCandidateGenerator
from short_engine.candidates.models import Candidate
from short_engine.candidates.pool import CandidatePoolSampler
from short_engine.core.config import Settings
from short_engine.core.errors import InputError
from short_engine.core.models import AspectRatio
from short_engine.ingest.media import FFmpegMediaService
from short_engine.ingest.models import SourceRequest
from short_engine.ingest.resolver import SourceResolver
from short_engine.pipeline.story_renderer import StoryRenderService
from short_engine.ranking.frames import FrameSampler
from short_engine.ranking.gemini import GeminiRanker
from short_engine.ranking.models import Selection
from short_engine.ranking.selector import CandidateSelector
from short_engine.run.manifest import Artifact, ManifestStore, RunManifest
from short_engine.segmentation.adapters import SceneDetector, SileroSpeechDetector
from short_engine.segmentation.timeline import TimelineBuilder
from short_engine.system.process import SubprocessRunner
from short_engine.transcription.mlx import MLXWhisperTranscriber
from short_engine.transcription.models import ASRConfig, Transcript


class EngineResult(BaseModel):
    manifest: Path
    renders: list[Path]
    selection: Selection


class Engine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.runner = SubprocessRunner()

    def run(
        self,
        source: str,
        clips: int = 3,
        aspect: AspectRatio = AspectRatio.VERTICAL,
        language: str | None = None,
        cookies_from_browser: str | None = None,
        render_outputs: bool = True,
    ) -> EngineResult:
        run_id = hashlib.sha256(f"{source}:{aspect}:{language}".encode()).hexdigest()[:12]
        run_dir = self.settings.output_root / run_id
        store = ManifestStore(run_dir / "manifest.json")
        store.initialize(RunManifest(run_id=run_id, source=source))
        asset = SourceResolver(self.runner).resolve(
            SourceRequest(source=source, cookies_from_browser=cookies_from_browser), run_dir
        )
        media = FFmpegMediaService(self.runner)
        audio = run_dir / "analysis" / "audio.wav"
        proxy = run_dir / "analysis" / "proxy.mp4"
        store.execute(
            "media",
            asset.source_fingerprint,
            lambda: [
                Artifact.from_path(media.extract_analysis_audio(asset.path, audio), kind="audio"),
                Artifact.from_path(media.create_proxy(asset.path, proxy), kind="proxy"),
            ],
        )
        transcript_path = run_dir / "analysis" / "transcript.json"

        def transcribe() -> list[Artifact]:
            transcript = MLXWhisperTranscriber().transcribe(
                audio, ASRConfig(model=self.settings.asr_model, language=language)
            )
            transcript_path.write_text(transcript.model_dump_json(indent=2))
            return [Artifact.from_path(transcript_path, kind="transcript")]

        store.execute(
            "transcription",
            f"{asset.source_fingerprint}:{self.settings.asr_model}:{language}",
            transcribe,
        )
        transcript = Transcript.model_validate_json(transcript_path.read_text())
        timeline_path = run_dir / "analysis" / "timeline.json"

        def segment() -> list[Artifact]:
            boundaries = SceneDetector().detect(proxy) + SileroSpeechDetector().detect(audio)
            timeline = TimelineBuilder().build(transcript, boundaries)
            timeline_path.write_text(timeline.model_dump_json(indent=2))
            return [Artifact.from_path(timeline_path, kind="timeline")]

        store.execute(
            "segmentation",
            f"{asset.source_fingerprint}:adaptive-scene:silero-v6",
            segment,
        )
        candidates = TranscriptCandidateGenerator().generate(
            transcript,
            CandidateConfig(
                min_duration_seconds=15,
                target_duration_seconds=35,
                max_duration_seconds=60,
                stride_segments=1,
            ),
        )
        if not candidates:
            raise InputError("No coherent candidate windows were found")
        candidates = CandidatePoolSampler().select(candidates, 24)
        candidates_path = run_dir / "candidates" / "candidates.json"
        candidates_path.parent.mkdir(parents=True, exist_ok=True)
        candidates_path.write_text(
            TypeAdapter(list[Candidate]).dump_json(candidates, indent=2).decode()
        )
        key = self.settings.gemini_api_key
        if key is None:
            raise InputError("GEMINI_API_KEY is required for ranking")
        ranking_path = run_dir / "candidates" / "selection.json"

        def rank() -> list[Artifact]:
            sampler = FrameSampler(self.runner)
            evidence = {
                item.id: sampler.sample(proxy, item, run_dir / "candidates" / "frames")
                for item in candidates
            }
            assessments = GeminiRanker(
                key.get_secret_value(),
                self.settings.gemini_model,
                debug_directory=run_dir / "debug",
                assessment_directory=run_dir / "candidates" / "assessments-v1",
            ).rank(candidates, evidence, video=proxy)
            selection = CandidateSelector(self.settings.editorial_quality).select(
                candidates, assessments, clips
            )
            ranking_path.write_text(selection.model_dump_json(indent=2))
            return [Artifact.from_path(ranking_path, kind="selection")]

        candidate_fingerprint = ":".join(item.id for item in candidates)
        store.execute(
            "ranking",
            f"{candidate_fingerprint}:{self.settings.gemini_model}:{clips}:"
            f"{self.settings.editorial_quality.model_dump_json()}:viral-temporal-v2",
            rank,
        )
        selection = Selection.model_validate_json(ranking_path.read_text())
        if not selection.selected:
            raise InputError(
                "No excerpt passed the editorial quality gate: a complete standalone hook, "
                "narrative arc, and explicit payoff are required"
            )
        renders = (
            StoryRenderService(self.settings, self.runner).render(
                asset.path, run_dir, transcript, candidates, selection, aspect, store
            )
            if render_outputs
            else []
        )
        return EngineResult(manifest=store.path, renders=renders, selection=selection)

    def render(
        self,
        manifest_path: Path,
        candidate_ids: list[str] | None = None,
        aspect: AspectRatio = AspectRatio.VERTICAL,
    ) -> EngineResult:
        store = ManifestStore(manifest_path)
        manifest = store.load()
        run_dir = manifest_path.parent
        source = Path(manifest.source).expanduser()
        if not source.is_file():
            downloads = sorted((run_dir / "source").glob("source_*.mp4"))
            if not downloads:
                raise InputError("Downloaded source is missing from this run")
            source = downloads[0]
        transcript = Transcript.model_validate_json(
            (run_dir / "analysis" / "transcript.json").read_text()
        )
        candidates = TypeAdapter(list[Candidate]).validate_json(
            (run_dir / "candidates" / "candidates.json").read_text()
        )
        selection = Selection.model_validate_json(
            (run_dir / "candidates" / "selection.json").read_text()
        )
        if candidate_ids:
            wanted = set(candidate_ids)
            selection = Selection(
                selected=[item for item in selection.selected if item.candidate_id in wanted],
                rejections=selection.rejections,
            )
            missing = wanted - {item.candidate_id for item in selection.selected}
            if missing:
                raise InputError(
                    f"Candidate IDs are not selected in this manifest: {', '.join(sorted(missing))}"
                )
        renders = StoryRenderService(self.settings, self.runner).render(
            source, run_dir, transcript, candidates, selection, aspect, store
        )
        return EngineResult(manifest=manifest_path, renders=renders, selection=selection)
