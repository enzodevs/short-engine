"""Local-first end-to-end short generation pipeline."""

import hashlib
from pathlib import Path

from pydantic import BaseModel

from short_engine.candidates.generator import CandidateConfig, TranscriptCandidateGenerator
from short_engine.candidates.models import Candidate
from short_engine.core.config import Settings
from short_engine.core.errors import InputError
from short_engine.core.models import AspectRatio
from short_engine.ingest.media import FFmpegMediaService
from short_engine.ingest.models import SourceRequest
from short_engine.ingest.probe import FFprobe
from short_engine.ingest.resolver import SourceResolver
from short_engine.ranking.frames import FrameSampler
from short_engine.ranking.gemini import GeminiRanker
from short_engine.ranking.models import Selection
from short_engine.ranking.selector import CandidateSelector
from short_engine.reframing.models import SubjectTrack
from short_engine.reframing.planner import CropPlanner
from short_engine.reframing.tracker import UltralyticsSubjectTracker
from short_engine.rendering.captions import AssCaptionWriter
from short_engine.rendering.renderer import FFmpegRenderer
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
        candidates = TranscriptCandidateGenerator().generate(transcript, CandidateConfig())
        if not candidates:
            raise InputError("No coherent candidate windows were found")
        candidates = candidates[:24]
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
            ).rank(candidates, evidence)
            selection = CandidateSelector().select(candidates, assessments, clips)
            ranking_path.write_text(selection.model_dump_json(indent=2))
            return [Artifact.from_path(ranking_path, kind="selection")]

        candidate_fingerprint = ":".join(item.id for item in candidates)
        store.execute(
            "ranking",
            f"{candidate_fingerprint}:{self.settings.gemini_model}:{clips}",
            rank,
        )
        selection = Selection.model_validate_json(ranking_path.read_text())
        probe = FFprobe(self.runner).inspect(asset.path)
        by_id = {item.id: item for item in candidates}
        renders: list[Path] = []
        for index, assessment in enumerate(selection.selected, start=1):
            candidate = by_id[assessment.candidate_id]
            output_path = run_dir / "renders" / f"short-{index:02d}.mp4"

            def render(
                candidate: Candidate = candidate,
                index: int = index,
                output_path: Path = output_path,
            ) -> list[Artifact]:
                try:
                    track = UltralyticsSubjectTracker(self.settings.tracker_model).track(
                        asset.path, candidate.time_range
                    )
                except RuntimeError:
                    track = SubjectTrack(observations=[])
                crop = CropPlanner().plan(
                    candidate.time_range,
                    track,
                    probe.width or 1920,
                    probe.height or 1080,
                    aspect,
                )
                words = [
                    word
                    for segment in transcript.segments
                    for word in segment.words
                    if candidate.time_range.start_seconds
                    <= word.start_seconds
                    < candidate.time_range.end_seconds
                ]
                caption_path = AssCaptionWriter().write(
                    run_dir / "renders" / f"short-{index:02d}.ass",
                    words,
                    candidate.time_range.start_seconds,
                )
                rendered = FFmpegRenderer(self.runner).render(
                    asset.path,
                    output_path,
                    candidate.time_range,
                    crop,
                    caption_path,
                )
                return [Artifact.from_path(rendered, kind="render")]

            store.execute(
                f"render:{candidate.id}",
                f"{candidate.id}:{aspect}:{self.settings.tracker_model}",
                render,
            )
            renders.append(output_path)
        return EngineResult(manifest=store.path, renders=renders, selection=selection)
