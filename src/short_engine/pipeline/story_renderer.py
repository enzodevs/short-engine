"""Render selected candidates or composed stories."""

from pathlib import Path

from short_engine.candidates.models import Candidate
from short_engine.core.config import Settings
from short_engine.core.models import AspectRatio, TimeRange
from short_engine.editing.jumpcuts import JumpCutPlanner
from short_engine.editing.story import StoryPackage
from short_engine.ingest.probe import FFprobe
from short_engine.ranking.models import Selection
from short_engine.reframing.models import SubjectTrack
from short_engine.reframing.planner import CropPlanner
from short_engine.reframing.tracker import UltralyticsSubjectTracker
from short_engine.rendering.captions import AssCaptionWriter
from short_engine.rendering.renderer import FFmpegRenderer
from short_engine.run.manifest import Artifact, ManifestStore
from short_engine.segmentation.models import BoundaryKind, Timeline
from short_engine.system.process import SubprocessRunner
from short_engine.transcription.models import Transcript


class StoryRenderService:
    def __init__(self, settings: Settings, runner: SubprocessRunner) -> None:
        self.settings = settings
        self.runner = runner

    def render(
        self,
        source: Path,
        run_dir: Path,
        transcript: Transcript,
        candidates: list[Candidate],
        selection: Selection,
        aspect: AspectRatio,
        store: ManifestStore,
        stories: StoryPackage | None = None,
    ) -> list[Path]:
        probe = FFprobe(self.runner).inspect(source)
        timeline = Timeline.model_validate_json(
            (run_dir / "analysis" / "timeline.json").read_text()
        )
        scene_boundaries = [
            boundary.at_seconds
            for boundary in timeline.boundaries
            if BoundaryKind.SCENE in boundary.kinds
        ]
        renders: list[Path] = []
        for index, (item_id, overall, source_ranges) in enumerate(
            self._items(candidates, selection, stories), start=1
        ):
            output = run_dir / "renders" / f"short-{index:02d}.mp4"
            fingerprint = ",".join(
                f"{item.start_seconds:.3f}-{item.end_seconds:.3f}" for item in source_ranges
            )

            def execute(
                index: int = index,
                item_id: str = item_id,
                overall: TimeRange = overall,
                source_ranges: list[TimeRange] = source_ranges,
                output: Path = output,
            ) -> list[Artifact]:
                words = [
                    word
                    for segment in transcript.segments
                    for word in segment.words
                    if any(
                        item.start_seconds <= word.start_seconds < item.end_seconds
                        for item in source_ranges
                    )
                ]
                edit_plan = JumpCutPlanner().plan_many(source_ranges, words)
                edit_ranges = [segment.source for segment in edit_plan.segments]
                scenes = [
                    value
                    for value in scene_boundaries
                    if any(
                        item.start_seconds <= value <= item.end_seconds for item in source_ranges
                    )
                ]
                try:
                    track = UltralyticsSubjectTracker(self.settings.tracker_model).track(
                        source, overall, scenes
                    )
                except RuntimeError:
                    track = SubjectTrack(observations=[])
                crop = CropPlanner().plan(
                    overall,
                    track,
                    probe.width or 1920,
                    probe.height or 1080,
                    aspect,
                    takes=edit_ranges,
                    hard_cuts_seconds=scenes,
                )
                render_dir = run_dir / "renders"
                render_dir.mkdir(parents=True, exist_ok=True)
                crop_path = render_dir / f"short-{index:02d}.crop.json"
                crop_path.write_text(crop.model_dump_json(indent=2))
                edit_path = render_dir / f"short-{index:02d}.edit.json"
                edit_path.write_text(edit_plan.model_dump_json(indent=2))
                captions = AssCaptionWriter().write(
                    render_dir / f"short-{index:02d}.ass", edit_plan.remap_words(words), 0
                )
                rendered = FFmpegRenderer(self.runner).render(
                    source, output, overall, crop, captions, edits=edit_ranges
                )
                return [
                    Artifact.from_path(crop_path, kind="crop-plan"),
                    Artifact.from_path(edit_path, kind="edit-plan"),
                    Artifact.from_path(rendered, kind="render"),
                ]

            store.execute(
                f"render:{item_id}",
                (
                    f"{item_id}:{fingerprint}:{aspect}:{self.settings.tracker_model}:"
                    "render-v13-story-composition"
                ),
                execute,
            )
            renders.append(output)
        return renders

    @staticmethod
    def _items(
        candidates: list[Candidate], selection: Selection, stories: StoryPackage | None
    ) -> list[tuple[str, TimeRange, list[TimeRange]]]:
        if stories is None:
            by_id = {item.id: item for item in candidates}
            return [
                (
                    by_id[item.candidate_id].id,
                    by_id[item.candidate_id].time_range,
                    [by_id[item.candidate_id].time_range],
                )
                for item in selection.selected
            ]
        by_id = {item.id: item for item in stories.variants}
        result: list[tuple[str, TimeRange, list[TimeRange]]] = []
        for identifier in stories.selected_variant_ids:
            ranges = [beat.source for beat in by_id[identifier].beats]
            result.append(
                (
                    identifier,
                    TimeRange(
                        start_seconds=min(item.start_seconds for item in ranges),
                        end_seconds=max(item.end_seconds for item in ranges),
                    ),
                    ranges,
                )
            )
        return result
