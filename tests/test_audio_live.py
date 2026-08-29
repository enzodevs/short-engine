from pathlib import Path

from short_engine.audio.director import AudioDirector
from short_engine.audio.models import AudioCatalog, MusicTrack, SoundEffect
from short_engine.core.models import TimeRange
from short_engine.editing.story import (
    BeatRole,
    ContentGenre,
    RetentionPoint,
    StoryBeat,
    StoryVariant,
)
from short_engine.live.detector import LiveHighlightDetector
from short_engine.live.models import LiveHighlightState, LiveSignal


def story() -> StoryVariant:
    return StoryVariant(
        id="story",
        title="Story",
        genre=ContentGenre.TECHNICAL,
        strategy="Result first",
        hook_text="It worked",
        beats=[
            StoryBeat(
                role=BeatRole.HOOK,
                source=TimeRange(start_seconds=10, end_seconds=18),
                rationale="result",
            ),
            StoryBeat(
                role=BeatRole.PAYOFF,
                source=TimeRange(start_seconds=0, end_seconds=10),
                rationale="proof",
            ),
        ],
        retention_map=[
            RetentionPoint(
                output_start_seconds=0,
                output_end_seconds=18,
                attention_reason="proof",
                drop_off_risk=20,
                edit_action="keep",
            )
        ],
        predicted_retention_score=80,
        fatal_flaw="jargon",
    )


def test_audio_director_selects_licensed_track_and_sparse_cues() -> None:
    catalog = AudioCatalog(
        music=[
            MusicTrack(
                id="pulse",
                path=Path("pulse.mp3"),
                genres={ContentGenre.TECHNICAL},
                moods={"tense"},
                bpm=120,
                energy=0.8,
                license="owned",
            )
        ],
        effects=[
            SoundEffect(
                id="impact",
                path=Path("impact.wav"),
                beat_roles={BeatRole.HOOK, BeatRole.PAYOFF},
                license="owned",
            )
        ],
    )

    plan = AudioDirector().plan(story(), catalog)

    assert plan.music_id == "pulse"
    assert [cue.at_seconds for cue in plan.cues] == [0, 8]


def test_live_detector_waits_for_payoff_before_finalizing() -> None:
    detector = LiveHighlightDetector()

    hook = detector.observe(
        LiveSignal(at_seconds=10, speech_energy=0.9, visual_novelty=0.8, semantic_stakes=0.9)
    )
    middle = detector.observe(
        LiveSignal(at_seconds=15, speech_energy=0.6, visual_novelty=0.4, semantic_stakes=0.6)
    )
    payoff = detector.observe(
        LiveSignal(at_seconds=20, speech_energy=0.8, visual_novelty=0.9, semantic_stakes=0.9)
    )

    assert hook.state is LiveHighlightState.HOOK_DETECTED
    assert middle.state is LiveHighlightState.ESCALATING
    assert payoff.state is LiveHighlightState.READY
    assert payoff.source == TimeRange(start_seconds=7, end_seconds=22)
