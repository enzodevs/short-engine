"""Conservative music and sound-design planner."""

from short_engine.audio.models import AudioCatalog, AudioPlan, SoundCue
from short_engine.editing.story import BeatRole, StoryVariant


class AudioDirector:
    def plan(self, story: StoryVariant, catalog: AudioCatalog) -> AudioPlan:
        tracks = [item for item in catalog.music if story.genre in item.genres]
        target_energy = min(1.0, max(0.2, story.predicted_retention_score / 100))
        music = min(tracks, key=lambda item: abs(item.energy - target_energy), default=None)
        cues: list[SoundCue] = []
        cursor = 0.0
        for beat in story.beats:
            effects = [item for item in catalog.effects if beat.role in item.beat_roles]
            if effects and beat.role in {BeatRole.HOOK, BeatRole.PAYOFF}:
                cues.append(SoundCue(effect_id=effects[0].id, at_seconds=cursor))
            cursor += beat.source.duration_seconds
        return AudioPlan(music_id=music.id if music else None, cues=cues)
