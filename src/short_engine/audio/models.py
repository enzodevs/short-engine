"""Licensed audio-library contracts."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from short_engine.editing.story import BeatRole, ContentGenre


class MusicTrack(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    path: Path
    genres: set[ContentGenre]
    moods: set[str]
    bpm: float = Field(gt=0)
    energy: float = Field(ge=0, le=1)
    license: str
    attribution: str | None = None


class SoundEffect(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    path: Path
    beat_roles: set[BeatRole]
    license: str


class AudioCatalog(BaseModel):
    music: list[MusicTrack] = Field(default_factory=list)
    effects: list[SoundEffect] = Field(default_factory=list)


class SoundCue(BaseModel):
    effect_id: str
    at_seconds: float = Field(ge=0)
    gain_db: float = Field(default=-8, ge=-30, le=6)


class AudioPlan(BaseModel):
    music_id: str | None = None
    music_gain_db: float = Field(default=-22, ge=-40, le=0)
    ducking_db: float = Field(default=-10, ge=-30, le=0)
    cues: list[SoundCue] = Field(default_factory=list)
