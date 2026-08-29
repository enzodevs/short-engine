"""Genre-specific editorial grammar."""

from dataclasses import dataclass

from short_engine.editing.story import ContentGenre


@dataclass(frozen=True)
class GenreProfile:
    target_seconds: int
    max_context_seconds: int
    cut_silence_seconds: float
    visual_priority: str
    preferred_hook: str


GENRE_PROFILES = {
    ContentGenre.PODCAST: GenreProfile(42, 7, 0.5, "active-speaker reaction", "conflict"),
    ContentGenre.INTERVIEW: GenreProfile(40, 6, 0.5, "speaker-response", "bold answer"),
    ContentGenre.TUTORIAL: GenreProfile(35, 5, 0.65, "screen evidence", "result first"),
    ContentGenre.TECHNICAL: GenreProfile(38, 6, 0.6, "proof and readable UI", "stakes"),
    ContentGenre.GAMEPLAY: GenreProfile(28, 3, 0.35, "event and reaction", "peak action"),
    ContentGenre.REACTION: GenreProfile(25, 3, 0.4, "face and source", "emotion"),
    ContentGenre.VLOG: GenreProfile(35, 5, 0.5, "environment and face", "surprise"),
    ContentGenre.STORY: GenreProfile(50, 8, 0.55, "speaker and illustrative b-roll", "open loop"),
}
