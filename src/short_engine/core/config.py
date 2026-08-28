"""Single configuration boundary for environment and defaults."""

from pathlib import Path

from platformdirs import user_cache_path
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Effective engine settings loaded once at the composition root."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SHORT_ENGINE_",
        extra="ignore",
    )

    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "SHORT_ENGINE_GEMINI_API_KEY"),
    )
    gemini_model: str = Field(
        default="gemini-3.1-flash-lite",
        validation_alias=AliasChoices("GEMINI_MODEL", "SHORT_ENGINE_GEMINI_MODEL"),
    )
    output_root: Path = Path("output")
    cache_root: Path = Field(default_factory=lambda: user_cache_path("short-engine"))
    asr_model: str = "mlx-community/whisper-large-v3-turbo"
    tracker_model: str = "yolo11n.pt"

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.get_secret_value())
