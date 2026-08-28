"""Short Engine public package."""

from short_engine.cli.app import app

__all__ = ["app", "main"]


def main() -> None:
    """Run the command-line application."""
    app()
