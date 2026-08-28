"""Typer composition root."""

from pathlib import Path
from typing import Annotated

import typer

from short_engine.cli.doctor import collect_checks
from short_engine.core.config import Settings
from short_engine.core.models import AspectRatio

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.command()
def doctor() -> None:
    """Report whether this Mac can run the engine."""
    checks = collect_checks(Settings())
    for check in checks:
        status = "OK" if check.ok else "WARN"
        typer.echo(f"[{status}] {check.name}: {check.detail}")


@app.command()
def run(
    source: Annotated[Path, typer.Argument(help="Local media path")],
    clips: Annotated[int, typer.Option(min=1, max=20)] = 3,
    aspect: Annotated[AspectRatio, typer.Option()] = AspectRatio.VERTICAL,
    language: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Run the full local pipeline (implemented by subsequent tasks)."""
    del source, clips, aspect, language
    raise typer.BadParameter("pipeline implementation is not installed yet")
