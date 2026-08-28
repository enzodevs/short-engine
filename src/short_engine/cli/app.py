"""Typer composition root."""

from pathlib import Path
from typing import Annotated

import typer

from short_engine.cli.doctor import collect_checks
from short_engine.core.config import Settings
from short_engine.core.models import AspectRatio
from short_engine.pipeline import Engine
from short_engine.run.manifest import ManifestStore

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
    source: Annotated[str, typer.Argument(help="Local media path or URL")],
    clips: Annotated[int, typer.Option(min=1, max=20)] = 3,
    aspect: Annotated[AspectRatio, typer.Option()] = AspectRatio.VERTICAL,
    language: Annotated[str | None, typer.Option()] = None,
    cookies_from_browser: Annotated[
        str | None, typer.Option(help="Browser profile, e.g. chrome:Profile 3")
    ] = None,
) -> None:
    """Analyze media and render ranked short clips."""
    result = Engine().run(source, clips, aspect, language, cookies_from_browser)
    for output in result.renders:
        typer.echo(str(output))
    typer.echo(f"Manifest: {result.manifest}")


@app.command()
def inspect(manifest: Annotated[str, typer.Argument(help="Run manifest path")]) -> None:
    """Print safe run status without model prompts or secrets."""
    value = ManifestStore(Path(manifest)).load()
    typer.echo(f"Run: {value.run_id}\nSource: {value.source}")
    for name, stage in value.stages.items():
        typer.echo(f"{name}: {stage.status} ({stage.elapsed_seconds or 0:.2f}s)")


@app.command()
def analyze(
    source: Annotated[str, typer.Argument(help="Local media path or URL")],
    clips: Annotated[int, typer.Option(min=1, max=20)] = 3,
    language: Annotated[str | None, typer.Option()] = None,
    cookies_from_browser: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Analyze and rank candidates without rendering video."""
    result = Engine().run(
        source,
        clips,
        AspectRatio.VERTICAL,
        language,
        cookies_from_browser,
        render_outputs=False,
    )
    typer.echo(f"Manifest: {result.manifest}")


@app.command("render")
def render_command(
    manifest: Annotated[Path, typer.Argument(help="Run manifest path")],
    candidate: Annotated[list[str] | None, typer.Option("--candidate")] = None,
    aspect: Annotated[AspectRatio, typer.Option()] = AspectRatio.VERTICAL,
) -> None:
    """Render selected candidates without rerunning analysis or ranking."""
    result = Engine().render(manifest, candidate, aspect)
    for output in result.renders:
        typer.echo(str(output))
