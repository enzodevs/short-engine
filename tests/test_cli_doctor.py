from typer.testing import CliRunner

from short_engine.cli.app import app

runner = CliRunner()


def test_doctor_reports_required_capabilities_without_secrets() -> None:
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "FFmpeg" in result.stdout
    assert "ffprobe" in result.stdout
    assert "Apple Silicon" in result.stdout
    assert "Gemini API key" in result.stdout
    assert "AIza" not in result.stdout
    assert "AQ." not in result.stdout


def test_invalid_aspect_is_rejected_before_pipeline_execution() -> None:
    result = runner.invoke(app, ["run", "missing.mp4", "--aspect", "4:5"])

    assert result.exit_code != 0
    assert "Invalid value" in result.stderr
