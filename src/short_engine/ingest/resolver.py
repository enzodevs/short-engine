"""Local and yt-dlp source resolution."""

import hashlib
from pathlib import Path

from short_engine.core.errors import InputError, MediaError
from short_engine.ingest.models import SourceAsset, SourceRequest
from short_engine.run.manifest import hash_file
from short_engine.system.process import CommandRunner


class SourceResolver:
    def __init__(self, runner: CommandRunner) -> None:
        self.runner = runner

    def resolve(self, request: SourceRequest, run_dir: Path) -> SourceAsset:
        if request.is_url:
            return self._download(request, run_dir)
        return self._local(request)

    def _local(self, request: SourceRequest) -> SourceAsset:
        path = Path(request.source).expanduser()
        if not path.is_file():
            raise InputError(f"Local source does not exist or is not a file: {path}")
        resolved = path.resolve()
        return SourceAsset(
            path=resolved,
            source_fingerprint=hash_file(resolved),
            original_source=request.source,
            downloaded=False,
        )

    def _download(self, request: SourceRequest, run_dir: Path) -> SourceAsset:
        source_dir = run_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(source_dir / "source_%(id)s.%(ext)s")
        args = [
            "yt-dlp",
            "--remote-components",
            "ejs:github",
            "--no-playlist",
            "--merge-output-format",
            "mp4",
            "--format",
            f"bv*[height<={request.download_height}]+ba/b[height<={request.download_height}]",
            "--output",
            output_template,
            "--print",
            "after_move:filepath",
        ]
        if request.cookies_from_browser:
            args.extend(["--cookies-from-browser", request.cookies_from_browser])
        args.append(request.source)
        result = self.runner.run(args)
        if result.returncode != 0:
            message = (
                result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
            )
            raise MediaError(f"yt-dlp failed: {message}")
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise MediaError("yt-dlp did not report the downloaded file path")
        path = Path(lines[-1]).expanduser().resolve()
        if not path.is_file():
            raise MediaError(f"yt-dlp reported a file that does not exist: {path}")
        url_hash = hashlib.sha256(request.source.encode()).hexdigest()
        return SourceAsset(
            path=path,
            source_fingerprint=url_hash,
            original_source=request.source,
            downloaded=True,
        )
