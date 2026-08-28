"""ASS caption generation from word timestamps."""

from pathlib import Path

from short_engine.transcription.models import TimedWord


class AssCaptionWriter:
    def write(self, path: Path, words: list[TimedWord], offset_seconds: float = 0) -> Path:
        header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
[V4+ Styles]
""" + (
            "Format: Name,Fontname,Fontsize,PrimaryColour,OutlineColour,BorderStyle,"
            "Outline,Shadow,Alignment,MarginL,MarginR,MarginV\n"
            "Style: Default,Arial,64,&H00FFFFFF,&H00000000,1,4,0,2,80,80,180\n"
            "[Events]\nFormat: Layer,Start,End,Style,Text\n"
        )
        lines = [header]
        for word in words:
            start = max(0, word.start_seconds - offset_seconds)
            end = max(start + 0.01, word.end_seconds - offset_seconds)
            text = word.text.replace("{", r"\{").replace("}", r"\}").replace("\n", " ")
            lines.append(f"Dialogue: 0,{self._time(start)},{self._time(end)},Default,{text}\n")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines))
        return path

    @staticmethod
    def _time(seconds: float) -> str:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes):02d}:{seconds:05.2f}"
