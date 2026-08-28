"""ASS caption generation from word timestamps."""

from pathlib import Path

from short_engine.transcription.models import TimedWord


class AssCaptionWriter:
    def __init__(self, words_per_chunk: int = 4) -> None:
        if words_per_chunk < 1:
            raise ValueError("words_per_chunk must be positive")
        self.words_per_chunk = words_per_chunk

    def write(self, path: Path, words: list[TimedWord], offset_seconds: float = 0) -> Path:
        header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
[V4+ Styles]
""" + (
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
            "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
            "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
            "Style: Default,Arial Black,86,&H00FFFFFF,&H003BEBFF,&H00000000,"
            "&H64000000,-1,0,0,0,100,100,1,0,1,5,2,2,90,90,330,1\n"
            "[Events]\n"
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
        )
        lines = [header]
        for chunk_start in range(0, len(words), self.words_per_chunk):
            chunk = words[chunk_start : chunk_start + self.words_per_chunk]
            for active_index, word in enumerate(chunk):
                start = max(0, word.start_seconds - offset_seconds)
                next_start = (
                    chunk[active_index + 1].start_seconds
                    if active_index + 1 < len(chunk)
                    else word.end_seconds
                )
                end = max(start + 0.01, next_start - offset_seconds)
                text = self._highlighted_chunk(chunk, active_index)
                lines.append(
                    f"Dialogue: 0,{self._time(start)},{self._time(end)},Default,,0,0,0,,{text}\n"
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines))
        return path

    @classmethod
    def _highlighted_chunk(cls, words: list[TimedWord], active_index: int) -> str:
        rendered: list[str] = []
        for index, word in enumerate(words):
            text = cls._escape(word.text)
            if index == active_index:
                text = rf"{{\c&H003BEBFF&\fscx112\fscy112}}{text}{{\r}}"
            rendered.append(text)
        return " ".join(rendered)

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("{", r"\{").replace("}", r"\}").replace("\n", " ")

    @staticmethod
    def _time(seconds: float) -> str:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes):02d}:{seconds:05.2f}"
