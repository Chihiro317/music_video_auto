from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import srt


@dataclass
class SubtitleItem:
    start: float
    end: float
    text: str


def load_srt(srt_path: str | Path | None) -> List[SubtitleItem]:
    """Load SRT subtitles. Missing path returns an empty list."""
    if not srt_path:
        return []
    path = Path(srt_path)
    if not path.exists():
        raise FileNotFoundError(f"字幕文件不存在：{path}")

    content = path.read_text(encoding="utf-8-sig")
    items: list[SubtitleItem] = []
    for sub in srt.parse(content):
        text = sub.content.replace("\n", " ").strip()
        if text:
            items.append(
                SubtitleItem(
                    start=sub.start.total_seconds(),
                    end=sub.end.total_seconds(),
                    text=text,
                )
            )
    return items


def find_subtitle_at(t: float, subtitles: list[SubtitleItem]) -> str:
    """Return subtitle text displayed at time t."""
    for item in subtitles:
        if item.start <= t <= item.end:
            return item.text
    return ""
