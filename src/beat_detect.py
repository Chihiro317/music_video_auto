from __future__ import annotations

from pathlib import Path
from typing import List

import librosa


def get_audio_duration(audio_path: str | Path) -> float:
    """Return audio duration in seconds."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"音乐文件不存在：{path}")
    return float(librosa.get_duration(path=str(path)))


def detect_beats(audio_path: str | Path, max_beats: int | None = None) -> List[float]:
    """Detect beat timestamps from an audio file.

    Returns a list of seconds. If beat detection fails, returns an empty list.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"音乐文件不存在：{path}")

    try:
        y, sr = librosa.load(str(path), sr=None, mono=True)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).astype(float).tolist()
    except Exception:
        return []

    if max_beats is not None:
        beat_times = beat_times[:max_beats]
    return beat_times


def is_near_beat(t: float, beats: list[float], window: float = 0.12) -> bool:
    """Return whether timestamp t is near a detected beat."""
    return any(abs(t - beat) <= window for beat in beats)
