from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
)

from .beat_detect import detect_beats, get_audio_duration
from .effects import beat_scale, flash_opacity, shake_offset
from .subtitle import load_srt


@dataclass
class RenderConfig:
    music: str
    character: str
    lyrics: str | None
    background: str | None
    title: str
    version: str
    account: str
    output: str
    width: int = 1080
    height: int = 1920
    fps: int = 30
    outro_seconds: float = 2.0


def _make_background(path: str | None, duration: float, size: tuple[int, int]):
    width, height = size
    if not path:
        return ColorClip(size, color=(12, 8, 24)).set_duration(duration)

    bg_path = Path(path)
    if not bg_path.exists():
        raise FileNotFoundError(f"背景素材不存在：{bg_path}")

    if bg_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        clip = VideoFileClip(str(bg_path)).without_audio()
        if clip.duration < duration:
            clip = clip.loop(duration=duration)
        else:
            clip = clip.subclip(0, duration)
        return clip.resize(height=height).crop(x_center=clip.w / 2, y_center=clip.h / 2, width=width, height=height)

    return ImageClip(str(bg_path)).set_duration(duration).resize(height=height).crop(
        x_center=width / 2, y_center=height / 2, width=width, height=height
    )


def _safe_text_clip(text: str, fontsize: int, color: str, stroke_color: str = "black", stroke_width: int = 2):
    return TextClip(
        txt=text,
        fontsize=fontsize,
        color=color,
        font="Microsoft-YaHei",
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        method="caption",
    )


def render_video(config: RenderConfig) -> Path:
    """Render a music video according to config."""
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    duration = get_audio_duration(config.music)
    total_duration = duration + config.outro_seconds
    size = (config.width, config.height)
    beats = detect_beats(config.music)
    subtitles = load_srt(config.lyrics)

    audio = AudioFileClip(config.music)
    background = _make_background(config.background, total_duration, size)

    character = ImageClip(config.character).set_duration(duration)
    max_character_height = int(config.height * 0.58)
    character = character.resize(height=max_character_height)

    def char_position(t: float):
        x_shake, y_shake = shake_offset(t, beats)
        return ("center", int(config.height * 0.43 - character.h / 2 + y_shake))

    def char_resize(t: float):
        return beat_scale(t, beats, base=1.0, pulse=0.11)

    character = character.resize(char_resize).set_position(char_position)

    title_text = f"{config.title} {config.version}".strip()
    title_clip = _safe_text_clip(title_text, fontsize=42, color="white", stroke_width=3)
    title_clip = title_clip.set_duration(duration).set_position((40, 60))

    subtitle_clips = []
    for sub in subtitles:
        txt = _safe_text_clip(sub.text, fontsize=48, color="white", stroke_width=3)
        txt = txt.set_start(sub.start).set_end(min(sub.end, duration)).set_position(("center", int(config.height * 0.78)))
        subtitle_clips.append(txt)

    flash = ColorClip(size, color=(255, 255, 255)).set_duration(duration)
    flash = flash.set_opacity(lambda t: flash_opacity(t, beats)).set_position((0, 0))

    outro_bg = ColorClip(size, color=(10, 10, 18)).set_start(duration).set_duration(config.outro_seconds)
    outro_title = _safe_text_clip("来抖音 发现更多创作者", fontsize=52, color="white", stroke_width=2)
    outro_title = outro_title.set_start(duration).set_duration(config.outro_seconds).set_position(("center", int(config.height * 0.38)))
    search_text = f"搜索：{config.account}" if config.account else "搜索：你的账号"
    outro_search = _safe_text_clip(search_text, fontsize=46, color="white", stroke_width=2)
    outro_search = outro_search.set_start(duration).set_duration(config.outro_seconds).set_position(("center", int(config.height * 0.48)))

    composite = CompositeVideoClip(
        [background, character, title_clip, *subtitle_clips, flash, outro_bg, outro_title, outro_search],
        size=size,
    ).set_duration(total_duration)
    composite = composite.set_audio(audio.set_duration(duration))

    composite.write_videofile(
        str(output),
        fps=config.fps,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4,
    )
    audio.close()
    composite.close()
    return output
