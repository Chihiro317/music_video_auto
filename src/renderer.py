from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy.editor import AudioFileClip, ColorClip, CompositeVideoClip, ImageClip, VideoFileClip
from PIL import Image

from .beat_detect import detect_beats, get_audio_duration
from .effects import flash_opacity, shake_offset
from .subtitle import load_srt
from .text_utils import make_text_clip


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


def _resample_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC


def _pil_cover_image(path: str | Path, size: tuple[int, int]) -> Image.Image:
    """Resize and center-crop an image to cover target size with PIL."""
    width, height = size
    image = Image.open(path).convert("RGB")
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    image = image.resize((new_w, new_h), _resample_filter())
    left = max(0, (new_w - width) // 2)
    top = max(0, (new_h - height) // 2)
    return image.crop((left, top, left + width, top + height))


def _pil_character_image(path: str | Path, target_height: int) -> Image.Image:
    """Resize a character image by height with PIL while preserving alpha."""
    image = Image.open(path).convert("RGBA")
    src_w, src_h = image.size
    scale = target_height / src_h
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    return image.resize((new_w, new_h), _resample_filter())


def _make_background(path: str | None, duration: float, size: tuple[int, int]):
    width, height = size
    if not path:
        return ColorClip(size, color=(12, 8, 24)).set_duration(duration)

    bg_path = Path(path)
    if not bg_path.exists():
        raise FileNotFoundError(f"背景素材不存在：{bg_path}")

    if bg_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        # Video backgrounds still use MoviePy resize/crop. Image backgrounds avoid MoviePy resize for CI stability.
        clip = VideoFileClip(str(bg_path)).without_audio()
        if clip.duration < duration:
            clip = clip.loop(duration=duration)
        else:
            clip = clip.subclip(0, duration)
        clip = clip.resize(height=height)
        return clip.crop(x_center=clip.w / 2, y_center=clip.h / 2, width=width, height=height)

    bg_image = _pil_cover_image(bg_path, size)
    return ImageClip(np.array(bg_image)).set_duration(duration)


def _safe_text_clip(text: str, fontsize: int, max_width: int, stroke_width: int = 2, align: str = "center"):
    """Create a text clip via PIL instead of MoviePy TextClip/ImageMagick."""
    return make_text_clip(
        text=text,
        fontsize=fontsize,
        max_width=max_width,
        stroke_width=stroke_width,
        align=align,
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

    max_character_height = int(config.height * 0.58)
    character_image = _pil_character_image(config.character, target_height=max_character_height)
    character = ImageClip(np.array(character_image)).set_duration(duration)

    def char_position(t: float):
        x_shake, y_shake = shake_offset(t, beats)
        return ("center", int(config.height * 0.43 - character.h / 2 + y_shake))

    # Keep beat-driven position shake. Stable beat-driven scale will be added via a custom frame function later.
    character = character.set_position(char_position)

    title_text = f"{config.title} {config.version}".strip()
    title_clip = _safe_text_clip(title_text, fontsize=max(24, int(config.height * 0.022)), max_width=int(config.width * 0.82), stroke_width=3, align="left")
    title_clip = title_clip.set_duration(duration).set_position((40, 60))

    subtitle_clips = []
    for sub in subtitles:
        txt = _safe_text_clip(sub.text, fontsize=max(28, int(config.height * 0.028)), max_width=int(config.width * 0.88), stroke_width=3)
        txt = txt.set_start(sub.start).set_end(min(sub.end, duration)).set_position(("center", int(config.height * 0.78)))
        subtitle_clips.append(txt)

    flash = ColorClip(size, color=(255, 255, 255)).set_duration(duration)
    flash = flash.set_opacity(lambda t: flash_opacity(t, beats)).set_position((0, 0))

    outro_bg = ColorClip(size, color=(10, 10, 18)).set_start(duration).set_duration(config.outro_seconds)
    outro_title = _safe_text_clip("来抖音 发现更多创作者", fontsize=max(32, int(config.height * 0.03)), max_width=int(config.width * 0.9), stroke_width=2)
    outro_title = outro_title.set_start(duration).set_duration(config.outro_seconds).set_position(("center", int(config.height * 0.38)))
    search_text = f"搜索：{config.account}" if config.account else "搜索：你的账号"
    outro_search = _safe_text_clip(search_text, fontsize=max(30, int(config.height * 0.028)), max_width=int(config.width * 0.88), stroke_width=2)
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
