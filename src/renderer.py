from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from moviepy.editor import AudioFileClip, ColorClip, CompositeVideoClip

from .beat_detect import detect_beats, get_audio_duration
from .subtitle import load_srt
from .text_utils import make_text_clip
from .visuals import make_animated_main_visual


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


def _safe_text_clip(text: str, fontsize: int, max_width: int, stroke_width: int = 2, align: str = "center"):
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
    main_visual = make_animated_main_visual(
        background_path=config.background,
        character_path=config.character,
        duration=duration,
        beats=beats,
        size=size,
    )

    title_text = f"{config.title} {config.version}".strip()
    title_clip = _safe_text_clip(
        title_text,
        fontsize=max(24, int(config.height * 0.022)),
        max_width=int(config.width * 0.82),
        stroke_width=3,
        align="left",
    )
    title_clip = title_clip.set_duration(duration).set_position((40, 60))

    subtitle_clips = []
    for sub in subtitles:
        txt = _safe_text_clip(
            sub.text,
            fontsize=max(28, int(config.height * 0.028)),
            max_width=int(config.width * 0.88),
            stroke_width=3,
        )
        txt = txt.set_start(sub.start).set_end(min(sub.end, duration)).set_position(("center", int(config.height * 0.78)))
        subtitle_clips.append(txt)

    outro_bg = ColorClip(size, color=(8, 9, 18)).set_start(duration).set_duration(config.outro_seconds)
    outro_title = _safe_text_clip(
        "来抖音 发现更多创作者",
        fontsize=max(28, int(config.height * 0.027)),
        max_width=int(config.width * 0.96),
        stroke_width=2,
    )
    outro_title = outro_title.set_start(duration).set_duration(config.outro_seconds).set_position(("center", int(config.height * 0.36)))
    search_text = f"搜索：{config.account}" if config.account else "搜索：你的账号"
    outro_search = _safe_text_clip(
        search_text,
        fontsize=max(30, int(config.height * 0.028)),
        max_width=int(config.width * 0.9),
        stroke_width=2,
    )
    outro_search = outro_search.set_start(duration).set_duration(config.outro_seconds).set_position(("center", int(config.height * 0.48)))

    composite = CompositeVideoClip(
        [main_visual, title_clip, *subtitle_clips, outro_bg, outro_title, outro_search],
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
