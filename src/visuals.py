from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from moviepy.editor import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageSequence

from .effects import beat_scale, flash_opacity, shake_offset

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _resample_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC


def _pil_cover_image(path: str | Path, size: tuple[int, int]) -> Image.Image:
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


def _resize_rgba_by_height(image: Image.Image, target_height: int) -> Image.Image:
    image = image.convert("RGBA")
    src_w, src_h = image.size
    scale = target_height / src_h
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    return image.resize((new_w, new_h), _resample_filter())


def _load_character_frames(character_path: str | Path, target_height: int) -> list[Image.Image]:
    path = Path(character_path)
    if path.is_dir():
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
        if not files:
            raise FileNotFoundError(f"角色帧目录为空：{path}")
        return [_resize_rgba_by_height(Image.open(p), target_height) for p in files]

    if not path.exists():
        raise FileNotFoundError(f"角色素材不存在：{path}")

    if path.suffix.lower() == ".gif":
        image = Image.open(path)
        frames = [_resize_rgba_by_height(frame.copy(), target_height) for frame in ImageSequence.Iterator(image)]
        return frames or [_resize_rgba_by_height(image, target_height)]

    return [_resize_rgba_by_height(Image.open(path), target_height)]


def _make_default_background(size: tuple[int, int]) -> Image.Image:
    """Create a near-black background with a slight purple center vignette."""
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width * 0.50, height * 0.46
    dist = np.sqrt(((xx - cx) / max(width, 1)) ** 2 + ((yy - cy) / max(height, 1)) ** 2)
    glow = np.clip(1.0 - dist * 2.9, 0, 1)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (2 + 32 * glow).astype(np.uint8)
    frame[:, :, 1] = (1 + 10 * glow).astype(np.uint8)
    frame[:, :, 2] = (4 + 34 * glow).astype(np.uint8)
    return Image.fromarray(frame, mode="RGB")


def _background_image(background_path: str | None, size: tuple[int, int]) -> Image.Image:
    # If no custom background is supplied, use the black sakura style.
    # If a user supplies a background image, keep it but still add petals/light streaks over it.
    if background_path and Path(background_path).exists() and Path(background_path).suffix.lower() not in VIDEO_SUFFIXES:
        return _pil_cover_image(background_path, size).convert("RGB")
    return _make_default_background(size)


def _draw_petal(draw: ImageDraw.ImageDraw, x: float, y: float, size: float, angle: float, alpha: int, blur_color: bool = False) -> None:
    """Draw one stylized sakura petal as a rotated translucent ellipse/polygon."""
    color = (255, 185, 238, alpha) if not blur_color else (255, 105, 220, alpha)
    r = max(2, int(size))
    # Approximate rotated petal with polygon points.
    ca, sa = math.cos(angle), math.sin(angle)
    pts = []
    raw = [(0, -1.2), (0.72, -0.25), (0.45, 0.95), (0, 1.25), (-0.45, 0.95), (-0.72, -0.25)]
    for px, py in raw:
        rx = x + (px * ca - py * sa) * r
        ry = y + (px * sa + py * ca) * r
        pts.append((rx, ry))
    draw.polygon(pts, fill=color)


def _petal_overlay(size: tuple[int, int], t: float) -> Image.Image:
    """Generate black-background sakura petals with depth and radial motion."""
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = width * 0.50, height * 0.43

    # Large blurred petals / bokeh blobs like the reference image.
    bokeh = Image.new("RGBA", size, (0, 0, 0, 0))
    bd = ImageDraw.Draw(bokeh)
    for i in range(20):
        seed = i * 37.17
        angle = seed + t * (0.55 + (i % 5) * 0.08)
        radius = (min(width, height) * 0.10 + ((i * 97 + int(t * 90)) % int(max(width, height) * 0.74)))
        x = cx + math.cos(angle) * radius * (1.15 + (i % 3) * 0.18)
        y = cy + math.sin(angle * 0.92) * radius * (0.70 + (i % 4) * 0.10)
        petal_size = 10 + (i % 5) * 7
        alpha = 42 + (i % 4) * 18
        _draw_petal(bd, x, y, petal_size, angle + t * 2.0, alpha, blur_color=True)
    bokeh = bokeh.filter(ImageFilter.GaussianBlur(radius=5.5))
    overlay.alpha_composite(bokeh)

    # Sharp foreground petals flying across the frame.
    for i in range(34):
        seed = i * 19.73
        progress = (t * (0.32 + (i % 6) * 0.045) + (i * 0.077)) % 1.0
        angle = seed * 0.17 + math.sin(t * 0.8 + i) * 0.35
        start_radius = min(width, height) * 0.08
        end_radius = max(width, height) * 0.86
        radius = start_radius + progress * end_radius
        side = -1 if i % 2 else 1
        x = cx + math.cos(angle) * radius + side * progress * width * 0.14
        y = cy + math.sin(angle) * radius * 0.72 + math.sin(t * 2.2 + i) * 22
        petal_size = 4 + (i % 7) * 2.2 + progress * 8
        alpha = int(75 + progress * 120)
        _draw_petal(draw, x, y, petal_size, angle + t * 4.8, min(alpha, 220))

    return overlay


def _speed_line_overlay(size: tuple[int, int], t: float, beats: list[float]) -> Image.Image:
    """Draw faint radial white/pink speed lines on black background."""
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = width * 0.50, height * 0.44
    beat_boost = 1.0 + 1.7 * flash_opacity(t, beats, duration=0.12)
    for i in range(46):
        angle = (i / 46) * math.tau + 0.13 * math.sin(t * 1.7)
        inner = min(width, height) * (0.18 + 0.04 * math.sin(i + t * 2.0))
        outer = max(width, height) * (0.62 + 0.12 * ((i % 5) / 5))
        x1 = cx + math.cos(angle) * inner
        y1 = cy + math.sin(angle) * inner * 0.68
        x2 = cx + math.cos(angle) * outer
        y2 = cy + math.sin(angle) * outer * 0.76
        alpha = int((18 + (i % 4) * 8) * beat_boost)
        color = (255, 210, 245, min(alpha, 90)) if i % 3 else (255, 255, 255, min(alpha, 75))
        draw.line((x1, y1, x2, y2), fill=color, width=1 if i % 4 else 2)
    return overlay.filter(ImageFilter.GaussianBlur(radius=0.6))


def _light_overlay(size: tuple[int, int], t: float, beats: list[float]) -> Image.Image:
    """Generate black sakura-style background effects."""
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))

    # Very soft center glow only; avoid the old blue vertical-line look.
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx = width // 2 + int(10 * math.sin(t * 1.2))
    cy = int(height * 0.43)
    for i in range(8):
        r = int(min(width, height) * (0.26 + i * 0.055))
        alpha = max(0, 20 - i * 2)
        gd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(170, 55, 160, alpha))
    overlay.alpha_composite(glow.filter(ImageFilter.GaussianBlur(radius=20)))
    overlay.alpha_composite(_speed_line_overlay(size, t, beats))
    overlay.alpha_composite(_petal_overlay(size, t))

    flash = flash_opacity(t, beats, duration=0.10)
    if flash > 0:
        beat = Image.new("RGBA", size, (255, 210, 245, int(120 * flash)))
        overlay.alpha_composite(beat)
    return overlay


def _paste_with_shadow(base: Image.Image, character: Image.Image, x: int, y: int, strength: int = 130) -> None:
    if character.mode != "RGBA":
        character = character.convert("RGBA")
    alpha = character.getchannel("A")
    # Pink/purple glow to fit sakura background.
    glow = Image.new("RGBA", character.size, (255, 95, 210, strength))
    glow.putalpha(alpha.filter(ImageFilter.GaussianBlur(radius=13)))
    base.alpha_composite(glow, (x, y))
    base.alpha_composite(character, (x, y))


def _character_frame(frames: list[Image.Image], t: float, beats: list[float], fps: float = 10.0) -> Image.Image:
    if len(frames) == 1:
        return frames[0]
    base_idx = int(t * fps)
    near_beat = any(0 <= t - beat <= 0.18 for beat in beats)
    if near_beat:
        base_idx += 1
    return frames[base_idx % len(frames)]


def make_animated_main_visual(
    background_path: str | None,
    character_path: str,
    duration: float,
    beats: list[float],
    size: tuple[int, int],
) -> VideoClip:
    """Create an animated main visual layer frame-by-frame.

    Static PNG only supports pulse/shake. For real dance/turn effects, pass a GIF
    or a directory containing transparent PNG animation frames.
    """
    width, height = size
    bg = _background_image(background_path, size)
    base_character_height = int(height * 0.58)
    character_frames = _load_character_frames(character_path, target_height=base_character_height)
    resample = _resample_filter()

    def make_frame(t: float) -> np.ndarray:
        frame = bg.copy().convert("RGBA")
        # If a custom image background is provided, darken it so petals/character remain visible.
        frame = Image.blend(Image.new("RGBA", size, (0, 0, 0, 255)), frame, 0.38)
        frame.alpha_composite(_light_overlay(size, t, beats))

        character_base = _character_frame(character_frames, t, beats)
        scale = beat_scale(t, beats, base=1.0, pulse=0.10, decay=0.20)
        target_h = max(1, int(character_base.height * scale))
        target_w = max(1, int(character_base.width * scale))
        char = character_base.resize((target_w, target_h), resample)

        sx, sy = shake_offset(t, beats, amount=max(3, width // 55))
        x = int(width / 2 - target_w / 2 + sx)
        y = int(height * 0.43 - target_h / 2 + sy)

        if len(character_frames) == 1:
            phase = math.sin(t * 2.5)
            squeeze = 0.94 + 0.06 * abs(phase)
            squeeze_w = max(1, int(char.width * squeeze))
            char = char.resize((squeeze_w, char.height), resample)
            if phase < -0.88:
                char = char.transpose(Image.FLIP_LEFT_RIGHT)
            x = int(width / 2 - char.width / 2 + sx)

        _paste_with_shadow(frame, char, x, y, strength=120)
        return np.array(frame.convert("RGB"))

    return VideoClip(make_frame=make_frame, duration=duration)
