from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

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
    """Load a static PNG, GIF, or directory of PNG frames as character animation.

    A static image still works, but real dance/turn motion requires either a GIF
    or a folder of transparent PNG frames exported from another animation tool.
    """
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
    """Create a dark blue-purple vertical gradient background."""
    width, height = size
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (5 + 25 * x + 30 * y).astype(np.uint8)
    frame[:, :, 1] = (4 + 14 * y).astype(np.uint8)
    frame[:, :, 2] = (30 + 115 * (1 - y) + 80 * x).clip(0, 255).astype(np.uint8)
    return Image.fromarray(frame, mode="RGB")


def _background_image(background_path: str | None, size: tuple[int, int]) -> Image.Image:
    if background_path and Path(background_path).exists() and Path(background_path).suffix.lower() not in VIDEO_SUFFIXES:
        return _pil_cover_image(background_path, size).convert("RGB")
    return _make_default_background(size)


def _light_overlay(size: tuple[int, int], t: float, beats: list[float]) -> Image.Image:
    """Generate moving stage beams, speed lines and center glow."""
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Strong diagonal stage beams coming from the top center.
    top_x = width // 2 + int(32 * math.sin(t * 1.7))
    speed = t * 180
    for i in range(-6, 7):
        end_x = int(width / 2 + i * width * 0.22 + 38 * math.sin(t * 2.1 + i))
        alpha = 40 + int(25 * math.sin(t * 3.0 + i))
        color = (105, 75, 255, max(18, alpha)) if i % 2 else (50, 190, 255, max(14, alpha - 12))
        draw.polygon(
            [
                (top_x - 8, 0),
                (top_x + 8, 0),
                (end_x + 16, height),
                (end_x - 16, height),
            ],
            fill=color,
        )

    # Fast vertical light/speed lines.
    spacing = max(24, width // 14)
    shift = int(speed % spacing)
    for x in range(-spacing, width + spacing, spacing):
        xx = x + shift
        alpha = 65 + int(35 * math.sin(t * 4.0 + xx * 0.05))
        draw.line((xx, 0, xx + int(width * 0.12), height), fill=(165, 125, 255, max(25, alpha)), width=3)
        draw.line((xx + 5, 0, xx + int(width * 0.12) + 5, height), fill=(95, 210, 255, 28), width=1)

    # Center aura.
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx = width // 2 + int(24 * math.sin(t * 1.4))
    cy = int(height * 0.43)
    radius = int(min(width, height) * (0.5 + 0.05 * math.sin(t * 2.3)))
    for i in range(9):
        r = int(radius * (1 - i * 0.08))
        alpha = max(0, 24 - i * 2)
        gd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(125, 70, 255, alpha))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=24))
    overlay.alpha_composite(glow)

    # Beat flash radial ring.
    flash = flash_opacity(t, beats, duration=0.12)
    if flash > 0:
        ring = Image.new("RGBA", size, (0, 0, 0, 0))
        rd = ImageDraw.Draw(ring)
        rr = int(min(width, height) * (0.22 + flash * 0.6))
        rd.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=(255, 255, 255, int(180 * flash)), width=5)
        overlay.alpha_composite(ring.filter(ImageFilter.GaussianBlur(radius=2)))

    return overlay.filter(ImageFilter.GaussianBlur(radius=0.4))


def _paste_with_shadow(base: Image.Image, character: Image.Image, x: int, y: int, strength: int = 130) -> None:
    if character.mode != "RGBA":
        character = character.convert("RGBA")
    alpha = character.getchannel("A")
    glow = Image.new("RGBA", character.size, (135, 85, 255, strength))
    glow.putalpha(alpha.filter(ImageFilter.GaussianBlur(radius=14)))
    base.alpha_composite(glow, (x, y))
    base.alpha_composite(character, (x, y))


def _character_frame(frames: list[Image.Image], t: float, beats: list[float], fps: float = 10.0) -> Image.Image:
    if len(frames) == 1:
        return frames[0]
    # Keep animation moving even without reliable beat detection.
    base_idx = int(t * fps)
    # Snap a little faster around beats.
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
    base_character_height = int(height * 0.56)
    character_frames = _load_character_frames(character_path, target_height=base_character_height)
    resample = _resample_filter()

    def make_frame(t: float) -> np.ndarray:
        frame = bg.copy().convert("RGBA")
        frame.alpha_composite(_light_overlay(size, t, beats))

        character_base = _character_frame(character_frames, t, beats)
        scale = beat_scale(t, beats, base=1.0, pulse=0.12, decay=0.22)
        target_h = max(1, int(character_base.height * scale))
        target_w = max(1, int(character_base.width * scale))
        char = character_base.resize((target_w, target_h), resample)

        sx, sy = shake_offset(t, beats, amount=max(4, width // 42))
        x = int(width / 2 - target_w / 2 + sx)
        y = int(height * 0.43 - target_h / 2 + sy)

        # Simple turn illusion for static frames: slight horizontal squeeze and flip.
        if len(character_frames) == 1:
            phase = math.sin(t * 2.5)
            squeeze = 0.92 + 0.08 * abs(phase)
            squeeze_w = max(1, int(char.width * squeeze))
            char = char.resize((squeeze_w, char.height), resample)
            if phase < -0.85:
                char = char.transpose(Image.FLIP_LEFT_RIGHT)
            x = int(width / 2 - char.width / 2 + sx)

        _paste_with_shadow(frame, char, x, y, strength=150)

        flash = flash_opacity(t, beats, duration=0.1)
        if flash > 0:
            white = Image.new("RGBA", size, (255, 255, 255, int(255 * flash)))
            frame.alpha_composite(white)

        return np.array(frame.convert("RGB"))

    return VideoClip(make_frame=make_frame, duration=duration)
