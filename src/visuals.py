from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from moviepy.editor import VideoClip
from PIL import Image, ImageDraw, ImageFilter

from .effects import beat_scale, flash_opacity, shake_offset
from .renderer import _pil_character_image, _pil_cover_image


def _make_default_background(size: tuple[int, int]) -> Image.Image:
    """Create a dark blue-purple vertical gradient background."""
    width, height = size
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (8 + 34 * x + 22 * y).astype(np.uint8)
    frame[:, :, 1] = (6 + 18 * y).astype(np.uint8)
    frame[:, :, 2] = (35 + 120 * (1 - y) + 60 * x).clip(0, 255).astype(np.uint8)
    return Image.fromarray(frame, mode="RGB")


def _background_image(background_path: str | None, size: tuple[int, int]) -> Image.Image:
    if background_path and Path(background_path).exists() and Path(background_path).suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        return _pil_cover_image(background_path, size).convert("RGB")
    return _make_default_background(size)


def _light_overlay(size: tuple[int, int], t: float) -> Image.Image:
    """Generate moving light streaks and soft glow."""
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Vertical neon streaks.
    spacing = max(36, width // 9)
    shift = int((t * 75) % spacing)
    for x in range(-spacing, width + spacing, spacing):
        xx = x + shift
        alpha = 52 + int(26 * math.sin(t * 2.0 + xx * 0.03))
        draw.line((xx, 0, xx + int(width * 0.08), height), fill=(140, 96, 255, max(20, alpha)), width=3)
        draw.line((xx + 4, 0, xx + int(width * 0.08) + 4, height), fill=(80, 190, 255, 22), width=1)

    # Radial glow around the character zone.
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx = width // 2 + int(20 * math.sin(t * 1.4))
    cy = int(height * 0.43)
    radius = int(min(width, height) * (0.42 + 0.04 * math.sin(t * 2.3)))
    for i in range(8):
        r = int(radius * (1 - i * 0.085))
        alpha = max(0, 18 - i * 2)
        gd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(110, 60, 255, alpha))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=22))
    overlay.alpha_composite(glow)
    return overlay


def _paste_with_shadow(base: Image.Image, character: Image.Image, x: int, y: int) -> None:
    """Paste character with a soft purple glow/shadow."""
    shadow = Image.new("RGBA", character.size, (0, 0, 0, 0))
    if character.mode != "RGBA":
        character = character.convert("RGBA")
    alpha = character.getchannel("A")
    glow = Image.new("RGBA", character.size, (120, 70, 255, 120))
    glow.putalpha(alpha.filter(ImageFilter.GaussianBlur(radius=12)))
    shadow.alpha_composite(glow)
    base.alpha_composite(shadow, (x, y))
    base.alpha_composite(character, (x, y))


def make_animated_main_visual(
    background_path: str | None,
    character_path: str,
    duration: float,
    beats: list[float],
    size: tuple[int, int],
) -> VideoClip:
    """Create the animated main visual layer as frame-by-frame PIL images.

    This avoids MoviePy callable resize/position effects while still allowing
    beat-driven scale, shake, flash, glow, and moving background light streaks.
    """
    width, height = size
    bg = _background_image(background_path, size)
    base_character_height = int(height * 0.56)
    character_base = _pil_character_image(character_path, target_height=base_character_height)

    def make_frame(t: float) -> np.ndarray:
        frame = bg.copy().convert("RGBA")
        frame.alpha_composite(_light_overlay(size, t))

        # Beat pulse and small idle breathing.
        scale = beat_scale(t, beats, base=1.0, pulse=0.12, decay=0.22)
        target_h = max(1, int(character_base.height * scale))
        target_w = max(1, int(character_base.width * scale))
        char = character_base.resize((target_w, target_h), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.BICUBIC)

        sx, sy = shake_offset(t, beats, amount=max(4, width // 45))
        x = int(width / 2 - target_w / 2 + sx)
        y = int(height * 0.43 - target_h / 2 + sy)
        _paste_with_shadow(frame, char, x, y)

        flash = flash_opacity(t, beats, duration=0.09)
        if flash > 0:
            white = Image.new("RGBA", size, (255, 255, 255, int(255 * flash)))
            frame.alpha_composite(white)

        return np.array(frame.convert("RGB"))

    return VideoClip(make_frame=make_frame, duration=duration)
