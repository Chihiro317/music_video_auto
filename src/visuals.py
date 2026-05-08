from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from moviepy.editor import VideoClip
from PIL import Image, ImageDraw, ImageFilter, ImageSequence

from .effects import beat_scale, flash_opacity

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _resample_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC


def _pil_cover_image(path: str | Path, size: tuple[int, int], zoom: float = 1.0) -> Image.Image:
    """Resize and center-crop an image to cover target size with optional zoom."""
    width, height = size
    image = Image.open(path).convert("RGB")
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h) * zoom
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


def _beat_strength(t: float, beats: list[float], duration: float = 0.22) -> float:
    """Return a decayed strength after the closest previous beat."""
    strength = 0.0
    for beat in beats:
        dt = t - beat
        if 0 <= dt <= duration:
            strength = max(strength, 1.0 - dt / duration)
        elif beat > t:
            break
    return strength


def _time_since_last_beat(t: float, beats: list[float]) -> float | None:
    previous = None
    for beat in beats:
        if beat <= t:
            previous = beat
        else:
            break
    return None if previous is None else t - previous


def _make_default_background(size: tuple[int, int]) -> Image.Image:
    """Pure black base with slight pink-purple center vignette."""
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width * 0.50, height * 0.46
    dist = np.sqrt(((xx - cx) / max(width, 1)) ** 2 + ((yy - cy) / max(height, 1)) ** 2)
    glow = np.clip(1.0 - dist * 3.2, 0, 1)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (1 + 24 * glow).astype(np.uint8)
    frame[:, :, 1] = (0 + 7 * glow).astype(np.uint8)
    frame[:, :, 2] = (2 + 20 * glow).astype(np.uint8)
    return Image.fromarray(frame, mode="RGB")


def _background_image(background_path: str | None, size: tuple[int, int]) -> Image.Image:
    if background_path and Path(background_path).exists() and Path(background_path).suffix.lower() not in VIDEO_SUFFIXES:
        return _pil_cover_image(background_path, size).convert("RGB")
    return _make_default_background(size)


def _draw_realistic_petal(draw: ImageDraw.ImageDraw, x: float, y: float, size: float, angle: float, alpha: int) -> None:
    """Draw a softer, asymmetric petal instead of a simple polygon."""
    r = max(3, float(size))
    ca, sa = math.cos(angle), math.sin(angle)
    raw = [
        (0.00, -1.35),
        (0.34, -0.95),
        (0.72, -0.20),
        (0.50, 0.70),
        (0.12, 1.18),
        (-0.25, 0.96),
        (-0.60, 0.20),
        (-0.42, -0.78),
    ]
    pts = []
    for px, py in raw:
        rx = x + (px * ca - py * sa) * r
        ry = y + (px * sa + py * ca) * r
        pts.append((rx, ry))
    fill = (255, 188, 230, alpha)
    edge = (255, 232, 248, min(255, alpha + 25))
    draw.polygon(pts, fill=fill)
    draw.line(pts + [pts[0]], fill=edge, width=max(1, int(r * 0.10)))
    # small vein
    x1 = x - math.sin(angle) * r * 0.65
    y1 = y + math.cos(angle) * r * 0.65
    x2 = x + math.sin(angle) * r * 0.75
    y2 = y - math.cos(angle) * r * 0.75
    draw.line((x1, y1, x2, y2), fill=(255, 245, 250, max(20, alpha // 3)), width=1)


def _petal_layer(size: tuple[int, int], t: float, layer: str) -> Image.Image:
    """Slow upward petals with depth: far/mid/near layers."""
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if layer == "far":
        count, speed, base_size, blur, alpha_base = 34, 0.035, 4.0, 0.5, 70
    elif layer == "mid":
        count, speed, base_size, blur, alpha_base = 22, 0.055, 7.0, 1.2, 105
    else:
        count, speed, base_size, blur, alpha_base = 13, 0.075, 15.0, 4.0, 135

    for i in range(count):
        seed = i * 91.731 + (0 if layer == "far" else 300 if layer == "mid" else 800)
        # progress goes upward: y moves from below screen to above screen slowly.
        progress = (t * speed + (i * 0.61803398875)) % 1.0
        x_base = ((math.sin(seed) * 0.5 + 0.5) * width)
        drift = math.sin(t * (0.35 + i * 0.003) + seed) * width * (0.025 if layer == "far" else 0.045 if layer == "mid" else 0.075)
        x = x_base + drift
        y = height + base_size * 6 - progress * (height + base_size * 12)
        angle = seed * 0.11 + t * (0.45 + i * 0.004)
        size_factor = 0.75 + 0.55 * (math.sin(seed * 0.07) * 0.5 + 0.5)
        petal_size = base_size * size_factor
        alpha = int(alpha_base * (0.55 + 0.45 * math.sin(progress * math.pi)))
        _draw_realistic_petal(draw, x, y, petal_size, angle, alpha)

    if blur > 0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur))
    return overlay


def _speed_line_overlay(size: tuple[int, int], t: float, beats: list[float]) -> Image.Image:
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = width * 0.50, height * 0.44
    boost = _beat_strength(t, beats, duration=0.16)
    if boost <= 0.01:
        return overlay
    for i in range(36):
        angle = (i / 36) * math.tau + 0.06 * math.sin(t)
        inner = min(width, height) * (0.16 + 0.04 * math.sin(i))
        outer = max(width, height) * (0.54 + 0.18 * ((i % 5) / 5))
        x1 = cx + math.cos(angle) * inner
        y1 = cy + math.sin(angle) * inner * 0.70
        x2 = cx + math.cos(angle) * outer
        y2 = cy + math.sin(angle) * outer * 0.78
        alpha = int((22 + (i % 4) * 8) * boost)
        draw.line((x1, y1, x2, y2), fill=(255, 220, 248, min(alpha, 100)), width=1 if i % 4 else 2)
    return overlay.filter(ImageFilter.GaussianBlur(radius=0.7))


def _background_effects(size: tuple[int, int], t: float, beats: list[float]) -> Image.Image:
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    # soft center glow
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy = width // 2, int(height * 0.43)
    boost = _beat_strength(t, beats, duration=0.20)
    for i in range(8):
        r = int(min(width, height) * (0.20 + i * 0.055 + boost * 0.018))
        alpha = max(0, 18 - i * 2) + int(10 * boost)
        gd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(160, 45, 135, alpha))
    overlay.alpha_composite(glow.filter(ImageFilter.GaussianBlur(radius=22)))
    overlay.alpha_composite(_speed_line_overlay(size, t, beats))
    # far and mid petals are behind the character.
    overlay.alpha_composite(_petal_layer(size, t, "far"))
    overlay.alpha_composite(_petal_layer(size, t, "mid"))
    flash = flash_opacity(t, beats, duration=0.10)
    if flash > 0:
        overlay.alpha_composite(Image.new("RGBA", size, (255, 210, 245, int(80 * flash))))
    return overlay


def _foreground_petal_effects(size: tuple[int, int], t: float) -> Image.Image:
    # near petals pass in front and are intentionally blurred.
    return _petal_layer(size, t, "near")


def _paste_with_shadow(base: Image.Image, character: Image.Image, x: int, y: int, strength: int = 130) -> None:
    if character.mode != "RGBA":
        character = character.convert("RGBA")
    alpha = character.getchannel("A")
    glow = Image.new("RGBA", character.size, (255, 95, 210, strength))
    glow.putalpha(alpha.filter(ImageFilter.GaussianBlur(radius=13)))
    base.alpha_composite(glow, (x, y))
    base.alpha_composite(character, (x, y))


def _slow_motion_character_frame(frames: list[Image.Image], t: float, beats: list[float], fps: float = 10.0) -> Image.Image:
    """Character frame selection: slow down briefly after each beat."""
    if len(frames) == 1:
        return frames[0]
    dt = _time_since_last_beat(t, beats)
    if dt is not None and 0 <= dt <= 0.24:
        # Hold/slow the frame immediately after beat for a staccato slow-motion feel.
        idx = int((t - dt * 0.62) * fps)
    else:
        idx = int(t * fps)
    return frames[idx % len(frames)]


def _jump_offset(t: float, beats: list[float], height: int) -> int:
    """Every beat makes the character jump upward then fall back."""
    best = 0.0
    for beat in beats:
        dt = t - beat
        if 0 <= dt <= 0.36:
            # Parabolic jump: peak around 0.16s.
            phase = dt / 0.36
            jump = math.sin(math.pi * phase)
            best = max(best, jump)
        elif beat > t:
            break
    return -int(height * 0.075 * best)


def make_animated_main_visual(
    background_path: str | None,
    character_path: str,
    duration: float,
    beats: list[float],
    size: tuple[int, int],
) -> VideoClip:
    """Create animated visuals.

    Visual design:
    - black background
    - slow upward sakura petals with far/mid/near depth
    - foreground petals are blurred
    - background beat zoom
    - character beat slow-motion and jump on every beat
    """
    width, height = size
    bg_base = _background_image(background_path, size)
    base_character_height = int(height * 0.58)
    character_frames = _load_character_frames(character_path, target_height=base_character_height)
    resample = _resample_filter()

    def make_frame(t: float) -> np.ndarray:
        beat = _beat_strength(t, beats, duration=0.20)
        bg_zoom = 1.0 + 0.055 * beat
        if background_path and Path(background_path).exists() and Path(background_path).suffix.lower() not in VIDEO_SUFFIXES:
            bg = _pil_cover_image(background_path, size, zoom=bg_zoom).convert("RGBA")
            bg = Image.blend(Image.new("RGBA", size, (0, 0, 0, 255)), bg, 0.35)
        else:
            # Zoom generated background by resizing/cropping.
            zoomed_size = (int(width * bg_zoom), int(height * bg_zoom))
            zoomed = bg_base.resize(zoomed_size, resample)
            left = max(0, (zoomed.width - width) // 2)
            top = max(0, (zoomed.height - height) // 2)
            bg = zoomed.crop((left, top, left + width, top + height)).convert("RGBA")

        frame = bg
        frame.alpha_composite(_background_effects(size, t, beats))

        character_base = _slow_motion_character_frame(character_frames, t, beats)
        scale = beat_scale(t, beats, base=1.0, pulse=0.075, decay=0.20)
        target_h = max(1, int(character_base.height * scale))
        target_w = max(1, int(character_base.width * scale))
        char = character_base.resize((target_w, target_h), resample)

        # Gentle side sway only; jump is beat-driven and visible.
        sway = int(math.sin(t * 1.6) * width * 0.018)
        jump = _jump_offset(t, beats, height)
        x = int(width / 2 - target_w / 2 + sway)
        y = int(height * 0.44 - target_h / 2 + jump)
        _paste_with_shadow(frame, char, x, y, strength=120)

        frame.alpha_composite(_foreground_petal_effects(size, t))
        return np.array(frame.convert("RGB"))

    return VideoClip(make_frame=make_frame, duration=duration)
