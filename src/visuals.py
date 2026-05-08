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


def _zoom_rgba_layer(layer: Image.Image, zoom: float) -> Image.Image:
    if zoom <= 1.001:
        return layer
    width, height = layer.size
    resample = _resample_filter()
    zw, zh = max(1, int(width * zoom)), max(1, int(height * zoom))
    enlarged = layer.resize((zw, zh), resample)
    left = max(0, (zw - width) // 2)
    top = max(0, (zh - height) // 2)
    return enlarged.crop((left, top, left + width, top + height))


def _make_default_background(size: tuple[int, int]) -> Image.Image:
    """Horizontal cinematic black background with subtle red/purple vignette."""
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width * 0.52, height * 0.50
    dist = np.sqrt(((xx - cx) / max(width, 1)) ** 2 + ((yy - cy) / max(height, 1)) ** 2)
    glow = np.clip(1.0 - dist * 2.55, 0, 1)
    side_glow = np.clip(1.0 - np.abs((xx - width * 0.52) / max(width, 1)) * 2.2, 0, 1) * 0.30
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (1 + 18 * glow + 5 * side_glow).astype(np.uint8)
    frame[:, :, 1] = (0 + 4 * glow).astype(np.uint8)
    frame[:, :, 2] = (2 + 15 * glow + 4 * side_glow).astype(np.uint8)
    return Image.fromarray(frame, mode="RGB")


def _background_image(background_path: str | None, size: tuple[int, int]) -> Image.Image:
    if background_path and Path(background_path).exists() and Path(background_path).suffix.lower() not in VIDEO_SUFFIXES:
        return _pil_cover_image(background_path, size).convert("RGB")
    return _make_default_background(size)


def _draw_realistic_petal(draw: ImageDraw.ImageDraw, x: float, y: float, size: float, angle: float, alpha: int, tone: int = 0) -> None:
    """Draw a soft loose petal."""
    r = max(3, float(size))
    ca, sa = math.cos(angle), math.sin(angle)
    raw = [
        (0.00, -1.42), (0.25, -1.10), (0.62, -0.28),
        (0.48, 0.60), (0.10, 1.18), (-0.22, 0.92),
        (-0.58, 0.20), (-0.36, -0.80),
    ]
    pts = []
    for px, py in raw:
        rx = x + (px * ca - py * sa) * r
        ry = y + (px * sa + py * ca) * r
        pts.append((rx, ry))
    if tone == 1:
        fill = (248, 210, 232, alpha)
        edge = (255, 242, 250, min(255, alpha + 22))
    else:
        fill = (232, 150, 205, alpha)
        edge = (255, 220, 240, min(255, alpha + 18))
    draw.polygon(pts, fill=fill)
    draw.line(pts + [pts[0]], fill=edge, width=max(1, int(r * 0.07)))
    x1 = x - math.sin(angle) * r * 0.62
    y1 = y + math.cos(angle) * r * 0.62
    x2 = x + math.sin(angle) * r * 0.72
    y2 = y - math.cos(angle) * r * 0.72
    draw.line((x1, y1, x2, y2), fill=(255, 245, 250, max(16, alpha // 4)), width=1)


def _draw_sakura_blossom(draw: ImageDraw.ImageDraw, x: float, y: float, size: float, angle: float, alpha: int) -> None:
    """Draw a small five-petal sakura blossom mixed with loose petals."""
    r = max(4, float(size))
    center_color = (255, 230, 110, min(255, alpha + 20))
    for k in range(5):
        a = angle + k * math.tau / 5
        px = x + math.cos(a) * r * 0.42
        py = y + math.sin(a) * r * 0.42
        petal_angle = a + math.pi / 2
        ca, sa = math.cos(petal_angle), math.sin(petal_angle)
        raw = [(0, -1.05), (0.45, -0.25), (0.34, 0.72), (0, 0.95), (-0.34, 0.72), (-0.45, -0.25)]
        pts = []
        for rx0, ry0 in raw:
            rx = px + (rx0 * ca - ry0 * sa) * r * 0.55
            ry = py + (rx0 * sa + ry0 * ca) * r * 0.55
            pts.append((rx, ry))
        draw.polygon(pts, fill=(255, 188, 226, alpha), outline=(255, 232, 246, min(255, alpha + 15)))
    draw.ellipse((x - r * 0.13, y - r * 0.13, x + r * 0.13, y + r * 0.13), fill=center_color)


def _floral_layer(size: tuple[int, int], t: float, layer: str, beat: float = 0.0) -> Image.Image:
    """Slow rising sakura blossoms and petals with strong foreground/background depth."""
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if layer == "far":
        count, speed, base_size, blur, alpha_base, drift_scale, blossom_ratio = 58, 0.020, height * 0.008, 1.0, 46, 0.018, 0.28
    elif layer == "mid":
        count, speed, base_size, blur, alpha_base, drift_scale, blossom_ratio = 42, 0.032, height * 0.017, 1.4, 84, 0.034, 0.34
    elif layer == "near":
        count, speed, base_size, blur, alpha_base, drift_scale, blossom_ratio = 24, 0.045, height * 0.043, 5.2, 125, 0.065, 0.22
    else:
        count, speed, base_size, blur, alpha_base, drift_scale, blossom_ratio = 10, 0.054, height * 0.085, 10.0, 90, 0.100, 0.16

    for i in range(count):
        seed = i * 101.37 + {"far": 0, "mid": 500, "near": 1000}.get(layer, 1500)
        progress = (t * speed + (i * 0.61803398875)) % 1.0
        x_base = (math.sin(seed) * 0.5 + 0.5) * width
        drift = math.sin(t * (0.24 + i * 0.002) + seed) * width * drift_scale
        expand = beat * width * (0.016 if layer == "far" else 0.032 if layer == "mid" else 0.060)
        side = -1 if x_base < width / 2 else 1
        x = x_base + drift + side * expand
        y = height + base_size * 7 - progress * (height + base_size * 14) - beat * height * (0.014 if layer == "far" else 0.030 if layer == "mid" else 0.052)
        angle = seed * 0.08 + t * (0.25 + i * 0.002) + beat * 0.24
        size_factor = 0.70 + 0.50 * (math.sin(seed * 0.07) * 0.5 + 0.5)
        item_size = base_size * size_factor * (1.0 + beat * (0.12 if layer == "far" else 0.22 if layer == "mid" else 0.34))
        alpha = int(alpha_base * (0.48 + 0.52 * math.sin(progress * math.pi)) * (1.0 + beat * 0.22))
        use_blossom = (abs(math.sin(seed * 0.13)) < blossom_ratio)
        if use_blossom:
            _draw_sakura_blossom(draw, x, y, item_size * 1.15, angle, min(alpha, 210))
        else:
            _draw_realistic_petal(draw, x, y, item_size, angle, min(alpha, 220), tone=i % 2)

    if blur > 0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur))
    return overlay


def _speed_line_overlay(size: tuple[int, int], t: float, beats: list[float]) -> Image.Image:
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = width * 0.52, height * 0.50
    boost = _beat_strength(t, beats, duration=0.16)
    if boost <= 0.01:
        return overlay
    for i in range(42):
        angle = (i / 42) * math.tau + 0.03 * math.sin(t)
        inner = min(width, height) * (0.15 + 0.03 * boost)
        outer = max(width, height) * (0.50 + 0.20 * ((i % 5) / 5) + 0.09 * boost)
        x1 = cx + math.cos(angle) * inner
        y1 = cy + math.sin(angle) * inner * 0.58
        x2 = cx + math.cos(angle) * outer
        y2 = cy + math.sin(angle) * outer * 0.65
        alpha = int((18 + (i % 4) * 6) * boost)
        draw.line((x1, y1, x2, y2), fill=(255, 225, 245, min(alpha, 86)), width=1)
    return overlay.filter(ImageFilter.GaussianBlur(radius=0.55))


def _background_effects(size: tuple[int, int], t: float, beats: list[float]) -> Image.Image:
    width, height = size
    beat = _beat_strength(t, beats, duration=0.20)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy = width // 2, height // 2
    for i in range(7):
        rx = int(width * (0.18 + i * 0.050 + beat * 0.025))
        ry = int(height * (0.24 + i * 0.060 + beat * 0.030))
        alpha = max(0, 16 - i * 2) + int(13 * beat)
        gd.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(150, 42, 118, alpha))
    overlay.alpha_composite(glow.filter(ImageFilter.GaussianBlur(radius=26)))
    overlay.alpha_composite(_speed_line_overlay(size, t, beats))
    overlay.alpha_composite(_floral_layer(size, t, "far", beat))
    overlay.alpha_composite(_floral_layer(size, t, "mid", beat))
    overlay = _zoom_rgba_layer(overlay, 1.0 + 0.065 * beat)
    flash = flash_opacity(t, beats, duration=0.10)
    if flash > 0:
        overlay.alpha_composite(Image.new("RGBA", size, (255, 220, 245, int(58 * flash))))
    return overlay


def _foreground_floral_effects(size: tuple[int, int], t: float, beats: list[float]) -> Image.Image:
    beat = _beat_strength(t, beats, duration=0.18)
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    layer.alpha_composite(_floral_layer(size, t, "near", beat))
    layer.alpha_composite(_floral_layer(size, t, "front", beat))
    return _zoom_rgba_layer(layer, 1.0 + 0.045 * beat)


def _paste_with_shadow(base: Image.Image, character: Image.Image, x: int, y: int, strength: int = 105) -> None:
    if character.mode != "RGBA":
        character = character.convert("RGBA")
    alpha = character.getchannel("A")
    glow = Image.new("RGBA", character.size, (220, 80, 180, strength))
    glow.putalpha(alpha.filter(ImageFilter.GaussianBlur(radius=12)))
    base.alpha_composite(glow, (x, y))
    base.alpha_composite(character, (x, y))


def _slow_motion_character_frame(frames: list[Image.Image], t: float, beats: list[float], fps: float = 10.0) -> Image.Image:
    if len(frames) == 1:
        return frames[0]
    dt = _time_since_last_beat(t, beats)
    if dt is not None and 0 <= dt <= 0.24:
        idx = int((t - dt * 0.62) * fps)
    else:
        idx = int(t * fps)
    return frames[idx % len(frames)]


def _jump_offset(t: float, beats: list[float], height: int) -> int:
    best = 0.0
    for beat in beats:
        dt = t - beat
        if 0 <= dt <= 0.36:
            phase = dt / 0.36
            best = max(best, math.sin(math.pi * phase))
        elif beat > t:
            break
    return -int(height * 0.095 * best)


def make_animated_main_visual(
    background_path: str | None,
    character_path: str,
    duration: float,
    beats: list[float],
    size: tuple[int, int],
) -> VideoClip:
    """Horizontal cinematic visual layer.

    - black cinematic base
    - slow rising petals plus full sakura blossoms
    - far/mid/near/front depth layers
    - near/front layers are intentionally blurred
    - background and floral layers scale on beat
    - character slows briefly and jumps on every beat
    """
    width, height = size
    bg_base = _background_image(background_path, size)
    base_character_height = int(height * 0.74)
    character_frames = _load_character_frames(character_path, target_height=base_character_height)
    resample = _resample_filter()

    def make_frame(t: float) -> np.ndarray:
        beat = _beat_strength(t, beats, duration=0.20)
        bg_zoom = 1.0 + 0.078 * beat
        if background_path and Path(background_path).exists() and Path(background_path).suffix.lower() not in VIDEO_SUFFIXES:
            bg = _pil_cover_image(background_path, size, zoom=bg_zoom).convert("RGBA")
            bg = Image.blend(Image.new("RGBA", size, (0, 0, 0, 255)), bg, 0.30)
        else:
            zoomed_size = (int(width * bg_zoom), int(height * bg_zoom))
            zoomed = bg_base.resize(zoomed_size, resample)
            left = max(0, (zoomed.width - width) // 2)
            top = max(0, (zoomed.height - height) // 2)
            bg = zoomed.crop((left, top, left + width, top + height)).convert("RGBA")

        frame = bg
        frame.alpha_composite(_background_effects(size, t, beats))

        character_base = _slow_motion_character_frame(character_frames, t, beats)
        scale = beat_scale(t, beats, base=1.0, pulse=0.060, decay=0.20)
        target_h = max(1, int(character_base.height * scale))
        target_w = max(1, int(character_base.width * scale))
        char = character_base.resize((target_w, target_h), resample)

        sway = int(math.sin(t * 1.35) * width * 0.012)
        jump = _jump_offset(t, beats, height)
        x = int(width * 0.50 - target_w / 2 + sway)
        y = int(height * 0.54 - target_h / 2 + jump)
        _paste_with_shadow(frame, char, x, y, strength=105)

        frame.alpha_composite(_foreground_floral_effects(size, t, beats))
        return np.array(frame.convert("RGB"))

    return VideoClip(make_frame=make_frame, duration=duration)
