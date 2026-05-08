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


def _beat_strength(t: float, beats: list[float], duration: float = 0.26) -> float:
    strength = 0.0
    for beat in beats:
        dt = t - beat
        if 0 <= dt <= duration:
            strength = max(strength, (1.0 - dt / duration) ** 0.65)
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


def _motion_blur(layer: Image.Image, beat: float) -> Image.Image:
    if beat <= 0.02:
        return layer
    base = layer.convert("RGBA")
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    out.alpha_composite(base)
    for k, alpha_mul in enumerate((0.38, 0.25, 0.16), start=1):
        z = 1.0 + beat * (0.030 + k * 0.030)
        ghost = _zoom_rgba_layer(base, z).filter(ImageFilter.GaussianBlur(radius=beat * (1.3 + k * 1.2)))
        a = ghost.getchannel("A").point(lambda v, m=alpha_mul: int(v * m))
        ghost.putalpha(a)
        out.alpha_composite(ghost)
    return out


def _make_default_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width * 0.52, height * 0.50
    dist = np.sqrt(((xx - cx) / max(width, 1)) ** 2 + ((yy - cy) / max(height, 1)) ** 2)
    glow = np.clip(1.0 - dist * 2.25, 0, 1)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (1 + 25 * glow).astype(np.uint8)
    frame[:, :, 1] = (0 + 5 * glow).astype(np.uint8)
    frame[:, :, 2] = (2 + 18 * glow).astype(np.uint8)
    return Image.fromarray(frame, mode="RGB")


def _background_image(background_path: str | None, size: tuple[int, int]) -> Image.Image:
    if background_path and Path(background_path).exists() and Path(background_path).suffix.lower() not in VIDEO_SUFFIXES:
        return _pil_cover_image(background_path, size).convert("RGB")
    return _make_default_background(size)


def _draw_realistic_petal(draw: ImageDraw.ImageDraw, x: float, y: float, size: float, angle: float, alpha: int, tone: int = 0) -> None:
    r = max(3, float(size))
    ca, sa = math.cos(angle), math.sin(angle)
    raw = [(0.00, -1.42), (0.25, -1.10), (0.62, -0.28), (0.48, 0.60), (0.10, 1.18), (-0.22, 0.92), (-0.58, 0.20), (-0.36, -0.80)]
    pts = []
    for px, py in raw:
        pts.append((x + (px * ca - py * sa) * r, y + (px * sa + py * ca) * r))
    if tone == 1:
        fill = (255, 220, 242, alpha)
        edge = (255, 250, 255, min(255, alpha + 32))
    else:
        fill = (255, 120, 214, alpha)
        edge = (255, 210, 245, min(255, alpha + 28))
    draw.polygon(pts, fill=fill)
    draw.line(pts + [pts[0]], fill=edge, width=max(1, int(r * 0.08)))
    x1 = x - math.sin(angle) * r * 0.62
    y1 = y + math.cos(angle) * r * 0.62
    x2 = x + math.sin(angle) * r * 0.72
    y2 = y - math.cos(angle) * r * 0.72
    draw.line((x1, y1, x2, y2), fill=(255, 255, 255, max(22, alpha // 3)), width=1)


def _draw_sakura_blossom(draw: ImageDraw.ImageDraw, x: float, y: float, size: float, angle: float, alpha: int) -> None:
    r = max(4, float(size))
    center_color = (255, 235, 125, min(255, alpha + 35))
    for k in range(5):
        a = angle + k * math.tau / 5
        px = x + math.cos(a) * r * 0.42
        py = y + math.sin(a) * r * 0.42
        petal_angle = a + math.pi / 2
        ca, sa = math.cos(petal_angle), math.sin(petal_angle)
        raw = [(0, -1.05), (0.45, -0.25), (0.34, 0.72), (0, 0.95), (-0.34, 0.72), (-0.45, -0.25)]
        pts = []
        for rx0, ry0 in raw:
            pts.append((px + (rx0 * ca - ry0 * sa) * r * 0.55, py + (rx0 * sa + ry0 * ca) * r * 0.55))
        draw.polygon(pts, fill=(255, 160, 225, alpha), outline=(255, 235, 250, min(255, alpha + 24)))
    draw.ellipse((x - r * 0.14, y - r * 0.14, x + r * 0.14, y + r * 0.14), fill=center_color)


def _draw_heart(draw: ImageDraw.ImageDraw, x: float, y: float, size: float, alpha: int, outline_only: bool) -> None:
    r = max(4, float(size))
    pts = []
    for i in range(34):
        tt = math.tau * i / 34
        hx = 16 * math.sin(tt) ** 3
        hy = -(13 * math.cos(tt) - 5 * math.cos(2 * tt) - 2 * math.cos(3 * tt) - math.cos(4 * tt))
        pts.append((x + hx * r / 18, y + hy * r / 18))
    if outline_only:
        draw.line(pts + [pts[0]], fill=(255, 120, 220, alpha), width=max(1, int(r * 0.10)))
    else:
        draw.polygon(pts, fill=(255, 120, 220, alpha))
        draw.line(pts + [pts[0]], fill=(255, 235, 250, min(255, alpha + 40)), width=max(1, int(r * 0.06)))


def _floral_layer(size: tuple[int, int], t: float, layer: str, beat: float = 0.0) -> Image.Image:
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if layer == "far":
        count, speed, base_size, blur, alpha_base, drift_scale, blossom_ratio, heart_ratio = 98, 0.020, height * 0.0075, 0.9, 62, 0.020, 0.22, 0.18
    elif layer == "mid":
        count, speed, base_size, blur, alpha_base, drift_scale, blossom_ratio, heart_ratio = 70, 0.033, height * 0.015, 1.25, 112, 0.038, 0.30, 0.18
    elif layer == "near":
        count, speed, base_size, blur, alpha_base, drift_scale, blossom_ratio, heart_ratio = 38, 0.048, height * 0.038, 4.5, 158, 0.070, 0.20, 0.12
    else:
        count, speed, base_size, blur, alpha_base, drift_scale, blossom_ratio, heart_ratio = 16, 0.058, height * 0.080, 8.0, 125, 0.110, 0.10, 0.08
    for i in range(count):
        seed = i * 101.37 + {"far": 0, "mid": 500, "near": 1000}.get(layer, 1500)
        progress = (t * speed + (i * 0.61803398875)) % 1.0
        x_base = (math.sin(seed) * 0.5 + 0.5) * width
        drift = math.sin(t * (0.28 + i * 0.002) + seed) * width * drift_scale
        expand = beat * width * (0.030 if layer == "far" else 0.055 if layer == "mid" else 0.095)
        side = -1 if x_base < width / 2 else 1
        x = x_base + drift + side * expand
        y = height + base_size * 7 - progress * (height + base_size * 14) - beat * height * (0.028 if layer == "far" else 0.052 if layer == "mid" else 0.080)
        angle = seed * 0.08 + t * (0.30 + i * 0.002) + beat * 0.36
        size_factor = 0.70 + 0.60 * (math.sin(seed * 0.07) * 0.5 + 0.5)
        item_size = base_size * size_factor * (1.0 + beat * (0.22 if layer == "far" else 0.36 if layer == "mid" else 0.52))
        alpha = int(alpha_base * (0.55 + 0.45 * math.sin(progress * math.pi)) * (1.0 + beat * 0.36))
        gate = abs(math.sin(seed * 0.13))
        if gate < heart_ratio:
            _draw_heart(draw, x, y, item_size * 1.55, min(alpha, 230), outline_only=(i % 3 != 0))
        elif gate < heart_ratio + blossom_ratio:
            _draw_sakura_blossom(draw, x, y, item_size * 1.20, angle, min(alpha, 225))
        else:
            _draw_realistic_petal(draw, x, y, item_size, angle, min(alpha, 235), tone=i % 2)
    if blur > 0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur))
    return overlay


def _star_layer(size: tuple[int, int], t: float, beat: float) -> Image.Image:
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i in range(130):
        seed = i * 72.31
        x = (math.sin(seed) * 0.5 + 0.5) * width
        y = (math.sin(seed * 1.37) * 0.5 + 0.5) * height
        twinkle = 0.45 + 0.55 * math.sin(t * (2.2 + (i % 5) * 0.35) + seed)
        alpha = int((40 + 80 * twinkle) * (1.0 + 0.45 * beat))
        r = 1 if i % 4 else 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 235, 255, min(alpha, 180)))
    return overlay


def _speed_line_overlay(size: tuple[int, int], t: float, beats: list[float]) -> Image.Image:
    width, height = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = width * 0.52, height * 0.50
    boost = _beat_strength(t, beats, duration=0.18)
    if boost <= 0.01:
        return overlay
    for i in range(56):
        angle = (i / 56) * math.tau + 0.03 * math.sin(t)
        inner = min(width, height) * (0.12 + 0.04 * boost)
        outer = max(width, height) * (0.58 + 0.24 * ((i % 5) / 5) + 0.16 * boost)
        x1 = cx + math.cos(angle) * inner
        y1 = cy + math.sin(angle) * inner * 0.58
        x2 = cx + math.cos(angle) * outer
        y2 = cy + math.sin(angle) * outer * 0.65
        alpha = int((24 + (i % 4) * 7) * boost)
        draw.line((x1, y1, x2, y2), fill=(255, 225, 245, min(alpha, 120)), width=1 if i % 3 else 2)
    return overlay.filter(ImageFilter.GaussianBlur(radius=0.75 + 1.2 * boost))


def _background_effects(size: tuple[int, int], t: float, beats: list[float]) -> Image.Image:
    width, height = size
    beat = _beat_strength(t, beats, duration=0.22)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy = width // 2, height // 2
    for i in range(8):
        rx = int(width * (0.16 + i * 0.055 + beat * 0.045))
        ry = int(height * (0.22 + i * 0.065 + beat * 0.050))
        alpha = max(0, 18 - i * 2) + int(23 * beat)
        gd.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(170, 45, 140, alpha))
    overlay.alpha_composite(glow.filter(ImageFilter.GaussianBlur(radius=30 + beat * 8)))
    overlay.alpha_composite(_star_layer(size, t, beat))
    overlay.alpha_composite(_speed_line_overlay(size, t, beats))
    overlay.alpha_composite(_floral_layer(size, t, "far", beat))
    overlay.alpha_composite(_floral_layer(size, t, "mid", beat))
    overlay = _zoom_rgba_layer(overlay, 1.0 + 0.105 * beat)
    overlay = _motion_blur(overlay, beat)
    flash = flash_opacity(t, beats, duration=0.11)
    if flash > 0:
        overlay.alpha_composite(Image.new("RGBA", size, (255, 220, 245, int(75 * flash))))
    return overlay


def _foreground_floral_effects(size: tuple[int, int], t: float, beats: list[float]) -> Image.Image:
    beat = _beat_strength(t, beats, duration=0.20)
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    layer.alpha_composite(_floral_layer(size, t, "near", beat))
    layer.alpha_composite(_floral_layer(size, t, "front", beat))
    layer = _zoom_rgba_layer(layer, 1.0 + 0.075 * beat)
    return _motion_blur(layer, beat * 0.9)


def _paste_with_shadow(base: Image.Image, character: Image.Image, x: int, y: int, strength: int = 120) -> None:
    if character.mode != "RGBA":
        character = character.convert("RGBA")
    alpha = character.getchannel("A")
    glow = Image.new("RGBA", character.size, (235, 82, 190, strength))
    glow.putalpha(alpha.filter(ImageFilter.GaussianBlur(radius=13)))
    base.alpha_composite(glow, (x, y))
    base.alpha_composite(character, (x, y))


def _slow_motion_character_frame(frames: list[Image.Image], t: float, beats: list[float], fps: float = 10.0) -> Image.Image:
    if len(frames) == 1:
        return frames[0]
    dt = _time_since_last_beat(t, beats)
    if dt is not None and 0 <= dt <= 0.26:
        idx = int((t - dt * 0.68) * fps)
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
    return -int(height * 0.105 * best)


def make_animated_main_visual(
    background_path: str | None,
    character_path: str,
    duration: float,
    beats: list[float],
    size: tuple[int, int],
) -> VideoClip:
    """Horizontal anime music visual layer.

    Dense black-pink background similar to reference images: hearts, stars,
    sakura blossoms and petals, strong beat zoom, and motion blur.
    """
    width, height = size
    bg_base = _background_image(background_path, size)
    base_character_height = int(height * 0.74)
    character_frames = _load_character_frames(character_path, target_height=base_character_height)
    resample = _resample_filter()

    def make_frame(t: float) -> np.ndarray:
        beat = _beat_strength(t, beats, duration=0.22)
        bg_zoom = 1.0 + 0.135 * beat
        if background_path and Path(background_path).exists() and Path(background_path).suffix.lower() not in VIDEO_SUFFIXES:
            bg = _pil_cover_image(background_path, size, zoom=bg_zoom).convert("RGBA")
            bg = Image.blend(Image.new("RGBA", size, (0, 0, 0, 255)), bg, 0.30)
        else:
            zoomed_size = (int(width * bg_zoom), int(height * bg_zoom))
            zoomed = bg_base.resize(zoomed_size, resample)
            left = max(0, (zoomed.width - width) // 2)
            top = max(0, (zoomed.height - height) // 2)
            bg = zoomed.crop((left, top, left + width, top + height)).convert("RGBA")
            if beat > 0.05:
                bg = bg.filter(ImageFilter.GaussianBlur(radius=beat * 2.2))

        frame = bg
        frame.alpha_composite(_background_effects(size, t, beats))

        character_base = _slow_motion_character_frame(character_frames, t, beats)
        scale = beat_scale(t, beats, base=1.0, pulse=0.075, decay=0.20)
        target_h = max(1, int(character_base.height * scale))
        target_w = max(1, int(character_base.width * scale))
        char = character_base.resize((target_w, target_h), resample)
        if beat > 0.20:
            ghost = char.filter(ImageFilter.GaussianBlur(radius=beat * 2.0))
            char_layer = Image.new("RGBA", char.size, (0, 0, 0, 0))
            a = ghost.getchannel("A").point(lambda v: int(v * 0.25))
            ghost.putalpha(a)
            char_layer.alpha_composite(ghost)
            char_layer.alpha_composite(char)
            char = char_layer

        sway = int(math.sin(t * 1.35) * width * 0.012)
        jump = _jump_offset(t, beats, height)
        x = int(width * 0.50 - target_w / 2 + sway)
        y = int(height * 0.54 - target_h / 2 + jump)
        _paste_with_shadow(frame, char, x, y, strength=120)

        frame.alpha_composite(_foreground_floral_effects(size, t, beats))
        return np.array(frame.convert("RGB"))

    return VideoClip(make_frame=make_frame, duration=duration)
