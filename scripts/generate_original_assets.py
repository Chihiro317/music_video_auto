from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from moviepy.editor import VideoClip
from PIL import Image, ImageDraw, ImageFilter


def _resample_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC


def make_petal(seed: int, size: int = 256) -> Image.Image:
    """Generate one original transparent sakura petal PNG."""
    rng = random.Random(seed)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    cx = size * 0.50 + rng.uniform(-size * 0.04, size * 0.04)
    cy = size * 0.52 + rng.uniform(-size * 0.04, size * 0.04)
    radius = size * rng.uniform(0.30, 0.42)
    notch = rng.uniform(0.12, 0.23)
    angle = rng.uniform(-0.35, 0.35)

    points = []
    for i in range(80):
        theta = 2 * math.pi * i / 80
        # Heart-like petal with a tiny top notch.
        r = radius * (1.0 + 0.18 * math.sin(theta) - notch * max(0, math.cos(theta)) ** 4)
        x = r * 0.72 * math.sin(theta)
        y = -r * (0.92 * math.cos(theta) + 0.18 * math.cos(2 * theta))
        ca, sa = math.cos(angle), math.sin(angle)
        px = cx + x * ca - y * sa
        py = cy + x * sa + y * ca
        points.append((px, py))

    base_color = (
        rng.randint(238, 255),
        rng.randint(150, 205),
        rng.randint(210, 245),
        rng.randint(205, 238),
    )
    edge_color = (255, 225, 245, rng.randint(185, 225))
    draw.polygon(points, fill=base_color)
    draw.line(points + [points[0]], fill=edge_color, width=max(2, size // 80), joint="curve")

    # Central vein and subtle sub-veins.
    vein = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vein)
    vx1 = cx + math.sin(angle) * radius * 0.55
    vy1 = cy + math.cos(angle) * radius * 0.55
    vx2 = cx - math.sin(angle) * radius * 0.35
    vy2 = cy - math.cos(angle) * radius * 0.55
    vd.line((vx1, vy1, vx2, vy2), fill=(255, 245, 255, 100), width=max(1, size // 90))
    for k in range(4):
        f = 0.2 + k * 0.14
        sx = vx1 * (1 - f) + vx2 * f
        sy = vy1 * (1 - f) + vy2 * f
        branch_angle = angle + (-1 if k % 2 else 1) * rng.uniform(0.55, 0.95)
        ex = sx + math.cos(branch_angle) * radius * 0.20
        ey = sy + math.sin(branch_angle) * radius * 0.12
        vd.line((sx, sy, ex, ey), fill=(255, 238, 255, 45), width=1)
    layer.alpha_composite(vein)

    # Soft edge blur and highlight.
    soft = layer.filter(ImageFilter.GaussianBlur(radius=size * 0.006))
    canvas.alpha_composite(soft)
    highlight = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    hd = ImageDraw.Draw(highlight)
    hd.ellipse(
        (size * 0.30, size * 0.24, size * 0.62, size * 0.58),
        fill=(255, 255, 255, rng.randint(22, 46)),
    )
    highlight = highlight.filter(ImageFilter.GaussianBlur(radius=size * 0.045))
    canvas.alpha_composite(highlight)
    return canvas


def generate_petals(output_dir: Path, count: int = 30) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        size = 192 + (i % 4) * 32
        petal = make_petal(seed=20260508 + i, size=size)
        petal.save(output_dir / f"petal_{i + 1:03d}.png")


def make_bokeh(seed: int, size: int = 256) -> Image.Image:
    rng = random.Random(seed)
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    cx = size / 2 + rng.uniform(-size * 0.08, size * 0.08)
    cy = size / 2 + rng.uniform(-size * 0.08, size * 0.08)
    color = (255, rng.randint(130, 190), rng.randint(220, 255), rng.randint(90, 155))
    r = rng.uniform(size * 0.22, size * 0.43)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(12, 26)))


def generate_bokeh(output_dir: Path, count: int = 12) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        make_bokeh(seed=9090 + i, size=256).save(output_dir / f"bokeh_{i + 1:03d}.png")


def make_background_frame(t: float, petals: list[Image.Image], bokehs: list[Image.Image], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width * 0.5, height * 0.45
    dist = np.sqrt(((xx - cx) / width) ** 2 + ((yy - cy) / height) ** 2)
    glow = np.clip(1.0 - dist * 3.0, 0, 1)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (2 + 28 * glow).astype(np.uint8)
    frame[:, :, 1] = (1 + 6 * glow).astype(np.uint8)
    frame[:, :, 2] = (4 + 24 * glow).astype(np.uint8)
    base = Image.fromarray(frame, mode="RGB").convert("RGBA")

    # Radial speed lines.
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i in range(52):
        angle = i / 52 * math.tau + 0.08 * math.sin(t)
        inner = min(width, height) * 0.15
        outer = max(width, height) * (0.55 + 0.2 * ((i % 5) / 5))
        x1 = cx + math.cos(angle) * inner
        y1 = cy + math.sin(angle) * inner * 0.72
        x2 = cx + math.cos(angle) * outer
        y2 = cy + math.sin(angle) * outer * 0.78
        draw.line((x1, y1, x2, y2), fill=(255, 210, 245, 26), width=1)
    base.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(radius=0.8)))

    # Bokeh layer.
    for i, bokeh in enumerate(bokehs):
        p = (t * (0.06 + i * 0.003) + i * 0.137) % 1.0
        angle = i * 1.93 + t * 0.28
        radius = min(width, height) * (0.22 + 0.65 * p)
        x = int(cx + math.cos(angle) * radius * 1.3)
        y = int(cy + math.sin(angle) * radius * 0.75)
        scale = 0.35 + p * 1.2
        img = bokeh.resize((max(1, int(bokeh.width * scale)), max(1, int(bokeh.height * scale))), _resample_filter())
        base.alpha_composite(img, (x - img.width // 2, y - img.height // 2))

    # Petal layer.
    for i, petal in enumerate(petals):
        p = (t * (0.16 + (i % 7) * 0.018) + i * 0.071) % 1.0
        angle = i * 2.11 + math.sin(t * 0.9 + i) * 0.4
        radius = min(width, height) * (0.08 + 0.9 * p)
        x = int(cx + math.cos(angle) * radius * 1.45)
        y = int(cy + math.sin(angle) * radius * 0.82 + 22 * math.sin(t * 1.7 + i))
        scale = 0.16 + 0.58 * p + (i % 3) * 0.06
        img = petal.resize((max(1, int(petal.width * scale)), max(1, int(petal.height * scale))), _resample_filter())
        img = img.rotate(math.degrees(angle + t * 2.5), expand=True, resample=_resample_filter())
        if i % 5 == 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=1.5 + p * 2.0))
        base.alpha_composite(img, (x - img.width // 2, y - img.height // 2))

    return np.array(base.convert("RGB"))


def generate_background_video(output_path: Path, petals_dir: Path, bokeh_dir: Path, duration: float, fps: int, size: tuple[int, int]) -> None:
    petals = [Image.open(p).convert("RGBA") for p in sorted(petals_dir.glob("*.png"))]
    bokehs = [Image.open(p).convert("RGBA") for p in sorted(bokeh_dir.glob("*.png"))]
    clip = VideoClip(lambda t: make_background_frame(t, petals, bokehs, size), duration=duration)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(output_path), fps=fps, codec="libx264", audio=False, preset="medium")
    clip.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate original sakura assets for music_video_auto.")
    parser.add_argument("--root", default="assets/original", help="Output asset root directory")
    parser.add_argument("--petals", type=int, default=30, help="Number of petal PNGs")
    parser.add_argument("--bokeh", type=int, default=12, help="Number of bokeh PNGs")
    parser.add_argument("--background", action="store_true", help="Also render a background preview MP4")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    root = Path(args.root)
    petals_dir = root / "petals"
    bokeh_dir = root / "bokeh"
    generate_petals(petals_dir, args.petals)
    generate_bokeh(bokeh_dir, args.bokeh)
    if args.background:
        generate_background_video(root / "sakura_black_original_loop.mp4", petals_dir, bokeh_dir, args.duration, args.fps, (args.width, args.height))
    print(f"Generated original assets under: {root.resolve()}")


if __name__ == "__main__":
    main()
