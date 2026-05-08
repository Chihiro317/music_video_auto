from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from moviepy.editor import ImageClip
from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    # GitHub Actions / common Linux CJK fonts
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
]


@lru_cache(maxsize=64)
def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        path = Path(font_path)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Wrap Chinese/English text by pixel width."""
    if not text:
        return [""]
    lines: list[str] = []
    current = ""
    dummy = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(dummy)
    for char in text:
        candidate = current + char
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=0)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def make_text_image(
    text: str,
    fontsize: int,
    color: tuple[int, int, int, int] = (255, 255, 255, 255),
    stroke_color: tuple[int, int, int, int] = (0, 0, 0, 220),
    stroke_width: int = 2,
    max_width: int = 980,
    padding: int = 18,
    align: str = "center",
) -> Image.Image:
    """Render text to a transparent PIL image without ImageMagick."""
    font = load_font(fontsize)
    lines = _wrap_text(text, font, max_width=max_width)
    dummy = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)

    line_boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width) for line in lines]
    line_widths = [box[2] - box[0] for box in line_boxes]
    line_heights = [box[3] - box[1] for box in line_boxes]
    width = min(max(max(line_widths, default=1) + padding * 2, 1), max_width + padding * 2)
    line_gap = max(6, int(fontsize * 0.22))
    height = sum(line_heights) + line_gap * max(len(lines) - 1, 0) + padding * 2

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    y = padding
    for line, box, line_width, line_height in zip(lines, line_boxes, line_widths, line_heights):
        if align == "left":
            x = padding
        elif align == "right":
            x = width - padding - line_width
        else:
            x = (width - line_width) // 2
        draw.text(
            (x, y - box[1]),
            line,
            font=font,
            fill=color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
        )
        y += line_height + line_gap
    return image


def make_text_clip(
    text: str,
    fontsize: int,
    max_width: int,
    color: tuple[int, int, int, int] = (255, 255, 255, 255),
    stroke_color: tuple[int, int, int, int] = (0, 0, 0, 220),
    stroke_width: int = 2,
    padding: int = 18,
    align: str = "center",
) -> ImageClip:
    image = make_text_image(
        text=text,
        fontsize=fontsize,
        color=color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        max_width=max_width,
        padding=padding,
        align=align,
    )
    return ImageClip(np.array(image))
