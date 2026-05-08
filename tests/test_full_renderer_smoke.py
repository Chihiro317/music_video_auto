from pathlib import Path

import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import ImageClip
from PIL import Image, ImageDraw

from src.renderer import RenderConfig, render_video


def _write_test_audio(path: Path, duration: float = 3.0, sample_rate: int = 44100) -> None:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = 0.12 * np.sin(2 * np.pi * 440 * t)
    audio_array = np.column_stack([tone, tone])
    audio = AudioArrayClip(audio_array, fps=sample_rate)
    audio.write_audiofile(str(path), fps=sample_rate, codec="pcm_s16le", logger=None)
    audio.close()


def _write_test_character(path: Path) -> None:
    image = Image.new("RGBA", (320, 420), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((70, 20, 250, 200), fill=(255, 235, 210, 255), outline=(255, 255, 255, 255), width=6)
    draw.rectangle((110, 190, 210, 390), fill=(115, 90, 255, 255), outline=(255, 255, 255, 255), width=6)
    draw.ellipse((118, 80, 138, 100), fill=(20, 20, 30, 255))
    draw.ellipse((182, 80, 202, 100), fill=(20, 20, 30, 255))
    draw.arc((125, 105, 195, 155), 0, 180, fill=(80, 20, 80, 255), width=4)
    image.save(path)


def _write_test_background(path: Path) -> None:
    width, height = 360, 640
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (20 + 70 * x).astype(np.uint8)
    frame[:, :, 1] = (10 + 40 * y).astype(np.uint8)
    frame[:, :, 2] = (80 + 130 * (1 - y)).astype(np.uint8)
    for i in range(0, width, 40):
        frame[:, i : i + 3] = np.array([110, 80, 220], dtype=np.uint8)
    Image.fromarray(frame).save(path)


def _write_test_srt(path: Path) -> None:
    path.write_text(
        "1\n00:00:00,000 --> 00:00:01,400\n第一句测试歌词\n\n"
        "2\n00:00:01,500 --> 00:00:03,000\n第二句测试歌词\n",
        encoding="utf-8",
    )


def test_full_renderer_smoke(tmp_path: Path):
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    music = asset_dir / "test.wav"
    character = asset_dir / "character.png"
    background = asset_dir / "background.png"
    lyrics = asset_dir / "lyrics.srt"
    output = Path("output/test_artifacts/full_renderer_3s_render.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    _write_test_audio(music)
    _write_test_character(character)
    _write_test_background(background)
    _write_test_srt(lyrics)

    cfg = RenderConfig(
        music=str(music),
        character=str(character),
        lyrics=str(lyrics),
        background=str(background),
        title="测试歌曲",
        version="自动渲染版",
        account="Chihiro317",
        output=str(output),
        width=360,
        height=640,
        fps=12,
        outro_seconds=1.0,
    )
    result = render_video(cfg)

    assert result.exists()
    assert result.stat().st_size > 20_000
