from pathlib import Path

import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from PIL import Image, ImageDraw

from src.renderer import RenderConfig, render_video


def _write_test_audio(path: Path, duration: float = 4.0, sample_rate: int = 44100) -> None:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = 0.10 * np.sin(2 * np.pi * 440 * t)
    pulse = ((np.sin(2 * np.pi * 2.0 * t) > 0.92).astype(float)) * 0.20
    audio_array = np.column_stack([tone + pulse, tone + pulse])
    audio = AudioArrayClip(audio_array, fps=sample_rate)
    audio.write_audiofile(str(path), fps=sample_rate, codec="pcm_s16le", logger=None)
    audio.close()


def _draw_dance_frame(path: Path, phase: float, turn: bool = False) -> None:
    image = Image.new("RGBA", (360, 460), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    cx = 180
    head_x = cx + int(22 * np.sin(phase))
    body_x = cx - int(12 * np.sin(phase))
    draw.ellipse((head_x - 78, 18, head_x + 78, 174), fill=(255, 235, 210, 255), outline=(255, 255, 255, 255), width=6)
    if turn:
        draw.rectangle((body_x - 54, 172, body_x + 54, 350), fill=(90, 85, 245, 255), outline=(255, 255, 255, 255), width=6)
        draw.arc((head_x - 30, 80, head_x + 30, 140), 260, 90, fill=(80, 20, 80, 255), width=4)
        draw.ellipse((head_x + 18, 76, head_x + 38, 96), fill=(20, 20, 30, 255))
    else:
        draw.rectangle((body_x - 58, 176, body_x + 58, 354), fill=(115, 90, 255, 255), outline=(255, 255, 255, 255), width=6)
        draw.ellipse((head_x - 32, 78, head_x - 12, 98), fill=(20, 20, 30, 255))
        draw.ellipse((head_x + 18, 78, head_x + 38, 98), fill=(20, 20, 30, 255))
        draw.arc((head_x - 35, 108, head_x + 35, 156), 0, 180, fill=(80, 20, 80, 255), width=4)
    arm_l_y = 198 + int(80 * np.sin(phase))
    arm_r_y = 198 - int(80 * np.sin(phase))
    draw.line((body_x - 56, 205, body_x - 130, arm_l_y), fill=(255, 255, 255, 255), width=14)
    draw.line((body_x + 56, 205, body_x + 130, arm_r_y), fill=(255, 255, 255, 255), width=14)
    draw.line((body_x - 24, 350, body_x - 82, 430 - int(35 * np.sin(phase))), fill=(255, 255, 255, 255), width=16)
    draw.line((body_x + 24, 350, body_x + 82, 430 + int(35 * np.sin(phase))), fill=(255, 255, 255, 255), width=16)
    image.save(path)


def _write_test_character_frames(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for i in range(12):
        phase = 2 * np.pi * i / 12
        _draw_dance_frame(path / f"dance_{i:02d}.png", phase=float(phase), turn=(i in {3, 4, 5, 9, 10}))


def _write_test_background(path: Path) -> None:
    width, height = 640, 360
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (2 + 20 * (1 - abs(x - 0.5) * 2)).clip(0, 255).astype(np.uint8)
    frame[:, :, 1] = (1 + 5 * y).clip(0, 255).astype(np.uint8)
    frame[:, :, 2] = (4 + 18 * (1 - y)).clip(0, 255).astype(np.uint8)
    Image.fromarray(frame).save(path)


def _write_test_srt(path: Path) -> None:
    path.write_text(
        "1\n00:00:00,000 --> 00:00:01,900\n第一句测试歌词\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\n第二句测试歌词\n",
        encoding="utf-8",
    )


def test_full_renderer_smoke(tmp_path: Path):
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    music = asset_dir / "test.wav"
    character_dir = asset_dir / "character_frames"
    background = asset_dir / "background.png"
    lyrics = asset_dir / "lyrics.srt"
    output = Path("output/test_artifacts/full_renderer_3s_render.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    _write_test_audio(music)
    _write_test_character_frames(character_dir)
    _write_test_background(background)
    _write_test_srt(lyrics)

    cfg = RenderConfig(
        music=str(music),
        character=str(character_dir),
        lyrics=str(lyrics),
        background=str(background),
        title="测试歌曲",
        version="横版高级感",
        account="Chihiro317",
        output=str(output),
        width=640,
        height=360,
        fps=12,
        outro_seconds=1.0,
    )
    result = render_video(cfg)

    assert result.exists()
    assert result.stat().st_size > 20_000
