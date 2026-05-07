from pathlib import Path

import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoClip


def _make_demo_frame(t: float) -> np.ndarray:
    """Create a visible animated frame without TextClip/ImageMagick."""
    width, height = 360, 640
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (25 + 80 * x).astype(np.uint8)
    frame[:, :, 1] = (20 + 50 * y).astype(np.uint8)
    frame[:, :, 2] = (70 + 120 * (1 - y)).astype(np.uint8)

    # Moving bright circle.
    cx = int(60 + 240 * (t / 3.0))
    cy = int(190 + 45 * np.sin(t * 2 * np.pi))
    yy, xx = np.ogrid[:height, :width]
    circle_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 42**2
    frame[circle_mask] = np.array([245, 245, 255], dtype=np.uint8)

    # Beat-like pulsing square in the middle.
    pulse = int(55 + 25 * abs(np.sin(t * 2 * np.pi)))
    x1, x2 = width // 2 - pulse, width // 2 + pulse
    y1, y2 = height // 2 - pulse, height // 2 + pulse
    frame[y1:y2, x1:x2] = np.array([180, 80, 255], dtype=np.uint8)

    # Bottom progress bar.
    bar_w = int(width * min(max(t / 3.0, 0), 1))
    frame[height - 42 : height - 24, 24 : 24 + bar_w] = np.array([255, 255, 255], dtype=np.uint8)

    # Border markers make it obvious the frame is not blank.
    frame[20:26, 20 : width - 20] = np.array([255, 255, 255], dtype=np.uint8)
    frame[height - 26 : height - 20, 20 : width - 20] = np.array([255, 255, 255], dtype=np.uint8)
    frame[20 : height - 20, 20:26] = np.array([255, 255, 255], dtype=np.uint8)
    frame[20 : height - 20, width - 26 : width - 20] = np.array([255, 255, 255], dtype=np.uint8)
    return frame


def test_minimal_video_render(tmp_path: Path):
    """Render a visible 3-second MP4 to verify MoviePy and FFmpeg work in CI.

    This deliberately avoids TextClip/ImageMagick and custom fonts so the test
    focuses on the basic video rendering pipeline plus visible layered content.
    """
    output_dir = Path("output/test_artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "minimal_3s_render.mp4"

    duration = 3.0
    fps = 12
    sample_rate = 44100

    video = VideoClip(make_frame=_make_demo_frame, duration=duration)

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = 0.12 * np.sin(2 * np.pi * 440 * t)
    audio_array = np.column_stack([tone, tone])
    audio = AudioArrayClip(audio_array, fps=sample_rate)

    video = video.set_audio(audio)
    video.write_videofile(
        str(output_file),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        logger=None,
    )

    video.close()
    audio.close()

    assert output_file.exists()
    assert output_file.stat().st_size > 10_000
