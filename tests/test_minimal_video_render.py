from pathlib import Path

import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import ColorClip


def test_minimal_video_render(tmp_path: Path):
    """Render a tiny 3-second MP4 to verify MoviePy and FFmpeg work in CI.

    This deliberately avoids TextClip/ImageMagick and custom fonts so the test
    focuses only on the minimum video rendering pipeline.
    """
    output_dir = Path("output/test_artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "minimal_3s_render.mp4"

    duration = 3.0
    fps = 12
    sample_rate = 44100

    video = ColorClip(size=(360, 640), color=(18, 12, 35)).set_duration(duration)

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
