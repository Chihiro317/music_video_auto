from pathlib import Path

from src.beat_detect import is_near_beat
from src.effects import beat_scale, flash_opacity, shake_offset
from src.renderer import RenderConfig
from src.subtitle import find_subtitle_at, load_srt


def test_subtitle_loader(tmp_path: Path):
    srt_file = tmp_path / "demo.srt"
    srt_file.write_text(
        "1\n00:00:00,000 --> 00:00:01,500\n第一句歌词\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\n第二句歌词\n",
        encoding="utf-8",
    )
    subtitles = load_srt(srt_file)
    assert len(subtitles) == 2
    assert find_subtitle_at(0.5, subtitles) == "第一句歌词"
    assert find_subtitle_at(2.5, subtitles) == "第二句歌词"
    assert find_subtitle_at(5.0, subtitles) == ""


def test_effect_helpers():
    beats = [0.5, 1.0, 1.5]
    assert is_near_beat(1.02, beats, window=0.05)
    assert not is_near_beat(1.2, beats, window=0.05)
    assert beat_scale(1.01, beats) >= 1.0
    assert isinstance(shake_offset(1.01, beats), tuple)
    assert flash_opacity(1.01, beats) >= 0.0


def test_render_config_defaults():
    cfg = RenderConfig(
        music="input/music/demo.mp3",
        character="input/characters/role.png",
        lyrics="input/lyrics/demo.srt",
        background="input/backgrounds/bg.mp4",
        title="歌曲名",
        version="卡点版",
        account="账号",
        output="output/demo.mp4",
    )
    assert cfg.width == 1080
    assert cfg.height == 1920
    assert cfg.fps == 30
