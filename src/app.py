from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from renderer import RenderConfig, render_video


st.set_page_config(page_title="音乐卡点视频自动生成器", layout="wide")
st.title("🎵 音乐卡点歌词视频自动生成器")
st.caption("上传音乐、角色图、歌词和背景，一键生成二次元风格音乐视频。")

with st.sidebar:
    st.header("输出设置")
    width = st.number_input("视频宽度", min_value=360, max_value=2160, value=1080, step=10)
    height = st.number_input("视频高度", min_value=360, max_value=3840, value=1920, step=10)
    fps = st.number_input("帧率", min_value=15, max_value=60, value=30, step=1)

col1, col2 = st.columns(2)

with col1:
    music_file = st.file_uploader("选择音乐 mp3/wav", type=["mp3", "wav"])
    character_file = st.file_uploader("选择角色图片 PNG/JPG", type=["png", "jpg", "jpeg"])
    lyrics_file = st.file_uploader("选择 SRT 歌词字幕，可选", type=["srt"])
    background_file = st.file_uploader("选择背景视频或图片，可选", type=["mp4", "mov", "mkv", "avi", "webm", "png", "jpg", "jpeg"])

with col2:
    title = st.text_input("歌曲名", value="歌曲名")
    version = st.text_input("版本名", value="卡点版")
    account = st.text_input("片尾账号名", value="你的账号")
    output_name = st.text_input("输出文件名", value="music_video.mp4")


def save_upload(uploaded_file, directory: Path) -> str | None:
    if uploaded_file is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / uploaded_file.name
    path.write_bytes(uploaded_file.getbuffer())
    return str(path)


if st.button("开始生成视频", type="primary"):
    if music_file is None or character_file is None:
        st.error("请至少上传音乐文件和角色图片。")
    else:
        with st.spinner("正在生成视频，请等待。第一次运行会比较慢。"):
            temp_root = Path(tempfile.mkdtemp(prefix="music_video_auto_"))
            music_path = save_upload(music_file, temp_root / "music")
            character_path = save_upload(character_file, temp_root / "characters")
            lyrics_path = save_upload(lyrics_file, temp_root / "lyrics")
            background_path = save_upload(background_file, temp_root / "backgrounds")
            output_path = temp_root / "output" / output_name

            cfg = RenderConfig(
                music=music_path,
                character=character_path,
                lyrics=lyrics_path,
                background=background_path,
                title=title,
                version=version,
                account=account,
                output=str(output_path),
                width=int(width),
                height=int(height),
                fps=int(fps),
            )
            try:
                result = render_video(cfg)
                st.success(f"生成完成：{result}")
                st.video(str(result))
                st.download_button("下载视频", data=Path(result).read_bytes(), file_name=Path(result).name, mime="video/mp4")
            except Exception as exc:
                st.exception(exc)
                st.warning("如果报错与 ImageMagick 或字体有关，建议先用命令行版本排查，或安装 ImageMagick/微软雅黑字体。")
