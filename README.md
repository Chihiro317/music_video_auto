# music_video_auto

音乐卡点歌词视频自动生成器。项目目标是自动生成类似抖音二次元音乐视频的短视频：暗色/光效背景、中间角色 PNG、节拍缩放抖动、底部歌词字幕、左上角歌曲标题、片尾关注引导页。

## 第一版功能

- 支持 mp3/wav 音乐输入
- 支持 PNG/JPG 角色图输入
- 支持 SRT 歌词字幕输入
- 支持背景视频或背景图片输入
- 默认输出 1080x1920 竖屏 MP4
- 自动读取音乐时长
- 自动检测音乐节拍
- 根据节拍让角色轻微缩放
- 添加左上角标题
- 添加底部歌词字幕
- 最后 2 秒添加关注引导页
- 提供命令行入口和 Streamlit 中文界面

## 环境要求

- Python 3.11 推荐
- 本机需要安装 FFmpeg，并确保 `ffmpeg` 可以在命令行中直接运行

Windows 安装 FFmpeg 后，需要把 `ffmpeg.exe` 所在目录加入系统 Path。

## 安装

```bash
pip install -r requirements.txt
```

## 命令行使用

```bash
python main.py ^
  --music input/music/demo.mp3 ^
  --character input/characters/role.png ^
  --lyrics input/lyrics/demo.srt ^
  --background input/backgrounds/bg.mp4 ^
  --title "歌曲名" ^
  --version "卡点版" ^
  --account "你的账号" ^
  --output output/demo.mp4
```

PowerShell 可以写成一行：

```bash
python main.py --music input/music/demo.mp3 --character input/characters/role.png --lyrics input/lyrics/demo.srt --background input/backgrounds/bg.mp4 --title "歌曲名" --version "卡点版" --account "你的账号" --output output/demo.mp4
```

## 打开中文界面

```bash
streamlit run src/app.py
```

## 目录说明

```text
music_video_auto/
├─ input/
│  ├─ music/
│  ├─ characters/
│  ├─ lyrics/
│  └─ backgrounds/
├─ output/
├─ assets/
├─ src/
│  ├─ beat_detect.py
│  ├─ subtitle.py
│  ├─ effects.py
│  ├─ renderer.py
│  ├─ batch.py
│  └─ app.py
├─ main.py
├─ requirements.txt
├─ config.example.json
└─ README.md
```

## 注意

本项目只负责自动合成视频。音乐、角色图片、背景素材请使用你拥有版权或可商用授权的素材。