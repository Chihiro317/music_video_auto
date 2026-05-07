from __future__ import annotations

import argparse
from pathlib import Path

from src.batch import run_batch
from src.renderer import RenderConfig, render_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="音乐卡点歌词视频自动生成器")
    parser.add_argument("--music", help="音乐文件路径，支持 mp3/wav")
    parser.add_argument("--character", help="角色图片路径，推荐 PNG 透明图")
    parser.add_argument("--lyrics", default=None, help="SRT 歌词字幕路径，可选")
    parser.add_argument("--background", default=None, help="背景视频或图片路径，可选")
    parser.add_argument("--title", default="", help="歌曲名")
    parser.add_argument("--version", default="", help="版本名，例如 卡点版")
    parser.add_argument("--account", default="", help="片尾展示账号名")
    parser.add_argument("--output", default="output/video.mp4", help="输出视频路径")
    parser.add_argument("--width", type=int, default=1080, help="视频宽度")
    parser.add_argument("--height", type=int, default=1920, help="视频高度")
    parser.add_argument("--fps", type=int, default=30, help="帧率")
    parser.add_argument("--batch", default=None, help="批量配置文件路径，支持 json/xlsx")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.batch:
        outputs = run_batch(args.batch)
        print("批量生成完成：")
        for output in outputs:
            print(output)
        return

    missing = []
    if not args.music:
        missing.append("--music")
    if not args.character:
        missing.append("--character")
    if missing:
        parser.error("缺少必要参数：" + ", ".join(missing))

    cfg = RenderConfig(
        music=args.music,
        character=args.character,
        lyrics=args.lyrics,
        background=args.background,
        title=args.title,
        version=args.version,
        account=args.account,
        output=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    output = render_video(cfg)
    print(f"视频生成完成：{Path(output).resolve()}")


if __name__ == "__main__":
    main()
