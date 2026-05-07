from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .renderer import RenderConfig, render_video


def load_tasks(config_path: str | Path) -> list[dict[str, Any]]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"批量配置文件不存在：{path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("tasks", [])
        return data

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str).fillna("")
        return df.to_dict(orient="records")

    raise ValueError("仅支持 json、xlsx、xls 批量配置文件")


def run_batch(config_path: str | Path) -> list[Path]:
    outputs: list[Path] = []
    for task in load_tasks(config_path):
        cfg = RenderConfig(
            music=task["music"],
            character=task["character"],
            lyrics=task.get("lyrics") or None,
            background=task.get("background") or None,
            title=task.get("title", ""),
            version=task.get("version", ""),
            account=task.get("account", ""),
            output=task.get("output", "output/video.mp4"),
            width=int(task.get("width", 1080) or 1080),
            height=int(task.get("height", 1920) or 1920),
            fps=int(task.get("fps", 30) or 30),
        )
        outputs.append(render_video(cfg))
    return outputs
