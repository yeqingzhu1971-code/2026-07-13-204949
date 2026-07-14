#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "quick_visual_index"
OUT.mkdir(exist_ok=True)


def run(cmd: list[str], check: bool = True, capture: bool = False):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def source() -> Path:
    files = [p for p in ROOT.iterdir() if p.suffix.lower() in {".mov", ".mp4", ".m4v"}]
    if not files:
        raise SystemExit("Source video missing")
    return max(files, key=lambda p: p.stat().st_size)


def main() -> None:
    src = source()
    probe = run([
        "ffprobe", "-v", "error", "-show_format", "-show_streams",
        "-of", "json", str(src),
    ], capture=True)
    (OUT / "source_info.json").write_text(probe.stdout, encoding="utf-8")

    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    timeline_filter = (
        "fps=1/3,scale=180:-2,"
        f"drawtext=fontfile='{font}':text='%{{pts\\:hms}}':"
        "x=4:y=4:fontsize=14:fontcolor=white:borderw=2:bordercolor=black,"
        "tile=6x6:padding=3:margin=3:color=black"
    )
    run([
        "ffmpeg", "-y", "-i", str(src), "-vf", timeline_filter,
        "-fps_mode", "vfr", "-q:v", "3", str(OUT / "timeline_%03d.jpg"),
    ])

    scene = run([
        "ffmpeg", "-hide_banner", "-i", str(src),
        "-vf", "select='gt(scene,0.18)',showinfo", "-an",
        "-fps_mode", "vfr", "-f", "null", "-",
    ], check=False, capture=True)
    times = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", scene.stderr)]
    (OUT / "scene_times.json").write_text(
        json.dumps(times, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Created {len(list(OUT.glob('timeline_*.jpg')))} timeline sheets and {len(times)} scene markers")


if __name__ == "__main__":
    main()
