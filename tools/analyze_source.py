#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "source_analysis"
OUT.mkdir(exist_ok=True)


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def find_source() -> Path:
    candidates = [
        p for p in ROOT.iterdir()
        if p.suffix.lower() in {".mov", ".mp4", ".m4v"}
        and p.parent != OUT
    ]
    if not candidates:
        raise SystemExit("No source video found in repository root")
    return max(candidates, key=lambda p: p.stat().st_size)


def ffprobe(src: Path) -> dict:
    result = run([
        "ffprobe", "-v", "error", "-show_format", "-show_streams",
        "-of", "json", str(src),
    ], capture=True)
    return json.loads(result.stdout)


def make_proxy(src: Path) -> None:
    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    vf = (
        "scale=360:-2:force_original_aspect_ratio=decrease,"
        "pad=360:640:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"drawtext=fontfile='{font}':text='%{{pts\\:hms}}':"
        "x=12:y=12:fontsize=20:fontcolor=white:borderw=2:bordercolor=black"
    )
    run([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-maxrate", "520k", "-bufsize", "1040k",
        "-c:a", "aac", "-b:a", "64k", "-ac", "1",
        "-movflags", "+faststart",
        str(OUT / "source_proxy_timecode.mp4"),
    ])


def make_contact_sheets(src: Path) -> None:
    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    # One frame every five seconds; each sheet covers 125 seconds.
    vf = (
        "fps=1/5,scale=180:-2,"
        f"drawtext=fontfile='{font}':text='%{{pts\\:hms}}':"
        "x=5:y=5:fontsize=14:fontcolor=white:borderw=2:bordercolor=black,"
        "tile=5x5:padding=4:margin=4:color=black"
    )
    run([
        "ffmpeg", "-y", "-i", str(src), "-vf", vf,
        "-fps_mode", "vfr", "-q:v", "3",
        str(OUT / "timeline_%03d.jpg"),
    ])


def detect_scene_changes(src: Path) -> None:
    result = run([
        "ffmpeg", "-hide_banner", "-i", str(src),
        "-vf", "select='gt(scene,0.20)',showinfo",
        "-an", "-fps_mode", "vfr", "-f", "null", "-",
    ], check=False, capture=True)
    text = result.stderr
    rows: list[dict[str, float]] = []
    for line in text.splitlines():
        match = re.search(r"pts_time:([0-9.]+)", line)
        if match:
            rows.append({"time": float(match.group(1))})
    (OUT / "scene_changes.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def detect_speech_intervals(src: Path) -> None:
    result = run([
        "ffmpeg", "-hide_banner", "-i", str(src), "-vn",
        "-af", "silencedetect=noise=-35dB:d=0.40",
        "-f", "null", "-",
    ], check=False, capture=True)
    text = result.stderr
    (OUT / "silence_detection.log").write_text(text, encoding="utf-8", errors="ignore")


def write_srt(segments: list[dict]) -> None:
    def stamp(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: list[str] = []
    for idx, seg in enumerate(segments, 1):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        lines.extend([
            str(idx),
            f"{stamp(float(seg['start']))} --> {stamp(float(seg['end']))}",
            text,
            "",
        ])
    (OUT / "full_source_transcript.srt").write_text("\n".join(lines), encoding="utf-8")


def transcribe(src: Path) -> None:
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)
    segments_iter, info = model.transcribe(
        str(src),
        language="zh",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 350},
        word_timestamps=True,
        initial_prompt=(
            "北京舞蹈学院毕业生的日常生活Vlog。内容可能包括做饭、吃饭、换装、"
            "出门、电梯、街景、餐厅、酒吧、聚会、聊天、朋友和舞蹈。"
            "请准确转写为简体中文口语，不要自行概括。"
        ),
    )
    segments: list[dict] = []
    for seg in segments_iter:
        words = []
        for word in seg.words or []:
            words.append({
                "start": word.start,
                "end": word.end,
                "word": word.word,
                "probability": word.probability,
            })
        item = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
            "words": words,
        }
        segments.append(item)
        print(f"[{seg.start:8.2f} - {seg.end:8.2f}] {seg.text.strip()}", flush=True)

    payload = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": segments,
    }
    (OUT / "full_source_transcript.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_srt(segments)
    with (OUT / "dialogue_index.tsv").open("w", encoding="utf-8") as f:
        f.write("start\tend\tduration\ttext\n")
        for seg in segments:
            f.write(
                f"{seg['start']:.3f}\t{seg['end']:.3f}\t"
                f"{seg['end'] - seg['start']:.3f}\t{seg['text']}\n"
            )


def main() -> None:
    src = find_source()
    print(f"Source: {src} ({src.stat().st_size} bytes)")
    info = ffprobe(src)
    (OUT / "source_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_proxy(src)
    make_contact_sheets(src)
    detect_scene_changes(src)
    detect_speech_intervals(src)
    transcribe(src)
    print("Source analysis complete:", OUT)


if __name__ == "__main__":
    main()
