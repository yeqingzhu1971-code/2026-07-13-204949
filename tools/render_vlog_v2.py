#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import re
import struct
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output_v2"
WORK = ROOT / ".vlog_v2_work"
PLAN_PATH = ROOT / "tools" / "vlog_v2_plan.json"
OUT.mkdir(exist_ok=True)
WORK.mkdir(exist_ok=True)


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def find_source() -> Path:
    candidates = [p for p in ROOT.iterdir() if p.suffix.lower() in {".mov", ".mp4", ".m4v"}]
    if not candidates:
        raise SystemExit("No source video found")
    return max(candidates, key=lambda p: p.stat().st_size)


def probe_duration(path: Path) -> float:
    return float(run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path)
    ], capture=True).stdout.strip())


def atempo_chain(speed: float) -> str:
    parts: list[str] = []
    value = speed
    while value > 2.0:
        parts.append("atempo=2.0")
        value /= 2.0
    while value < 0.5:
        parts.append("atempo=0.5")
        value /= 0.5
    parts.append(f"atempo={value:.6f}")
    return ",".join(parts)


def render_segments(src: Path, plan: dict) -> tuple[list[Path], list[float]]:
    segments: list[Path] = []
    output_lengths: list[float] = []
    for idx, clip in enumerate(plan["clips"]):
        start = float(clip["start"])
        duration = float(clip["duration"])
        speed = float(clip.get("speed", 1.0))
        out_len = duration / speed
        seg = WORK / f"seg_{idx:03d}.mkv"
        zoom = 1.035 if idx % 3 == 1 else (1.018 if idx % 3 == 2 else 1.0)
        sw = int(round(720 * zoom / 2) * 2)
        sh = int(round(1280 * zoom / 2) * 2)
        x = max(0, (sw - 720) // 2 + (8 if idx % 4 == 1 else -8 if idx % 4 == 3 else 0))
        y = max(0, (sh - 1280) // 2)
        vf = (
            f"scale={sw}:{sh}:force_original_aspect_ratio=increase,"
            f"crop=720:1280:{x}:{y},"
            "eq=contrast=1.035:saturation=1.08:brightness=0.008,"
            f"setpts=PTS/{speed:.6f},fps=30"
        )
        volume = float(clip.get("volume", 1.0))
        af = f"{atempo_chain(speed)},loudnorm=I=-16:LRA=9:TP=-1.5,volume={volume:.4f}"
        run([
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src),
            "-t", f"{duration:.3f}", "-vf", vf, "-af", af,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(seg)
        ])
        segments.append(seg)
        output_lengths.append(out_len)
    return segments, output_lengths


def concat_segments(segments: list[Path]) -> Path:
    listing = WORK / "concat.txt"
    listing.write_text("\n".join(f"file '{p.as_posix()}'" for p in segments), encoding="utf-8")
    rough = WORK / "rough_v2.mkv"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(rough)])
    return rough


def transcribe(rough: Path) -> list[dict]:
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4)
        iterator, _ = model.transcribe(
            str(rough), language="zh", beam_size=5, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 260}, word_timestamps=False,
            condition_on_previous_text=False,
            initial_prompt=(
                "北京舞蹈学院高能量毕业生的日常生活Vlog。请准确转写简体中文口语，"
                "保留吐槽、笑话、做饭、出门、坐电梯、朋友聚会和聊天内容，不要概括。"
            )
        )
        items = []
        for seg in iterator:
            text = seg.text.strip()
            if text:
                items.append({"start": seg.start, "end": seg.end, "text": text})
        return items
    except Exception as exc:
        print("Transcription failed:", exc, file=sys.stderr)
        return []


def ass_time(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap_cn(text: str, width: int = 14) -> str:
    text = re.sub(r"\s+", "", text.strip())
    if len(text) <= width:
        return text
    if len(text) <= width * 2:
        split = len(text) // 2
        return text[:split] + r"\N" + text[split:]
    return text[:width] + r"\N" + text[width:width * 2]


def build_subtitles(segments: list[dict], captions: list[dict], overlays: list[dict], use_auto: bool) -> tuple[Path, Path]:
    ass = WORK / "vlog_v2.ass"
    font = "Noto Sans CJK SC"
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Dialogue,{font},38,&H00FFFFFF,&H000000FF,&H00101010,&H50000000,-1,0,0,0,100,100,0,0,1,3,1,2,48,48,105,1
Style: Hook,{font},54,&H00FFFFFF,&H000000FF,&H00101010,&H65000000,-1,0,0,0,100,100,1,0,1,4,1,8,38,38,92,1
Style: Punch,{font},48,&H0000D7FF,&H000000FF,&H00101010,&H00000000,-1,0,0,0,100,100,1,0,1,4,1,5,45,45,0,1
Style: Chapter,{font},34,&H00FFFFFF,&H000000FF,&H00101010,&H60000000,-1,0,0,0,100,100,1,0,1,3,1,7,38,38,66,1
Style: Counter,{font},25,&H00FFFFFF,&H000000FF,&H00101010,&H60000000,-1,0,0,0,100,100,1,0,1,2,1,9,30,30,48,1
Style: End,{font},42,&H00FFFFFF,&H000000FF,&H00101010,&H65000000,-1,0,0,0,100,100,1,0,1,4,1,5,45,45,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    dialogue_items = segments if use_auto else captions
    for seg in dialogue_items:
        start, end = float(seg["start"]), float(seg["end"])
        if end - start > 4.4:
            end = start + 4.4
        text = wrap_cn(str(seg["text"]))
        events.append(f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Dialogue,,0,0,0,,{text}")
    for item in overlays:
        style = item.get("style", "Punch")
        text = str(item["text"]).replace("\n", r"\N")
        events.append(
            f"Dialogue: 3,{ass_time(float(item['start']))},{ass_time(float(item['end']))},{style},,0,0,0,,{text}"
        )
    ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")

    srt = OUT / "北舞高能量毕业生_嘴闲不住的一天_v2字幕.srt"
    lines: list[str] = []
    for idx, seg in enumerate(dialogue_items, 1):
        lines.extend([
            str(idx),
            f"{srt_time(float(seg['start']))} --> {srt_time(float(seg['end']))}",
            str(seg["text"]).strip(), ""
        ])
    srt.write_text("\n".join(lines), encoding="utf-8")
    return ass, srt


def make_bgm(seconds: float, path: Path, sr: int = 44100) -> None:
    random.seed(20260713)
    n = int(seconds * sr)
    frames = bytearray()
    sections = [(0.0, 138, 0), (max(8.0, seconds * 0.30), 122, 1), (max(18.0, seconds * 0.66), 132, 2)]
    roots_by_section = [
        [220.00, 261.63, 196.00, 293.66],
        [174.61, 220.00, 196.00, 233.08],
        [196.00, 246.94, 220.00, 261.63],
    ]
    for i in range(n):
        t = i / sr
        section = max(j for j, (st, _, _) in enumerate(sections) if t >= st)
        st, bpm, mode = sections[section]
        beat = 60.0 / bpm
        roots = roots_by_section[mode]
        root = roots[int((t - st) / (beat * 4)) % len(roots)]
        local = t - st
        beat_phase = local % beat
        half_phase = local % (beat / 2)
        bar_phase = local % (beat * 4)
        bass = 0.105 * math.sin(2 * math.pi * (root / 2) * t)
        chord = 0.055 * math.sin(2 * math.pi * root * t) + 0.035 * math.sin(2 * math.pi * root * 1.5 * t)
        kick = 0.42 * math.exp(-beat_phase * 17) * math.sin(2 * math.pi * (78 - 30 * beat_phase) * t)
        hat = 0.035 * math.exp(-half_phase * 75) * (1 if math.sin(2 * math.pi * 7000 * t) >= 0 else -1)
        clap_phase = (local + beat) % (beat * 2)
        clap = 0.11 * math.exp(-clap_phase * 42) * math.sin(2 * math.pi * 1800 * t)
        pulse = 0.04 * math.sin(2 * math.pi * root * 2 * t) * (1.0 if bar_phase < beat * 2 else 0.35)
        section_fade = min(1.0, max(0.0, (t - st) / 0.35))
        next_st = sections[section + 1][0] if section + 1 < len(sections) else seconds
        section_fade *= min(1.0, max(0.0, (next_st - t) / 0.25)) if next_st < seconds else 1.0
        total_fade = min(1.0, t / 0.65, max(0.0, (seconds - t) / 0.9))
        x = (bass + chord + kick + hat + clap + pulse) * section_fade * total_fade
        v = int(max(-1.0, min(1.0, x)) * 32767)
        frames += struct.pack("<hh", v, v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames)


def make_sfx(path: Path, kind: str, seconds: float = 0.28, sr: int = 44100) -> None:
    random.seed(kind)
    frames = bytearray()
    for i in range(int(seconds * sr)):
        t = i / sr
        p = min(1.0, t / seconds)
        if kind == "pop":
            x = 0.55 * math.exp(-t * 24) * math.sin(2 * math.pi * (580 + 500 * p) * t)
        elif kind == "ding":
            x = 0.42 * math.exp(-t * 8) * (math.sin(2 * math.pi * 1050 * t) + 0.45 * math.sin(2 * math.pi * 1575 * t))
        else:
            noise = random.random() * 2 - 1
            x = (0.30 * noise + 0.32 * math.sin(2 * math.pi * (260 + 3500 * p * p) * t)) * math.sin(math.pi * p) * (1 - p)
        v = int(max(-1.0, min(1.0, x)) * 32767)
        frames += struct.pack("<hh", v, v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames)


def mux_final(rough: Path, ass: Path, plan: dict, duration: float) -> Path:
    bgm = WORK / "bgm_v2.wav"
    whoosh = WORK / "whoosh.wav"
    pop = WORK / "pop.wav"
    ding = WORK / "ding.wav"
    make_bgm(duration + 1.0, bgm)
    make_sfx(whoosh, "whoosh", 0.34)
    make_sfx(pop, "pop", 0.20)
    make_sfx(ding, "ding", 0.42)

    input_args = ["-i", str(rough), "-i", str(bgm)]
    filter_parts = [
        "[0:a]highpass=f=70,volume=1.02[voice]",
        "[1:a]volume=0.25[bgraw]",
        "[bgraw][voice]sidechaincompress=threshold=0.025:ratio=10:attack=12:release=260[bgduck]",
    ]
    mix_labels = ["[voice]", "[bgduck]"]
    for i, sfx in enumerate(plan.get("sfx", [])):
        kind = sfx.get("kind", "whoosh")
        source = {"whoosh": whoosh, "pop": pop, "ding": ding}.get(kind, whoosh)
        input_args += ["-i", str(source)]
        stream = i + 2
        delay = int(float(sfx["time"]) * 1000)
        vol = float(sfx.get("volume", 0.30))
        filter_parts.append(f"[{stream}:a]adelay={delay}|{delay},volume={vol:.3f}[fx{i}]")
        mix_labels.append(f"[fx{i}]")
    filter_parts.append(
        "".join(mix_labels) + f"amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0,alimiter=limit=0.94[aout]"
    )
    escaped_ass = str(ass).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    vf = f"ass='{escaped_ass}',fade=t=in:st=0:d=0.18,fade=t=out:st={max(0.0, duration-0.45):.3f}:d=0.45"
    final = OUT / "北舞高能量毕业生_嘴闲不住的一天_vlog_v2.mp4"
    run(["ffmpeg", "-y"] + input_args + [
        "-filter_complex", ";".join(filter_parts), "-vf", vf,
        "-map", "0:v", "-map", "[aout]", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(final)
    ])
    return final


def main() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    src = find_source()
    segments, _ = render_segments(src, plan)
    rough = concat_segments(segments)
    duration = probe_duration(rough)
    transcript = transcribe(rough) if plan.get("auto_transcribe", False) else []
    ass, _ = build_subtitles(
        transcript, plan.get("captions", []), plan.get("overlays", []),
        bool(plan.get("use_auto_captions", False))
    )
    final = mux_final(rough, ass, plan, duration)
    (OUT / "剪辑方案_v2.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "转写_v2.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "source": src.name,
        "clip_count": len(plan["clips"]),
        "duration": probe_duration(final),
        "output": str(final)
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
