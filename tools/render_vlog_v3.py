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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output_v3"
WORK = ROOT / ".vlog_v3_work"
PLAN_PATH = ROOT / "tools" / "vlog_v2_plan.json"
OUT.mkdir(exist_ok=True)
WORK.mkdir(exist_ok=True)


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def find_source() -> Path:
    candidates = [
        p for p in ROOT.iterdir()
        if p.suffix.lower() in {".mov", ".mp4", ".m4v"} and p.parent != OUT
    ]
    if not candidates:
        raise SystemExit("No source video found in repository root")
    return max(candidates, key=lambda p: p.stat().st_size)


def probe_duration(path: Path) -> float:
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path)
    ], capture=True)
    return float(result.stdout.strip())


def atempo_chain(speed: float) -> str:
    if speed <= 0:
        raise ValueError("speed must be positive")
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


def render_segments(src: Path, plan: dict[str, Any]) -> tuple[list[Path], list[dict[str, Any]]]:
    segments: list[Path] = []
    manifest: list[dict[str, Any]] = []
    timeline = 0.0

    for idx, clip in enumerate(plan["clips"]):
        start = float(clip["start"])
        duration = float(clip["duration"])
        speed = float(clip.get("speed", 1.0))
        volume = float(clip.get("volume", 1.0))
        clip_type = str(clip.get("type", "dialogue"))
        output_duration = duration / speed
        seg = WORK / f"seg_{idx:03d}.mkv"

        zoom = 1.0 if clip_type == "dialogue" else (1.018 if idx % 2 else 1.008)
        sw = int(round(1080 * zoom / 2) * 2)
        sh = int(round(1920 * zoom / 2) * 2)
        offset = 10 if clip_type == "montage" and idx % 4 == 1 else -10 if clip_type == "montage" and idx % 4 == 3 else 0
        x = max(0, (sw - 1080) // 2 + offset)
        y = max(0, (sh - 1920) // 2)
        vf = (
            f"scale={sw}:{sh}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop=1080:1920:{x}:{y},"
            "hqdn3d=0.65:0.65:2.8:2.8,"
            "eq=contrast=1.03:saturation=1.045:brightness=0.005:gamma=1.01,"
            "unsharp=3:3:0.26:3:3:0.0,"
            f"setpts=PTS/{speed:.6f},fps=30,format=yuv420p"
        )
        af = f"{atempo_chain(speed)},volume={volume:.4f},aresample=48000"

        run([
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src),
            "-vf", vf, "-af", af,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(seg)
        ])
        segments.append(seg)
        manifest.append({
            "index": idx,
            "source_start": start,
            "source_duration": duration,
            "speed": speed,
            "type": clip_type,
            "scene": clip.get("scene", ""),
            "timeline_start": timeline,
            "timeline_end": timeline + output_duration,
        })
        timeline += output_duration
    return segments, manifest


def concat_segments(segments: list[Path]) -> Path:
    listing = WORK / "concat.txt"
    listing.write_text("\n".join(f"file '{p.as_posix()}'" for p in segments), encoding="utf-8")
    rough = WORK / "rough_dialogue_v3.mkv"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(rough)])
    return rough


def clean_text(text: str) -> str:
    text = text.replace("字幕", "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", text)
    text = re.sub(r"([（【])\s+", r"\1", text)
    text = re.sub(r"\s+([）】])", r"\1", text)
    text = re.sub(r"([。！？!?])\1+", r"\1", text)
    return text.strip(" -—")


def is_useful_text(text: str) -> bool:
    compact = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    if not compact:
        return False
    if len(set(compact)) == 1 and len(compact) > 4:
        return False
    banned = {"谢谢观看", "请不吝点赞订阅转发打赏支持明镜与点点栏目"}
    return text not in banned


def transcribe_to_cues(rough: Path, model_name: str) -> list[dict[str, Any]]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
    segments, _ = model.transcribe(
        str(rough),
        language="zh",
        beam_size=5,
        best_of=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 180},
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=0.0,
        no_speech_threshold=0.55,
        initial_prompt=(
            "北京舞蹈学院毕业生日常Vlog。真实口语对话，涉及 duty、duty free、A室、"
            "薯条、意大利国旗、下雨、十二点四十六、蛋炒饭。请准确转写为简体中文，"
            "保留双方接话，不要添加解说。"
        ),
    )

    cues: list[dict[str, Any]] = []
    current_words: list[Any] = []
    previous_end: float | None = None

    def flush() -> None:
        nonlocal current_words, previous_end
        if not current_words:
            return
        text = clean_text("".join(w.word for w in current_words))
        start = float(current_words[0].start)
        end = float(current_words[-1].end)
        if is_useful_text(text) and end - start >= 0.25:
            end = max(end, start + 0.72)
            cues.append({"start": start, "end": end, "text": text})
        current_words = []
        previous_end = None

    for segment in segments:
        words = list(segment.words or [])
        if not words:
            text = clean_text(segment.text)
            if is_useful_text(text):
                cues.append({"start": float(segment.start), "end": float(segment.end), "text": text})
            continue
        for word in words:
            token = word.word
            token_clean = clean_text(token)
            gap = 0.0 if previous_end is None else max(0.0, float(word.start) - previous_end)
            tentative = clean_text("".join(w.word for w in current_words) + token)
            duration = 0.0 if not current_words else float(word.end) - float(current_words[0].start)
            punctuation_break = bool(current_words and re.search(r"[。！？!?]$", clean_text(current_words[-1].word)))
            if current_words and (gap > 0.46 or len(tentative) > 18 or duration > 3.6 or punctuation_break):
                flush()
            if token_clean:
                current_words.append(word)
                previous_end = float(word.end)
        flush()

    cleaned: list[dict[str, Any]] = []
    for cue in cues:
        if cleaned and cue["text"] == cleaned[-1]["text"] and cue["start"] - cleaned[-1]["end"] < 1.2:
            cleaned[-1]["end"] = max(cleaned[-1]["end"], cue["end"])
            continue
        if cleaned and cue["start"] < cleaned[-1]["end"]:
            cleaned[-1]["end"] = max(cleaned[-1]["start"] + 0.45, cue["start"] - 0.04)
        cleaned.append(cue)
    return cleaned


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


def wrap_caption(text: str, width: int = 16, ass: bool = True) -> str:
    text = clean_text(text)
    if len(text) <= width:
        return text
    center = len(text) // 2
    candidates = [i + 1 for i, ch in enumerate(text) if ch in "，。！？、,!?" and 5 <= i <= len(text) - 5]
    split = min(candidates, key=lambda i: abs(i - center)) if candidates else center
    sep = r"\N" if ass else "\n"
    return text[:split] + sep + text[split:]


def write_subtitles(cues: list[dict[str, Any]]) -> tuple[Path, Path]:
    ass = WORK / "dialogue_v3.ass"
    srt = OUT / "北舞高能量毕业生_嘴闲不住的一天_v3_对白字幕.srt"
    font = "Noto Sans CJK SC"
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Dialogue,{font},49,&H00FFFFFF,&H000000FF,&H00111111,&H55000000,-1,0,0,0,100,100,0,0,1,2.6,0.8,2,76,76,142,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    srt_lines: list[str] = []
    for idx, cue in enumerate(cues, 1):
        ass_text = wrap_caption(str(cue["text"]), ass=True)
        srt_text = wrap_caption(str(cue["text"]), ass=False)
        events.append(f"Dialogue: 0,{ass_time(float(cue['start']))},{ass_time(float(cue['end']))},Dialogue,,0,0,0,,{ass_text}")
        srt_lines.extend([
            str(idx),
            f"{srt_time(float(cue['start']))} --> {srt_time(float(cue['end']))}",
            srt_text,
            "",
        ])
    ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    srt.write_text("\n".join(srt_lines), encoding="utf-8")
    return ass, srt


def make_cheerful_bgm(seconds: float, path: Path, sr: int = 48000) -> None:
    import numpy as np

    rng = np.random.default_rng(20260714)
    bpm = 116.0
    beat = 60.0 / bpm
    roots = np.array([130.81, 98.00, 110.00, 87.31], dtype=np.float64)
    thirds = np.array([1.2599, 1.2599, 1.1892, 1.2599], dtype=np.float64)

    def tone(freq: np.ndarray, t: np.ndarray) -> np.ndarray:
        return (
            np.sin(2 * np.pi * freq * t)
            + 0.34 * np.sin(2 * np.pi * freq * 2 * t)
            + 0.13 * np.sin(2 * np.pi * freq * 3 * t)
            + 0.05 * np.sin(2 * np.pi * freq * 4 * t)
        ) / 1.52

    total = int(seconds * sr)
    chunk_samples = sr * 5
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        for base in range(0, total, chunk_samples):
            count = min(chunk_samples, total - base)
            t = (base + np.arange(count, dtype=np.float64)) / sr
            beat_index = np.floor(t / beat).astype(np.int64)
            bar_index = beat_index // 4
            chord_index = bar_index % 4
            root = roots[chord_index]
            third = thirds[chord_index]
            beat_phase = np.mod(t, beat)
            bar_phase = np.mod(t, beat * 4)

            chord_gate = 0.62 + 0.38 * (0.5 + 0.5 * np.cos(2 * np.pi * beat_phase / beat))
            pad = (tone(root * 2, t) + tone(root * third * 2, t) + tone(root * 1.4983 * 2, t)) / 3.0
            pad *= 0.105 * chord_gate
            bass = 0.12 * np.exp(-beat_phase * 4.8) * tone(root, t)
            kick_freq = 68.0 - 20.0 * np.minimum(1.0, beat_phase / 0.16)
            kick = 0.28 * np.exp(-beat_phase * 18.0) * np.sin(2 * np.pi * kick_freq * t)
            within_bar_beat = np.floor(bar_phase / beat).astype(np.int64)
            snare_mask = (within_bar_beat == 1) | (within_bar_beat == 3)
            snare = 0.075 * np.exp(-beat_phase * 26.0) * rng.uniform(-1.0, 1.0, count) * snare_mask
            rhythm_phase = np.mod(t, beat / 2)
            rhythm = 0.028 * np.exp(-rhythm_phase * 10.0) * tone(root * 2, t)
            fade_in = np.minimum(1.0, t / 0.75)
            fade_out = np.clip((seconds - t) / 1.1, 0.0, 1.0)
            samples = np.clip((pad + bass + kick + snare + rhythm) * fade_in * fade_out, -1.0, 1.0)
            pcm = (samples * 32767.0).astype('<i2')
            stereo = np.empty(count * 2, dtype='<i2')
            stereo[0::2] = pcm
            stereo[1::2] = pcm
            wav.writeframes(stereo.tobytes())


def mux_final(rough: Path, ass: Path, duration: float) -> Path:
    bgm = WORK / "cheerful_music_only.wav"
    make_cheerful_bgm(duration + 1.0, bgm)

    filter_complex = (
        "[0:a]highpass=f=65,acompressor=threshold=-19dB:ratio=3:attack=10:release=180,"
        "volume=1.06,asplit=2[voice_mix][voice_sc];"
        "[1:a]volume=0.19[bg];"
        "[bg][voice_sc]sidechaincompress=threshold=0.020:ratio=12:attack=10:release=300[bgduck];"
        "[voice_mix][bgduck]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
        "volume=1.28,alimiter=limit=0.94[aout]"
    )
    escaped_ass = str(ass).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    vf = f"ass='{escaped_ass}',fade=t=in:st=0:d=0.16,fade=t=out:st={max(0.0, duration - 0.42):.3f}:d=0.42"
    final = OUT / "北舞高能量毕业生_嘴闲不住的一天_vlog_v3_对话加强版.mp4"
    run([
        "ffmpeg", "-y", "-i", str(rough), "-i", str(bgm),
        "-filter_complex", filter_complex, "-vf", vf,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-profile:v", "high",
        "-c:a", "aac", "-b:a", "224k", "-movflags", "+faststart", str(final)
    ])
    return final


def make_preview(final: Path) -> Path:
    preview = OUT / "v3_抽帧预览.jpg"
    run([
        "ffmpeg", "-y", "-i", str(final),
        "-vf", "fps=1/10,scale=270:480,tile=4x4:padding=4:margin=4",
        "-frames:v", "1", "-q:v", "2", str(preview)
    ])
    return preview


def main() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    src = find_source()
    segments, manifest = render_segments(src, plan)
    rough = concat_segments(segments)
    duration = probe_duration(rough)
    cues = transcribe_to_cues(rough, str(plan.get("whisper_model", "medium")))
    if len(cues) < 12:
        raise RuntimeError(f"Too few subtitle cues were recognized: {len(cues)}")
    ass, srt = write_subtitles(cues)
    final = mux_final(rough, ass, duration)
    preview = make_preview(final)

    (OUT / "剪辑方案_v3.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "对话镜头清单_v3.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "自动转写_v3.json").write_text(json.dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "source": src.name,
        "clip_count": len(plan["clips"]),
        "dialogue_clip_count": sum(1 for c in plan["clips"] if c.get("type") == "dialogue"),
        "subtitle_cue_count": len(cues),
        "duration": probe_duration(final),
        "output": str(final),
        "srt": str(srt),
        "preview": str(preview),
        "added_sound_effects": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
