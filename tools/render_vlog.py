#!/usr/bin/env python3
import json
import math
import os
import re
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
WORK = ROOT / ".vlog_work"
OUT.mkdir(exist_ok=True)
WORK.mkdir(exist_ok=True)


def run(cmd, check=True, capture=False):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def probe(path):
    p = run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=index,codec_type,width,height,r_frame_rate",
        "-of", "json", str(path)
    ], capture=True)
    return json.loads(p.stdout)


def find_source():
    candidates = [p for p in ROOT.iterdir() if p.suffix.lower() in {".mov", ".mp4", ".m4v"}]
    candidates = [p for p in candidates if p.parent != OUT]
    if not candidates:
        raise SystemExit("No source video found in repository root")
    return max(candidates, key=lambda p: p.stat().st_size)


def non_silent_intervals(src, duration):
    log = WORK / "silence.log"
    with log.open("w", encoding="utf-8") as f:
        subprocess.run([
            "ffmpeg", "-hide_banner", "-i", str(src), "-vn",
            "-af", "silencedetect=noise=-34dB:d=0.45", "-f", "null", "-"
        ], stdout=subprocess.DEVNULL, stderr=f, check=False)
    text = log.read_text(encoding="utf-8", errors="ignore")
    starts = [float(x) for x in re.findall(r"silence_start: ([0-9.]+)", text)]
    ends = [float(x) for x in re.findall(r"silence_end: ([0-9.]+)", text)]
    events = []
    for s in starts:
        events.append((s, "start"))
    for e in ends:
        events.append((e, "end"))
    events.sort()
    intervals = []
    cursor = 0.0
    in_silence = False
    for t, kind in events:
        if kind == "start" and not in_silence:
            if t - cursor >= 1.2:
                intervals.append((cursor, t))
            in_silence = True
        elif kind == "end" and in_silence:
            cursor = t
            in_silence = False
    if not in_silence and duration - cursor >= 1.2:
        intervals.append((cursor, duration))
    return intervals or [(0.0, duration)]


def choose_clips(intervals, duration, target=72.0):
    # Keep the day moving: distribute clips across the full recording, favor speech-rich spans.
    bins = 14
    picks = []
    for i in range(bins):
        lo, hi = duration * i / bins, duration * (i + 1) / bins
        local = []
        for a, b in intervals:
            x, y = max(a, lo), min(b, hi)
            if y - x >= 1.4:
                local.append((x, y))
        if local:
            a, b = max(local, key=lambda z: z[1] - z[0])
            length = min(6.0 if i in {0, bins-1} else 5.0, b - a)
            start = a + max(0.0, (b - a - length) * 0.35)
            picks.append((start, length))
    if not picks:
        step = duration / bins
        picks = [(i * step, min(5.0, duration - i * step)) for i in range(bins)]
    # Cap around target duration.
    total = 0.0
    final = []
    for start, length in picks:
        if total >= target:
            break
        length = min(length, target - total)
        if length >= 1.0:
            final.append((start, length))
            total += length
    return final


def make_bgm(seconds, path, sr=44100):
    # Original royalty-free upbeat synth/percussion bed generated in-code.
    import struct
    bpm = 126
    beat = 60.0 / bpm
    n = int(seconds * sr)
    frames = bytearray()
    for i in range(n):
        t = i / sr
        phase = t % (beat * 4)
        chord_idx = int(t / (beat * 4)) % 4
        roots = [220.0, 246.94, 196.0, 261.63]
        root = roots[chord_idx]
        synth = 0.12 * math.sin(2 * math.pi * root * t) + 0.06 * math.sin(2 * math.pi * root * 1.5 * t)
        beat_phase = t % beat
        kick = 0.55 * math.exp(-beat_phase * 18) * math.sin(2 * math.pi * (72 - 35 * beat_phase) * t)
        clap_phase = (t + beat / 2) % beat
        clap = 0.16 * math.exp(-clap_phase * 45) * math.sin(2 * math.pi * 1600 * t)
        hat_phase = t % (beat / 2)
        hat = 0.05 * math.exp(-hat_phase * 70) * math.sin(2 * math.pi * 5800 * t)
        fade = min(1.0, t / 1.2, max(0.0, (seconds - t) / 1.4))
        x = max(-1.0, min(1.0, (synth + kick + clap + hat) * fade))
        v = int(x * 32767)
        frames += struct.pack("<hh", v, v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames)


def make_sfx(path, seconds=0.35, sr=44100):
    import random, struct
    n = int(seconds * sr)
    frames = bytearray()
    for i in range(n):
        t = i / sr
        env = math.sin(math.pi * min(1.0, t / seconds)) ** 2
        sweep = math.sin(2 * math.pi * (350 + 3200 * (t / seconds) ** 2) * t)
        noise = (random.random() * 2 - 1) * 0.25
        x = (0.45 * sweep + noise) * env * (1 - t / seconds)
        v = int(max(-1, min(1, x)) * 32767)
        frames += struct.pack("<hh", v, v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes(frames)


def main():
    src = find_source()
    info = probe(src)
    duration = float(info["format"]["duration"])
    intervals = non_silent_intervals(src, duration)
    clips = choose_clips(intervals, duration)
    print("Selected clips:", clips)

    segments = []
    for idx, (start, length) in enumerate(clips):
        seg = WORK / f"seg_{idx:02d}.mp4"
        vf = (
            "scale=720:1280:force_original_aspect_ratio=increase,"
            "crop=720:1280,"
            "eq=contrast=1.04:saturation=1.08,"
            "fps=30"
        )
        run([
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{length:.3f}",
            "-vf", vf, "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(seg)
        ])
        segments.append(seg)

    concat = WORK / "concat.txt"
    concat.write_text("\n".join(f"file '{p.as_posix()}'" for p in segments), encoding="utf-8")
    rough = WORK / "rough.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(rough)])
    rough_duration = float(probe(rough)["format"]["duration"])

    # Speech-to-text captions. Base model balances speed and accuracy on Actions runners.
    run([sys.executable, "-m", "whisper", str(rough), "--model", "base", "--language", "Chinese",
         "--task", "transcribe", "--output_format", "srt", "--output_dir", str(WORK)], check=False)
    srt = WORK / "rough.srt"
    if not srt.exists():
        srt.write_text("1\n00:00:00,000 --> 00:00:03,500\n北舞高能量毕业生，嘴闲不住的一天！\n", encoding="utf-8")

    bgm = WORK / "bgm.wav"
    sfx = WORK / "whoosh.wav"
    make_bgm(rough_duration + 1.0, bgm)
    make_sfx(sfx)

    # Add multiple quick whooshes at cut points.
    sfx_inputs = []
    filter_parts = ["[1:a]volume=0.16[bg]", "[0:a]volume=1.0[voice]"]
    input_args = ["-i", str(rough), "-i", str(bgm)]
    cursor = 0.0
    for i, (_, length) in enumerate(clips[:-1]):
        cursor += length
        input_args += ["-i", str(sfx)]
        delay = int(max(0, cursor - 0.08) * 1000)
        filter_parts.append(f"[{i+2}:a]adelay={delay}|{delay},volume=0.32[s{i}]")
        sfx_inputs.append(f"[s{i}]")
    mix_inputs = "[voice][bg]" + "".join(sfx_inputs)
    filter_parts.append(f"{mix_inputs}amix=inputs={2+len(sfx_inputs)}:duration=first:dropout_transition=0,alimiter=limit=0.95[aout]")

    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    title_font = font if Path(font).exists() else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    escaped_srt = str(srt).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    title = "北舞高能量毕业生\\N嘴闲不住的一天！"
    vf = (
        f"subtitles='{escaped_srt}':force_style='FontName=Noto Sans CJK SC,FontSize=18,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00101010,BorderStyle=1,Outline=3,Shadow=1,"
        "Alignment=2,MarginV=105',"
        f"drawtext=fontfile='{title_font}':text='{title}':fontsize=43:fontcolor=white:"
        "borderw=4:bordercolor=black:x=(w-text_w)/2:y=110:enable='between(t,0,3.2)',"
        "drawbox=x=36:y=54:w=210:h=46:color=black@0.58:t=fill:enable='between(t,0,5)',"
        f"drawtext=fontfile='{title_font}':text='BDA GRAD VLOG':fontsize=22:fontcolor=white:"
        "x=54:y=64:enable='between(t,0,5)',"
        "fade=t=in:st=0:d=0.35,fade=t=out:st=" + f"{max(0, rough_duration-0.55):.2f}" + ":d=0.55"
    )

    final = OUT / "北舞高能量毕业生_嘴闲不住的一天_vlog.mp4"
    cmd = ["ffmpeg", "-y"] + input_args + [
        "-filter_complex", ";".join(filter_parts), "-vf", vf,
        "-map", "0:v", "-map", "[aout]", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(final)
    ]
    run(cmd)
    (OUT / "字幕.srt").write_text(srt.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Rendered: {final}")

if __name__ == "__main__":
    main()
