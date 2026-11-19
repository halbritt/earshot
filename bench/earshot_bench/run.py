"""Run variants on a take and score them."""
import io
import os
import pathlib
import shlex
import subprocess
import tempfile
import time
import tomllib
import wave
from dataclasses import dataclass

from .text import read_reference, score, words


class BenchError(Exception):
    pass


@dataclass
class Take:
    name: str
    recording: pathlib.Path
    reference: list          # [(section, words)]
    playback: list           # words
    regions: list            # [(name, start_s, end_s)], may be empty
    chunk_s: float

    @classmethod
    def load(cls, path):
        path = pathlib.Path(path)
        with open(path, "rb") as f:
            m = tomllib.load(f)
        base = path.parent
        for key in ("recording", "reference"):
            if key not in m:
                raise BenchError(f"{path}: '{key}' is missing")
        reference = read_reference((base / m["reference"]).read_text())
        if not reference:
            raise BenchError(f"{path}: the reference text has no words")
        playback = []
        for p in m.get("playback", []):
            playback += words((base / p).read_text())
        regions = [(r["name"], float(r["start"]), float(r["end"])) for r in m.get("region", [])]
        for name, a, b in regions:
            if not b > a >= 0:
                raise BenchError(f"{path}: region '{name}' must have 0 <= start < end")
        return cls(m.get("name", path.stem), base / m["recording"], reference, playback, regions,
                   float(m.get("chunk_seconds", 25.0)))


def run_variant(command, recording, out_path):
    """Run a variant command. `{in}` and `{out}` in the command are replaced by the recording
    and the output path. Returns (cpu seconds of the command and its children, wall seconds)."""
    if "{out}" not in command:
        raise BenchError("a variant command must contain {out}")
    cmd = command.replace("{in}", shlex.quote(str(recording))).replace("{out}", shlex.quote(str(out_path)))
    before, t0 = os.times(), time.monotonic()
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    after, wall = os.times(), time.monotonic() - t0
    if proc.returncode != 0:
        raise BenchError(f"variant command failed with status {proc.returncode}: {cmd}\n{proc.stderr[-2000:]}")
    if not pathlib.Path(out_path).exists():
        raise BenchError(f"variant command wrote no output file: {cmd}")
    cpu = (after.children_user - before.children_user) + (after.children_system - before.children_system)
    return cpu, wall


def read_wav(path):
    try:
        with wave.open(str(path), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise BenchError(f"{path}: a variant must write one channel of 16-bit samples "
                                 f"(found {w.getnchannels()} channels, {8 * w.getsampwidth()} bits)")
            return w.getframerate(), w.readframes(w.getnframes())
    except wave.Error as e:
        raise BenchError(f"{path}: not a WAV file ({e})") from None


def wav_bytes(rate, pcm):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return b.getvalue()


def chunks(take, rate, pcm):
    """(name, pcm) pieces to transcribe: the take's regions, or fixed-length pieces."""
    total = len(pcm) // 2 / rate
    if take.regions:
        spans = take.regions
    else:
        n = max(1, int(-(-total // take.chunk_s)))
        spans = [(f"{i * take.chunk_s:.0f}s", i * take.chunk_s, min(total, (i + 1) * take.chunk_s)) for i in range(n)]
    return [(name, pcm[2 * int(a * rate): 2 * int(b * rate)]) for name, a, b in spans]


def score_audio(take, recogniser, wav_path):
    rate, pcm = read_wav(wav_path)
    texts = {}
    for name, piece in chunks(take, rate, pcm):
        texts[name] = recogniser.transcribe(wav_bytes(rate, piece)) if piece else ""
    return texts, len(pcm) / 2 / rate


def result(take, texts, cpu=None, wall=None, audio_s=None):
    hyp = words(" ".join(texts.values()))
    r = score(take.reference, hyp, take.playback).as_dict()
    r["transcript_words"] = len(hyp)
    r["texts"] = texts
    if cpu is not None:
        r.update(cpu_seconds=round(cpu, 2), wall_seconds=round(wall, 2), audio_seconds=round(audio_s, 2),
                 cpu_percent_of_real_time=round(100 * cpu / audio_s, 1) if audio_s else None)
    return r


def bench(take, variants, transcripts, recogniser, keep_dir=None):
    """variants: {name: command}; transcripts: {name: {chunk name: text}} scored as they are."""
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = pathlib.Path(keep_dir) if keep_dir else pathlib.Path(tmp)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, command in variants.items():
            out = out_dir / f"{name}.wav"
            if out.exists():
                out.unlink()
            cpu, wall = run_variant(command, take.recording, out)
            texts, audio_s = score_audio(take, recogniser, out)
            results[name] = result(take, texts, cpu, wall, audio_s)
    for name, texts in transcripts.items():
        results[name] = result(take, texts)
    return results


def table(results, sections=True):
    names = []
    for r in results.values():
        for s in r["sections"]:
            if s not in names:
                names.append(s)
    show_sections = sections and names != ["all"]
    head = ["variant", "WER %", "correct", "subst", "missing", "extra", "leaked", "CPU % of real time"]
    if show_sections:
        head += [f"WER % {s}" for s in names]
    rows = [head]
    for name, r in results.items():
        row = [name, f"{100 * r['wer']:.1f}", r["correct"], r["substituted"], r["missing"], r["extra"], r["leaked"],
               r.get("cpu_percent_of_real_time", "-")]
        if show_sections:
            row += [f"{100 * r['sections'][s]['wer']:.1f}" if s in r["sections"] else "-" for s in names]
        rows.append([str(x) for x in row])
    widths = [max(len(r[i]) for r in rows) for i in range(len(head))]
    lines = ["  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))) for r in rows]
    for name, r in results.items():
        for run in r["missing_runs"]:
            lines.append(f"{name}: {run['words']} consecutive words missing in '{run['section']}' "
                         f"from reference word {run['first_word']}")
    return "\n".join(lines)
