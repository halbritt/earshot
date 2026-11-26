"""Render the 5-minute multi-character test script through Gemini TTS (`announce --out`)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coach.contracts import CanonicalEvent, EventKind, SessionBrief, Stability
from coach.storage import export_events_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPT = Path(__file__).resolve().with_name("script.json")
DEFAULT_BRIEF = Path(__file__).resolve().with_name("brief.json")
DEFAULT_OUT_DIR = REPO_ROOT / "takes" / "coach_conversation_5min"
DEFAULT_ANNOUNCE_BIN = "/home/halbritt/.local/bin/announce"


@dataclass(frozen=True, slots=True)
class CharacterSpec:
    speaker_id: str
    name: str
    role: str
    voice: str
    azimuth_deg: float
    style: str


@dataclass(frozen=True, slots=True)
class TurnSpec:
    turn_id: int
    act: int
    speaker_id: str
    text: str
    pause_after_ms: int
    pinned_pause: bool
    coach_expectation: str


@dataclass(frozen=True, slots=True)
class RenderedTurn:
    spec: TurnSpec
    char: CharacterSpec
    request_id: str
    digest_hex: str
    mp3_path: Path
    wav_path: Path
    job_path: Path
    nframes: int
    duration_ms: int
    effective_pause_ms: int
    start_ms: int
    end_ms: int
    cached: bool
    usage: Mapping[str, Any]
    estimated_new_cost_usd: float


def _turn_digest(voice: str, style: str, text: str) -> str:
    payload = json.dumps(
        {"voice": voice, "style": style, "text": text, "model": "gemini-3.8-flash-tts"},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_script(path: Path) -> tuple[str, str, float, dict[str, CharacterSpec], list[TurnSpec]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    title = str(raw["title"])
    session_id = str(raw["session_id"])
    target_seconds = float(raw.get("target_seconds", 300.0))
    chars: dict[str, CharacterSpec] = {}
    for sid, c in raw["characters"].items():
        chars[str(sid)] = CharacterSpec(
            speaker_id=str(sid),
            name=str(c["name"]),
            role=str(c["role"]),
            voice=str(c["voice"]),
            azimuth_deg=float(c["azimuth_deg"]),
            style=str(c["style"]),
        )
    turns: list[TurnSpec] = []
    for t in raw["turns"]:
        turns.append(
            TurnSpec(
                turn_id=int(t["turn_id"]),
                act=int(t["act"]),
                speaker_id=str(t["speaker_id"]),
                text=str(t["text"]),
                pause_after_ms=int(t["pause_after_ms"]),
                pinned_pause=bool(t.get("pinned_pause", False)),
                coach_expectation=str(t.get("coach_expectation", "")),
            )
        )
    return title, session_id, target_seconds, chars, turns


def _wav_nframes(wav_path: Path) -> int:
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getframerate() != 16000 or wf.getsampwidth() != 2:
            raise ValueError(f"Expected 16 kHz mono PCM16 WAV at {wav_path}")
        return wf.getnframes()


def _convert_to_16k_wav(mp3_path: Path, wav_path: Path) -> int:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(mp3_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ],
        check=True,
    )
    return _wav_nframes(wav_path)


def _synthesize_turn(
    turn: TurnSpec,
    char: CharacterSpec,
    turns_dir: Path,
    announce_bin: str,
) -> tuple[str, str, Path, Path, Path, int, bool, Mapping[str, Any], float]:
    digest_hex = _turn_digest(char.voice, char.style, turn.text)
    request_id = f"earshot-conv5m-{digest_hex[:16]}"
    stem = turns_dir / f"turn_{turn.turn_id:03d}"
    txt_path = stem.with_suffix(".txt")
    mp3_path = stem.with_suffix(".mp3")
    wav_path = turns_dir / f"turn_{turn.turn_id:03d}_16k.wav"
    job_path = turns_dir / f"turn_{turn.turn_id:03d}.job.json"
    txt_path.write_text(turn.text, encoding="utf-8")

    cached = False
    job: dict[str, Any] | None = None
    if mp3_path.is_file() and job_path.is_file():
        existing = json.loads(job_path.read_text(encoding="utf-8"))
        if existing.get("digest_hex") == digest_hex and existing.get("status") == "rendered":
            job = existing
            cached = True

    if job is None:
        proc = subprocess.run(
            [
                announce_bin,
                "--style",
                char.style,
                "--voice",
                char.voice,
                "--ttl",
                "1800",
                "--request-id",
                request_id,
                "--out",
                str(mp3_path),
                "--text-file",
                str(txt_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        job = json.loads(proc.stdout)
        if job.get("status") != "rendered":
            raise RuntimeError(f"Turn {turn.turn_id} failed status: {job}")
        result = job.get("result", {})
        if result.get("model") != "gemini-3.8-flash-tts":
            raise RuntimeError(f"Turn {turn.turn_id} unexpected model: {result.get('model')}")
        job["digest_hex"] = digest_hex
        job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    nframes = (
        _wav_nframes(wav_path)
        if wav_path.is_file() and cached
        else _convert_to_16k_wav(mp3_path, wav_path)
    )
    result = job.get("result", {})
    parts = result.get("parts", [])
    usage: dict[str, Any] = (
        parts[0].get("usage", {}) if parts and isinstance(parts[0], Mapping) else {}
    )
    est_cost = float(result.get("estimated_new_cost_usd", 0.0))
    return request_id, digest_hex, mp3_path, wav_path, job_path, nframes, cached, usage, est_cost


def _calibrate_pauses(
    turns: Sequence[TurnSpec],
    durations_ms: Sequence[int],
    target_seconds: float,
) -> list[int]:
    speech_ms = sum(durations_ms)
    pinned_ms = sum(t.pause_after_ms for t in turns if t.pinned_pause)
    unpinned_indices = [i for i, t in enumerate(turns) if not t.pinned_pause]
    target_ms = round(target_seconds * 1000.0)
    desired_unpinned_ms = max(0, target_ms - speech_ms - pinned_ms)
    base_unpinned_ms = sum(turns[i].pause_after_ms for i in unpinned_indices)
    scale = (desired_unpinned_ms / base_unpinned_ms) if base_unpinned_ms > 0 else 1.0
    effective: list[int] = []
    for t in turns:
        if t.pinned_pause:
            effective.append(t.pause_after_ms)
        else:
            scaled = round(t.pause_after_ms * scale)
            effective.append(max(250, min(1800, scaled)))
    return effective


def _assemble_outputs(
    title: str,
    brief: SessionBrief,
    chars: Mapping[str, CharacterSpec],
    turns: Sequence[TurnSpec],
    synth_rows: Sequence[tuple[str, str, Path, Path, Path, int, bool, Mapping[str, Any], float]],
    out_dir: Path,
    target_seconds: float,
) -> dict[str, Any]:
    durations_ms = [round(row[5] * 1000 / 16000) for row in synth_rows]
    pauses_ms = _calibrate_pauses(turns, durations_ms, target_seconds)
    rendered: list[RenderedTurn] = []
    cursor_ms = 0
    for turn, row, dur_ms, pause_ms in zip(turns, synth_rows, durations_ms, pauses_ms, strict=True):
        start_ms = cursor_ms
        end_ms = start_ms + dur_ms
        cursor_ms = end_ms + pause_ms
        rendered.append(
            RenderedTurn(
                spec=turn,
                char=chars[turn.speaker_id],
                request_id=row[0],
                digest_hex=row[1],
                mp3_path=row[2],
                wav_path=row[3],
                job_path=row[4],
                nframes=row[5],
                duration_ms=dur_ms,
                effective_pause_ms=pause_ms,
                start_ms=start_ms,
                end_ms=end_ms,
                cached=row[6],
                usage=row[7],
                estimated_new_cost_usd=row[8],
            )
        )

    combined_wav = out_dir / "conversation_5min_16k.wav"
    with wave.open(str(combined_wav), "wb") as out_wf:
        out_wf.setnchannels(1)
        out_wf.setsampwidth(2)
        out_wf.setframerate(16000)
        for rt in rendered:
            with wave.open(str(rt.wav_path), "rb") as in_wf:
                out_wf.writeframes(in_wf.readframes(in_wf.getnframes()))
            silence_frames = int(round(rt.effective_pause_ms * 16))
            if silence_frames > 0:
                out_wf.writeframes(b"\x00\x00" * silence_frames)

    combined_mp3 = out_dir / "conversation_5min.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(combined_wav),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "3",
            str(combined_mp3),
        ],
        check=True,
    )
    return _write_metadata_and_events(title, brief, rendered, out_dir, combined_wav, combined_mp3)


def _write_metadata_and_events(
    title: str,
    brief: SessionBrief,
    rendered: Sequence[RenderedTurn],
    out_dir: Path,
    combined_wav: Path,
    combined_mp3: Path,
) -> dict[str, Any]:
    events: list[CanonicalEvent] = []
    direction_lines: list[str] = []
    seq = 0
    for idx, rt in enumerate(rendered):
        seq += 1
        dur_ms = rt.end_ms - rt.start_ms
        events.append(
            CanonicalEvent(
                schema_version=1,
                event_id=f"ev-utt-{rt.spec.turn_id:03d}",
                session_id=brief.session_id,
                source_id="gemini-tts-ground-truth",
                source_sequence=seq,
                received_ms=rt.end_ms + 120,
                kind=EventKind.SEGMENT,
                start_ms=rt.start_ms,
                end_ms=rt.end_ms,
                channel=0,
                speaker_id=rt.spec.speaker_id,
                segment_id=f"seg-{rt.spec.turn_id:03d}",
                source_revision=1,
                stability=Stability.COMMITTED,
                text=rt.spec.text,
                asr_confidence=1.0,
                speaker_confidence=0.95,
                speaker_uncertain=dur_ms < 1000,
            )
        )
        t_ms = rt.start_ms + 100
        while t_ms <= rt.end_ms:
            direction_lines.append(
                json.dumps(
                    {
                        "t": round(t_ms / 1000.0, 3),
                        "azimuth": rt.char.azimuth_deg,
                        "contrast": 3.5,
                        "level_db": -22.0,
                    }
                )
            )
            t_ms += 200
        next_spk = rendered[idx + 1].spec.speaker_id if idx + 1 < len(rendered) else None
        if rt.effective_pause_ms >= 600 or (
            next_spk is not None and next_spk != rt.spec.speaker_id
        ):
            seq += 1
            events.append(
                CanonicalEvent(
                    schema_version=1,
                    event_id=f"ev-bnd-{rt.spec.turn_id:03d}",
                    session_id=brief.session_id,
                    source_id="gemini-tts-ground-truth",
                    source_sequence=seq,
                    received_ms=rt.end_ms + 160,
                    kind=EventKind.BOUNDARY,
                    start_ms=rt.end_ms,
                    end_ms=rt.end_ms,
                    channel=0,
                    speaker_id=rt.spec.speaker_id,
                    segment_id=f"seg-{rt.spec.turn_id:03d}",
                    source_revision=1,
                    stability=Stability.COMMITTED,
                    boundary="pause" if rt.effective_pause_ms >= 600 else "speaker_change",
                )
            )

    events_path = out_dir / "events.jsonl"
    export_events_jsonl(events, events_path, brief=brief)
    direction_path = out_dir / "direction.jsonl"
    direction_path.write_text("\n".join(direction_lines) + "\n", encoding="utf-8")

    total_frames = _wav_nframes(combined_wav)
    total_s = total_frames / 16000.0
    speech_s = sum(rt.nframes for rt in rendered) / 16000.0
    pause_s = max(0.0, total_s - speech_s)
    total_in_tok = sum(int(rt.usage.get("total_input_tokens", 0)) for rt in rendered)
    total_out_tok = sum(int(rt.usage.get("total_output_tokens", 0)) for rt in rendered)
    total_cost = round(sum(rt.estimated_new_cost_usd for rt in rendered), 6)
    manifest: dict[str, Any] = {
        "title": title,
        "session_id": brief.session_id,
        "model": "gemini-3.8-flash-tts",
        "playback": "none (render-only --out)",
        "wav_path": str(combined_wav),
        "mp3_path": str(combined_mp3),
        "events_jsonl": str(events_path),
        "direction_jsonl": str(direction_path),
        "turn_count": len(rendered),
        "word_count": sum(len(rt.spec.text.split()) for rt in rendered),
        "total_duration_seconds": round(total_s, 3),
        "speech_seconds": round(speech_s, 3),
        "pause_seconds": round(pause_s, 3),
        "voices_used": sorted({rt.char.voice for rt in rendered}),
        "total_input_tokens": total_in_tok,
        "total_output_tokens": total_out_tok,
        "estimated_new_cost_usd": total_cost,
        "turns": [
            {
                "turn_id": rt.spec.turn_id,
                "act": rt.spec.act,
                "speaker_id": rt.spec.speaker_id,
                "character": rt.char.name,
                "voice": rt.char.voice,
                "start_ms": rt.start_ms,
                "end_ms": rt.end_ms,
                "duration_ms": rt.duration_ms,
                "pause_after_ms": rt.effective_pause_ms,
                "cached": rt.cached,
                "coach_expectation": rt.spec.coach_expectation,
            }
            for rt in rendered
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _write_markdown(
    path: Path,
    title: str,
    brief: SessionBrief,
    chars: Mapping[str, CharacterSpec],
    turns: Sequence[TurnSpec],
) -> None:
    total_words = sum(len(t.text.split()) for t in turns)
    lines = [
        f"# 5-Minute Conversation Script: {title}",
        "",
        f"- **Session ID:** `{brief.session_id}`",
        f"- **Mode:** `{brief.mode.value}`",
        f"- **Objective:** {brief.objective}",
        f"- **Turns / Words:** {len(turns)} turns, {total_words} spoken words (~5 min with natural pauses)",
        "- **TTS Stack:** `~/git/announce` (`gemini-3.8-flash-tts`, render-only via `announce --out`)",
        "",
        "## Characters",
        "",
        "| Speaker ID | Name | Role | Gemini Voice | Azimuth (deg) | Style Direction |",
        "|---|---|---|---|---:|---|",
    ]
    for sid in ("ME", "S1", "S2", "S3"):
        c = chars[sid]
        lines.append(
            f"| `{c.speaker_id}` | {c.name} | `{c.role}` | `{c.voice}` | {c.azimuth_deg:.0f} | {c.style} |"
        )
    lines.extend(["", "## Dialogue", ""])
    cur_act = 0
    for t in turns:
        if t.act != cur_act:
            cur_act = t.act
            lines.extend([f"### Act {cur_act}", ""])
        c = chars[t.speaker_id]
        lines.append(
            f"- **Turn {t.turn_id:02d} — {t.speaker_id} ({c.name}, `{c.voice}`)** "
            f"[{t.coach_expectation}, pause {t.pause_after_ms} ms"
            f"{' pinned' if t.pinned_pause else ''}]: {t.text}"
        )
    lines.extend(
        [
            "",
            "## Reproduce Audio & Coach Replay",
            "",
            "```sh",
            "uv run python scripts/conversation_5min/render_announce.py --write-markdown scripts/conversation_5min/SCRIPT.md",
            "uv run coach replay --session takes/coach_conversation_5min/events.jsonl --policy policies/policy-v0-a.yaml --db takes/coach_conversation_5min/replay.db --run-id conv5m-ground-truth --json",
            "uv run coach report --db takes/coach_conversation_5min/replay.db --session conv-5min-cutover-001 --run-id conv5m-ground-truth --json",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize the 5-minute multi-character test script via announce."
    )
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--brief", type=Path, default=DEFAULT_BRIEF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--announce-bin", default=DEFAULT_ANNOUNCE_BIN)
    parser.add_argument("--target-seconds", type=float, default=300.0)
    parser.add_argument("--write-markdown", type=Path, default=None)
    args = parser.parse_args(argv)

    brief = SessionBrief.from_dict(json.loads(args.brief.read_text(encoding="utf-8")))
    title, session_id, target_s, chars, turns = _load_script(args.script)
    if session_id != brief.session_id:
        raise SystemExit(f"session_id mismatch: {session_id!r} != {brief.session_id!r}")
    if args.write_markdown is not None:
        _write_markdown(args.write_markdown, title, brief, chars, turns)

    turns_dir = args.out_dir / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)
    synth_rows = [
        _synthesize_turn(turn, chars[turn.speaker_id], turns_dir, args.announce_bin)
        for turn in turns
    ]
    eff_target = target_s if args.target_seconds == 300.0 else float(args.target_seconds)
    manifest = _assemble_outputs(title, brief, chars, turns, synth_rows, args.out_dir, eff_target)
    print(json.dumps({k: v for k, v in manifest.items() if k != "turns"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
