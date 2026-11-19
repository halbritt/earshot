"""earshot-bench: score front-end variants on a recorded take.

  python3 -m earshot_bench TAKE.toml --variant NAME='COMMAND {in} {out}' ... [--transcript NAME=FILE.json]

A variant command reads the recording at {in} and writes one channel of 16-bit WAV to {out}.
The recogniser address comes from --stt-url or the EARSHOT_STT_URL environment variable.
"""
import argparse
import json
import os
import pathlib
import sys

from .run import BenchError, Take, bench, table
from .stt import Recogniser


def pairs(items, what):
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise BenchError(f"--{what} needs NAME=VALUE, got '{item}'")
        if name in out:
            raise BenchError(f"--{what} '{name}' is given twice")
        out[name] = value
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="earshot-bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("take", help="take description (TOML)")
    p.add_argument("--variant", action="append", default=[], metavar="NAME=COMMAND")
    p.add_argument("--transcript", action="append", default=[], metavar="NAME=FILE.json",
                   help="score an existing transcript: a JSON object of chunk name to text")
    p.add_argument("--stt-url", default=os.environ.get("EARSHOT_STT_URL"))
    p.add_argument("--language", default="en")
    p.add_argument("--cache", default=os.environ.get("EARSHOT_CACHE", ".earshot-cache"),
                   help="directory for cached transcripts ('' for none)")
    p.add_argument("--keep", metavar="DIR", help="keep each variant's output audio in DIR")
    p.add_argument("--json", metavar="FILE", help="write full results, with transcripts, to FILE")
    a = p.parse_args(argv)
    try:
        variants = pairs(a.variant, "variant")
        files = pairs(a.transcript, "transcript")
        if not variants and not files:
            raise BenchError("give at least one --variant or --transcript")
        if set(variants) & set(files):
            raise BenchError("a variant and a transcript have the same name")
        if variants and not a.stt_url:
            raise BenchError("no recogniser: give --stt-url or set EARSHOT_STT_URL")
        take = Take.load(a.take)
        transcripts = {}
        for name, path in files.items():
            data = json.loads(pathlib.Path(path).read_text())
            if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
                raise BenchError(f"{path}: expected a JSON object of chunk name to text")
            transcripts[name] = data
        recogniser = Recogniser(a.stt_url, a.cache or None, a.language) if variants else None
        results = bench(take, variants, transcripts, recogniser, a.keep)
    except (BenchError, OSError) as e:
        print(f"earshot-bench: {e}", file=sys.stderr)
        return 2
    words = sum(len(w) for _, w in take.reference)
    print(f"take {take.name}: {words} reference words, {len(take.playback)} playback words")
    print(table(results))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(dict(take=take.name, results=results), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
