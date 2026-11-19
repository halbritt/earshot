"""End to end: real variant commands, a real HTTP recogniser stand-in, real files."""
import array
import http.server
import json
import pathlib
import sys
import threading
import wave

import pytest

from earshot_bench.__main__ import main
from earshot_bench.pick import pick

BENCH = str(pathlib.Path(__file__).resolve().parent.parent / "bench")
PY = f"PYTHONPATH={BENCH} {sys.executable}"

# The stand-in recogniser "hears" a text chosen by the first sample of the audio it is sent.
HEARD = {1000: "the quick brown fox", 2000: "jumps over the lazy dog", 3000: "the quick fox", 0: ""}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        start = body.index(b"RIFF")
        first = int.from_bytes(body[start + 44:start + 46], "little", signed=True)
        self.server.seen.append(first)
        reply = json.dumps({"text": HEARD[first]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *a):
        pass


@pytest.fixture
def recogniser():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.seen = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def raw_recording(path, per_second, channels=2, rate=100):
    """`per_second[c]` lists channel c's constant sample value for each second."""
    a = array.array("i")
    for s in range(len(per_second[0])):
        for _ in range(rate):
            for c in range(channels):
                a.append(per_second[c][s] << 16)
    path.write_bytes(a.tobytes())


@pytest.fixture
def take(tmp_path):
    # channel 0 carries "the quick brown fox" for 2 s then "jumps over the lazy dog" for 2 s;
    # channel 1 carries a worse first half.
    raw_recording(tmp_path / "rec.raw", [[1000, 1000, 2000, 2000], [3000, 3000, 2000, 2000]])
    (tmp_path / "ref.txt").write_text("# region: first\nThe quick brown fox\n# region: second\njumps over the lazy dog.\n")
    (tmp_path / "take.toml").write_text(
        'name = "t"\nrecording = "rec.raw"\nreference = "ref.txt"\n'
        '[[region]]\nname = "first"\nstart = 0\nend = 2\n[[region]]\nname = "second"\nstart = 2\nend = 4\n'
    )
    return tmp_path


def pick_cmd(channel):
    return f"{PY} -m earshot_bench.pick --channels 2 --channel {channel} --rate 100 --no-normalise {{in}} {{out}}"


def run(take, recogniser, *args):
    out = take / "r.json"
    code = main([str(take / "take.toml"), "--stt-url", f"http://127.0.0.1:{recogniser.server_address[1]}/x",
                 "--cache", str(take / "cache"), "--json", str(out), *args])
    return code, (json.loads(out.read_text())["results"] if out.exists() else None)


def test_a_variant_is_run_transcribed_per_region_and_scored(take, recogniser):
    code, r = run(take, recogniser, "--variant", "mic0=" + pick_cmd(0), "--variant", "mic1=" + pick_cmd(1))
    assert code == 0
    assert r["mic0"]["wer"] == 0 and r["mic0"]["correct"] == 9
    # mic1 heard "the quick fox" for "the quick brown fox": one of nine words missing
    assert r["mic1"]["missing"] == 1 and r["mic1"]["wer"] == round(1 / 9, 4)
    assert r["mic1"]["sections"]["first"]["missing"] == 1
    assert r["mic1"]["sections"]["second"]["wer"] == 0
    assert r["mic0"]["audio_seconds"] == 4.0


def test_the_table_names_each_variant_and_its_error_rate(take, recogniser, capsys):
    run(take, recogniser, "--variant", "mic1=" + pick_cmd(1))
    out = capsys.readouterr().out
    assert "9 reference words" in out
    line = next(l for l in out.splitlines() if l.startswith("mic1"))
    assert "11.1" in line


def test_an_existing_transcript_is_scored_without_a_recogniser(take, capsys):
    (take / "live.json").write_text(json.dumps({"first": "the quick brown fox", "second": "jumps over"}))
    code = main([str(take / "take.toml"), "--transcript", f"live={take / 'live.json'}", "--json", str(take / "r.json")])
    r = json.loads((take / "r.json").read_text())["results"]["live"]
    assert code == 0 and r["missing"] == 3 and "cpu_seconds" not in r


def test_unchanged_audio_is_not_sent_to_the_recogniser_twice(take, recogniser):
    run(take, recogniser, "--variant", "mic0=" + pick_cmd(0))
    first = len(recogniser.seen)
    code, r = run(take, recogniser, "--variant", "mic0=" + pick_cmd(0))
    assert first == 2 and len(recogniser.seen) == 2 and r["mic0"]["wer"] == 0


def test_the_cpu_time_of_the_variant_command_is_measured(take, recogniser):
    burn = f"{sys.executable} -c 'import time\nt=time.process_time()\nwhile time.process_time()-t<0.4: pass' && " + pick_cmd(0)
    code, r = run(take, recogniser, "--variant", "slow=" + burn, "--variant", "fast=" + pick_cmd(0))
    assert r["slow"]["cpu_seconds"] >= 0.4
    assert r["slow"]["cpu_seconds"] - r["fast"]["cpu_seconds"] >= 0.3
    assert r["slow"]["cpu_percent_of_real_time"] >= 10.0      # 0.4 s for 4 s of audio


def test_a_failing_variant_stops_the_run_with_its_error(take, recogniser, capsys):
    code, r = run(take, recogniser, "--variant", "bad=echo broken >&2; exit 3 # {out}")
    assert code == 2 and r is None
    err = capsys.readouterr().err
    assert "status 3" in err and "broken" in err


def test_a_variant_that_writes_stereo_is_rejected(take, recogniser, capsys):
    script = take / "stereo.py"
    script.write_text("import sys, wave\nw = wave.open(sys.argv[1], 'wb'); w.setnchannels(2); w.setsampwidth(2)\n"
                      "w.setframerate(100); w.writeframes(b'\\0' * 400); w.close()\n")
    code, _ = run(take, recogniser, "--variant", f"st={sys.executable} {script} {{out}}")
    assert code == 2 and "one channel" in capsys.readouterr().err


def test_without_regions_the_audio_is_cut_into_fixed_pieces(take, recogniser):
    (take / "take.toml").write_text('recording = "rec.raw"\nreference = "ref.txt"\nchunk_seconds = 2\n')
    code, r = run(take, recogniser, "--variant", "mic0=" + pick_cmd(0))
    assert list(r["mic0"]["texts"]) == ["0s", "2s"] and r["mic0"]["wer"] == 0


def test_a_missing_recogniser_address_is_an_error(take, capsys, monkeypatch):
    monkeypatch.delenv("EARSHOT_STT_URL", raising=False)
    assert main([str(take / "take.toml"), "--variant", "mic0=" + pick_cmd(0)]) == 2
    assert "EARSHOT_STT_URL" in capsys.readouterr().err


def test_pick_takes_the_chosen_channel_and_normalises_to_nine_tenths():
    raw = array.array("i", [1 << 28, 5, -(1 << 29), 6]).tobytes()      # 2 channels, 2 frames
    out = array.array("h")
    out.frombytes(pick(raw, 2, 4, 0))
    assert out[1] == pytest.approx(-0.9 * 32768, abs=2) and out[0] == pytest.approx(0.45 * 32768, abs=2)


def test_pick_without_normalising_keeps_the_top_sixteen_bits():
    raw = array.array("i", [1000 << 16, 0, -2000 << 16, 0]).tobytes()
    out = array.array("h")
    out.frombytes(pick(raw, 2, 4, 0, normalise=False))
    assert list(out) == [1000, -2000]
