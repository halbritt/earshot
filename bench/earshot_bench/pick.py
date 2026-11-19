"""Take one channel from a raw interleaved recording and write it as 16-bit WAV.

  python3 -m earshot_bench.pick --channels 8 --width 4 --channel 0 IN.raw OUT.wav

The simplest variant: one microphone, nothing else. The level is raised so that the peak is
at 0.9 of full scale unless --no-normalise is given.
"""
import argparse
import array
import sys
import wave

CODES = {2: "h", 4: "i"}


def pick(raw, channels, width, channel, normalise=True):
    a = array.array(CODES[width])
    a.frombytes(raw[: len(raw) - len(raw) % (channels * width)])
    if sys.byteorder == "big":
        a.byteswap()
    one = a[channel::channels]
    shift = 8 * (width - 2)
    peak = max(max(one, default=0), -min(one, default=0)) or 1
    full = 1 << (8 * width - 1)
    gain = 0.9 * full / peak if normalise else 1.0
    out = array.array("h", (max(-32768, min(32767, int(v * gain) >> shift)) for v in one))
    if sys.byteorder == "big":
        out.byteswap()
    return out.tobytes()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", type=int, required=True)
    p.add_argument("--width", type=int, default=4, choices=sorted(CODES), help="bytes per sample")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--rate", type=int, default=16000)
    p.add_argument("--no-normalise", action="store_true")
    p.add_argument("input")
    p.add_argument("output")
    a = p.parse_args(argv)
    if not 0 <= a.channel < a.channels:
        p.error("--channel must be less than --channels")
    with open(a.input, "rb") as f:
        pcm = pick(f.read(), a.channels, a.width, a.channel, not a.no_normalise)
    with wave.open(a.output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(a.rate)
        w.writeframes(pcm)


if __name__ == "__main__":
    main()
