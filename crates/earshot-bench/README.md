# Replay bench

Scores audio front-end variants on a recorded take. A variant is any command that turns the
recording into one channel of audio. The bench runs it, measures its CPU time, sends the
audio to a speech recogniser, and compares the words with what was really said.

Build with `cargo build --release`; the binary is `target/release/earshot-bench`. The recogniser is a server with an OpenAI-style
transcription endpoint (multipart upload with a `file` field, JSON answer with a `text`
field); whisper.cpp's server is one.

## A take

A take is a directory with a recording, the text that was read, and optionally the text that
the device played, described by a TOML file:

```toml
name = "kitchen-1"
recording = "kitchen-1.raw"      # whatever the variant commands read
reference = "reference.txt"      # what the person said
playback = ["played.txt"]        # optional: what the device said
chunk_seconds = 25               # used when there are no regions

[[region]]                       # optional: where to cut the audio for the recogniser
name = "quiet"
start = 12.0                     # seconds
end = 48.5
```

Cut regions at pauses. Without regions the audio is cut every `chunk_seconds`, which can
split a word.

The reference is plain text. A line `# region: NAME` starts a section, and errors are then
also reported for each section, for example quiet parts against parts where the device was
talking. Trim the reference to what was really read.

Recordings of people are private. Keep takes out of the repository; `takes/` is ignored.

## Running

```sh
export EARSHOT_STT_URL=http://recogniser.example:8080/inference
earshot-bench run takes/kitchen-1/take.toml \
  --variant 'mic0=earshot-bench pick --channels 8 --channel 0 {in} {out}' \
  --variant 'beam=./my-beamformer --array respeaker.toml {in} {out}' \
  --transcript live=takes/kitchen-1/live.json \
  --json results.json
```

- `--variant NAME=COMMAND`: the command is run by `sh`. `{in}` is the recording, `{out}` is
  where it must write 16-bit, one-channel WAV. The CPU time is that of the command and its children.
- `--transcript NAME=FILE.json`: score text that already exists, for example from a running
  service. The file is a JSON object of chunk name to text.
- `earshot-bench pick` is the simplest variant: one channel of a raw interleaved recording.
- Transcripts are cached in `.earshot-cache` by the hash of the audio, so an unchanged
  variant is not sent to the recogniser again.

## What it reports

```
variant  WER %  correct  subst  missing  extra  leaked  CPU % of real time  WER % quiet  WER % over
mic0       5.5      437     18        4      3       0                 0.2          3.1         8.2
live      31.8      330     18      111     17       0                   -         25.0        40.1
live: 25 consecutive words missing in 'over' from reference word 139
```

- **WER**: substituted, missing and extra words, divided by the number of reference words.
- **leaked**: extra or wrong words that form a run of three or more words from the playback
  text. These are the device's own speech coming back.
- **missing runs**: four or more consecutive reference words with nothing heard. One long
  run points at detection or tracking; scattered single words point at audio quality.
- **CPU % of real time**: CPU seconds per second of audio, on the machine that ran the
  command. Compare variants on the same machine only.

## How much difference is real

The recogniser is not stable against changes of its input that nobody can hear. One
microphone of a 459-word reading was scored sixteen times: as it was, with gain changes from
-6 to +3 dB, delayed by 1 to 400 samples, and with one step of added noise.

- Word error rate: 5.0 to 7.4% (mean 6.6, standard deviation 0.9).
- Correct words: 436 to 438. Missing words: 2 or 3. Substituted: 19 to 21.
- Extra words: 1 to 12. Almost all of the spread is one run of about ten inserted words
  that the recogniser adds in some versions and not in others.

So on a take of that length a difference of less than about 2.5 points in word error rate
is nothing, while a difference of five or more correct or missing words is real. Look at
`correct` and `missing` first, and confirm any result on more than one take.

## Tests

`mise run test`, or `cargo test`. The end-to-end tests run the real binary against a stand-in
recogniser on the loopback interface.
