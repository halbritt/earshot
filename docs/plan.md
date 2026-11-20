# A general-purpose microphone-array front end: plan

Status: the bench and the first part of the core are built (see "Order of work"). The measurements come from the
[ReSpeaker Core v2 revival](https://github.com/halbritt/respeaker-core-v2-revival) project,
whose running service is the first array this tool will be tested on.

## Why

The pipeline in that project joins five programs (`arecord`, `rsp-aec`, ODAS, a Python segmenter and
a speech gate) and is specific to one board. Measurements on this board say where the value
is and where it is not:

- With one talker a few metres away, one microphone after the echo canceller transcribes as
  well as any beam (5–7% word error rate on a 459-word reading, beams and single microphone
  alike).
- The live pipeline lost a quarter of the same reading. The words were lost by tracking and
  speech detection, not by the beam and not by the echo rule.
- ODAS takes about half of one core all the time, in a quiet room too.
- On this 93 mm, 6-microphone array, a delay-and-sum beam has almost no directivity below
  1 kHz. A fixed superdirective beam does (see the simulation below).

The new tool keeps what earned its place, makes the array a configuration item, and is
written so that the same core can move to a low-power chip later.

## Two modes of use

| | Agent mode | Room mode |
|---|---|---|
| Purpose | One person talks to an agent, and the agent talks back | Put the array in a room and transcribe everyone |
| Playback | Yes, so echo cancellation is necessary | Usually none |
| Talkers | The one addressing the device | All of them, including overlap |
| Output | One stream, opened by a wake phrase or speech | One stream per tracked talker, with direction and time |
| What loses words | Slow or missed detection, echo | Distance, overlap, wrong talker assignment |
| Role of beamforming | Small at close range, more in noise | Central |

Both modes use the same core. They differ in the policy that decides which streams to open.

## Design rules

1. **Linear, slowly varying processing only on the path to the recogniser.** Masks,
   post-filters and fast adaptation make audio sound cleaner and make recognition worse.
2. **Never drop audio because a detector was late.** A ring buffer of 1–2 s sits in front of
   every decision, and a stream starts from the buffered past.
3. **Do as little as possible in silence.** One microphone and a cheap gate run all the
   time. Everything else starts when the gate opens.
4. **The array is data.** Microphone positions, channel order, sample rate, gain and the
   optional reference channel come from a description file. No geometry in code.
5. **Every change is scored by replay.** The same recordings, the same recogniser, word
   error rate against a reference text, CPU time, and words leaked from playback.
6. **Rust for everything, and the core is `no_std`.** The processing core is a library
   with a fixed block size, no allocation after start, no operating-system calls and
   single-precision arithmetic. This is what makes a later move to a small chip possible.
   C is used only where a chip's toolchain has no Rust support.

## Stages

```
capture ──► ring buffer ──► gate (1 mic) ──► echo canceller ──► STFT ──┬─► direction
(N ch + optional ref)                         (only with a ref)        ├─► beams
                                                                       └─► stream policy ──► outputs
```

| Stage | First version | Later options |
|---|---|---|
| Gate | Energy threshold, then Silero speech detector on one microphone | Wake phrase |
| Echo canceller | Speex, one filter per microphone (what `rsp-aec` does now) | One shared multichannel filter in the STFT domain |
| Direction | SRP-PHAT over azimuth only, 5–10 times a second, only while the gate is open | Elevation, multiple-source tracking for room mode |
| Beams | Superdirective weights calculated once from the geometry, on a grid of directions; optional fixed null on the loudspeaker | Slow MVDR with the noise estimate gated by the speech detector (memory of about 16 s) |
| Stream policy | Agent mode: the strongest speech direction | Room mode: one stream per tracked talker |

Not planned: neural multichannel enhancement (40–300 times over the CPU budget of this
board), and post-filters on the recogniser path.

## Interfaces

**Input.** Interleaved PCM from ALSA, a file or standard input, plus an array description:

```toml
rate = 16000
channels = 8
reference = [6, 7]            # optional playback loopback channels
[[mic]]
channel = 0
position = [0.0401, 0.0231, 0.0]   # metres
```

**Output.** Two forms:

- A native event stream: audio per stream, with stream identifier, direction, start and end
  time, and whether playback was active. Room mode needs all of these.
- A Wyoming adapter, so that existing recognisers, synthesisers and wake-word services work
  without changes.

### Wyoming

Wyoming is the protocol that Home Assistant uses between voice programs: one line of JSON
per event, with optional binary data, over TCP, a Unix socket or standard input and output.
Read at version 1.12.

Fits:

- `audio-start`, `audio-chunk` and `audio-stop` carry PCM with rate, width, channel count and
  a millisecond timestamp.
- `transcribe` and `transcript`, with streaming transcripts that can revise earlier words.
- `synthesize` with streaming text, `detect` and `detection` for wake words,
  `voice-started` and `voice-stopped`.
- Events carry a free-form `context` object, and there is a `user-event` type, so direction
  and talker identity can travel without breaking other programs.
- Recognisers, synthesisers and wake-word services that speak it already exist.

Does not fit as the only interface:

- It has no authentication or encryption by design. Room speech must not cross a network
  on it without a tunnel.
- It has no standard fields for direction or talker, and one connection carries one audio
  stream. Room mode would need one connection per talker and private conventions.
- Its satellite events assume the Home Assistant pipeline order (wake, recognise, intent,
  speak). An agent that can be interrupted does not follow that order.

Decision: speak Wyoming at the edge, as an adapter. Keep the native stream as the full
interface.

## Array simulation

Calculated, not measured (`sim/arrays.py`, results in `sim/arrays.json`). Free-field omnidirectional microphones, talker 20° above the
plane of the array, spherically diffuse noise, average over look directions. "Band" is the
mean directivity over 300–7000 Hz on a logarithmic scale. "Second talker" is the mean
power from a talker 90° to the side. Superdirective weights keep the white-noise gain at
0 dB or better, so microphone self-noise and mismatch are not amplified.

| Array | Delay-and-sum: 500 Hz | 1 kHz | band | Superdirective: 500 Hz | 1 kHz | band | Second talker at 90° |
|---|---|---|---|---|---|---|---|
| 4 on 64 mm | 0.3 | 1.1 | 2.8 | 3.2 | 5.8 | 5.1 | −5.7 |
| ReSpeaker Core v2: 6 on 93 mm | 0.6 | 2.3 | 4.0 | 5.5 | 7.1 | 6.9 | −8.7 |
| 6 on 93 mm and one in the centre | 0.5 | 1.9 | 4.1 | 5.5 | 7.1 | 7.2 | −10.0 |
| 8 on 70 mm | 0.3 | 1.3 | 3.6 | 5.0 | 6.5 | 7.4 | −10.1 |
| 8 on 100 mm | 0.7 | 2.6 | 4.4 | 5.9 | 7.8 | 7.9 | −11.6 |
| 8 on 140 mm | 1.3 | 4.5 | 5.4 | 6.5 | 8.7 | 8.4 | −12.0 |
| 4 on 40 mm and 8 on 120 mm | 0.7 | 2.6 | 4.9 | 6.3 | 8.4 | 8.9 | −14.4 |
| 6 on 60 mm and 10 on 188 mm | 1.6 | 4.8 | 6.4 | 8.0 | 10.2 | 10.2 | −16.3 |
| Whistler: 16 on 188 mm | 2.3 | 6.0 | 6.9 | 8.2 | 9.9 | 10.3 | −17.0 |

All values in dB.

What the table says:

- Delay-and-sum follows the diameter. Superdirective follows the number of microphones
  more than the diameter.
- The method matters as much as the hardware: superdirective weights on the ReSpeaker
  (6.9 dB) equal delay-and-sum on a 16-microphone, 188 mm array (6.9 dB).
- Returns diminish. From 6 to 8 microphones is worth about 1 dB, from 8 to 16 about 2 dB.
  Rejection of a second talker improves faster than directivity: 8 dB more from the
  ReSpeaker to the 16-microphone ring.
- The 36.7 mm spacing of the 16-microphone ring does not cost directivity above 4.7 kHz in
  diffuse noise (11.6 dB at 7 kHz).
- A second, inner ring gave nothing over a single ring with the same number of
  microphones.

What it does not say: the effect of the board and the table surface, reverberation, and
how many dB a recogniser needs. Only recordings answer those.

## Later: a low-power chip

Not decided now. Some candidate chips have no Rust toolchain; on those the core would be
linked from C or ported. The plan keeps the door open by rule 6 and by measuring the cost of each
stage in operations per second, not only as a share of this board's CPU. The gate is the
part that must run always, so it decides standby power; it should run on one microphone at
a few milliwatts. Capture chips of the kind used on 16-microphone USB arrays have signal
processing capacity of their own, so a split is possible: gate and capture on the small
chip, the rest on a host that sleeps.

## What the board's own voice taught us

Measured on the ReSpeaker with its speaker a few centimetres from the microphones, while
fixing a leak of the board's voice into the transcript. These decide the echo stage here.

- **The echo path hardly changes.** A fixed 96 ms filter fitted to one 5 s prompt removed 17
  to 19 dB of another prompt 90 s later, after the listener had moved. Fitted and tested on
  the same prompt it removed 20 to 21 dB; a longer filter added nothing. The adaptive
  filter in use (Speex) removed 13 to 16 dB. A filter that learns slowly from long averages
  should do better than one that follows every block, above all while someone talks over
  the playback.
- **The limit is the speaker, then the room's noise.** With the speaker inside its clean
  range a fixed filter removed everything down to the noise of the room (21 to 26 dB). Driven
  5 dB harder it removed 19 to 21 dB. Polynomial models of the speaker fitted well and
  predicted nothing on the next half of the same prompt.
- **What is left is still transcribed.** The leftover of a 6 s prompt came out as 17 wrong
  words with high confidence. The recogniser's own confidence did not tell it from a person
  (-0.03 average log probability, better than the real reader).
- **Masking the leftover makes it worse.** Attenuating time-frequency cells where the
  canceller's output was far below its input turned the leftover into different wrong
  sentences. This is design rule 1 again.
- **A spatial null on the speaker did not help**: 17 dB with one complex weight per
  frequency, with or without a person talking.
- **What worked is a decision, not a filter.** Block by block (30 ms), the canceller's
  output is compared with its input. Within 9 dB: something else is there. The speech
  detector ignores the other blocks while the device plays. On a 459-word reading this
  removed the leaked prompt and lost four of the reader's words; on an older take the error
  rate fell from 50% to 27%.

## Order of work

1. **Replay bench** (done: `crates/earshot-bench`). Recording, variant and reference text in; word error
   rate, CPU time and leaked playback words out.
2. **One-microphone detection in the ReSpeaker service**, in that project, as an option, replayed against all
   takes. It is the fix for the lost quarter and the first measured piece of the new design.
3. **Harder recordings:** two talkers, a noise source, more distance, no playback. These
   decide how much beamforming the room mode needs.
4. **Core library:** array description, STFT, superdirective beam set, azimuth direction.
   Checked against the simulation and replayed on the bench. Done so far
   (`crates/earshot-core`, `earshot beam`): array description, STFT and beam design, which
   gives the simulation's directivity to 0.05 dB. On the 459-word reading, with the beam
   steered by hand: one microphone 5.4%, delay-and-sum 5.2%, superdirective 5.4%, and a
   superdirective beam pointed 180 degrees away from the talker 8.3%. Direction
   (`earshot direction`), on the same recording, half-second intervals, louder half of each
   part: a reader at about 100 degrees is placed at 103, and after moving to about 305 at
   299 to 300, with every interval within 15 degrees while the board is silent and 76 to
   97% of them while it talks. The board talking alone reads 176 degrees, its own speaker.
   Steered by that direction (`earshot beam --steer auto`: the beam turns when three
   measurements in a row, 0.77 s, agree on a new direction, and ignores sound while the
   device plays), the beam made the two right turns in the 222 s and no other, and scored
   434 correct words against 436 steered by hand.
   CPU on the ReSpeaker (one Cortex-A7 core, statically linked build, whole recording):
   transform and one beam 3.3% of the core, direction finding another 5.6%, measured all the
   time and not only during speech. ODAS takes 46% on the same board.
   Not done: speech gate, echo stage, live capture.
5. **Agent mode end to end** on the ReSpeaker, replacing ODAS; then the Wyoming adapter.
6. **Room mode:** talker tracking and one stream per talker.
7. **A second array** to prove that the array really is data.

## Open decisions

- The name. `earshot` is a working name.
- Speech detector on a small chip: Silero needs a neural runtime; a smaller detector may
  be necessary there.
- Who labels talkers in room mode: direction alone, or voice identity from the recogniser
  side.
