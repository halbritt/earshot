# earshot (working name)

A general-purpose front end for microphone arrays: echo cancellation, direction finding and
beamforming for speech recognition, with the array given as a description file. It is meant
for two uses: an agent that listens and speaks through one device, and a device that is put
in a room to transcribe everyone.

Status: the replay bench exists, and the first part of the core: array description, beam
design, a fixed beam over a recording, and direction finding. Echo cancellation, speech
detection and live capture are not built yet. See [docs/plan.md](docs/plan.md).

- `docs/plan.md`: why, the design rules, the stages, the interfaces and the order of work.
- `crates/earshot-bench/`: the replay bench, which scores any front-end variant on a recorded
  take by word error rate and CPU time. See its [README](crates/earshot-bench/README.md).
- `crates/earshot-core/`: the processing core, `no_std`: the array as data, delay-and-sum and
  superdirective beam design, direction finding (SRP-PHAT over azimuth), and the transform. Its beam design is tested against the
  simulation below.
- `crates/earshot-cli/`: the `earshot` command. `earshot beam --array arrays/respeaker-core-v2.toml
  --steer 100 in.raw out.wav` puts one fixed beam over a raw recording; `--steer 0:100,113.5:305`
  turns it at given times. `earshot direction --array ... in.raw` prints where the sound
  comes from, twice a second. These work on recordings; there is no service yet.
- `arrays/`: array description files.
- `sim/arrays.py`: calculated directivity for candidate array geometries
  (`python3 sim/arrays.py`, needs NumPy; takes a few minutes). It is a one-off calculation;
  the core's beam design now gives the same numbers.

## Languages

Rust for everything. C only where a target chip's toolchain leaves no other choice.

## Licence

MIT. See [LICENSE](LICENSE).
