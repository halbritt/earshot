# earshot (working name)

A general-purpose front end for microphone arrays: echo cancellation, direction finding and
beamforming for speech recognition, with the array given as a description file. It is meant
for two uses: an agent that listens and speaks through one device, and a device that is put
in a room to transcribe everyone.

Status: the replay bench exists; the front end itself is not built yet.
See [docs/plan.md](docs/plan.md).

- `docs/plan.md`: why, the design rules, the stages, the interfaces and the order of work.
- `crates/earshot-bench/`: the replay bench, which scores any front-end variant on a recorded
  take by word error rate and CPU time. See its [README](crates/earshot-bench/README.md).
- `sim/arrays.py`: calculated directivity for candidate array geometries
  (`python3 sim/arrays.py`, needs NumPy; takes a few minutes). It is a one-off calculation
  that the front end's own beam design code will replace.

## Languages

Rust for everything. C only where a target chip's toolchain leaves no other choice.

## Licence

MIT. See [LICENSE](LICENSE).
