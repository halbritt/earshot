# earshot (working name)

A general-purpose front end for microphone arrays: echo cancellation, direction finding and
beamforming for speech recognition, with the array given as a description file. It is meant
for two uses: an agent that listens and speaks through one device, and a device that is put
in a room to transcribe everyone.

Status: planning. See [docs/plan.md](docs/plan.md).

- `docs/plan.md`: why, the design rules, the stages, the interfaces and the order of work.
- `sim/arrays.py`: calculated directivity for candidate array geometries
  (`python3 sim/arrays.py`, needs NumPy; takes a few minutes).

## Licence

MIT. See [LICENSE](LICENSE).
