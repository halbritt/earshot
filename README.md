# earshot + coach

Two-layer live conversation audio front end and conversation coach:

1. **Rust Audio Front End (`crates/`):** `no_std` microphone-array beamforming, SRP-PHAT direction finding, echo/speech gating, and WER/CPU replay benchmark (`earshot` CLI and `earshot-bench`).
2. **Python Live Conversation Coach (`coach/`, `policies/`, `coach/ui/`):** Python 3.14 (`uv`) `asyncio` 5-task coaching runtime with SQLite WAL persistence, organized along the 14 upward-only module boundaries in [`docs/design.md`](docs/design.md) §15, versioned YAML policies/templates (`policies/`), a private `127.0.0.1`-only token-gated WebSocket Card UI (`coach/ui/`), and a 2-pass locked annotation tool (`coach/annotation_tool.py`).

## Repository Layout

- `crates/earshot-core/`: `no_std` array geometry, delay-and-sum/superdirective beamforming, SRP-PHAT azimuth direction finding, and FFT transform.
- `crates/earshot-cli/`: `earshot` CLI (`beam`, `direction`).
- `crates/earshot-bench/`: replay benchmark scoring front-end variants by WER and CPU time.
- `coach/`: 14 §15 modules (`transcript_adapter`, `canonical_state` + `contracts` + `storage`, `context_builder`, `opportunity_classifier`, `calibration`, `routing_policy`, `template_renderer`, `question_generator`, `provider_adapter`, `candidate_validator`, `suggestion_controller`, `replay_evaluator`, `annotation_tool`, `telemetry`), `coach/ui/`, and `coach/cli.py` (`coach`).
- `policies/`: immutable versioned policy (`policies/policy-v0-a.yaml`) and grounded YAML templates (`policies/templates/*.yaml`).
- `arrays/`, `sim/`: array geometry descriptions and acoustic directivity simulations.
- `docs/plan.md`, `docs/design.md`: front-end architecture and Live Conversation Coach specification (v0.4).

## `coach` CLI Subcommands

Install/sync Python dependencies with `uv sync --frozen`, then invoke `uv run coach <subcommand>`:

- `coach serve --brief <brief.json> --policy <policy.yaml> --db <coach.db> [--source peecee|deepgram|replay] [--stream-file <path>] [--ws-url <url>] [--ui-port <port>] [--local-only]`: runs the 5-task `asyncio` runtime and private `127.0.0.1` token-gated Card UI.
- `coach replay --session <jsonl|db> --policy <policy.yaml> [--db <coach.db>] [--labels <labels.jsonl>] [--split-by conversation|series] [--eval-log <path>] [--sweep-t-display <csv>] [--json]`: deterministic policy-level replay and validity-interval opportunity matching.
- `coach diff --db <coach.db> --run-a <id> --run-b <id> [--labels <labels.jsonl>] [--bootstrap] [--json]`: compares gained/lost/changed cards across two runs, with optional paired conversation bootstrap.
- `coach report --db <coach.db> --session <id> [--run-id <id>] [--json]`: generates per-session report and checks all 6 §13.2 conservation invariants.
- `coach explain --db <coach.db> --session <id> --at-ms <int> [--window-ms <int>] [--json]`: queries anchors, scores, routing outcomes, and terminal reasons in a window.
- `coach annotate --session <jsonl|db> --out <labels.jsonl> [--pass 1|2] [--port <port>]` or `coach annotate --labels <first.jsonl> --kappa <retest.jsonl> [--json]`: 2-pass locked localhost annotation UI and blind test-retest Cohen's kappa.
- `coach bench --contexts <jsonl> --classifier <name> [--batch-sizes 1,4,8] [--tokens 128,256,512,1024,2048] [--json]`: latency and memory benchmark across real serialized contexts.
- `coach train --data <jsonl> --out <model_dir> [--backend embed_logreg|modernbert]`: trains `EmbedLogRegClassifier` or optional `ModernBERT`.
- `coach calibrate --data <labels.jsonl> --classifier <name> --mode <mode> --out <cal.json>`: fits per-head Platt calibration (requires `sample_source == "random"`).
- `coach delete-session --db <coach.db> --session <id>`: §14.2 session text deletion with WAL checkpoint + `VACUUM`, preserving nulled decision metadata.

### Environment Hooks

- `COACH_SEAT_AZIMUTHS`: JSON object mapping speaker IDs (`ME`, `S1`, ...) to seat azimuth degrees for `RecipientBinder`.
- `COACH_EARSHOT_DIRECTION_JSON`: path to `earshot direction` JSONL file or inline JSONL string.
- Provider API keys: passed via `--provider-api-key-env <ENV_VAR>` (never stored in policy/config files).

## Development, Lint & Tests

- `mise run lint`: runs Rust (`cargo fmt --check`, `cargo clippy`) and Python (`ruff check`, `ruff format --check`, `mypy --strict`) checks.
- `mise run test`: runs both `cargo test` (`mise run test:rust`) and `uv run --frozen pytest` (`mise run test:python`).

## Languages

Rust (`crates/`) for the real-time microphone-array DSP front end; Python 3.14 (`coach/`, `sim/`) for the conversation coach runtime, evaluation/training tooling, and array simulations.

## Licence

MIT. See [LICENSE](LICENSE).
