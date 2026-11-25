"""Opaque-box subprocess runner for coach and earshot CLIs."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


def default_repo_root() -> Path:
    """Return the workspace root containing tests/e2e."""
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class CliResult:
    """Captured CLI subprocess execution result."""

    returncode: int
    stdout: str
    stderr: str
    args: list[str]
    duration_s: float

    def json(self) -> Any:
        return json.loads(self.stdout)

    def jsonl(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in self.stdout.splitlines():
            stripped = line.strip()
            if stripped:
                parsed = json.loads(stripped)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected JSON object line, got {type(parsed).__name__}")
                rows.append(parsed)
        return rows


def resolve_coach_cmd(repo_root: Path | None = None) -> list[str]:
    """Locate the coach CLI executable or module entrypoint."""
    root = (repo_root or default_repo_root()).resolve()
    main_repo = Path("/home/halbritt/git/earshot").resolve()
    candidates: list[Path] = []
    uv_env = os.environ.get("UV_PROJECT_ENVIRONMENT", "").strip()
    if uv_env:
        candidates.append(Path(uv_env) / "bin" / "coach")
    candidates.append(root / ".venv" / "bin" / "coach")
    if root == main_repo:
        candidates.append(main_repo / ".venv" / "bin" / "coach")
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return [str(cand)]
    venv_py = root / ".venv" / "bin" / "python"
    if venv_py.is_file() and os.access(venv_py, os.X_OK):
        return [str(venv_py), "-m", "coach.cli"]
    main_venv_py = main_repo / ".venv" / "bin" / "python"
    if main_venv_py.is_file() and os.access(main_venv_py, os.X_OK):
        return [str(main_venv_py), "-m", "coach.cli"]
    which_coach = shutil.which("coach")
    if which_coach and root == main_repo:
        return [which_coach]
    return [sys.executable, "-m", "coach.cli"]


def resolve_earshot_cmd(repo_root: Path | None = None) -> list[str]:
    """Locate the earshot Rust CLI binary."""
    root = repo_root or default_repo_root()
    candidates: list[Path] = []
    target_dir = os.environ.get("CARGO_TARGET_DIR", "").strip()
    if target_dir:
        candidates.append(Path(target_dir) / "debug" / "earshot")
        candidates.append(Path(target_dir) / "release" / "earshot")
    for base in (root, Path("/home/halbritt/git/earshot")):
        candidates.append(base / "target" / "debug" / "earshot")
        candidates.append(base / "target" / "release" / "earshot")
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return [str(cand)]
    which_earshot = shutil.which("earshot")
    if which_earshot:
        return [which_earshot]
    raise FileNotFoundError("Could not locate earshot CLI binary in target/ or PATH")


def _merged_env(extra: Mapping[str, str] | None) -> dict[str, str]:
    merged = dict(os.environ)
    if extra:
        merged.update(extra)
    return merged


def run_cli(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    input_text: str | None = None,
    timeout: float = 30.0,
) -> CliResult:
    """Execute a CLI command synchronously and capture stdout/stderr/timing."""
    cmd_list = list(cmd)
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd_list,
        cwd=cwd,
        env=_merged_env(env),
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CliResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        args=cmd_list,
        duration_s=time.monotonic() - t0,
    )


def _coach_env(repo_root: Path | None, env: Mapping[str, str] | None) -> dict[str, str]:
    root = (repo_root or default_repo_root()).resolve()
    out = dict(env) if env else {}
    existing_pp = out.get("PYTHONPATH", os.environ.get("PYTHONPATH", "")).strip()
    root_str = str(root)
    if existing_pp:
        parts = existing_pp.split(os.pathsep)
        if root_str not in parts:
            out["PYTHONPATH"] = f"{root_str}{os.pathsep}{existing_pp}"
    else:
        out["PYTHONPATH"] = root_str
    return out


def run_coach(
    subcommand: str,
    args: Sequence[str] = (),
    *,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    input_text: str | None = None,
    timeout: float = 30.0,
) -> CliResult:
    """Run a coach CLI subcommand (serve, replay, diff, report, explain, bench, ...)."""
    root = (repo_root or default_repo_root()).resolve()
    cmd = [*resolve_coach_cmd(root), subcommand, *args]
    return run_cli(
        cmd,
        env=_coach_env(root, env),
        cwd=cwd or root,
        input_text=input_text,
        timeout=timeout,
    )


def run_earshot_beam(
    array_toml: Path,
    input_pcm: Path,
    output_wav: Path,
    *,
    steer: str = "0",
    elevation: float = 20.0,
    kind: str = "superdirective",
    white_noise_gain: float = 0.0,
    gain: float = 0.0,
    verbose: bool = False,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 30.0,
) -> CliResult:
    """Run `earshot beam` with exact crates/earshot-cli flags."""
    cmd = [
        *resolve_earshot_cmd(repo_root),
        "beam",
        "--array",
        str(array_toml),
        "--steer",
        steer,
        "--elevation",
        str(elevation),
        "--kind",
        kind,
        "--white-noise-gain",
        str(white_noise_gain),
        "--gain",
        str(gain),
    ]
    if verbose:
        cmd.append("--verbose")
    cmd.extend([str(input_pcm), str(output_wav)])
    return run_cli(cmd, env=env, cwd=repo_root, timeout=timeout)


def run_earshot_direction(
    array_toml: Path,
    input_pcm: Path,
    *,
    every: float = 0.5,
    elevation: float = 20.0,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 30.0,
) -> CliResult:
    """Run `earshot direction` returning JSONL lines on stdout."""
    cmd = [
        *resolve_earshot_cmd(repo_root),
        "direction",
        "--array",
        str(array_toml),
        "--every",
        str(every),
        "--elevation",
        str(elevation),
        str(input_pcm),
    ]
    return run_cli(cmd, env=env, cwd=repo_root, timeout=timeout)


class ManagedProcess:
    """Managed long-running subprocess with readiness polling and SIGTERM->SIGKILL cleanup."""

    def __init__(
        self,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        ready_url: str | None = None,
        ready_headers: Mapping[str, str] | None = None,
        ready_regex: str | None = None,
        startup_timeout: float = 10.0,
    ) -> None:
        self._cmd = list(cmd)
        self._env = _merged_env(env)
        self._cwd = cwd
        self._ready_url = ready_url
        self._ready_headers = dict(ready_headers or {})
        self._ready_pattern = re.compile(ready_regex) if ready_regex else None
        self._startup_timeout = startup_timeout
        self._proc: subprocess.Popen[str] | None = None
        self._stdout_lines: list[str] = []
        self._stderr_lines: list[str] = []
        self._lock = threading.Lock()
        self._readers: list[threading.Thread] = []

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def returncode(self) -> int | None:
        return self._proc.poll() if self._proc is not None else None

    @property
    def stdout_text(self) -> str:
        with self._lock:
            return "".join(self._stdout_lines)

    @property
    def stderr_text(self) -> str:
        with self._lock:
            return "".join(self._stderr_lines)

    def _pump(self, stream: Any, target: list[str]) -> None:
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                with self._lock:
                    target.append(line)
        finally:
            stream.close()

    def __enter__(self) -> ManagedProcess:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            self._cmd,
            cwd=self._cwd,
            env=self._env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        t_out = threading.Thread(
            target=self._pump, args=(self._proc.stdout, self._stdout_lines), daemon=True
        )
        t_err = threading.Thread(
            target=self._pump, args=(self._proc.stderr, self._stderr_lines), daemon=True
        )
        t_out.start()
        t_err.start()
        self._readers = [t_out, t_err]
        self._wait_ready()

    def _wait_ready(self) -> None:
        if self._ready_url is None and self._ready_pattern is None:
            return
        assert self._proc is not None
        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            rc = self._proc.poll()
            if rc is not None:
                for t in self._readers:
                    t.join(timeout=1.0)
                raise RuntimeError(
                    f"Process exited prematurely with code {rc}.\n"
                    f"stdout:\n{self.stdout_text}\nstderr:\n{self.stderr_text}"
                )
            if self._ready_pattern is not None:
                combined = self.stdout_text + "\n" + self.stderr_text
                if self._ready_pattern.search(combined):
                    return
            if self._ready_url is not None:
                try:
                    resp = httpx.get(
                        self._ready_url,
                        headers=self._ready_headers,
                        timeout=0.5,
                    )
                    if resp.status_code < 500:
                        return
                except httpx.HTTPError:
                    pass
            time.sleep(0.05)
        self.stop()
        raise TimeoutError(
            f"Process did not become ready within {self._startup_timeout}s.\n"
            f"stdout:\n{self.stdout_text}\nstderr:\n{self.stderr_text}"
        )

    def stop(self, timeout_s: float = 5.0) -> CliResult:
        if self._proc is None:
            return CliResult(0, self.stdout_text, self.stderr_text, self._cmd, 0.0)
        proc = self._proc
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=2.0)
        for t in self._readers:
            t.join(timeout=1.0)
        rc = proc.returncode if proc.returncode is not None else -1
        self._proc = None
        return CliResult(rc, self.stdout_text, self.stderr_text, self._cmd, 0.0)

    def wait_for_output(self, pattern: str, timeout: float = 5.0) -> re.Match[str]:
        """Wait until combined stdout+stderr matches regex pattern or fail on exit/timeout."""
        compiled = re.compile(pattern)
        combined = self.stdout_text + "\n" + self.stderr_text
        match = compiled.search(combined)
        if match is not None:
            return match
        if self._proc is None:
            raise RuntimeError(
                f"Process is not running and output did not match {pattern!r}.\n"
                f"stdout:\n{self.stdout_text}\nstderr:\n{self.stderr_text}"
            )
        deadline = time.monotonic() + timeout
        while True:
            rc = self._proc.poll()
            if rc is not None:
                for t in self._readers:
                    t.join(timeout=1.0)
                combined = self.stdout_text + "\n" + self.stderr_text
                match = compiled.search(combined)
                if match is not None:
                    return match
                raise RuntimeError(
                    f"Process exited with code {rc} before matching {pattern!r}.\n"
                    f"stdout:\n{self.stdout_text}\nstderr:\n{self.stderr_text}"
                )
            combined = self.stdout_text + "\n" + self.stderr_text
            match = compiled.search(combined)
            if match is not None:
                return match
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeout}s waiting for {pattern!r}.\n"
                    f"stdout:\n{self.stdout_text}\nstderr:\n{self.stderr_text}"
                )
            time.sleep(0.02)


def start_coach_process(
    subcommand: str,
    args: Sequence[str] = (),
    *,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    ready_url: str | None = None,
    ready_headers: Mapping[str, str] | None = None,
    ready_regex: str | None = None,
    startup_timeout: float = 10.0,
) -> ManagedProcess:
    """Create a ManagedProcess for a long-running coach subcommand (serve/annotate)."""
    root = (repo_root or default_repo_root()).resolve()
    cmd = [*resolve_coach_cmd(root), subcommand, *args]
    return ManagedProcess(
        cmd,
        env=_coach_env(root, env),
        cwd=cwd or root,
        ready_url=ready_url,
        ready_headers=ready_headers,
        ready_regex=ready_regex,
        startup_timeout=startup_timeout,
    )


def extract_ui_url_and_token(
    proc: ManagedProcess,
    default_port: int,
    timeout: float = 5.0,
) -> tuple[str, str]:
    """Wait for and extract the localhost UI base URL and session token from process output."""
    token_match = proc.wait_for_output(
        r"(?:token=|[\"']token[\"']\s*:\s*[\"'])([A-Za-z0-9._-]+)",
        timeout=timeout,
    )
    token = token_match.group(1)
    combined = proc.stdout_text + "\n" + proc.stderr_text
    url_match = re.search(r"http://127\.0\.0\.1:(\d+)", combined)
    port = int(url_match.group(1)) if url_match is not None else default_port
    return (f"http://127.0.0.1:{port}", token)


def build_coach_test_env(
    *,
    openai_base_url: str | None = None,
    nemo_ws_url: str | None = None,
    peecee_bin: Path | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build environment overrides wiring coach CLI subprocesses to local test mocks."""
    env: dict[str, str] = {}
    if openai_base_url is not None:
        env["OPENAI_BASE_URL"] = openai_base_url
        env["COACH_OPENAI_BASE_URL"] = openai_base_url
        env["OPENAI_API_KEY"] = "test-e2e-key"
    if nemo_ws_url is not None:
        env["PEECEE_WS_URL"] = nemo_ws_url
        env["NEMO_WS_URL"] = nemo_ws_url
        env["COACH_PEECEE_WS_URL"] = nemo_ws_url
        env["COACH_NEMO_WS_URL"] = nemo_ws_url
    if peecee_bin is not None:
        peecee_dir = str(peecee_bin.parent)
        existing_path = os.environ.get("PATH", "")
        env["PATH"] = f"{peecee_dir}{os.pathsep}{existing_path}" if existing_path else peecee_dir
        env["PEECEE_SPEECH_BIN"] = str(peecee_bin)
        env["COACH_PEECEE_BIN"] = str(peecee_bin)
    if extra_env:
        env.update(extra_env)
    return env
