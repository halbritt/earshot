"""Shared pytest fixtures for opaque-box Live Conversation Coach E2E tests."""

from __future__ import annotations

import socket
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from .helpers.cli_runner import default_repo_root
from .helpers.fixtures import (
    build_session_brief,
    write_brief_json,
    write_policy_yaml,
    write_templates_dir,
)
from .helpers.mock_nemo_ws import MockNemoServer, write_mock_peecee_speech
from .helpers.mock_openai import MockOpenAIServer


def pytest_configure() -> None:
    """Ensure repository root is on sys.path for bare `pytest` invocations."""
    root_str = str(Path(__file__).resolve().parents[2])
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


@pytest.fixture
def free_port_factory() -> Callable[[], int]:
    """Allocate an ephemeral unused TCP port on 127.0.0.1."""

    def _alloc() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    return _alloc


@pytest.fixture
def repo_root() -> Path:
    """Return the repository/worktree root directory."""
    return default_repo_root()


@pytest.fixture
def e2e_workspace(tmp_path: Path) -> dict[str, Path]:
    """Isolated per-test filesystem layout under pytest tmp_path."""
    sessions_dir = tmp_path / "sessions"
    policies_dir = tmp_path / "policies"
    templates_dir = policies_dir / "templates"
    logs_dir = tmp_path / "logs"
    for d in (sessions_dir, policies_dir, templates_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    write_templates_dir(templates_dir)
    policy_path = write_policy_yaml(policies_dir / "policy-v0-a.yaml")
    brief_path = write_brief_json(tmp_path / "brief.json", build_session_brief())
    return {
        "root": tmp_path,
        "db_path": tmp_path / "coach.db",
        "sessions_dir": sessions_dir,
        "policies_dir": policies_dir,
        "templates_dir": templates_dir,
        "logs_dir": logs_dir,
        "policy_path": policy_path,
        "brief_path": brief_path,
    }


@pytest.fixture
def mock_nemo_ws() -> Iterator[MockNemoServer]:
    """Start and stop a local NeMo-Speech.cpp (20999cc) WebSocket mock server."""
    with MockNemoServer() as server:
        yield server


@pytest.fixture
def mock_peecee_bin(tmp_path: Path) -> Path:
    """Create an executable mock peecee-speech script in tmp_path/bin."""
    return write_mock_peecee_speech(tmp_path / "bin")


@pytest.fixture
def mock_openai() -> Iterator[MockOpenAIServer]:
    """Start and stop a local OpenAI-compatible HTTP mock server."""
    with MockOpenAIServer() as server:
        yield server


@pytest.fixture
def sample_brief(e2e_workspace: dict[str, Path]) -> dict[str, Any]:
    """Return a default valid schema-v1 SessionBrief dict."""
    return build_session_brief()


@pytest.fixture
def sample_policy(e2e_workspace: dict[str, Path]) -> Path:
    """Return path to default policy-v0-a.yaml inside isolated workspace."""
    return e2e_workspace["policy_path"]
