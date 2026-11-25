"""Tests for §15 upward-only module dependency invariants."""

from __future__ import annotations

import ast
from pathlib import Path

COACH_DIR = Path(__file__).resolve().parents[2] / "coach"

FORBIDDEN_BELOW_CANONICAL = frozenset(
    {
        "context_builder",
        "opportunity_classifier",
        "calibration",
        "routing_policy",
        "template_renderer",
        "question_generator",
        "provider_adapter",
        "candidate_validator",
        "suggestion_controller",
        "replay_evaluator",
        "annotation_tool",
        "telemetry",
        "cli",
        "transcript_adapter",
    }
)


def _coach_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "coach" or alias.name.startswith("coach."):
                    imported.add(alias.name.removeprefix("coach.").split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level > 0:
                if mod:
                    imported.add(mod.split(".")[0])
                else:
                    for alias in node.names:
                        imported.add(alias.name)
            elif mod == "coach":
                for alias in node.names:
                    imported.add(alias.name)
            elif mod.startswith("coach."):
                imported.add(mod.removeprefix("coach.").split(".")[0])
    return imported


def test_contracts_has_no_coach_imports() -> None:
    assert _coach_imports(COACH_DIR / "contracts.py") == set()


def test_canonical_state_only_imports_contracts() -> None:
    imports = _coach_imports(COACH_DIR / "canonical_state.py")
    assert imports <= {"contracts"}
    assert not (imports & FORBIDDEN_BELOW_CANONICAL)


def test_storage_only_imports_contracts_and_canonical_state() -> None:
    imports = _coach_imports(COACH_DIR / "storage.py")
    assert imports <= {"contracts", "canonical_state"}
    assert not (imports & FORBIDDEN_BELOW_CANONICAL)
