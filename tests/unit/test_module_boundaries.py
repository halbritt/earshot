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


def test_m4_module_boundaries_upward_only() -> None:
    opp_imports = _coach_imports(COACH_DIR / "opportunity_classifier.py")
    assert opp_imports <= {"contracts", "canonical_state", "storage", "context_builder"}
    assert "provider_adapter" not in opp_imports

    cal_imports = _coach_imports(COACH_DIR / "calibration.py")
    assert cal_imports <= {
        "contracts",
        "canonical_state",
        "storage",
        "context_builder",
        "opportunity_classifier",
    }

    rp_imports = _coach_imports(COACH_DIR / "routing_policy.py")
    assert rp_imports <= {
        "contracts",
        "canonical_state",
        "storage",
        "context_builder",
        "opportunity_classifier",
        "calibration",
    }
    assert "candidate_validator" not in rp_imports

    re_imports = _coach_imports(COACH_DIR / "replay_evaluator.py")
    assert re_imports <= {
        "transcript_adapter",
        "contracts",
        "canonical_state",
        "storage",
        "context_builder",
        "opportunity_classifier",
        "calibration",
        "routing_policy",
        "template_renderer",
        "question_generator",
        "provider_adapter",
        "candidate_validator",
    }
    assert not (re_imports & {"suggestion_controller", "annotation_tool", "telemetry", "cli"})


def test_m5_module_boundaries_upward_only() -> None:
    sc_imports = _coach_imports(COACH_DIR / "suggestion_controller.py")
    assert sc_imports <= {
        "transcript_adapter",
        "contracts",
        "canonical_state",
        "storage",
        "context_builder",
        "opportunity_classifier",
        "calibration",
        "routing_policy",
        "template_renderer",
        "question_generator",
        "provider_adapter",
        "candidate_validator",
        "ui",
    }
    assert not (sc_imports & {"replay_evaluator", "annotation_tool", "telemetry", "cli"})

    for ui_file in sorted((COACH_DIR / "ui").glob("*.py")):
        ui_imports = _coach_imports(ui_file) - {"ui"}
        assert ui_imports <= {"contracts", "canonical_state"}
        assert not (
            ui_imports
            & {"suggestion_controller", "replay_evaluator", "annotation_tool", "telemetry", "cli"}
        )

    ann_imports = _coach_imports(COACH_DIR / "annotation_tool.py")
    assert ann_imports <= {
        "transcript_adapter",
        "contracts",
        "canonical_state",
        "storage",
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
    }
    assert not (ann_imports & {"telemetry", "cli"})

    tel_imports = _coach_imports(COACH_DIR / "telemetry.py")
    assert "cli" not in tel_imports
