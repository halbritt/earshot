"""Per-head and per-category Platt sigmoid calibration and diagnostics (§5.7, §12.4, §12.5, §15)."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from coach.context_builder import BUILDER_VERSION, PREFILTER_VERSION, ContextBuildResult
from coach.contracts import (
    PREDICTABLE_CATEGORIES,
    SCHEMA_VERSION,
    Category,
    ClassifierResult,
    DecisionContext,
    LabelRecord,
    SampleSource,
    ScoreSet,
    SessionMode,
)
from coach.opportunity_classifier import ClassifierErrorResult, ClassifierInput

InputSpace = Literal["logit", "probability"]


class CalibrationDataError(ValueError):
    """Raised when calibration fit/eval receives empty data or non-random sample_source (§12.4)."""


def _clip_prob(p: float, eps: float = 1e-6) -> float:
    return min(1.0 - eps, max(eps, float(p)))


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _to_logit_input(raw_score: float, input_space: InputSpace) -> float:
    val = float(raw_score)
    if not math.isfinite(val):
        raise ValueError(f"Non-finite raw score: {raw_score!r}")
    if input_space == "logit":
        return val
    if input_space == "probability":
        p = _clip_prob(val)
        return math.log(p / (1.0 - p))
    raise ValueError(f"Unsupported input_space: {input_space!r}")


def _extract_context(
    item: DecisionContext | ClassifierInput | ContextBuildResult,
) -> DecisionContext:
    if isinstance(item, DecisionContext):
        return item
    if isinstance(item, ClassifierInput):
        return item.context
    if isinstance(item, ContextBuildResult):
        if item.context is None:
            raise ValueError("Dropped ContextBuildResult cannot be used in calibration")
        return item.context
    raise TypeError(
        f"Expected DecisionContext | ClassifierInput | ContextBuildResult, got {type(item).__name__}"
    )


def _require_random_labels(labels: Sequence[LabelRecord]) -> None:
    if not labels:
        raise CalibrationDataError("Calibration requires at least one LabelRecord")
    bad = [
        (lbl.label_id, str(lbl.sample_source))
        for lbl in labels
        if lbl.sample_source is not SampleSource.RANDOM
    ]
    if bad:
        raise CalibrationDataError(
            f"Calibration fit/evaluation refuses non-random sample_source rows (§12.4): {bad[:5]}"
        )


@dataclass(frozen=True, slots=True)
class PlattHeadParams:
    """Sigmoid (Platt) scaling parameters P(y=1 | s) = sigmoid(a * x(s) + b)."""

    a: float
    b: float
    input_space: InputSpace
    n_samples: int
    n_positives: int
    degenerate_prior: bool = False

    def predict(self, raw_score: float) -> float:
        x = _to_logit_input(raw_score, self.input_space)
        return _clip_prob(_sigmoid(self.a * x + self.b), eps=1e-7)

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": float(self.a),
            "b": float(self.b),
            "input_space": self.input_space,
            "n_samples": int(self.n_samples),
            "n_positives": int(self.n_positives),
            "degenerate_prior": bool(self.degenerate_prior),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlattHeadParams:
        space = str(data["input_space"])
        if space not in {"logit", "probability"}:
            raise ValueError(f"Invalid input_space: {space!r}")
        return cls(
            a=float(data["a"]),
            b=float(data["b"]),
            input_space=space,  # type: ignore[arg-type]
            n_samples=int(data["n_samples"]),
            n_positives=int(data["n_positives"]),
            degenerate_prior=bool(data.get("degenerate_prior", False)),
        )


@dataclass(frozen=True, slots=True)
class CalibrationModel:
    """Version-bound Platt calibration model (§5.7, §16)."""

    calibration_version: str
    classifier_version: str
    builder_version: str
    prefilter_version: str
    mode: SessionMode
    heads: dict[str, PlattHeadParams]

    def matches(
        self,
        context: DecisionContext,
        result: ClassifierResult,
        mode: SessionMode,
    ) -> bool:
        return (
            result.classifier_version == self.classifier_version
            and context.builder_version == self.builder_version
            and context.prefilter_version == self.prefilter_version
            and mode is self.mode
        )

    def calibrate(
        self,
        context: DecisionContext,
        result: ClassifierResult,
        *,
        mode: SessionMode,
    ) -> ClassifierResult:
        if isinstance(result, ClassifierErrorResult):
            return result
        if not self.matches(context, result, mode):
            return ClassifierResult(
                schema_version=result.schema_version,
                candidate_id=result.candidate_id,
                based_on_state_version=result.based_on_state_version,
                classifier_version=result.classifier_version,
                raw_scores=result.raw_scores,
                calibration_version=None,
                calibrated_scores=None,
                latency_ms=result.latency_ms,
                input_refs=result.input_refs,
            )

        opp_head = self.heads.get("opportunity")
        if opp_head is not None:
            cal_opp = opp_head.predict(result.raw_scores.opportunity)
        else:
            cal_opp = _clip_prob(result.raw_scores.opportunity)

        cal_cats: dict[str, float] = {}
        for cat_name, raw_val in result.raw_scores.categories.items():
            if cat_name == Category.OTHER.value:
                continue
            head = self.heads.get(cat_name)
            if head is not None:
                cal_cats[cat_name] = head.predict(raw_val)

        return ClassifierResult(
            schema_version=result.schema_version,
            candidate_id=result.candidate_id,
            based_on_state_version=result.based_on_state_version,
            classifier_version=result.classifier_version,
            raw_scores=result.raw_scores,
            calibration_version=self.calibration_version,
            calibrated_scores=ScoreSet(opportunity=cal_opp, categories=cal_cats),
            latency_ms=result.latency_ms,
            input_refs=result.input_refs,
        )

    def to_metadata(self) -> dict[str, str]:
        return {
            "calibration_version": self.calibration_version,
            "classifier_version": self.classifier_version,
            "builder_version": self.builder_version,
            "prefilter_version": self.prefilter_version,
            "mode": str(self.mode),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "calibration_version": self.calibration_version,
            "classifier_version": self.classifier_version,
            "builder_version": self.builder_version,
            "prefilter_version": self.prefilter_version,
            "mode": str(self.mode),
            "heads": {k: v.to_dict() for k, v in self.heads.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalibrationModel:
        heads_raw = data["heads"]
        if not isinstance(heads_raw, Mapping):
            raise ValueError("CalibrationModel.heads must be a mapping")
        return cls(
            calibration_version=str(data["calibration_version"]),
            classifier_version=str(data["classifier_version"]),
            builder_version=str(data["builder_version"]),
            prefilter_version=str(data["prefilter_version"]),
            mode=SessionMode(str(data["mode"])),
            heads={str(k): PlattHeadParams.from_dict(v) for k, v in heads_raw.items()},
        )

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> CalibrationModel:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)


class CalibrationRegistry:
    """Registry of version-bound CalibrationModel instances."""

    def __init__(self, models: Sequence[CalibrationModel] = ()) -> None:
        self._by_version: dict[str, CalibrationModel] = {}
        for m in models:
            self.register(m)

    def register(self, model: CalibrationModel) -> None:
        self._by_version[model.calibration_version] = model

    def get(self, calibration_version: str) -> CalibrationModel | None:
        return self._by_version.get(calibration_version)

    def find_match(
        self,
        context: DecisionContext,
        result: ClassifierResult,
        mode: SessionMode,
    ) -> CalibrationModel | None:
        for model in self._by_version.values():
            if model.matches(context, result, mode):
                return model
        return None

    def calibrate(
        self,
        context: DecisionContext,
        result: ClassifierResult,
        *,
        mode: SessionMode,
        calibration_version: str | None = None,
    ) -> ClassifierResult:
        if isinstance(result, ClassifierErrorResult):
            return result
        model = (
            self.get(calibration_version)
            if calibration_version is not None
            else self.find_match(context, result, mode)
        )
        if model is None:
            return ClassifierResult(
                schema_version=result.schema_version,
                candidate_id=result.candidate_id,
                based_on_state_version=result.based_on_state_version,
                classifier_version=result.classifier_version,
                raw_scores=result.raw_scores,
                calibration_version=None,
                calibrated_scores=None,
                latency_ms=result.latency_ms,
                input_refs=result.input_refs,
            )
        return model.calibrate(context, result, mode=mode)

    def to_metadata(self) -> dict[str, dict[str, str]]:
        return {ver: model.to_metadata() for ver, model in self._by_version.items()}


def _fit_single_platt_head(
    raw_scores: Sequence[float],
    labels: Sequence[int],
    *,
    input_space: InputSpace,
    c_reg: float = 1e6,
) -> PlattHeadParams:
    n_samples = len(raw_scores)
    n_pos = sum(labels)
    if n_samples == 0:
        return PlattHeadParams(
            a=1.0,
            b=-4.0,
            input_space=input_space,
            n_samples=0,
            n_positives=0,
            degenerate_prior=True,
        )
    y_arr = np.asarray(labels, dtype=np.int64)
    if np.unique(y_arr).size < 2:
        prior = _clip_prob(float(n_pos) / float(n_samples), eps=1e-4)
        return PlattHeadParams(
            a=0.0,
            b=float(math.log(prior / (1.0 - prior))),
            input_space=input_space,
            n_samples=n_samples,
            n_positives=n_pos,
            degenerate_prior=True,
        )
    x_arr = np.asarray(
        [_to_logit_input(s, input_space) for s in raw_scores], dtype=np.float64
    ).reshape(-1, 1)
    from sklearn.linear_model import LogisticRegression

    # §5.5 / §12.5: natural prevalence only — never apply class_weight in calibration.
    lr = LogisticRegression(C=float(c_reg), class_weight=None, solver="lbfgs", max_iter=1000)
    lr.fit(x_arr, y_arr)
    return PlattHeadParams(
        a=float(lr.coef_[0][0]),
        b=float(lr.intercept_[0]),
        input_space=input_space,
        n_samples=n_samples,
        n_positives=n_pos,
        degenerate_prior=False,
    )


def fit_platt_calibrator(
    rows: Sequence[
        tuple[DecisionContext | ClassifierInput | ContextBuildResult, ClassifierResult, LabelRecord]
    ],
    *,
    calibration_version: str,
    classifier_version: str,
    builder_version: str = BUILDER_VERSION,
    prefilter_version: str = PREFILTER_VERSION,
    mode: SessionMode,
    input_space: InputSpace = "logit",
    active_categories: Sequence[Category] | None = None,
    c_reg: float = 1e6,
) -> CalibrationModel:
    """Fit per-head and per-category Platt sigmoid calibration on random-sample rows (§5.7, §12.4)."""
    if not rows:
        raise CalibrationDataError("Calibration fitting requires non-empty rows")
    labels_only = [lbl for _, _, lbl in rows]
    _require_random_labels(labels_only)

    opp_raw: list[float] = []
    opp_y: list[int] = []
    seen_cats: set[Category] = set(active_categories or ())

    for item, res, _lbl in rows:
        ctx = _extract_context(item)
        if isinstance(res, ClassifierErrorResult):
            raise CalibrationDataError(
                f"Cannot fit calibration on ClassifierErrorResult ({res.candidate_id})"
            )
        if res.classifier_version != classifier_version:
            raise CalibrationDataError(
                f"Row classifier_version {res.classifier_version!r} != {classifier_version!r}"
            )
        if ctx.builder_version != builder_version:
            raise CalibrationDataError(
                f"Row builder_version {ctx.builder_version!r} != {builder_version!r}"
            )
        if ctx.prefilter_version != prefilter_version:
            raise CalibrationDataError(
                f"Row prefilter_version {ctx.prefilter_version!r} != {prefilter_version!r}"
            )
        if active_categories is None:
            for k in res.raw_scores.categories:
                if k != Category.OTHER.value:
                    seen_cats.add(Category(k))

    ordered_cats = sorted(
        (c for c in seen_cats if c in PREDICTABLE_CATEGORIES), key=lambda c: c.value
    )
    cat_raw: dict[str, list[float]] = {c.value: [] for c in ordered_cats}
    cat_y: dict[str, list[int]] = {c.value: [] for c in ordered_cats}

    for _, res, lbl in rows:
        opp_raw.append(float(res.raw_scores.opportunity))
        opp_y.append(1 if lbl.worthwhile else 0)
        if lbl.worthwhile and not lbl.categories:
            continue
        for cat in ordered_cats:
            if cat.value in res.raw_scores.categories:
                cat_raw[cat.value].append(float(res.raw_scores.categories[cat.value]))
                cat_y[cat.value].append(1 if (lbl.worthwhile and cat in lbl.categories) else 0)

    heads: dict[str, PlattHeadParams] = {
        "opportunity": _fit_single_platt_head(opp_raw, opp_y, input_space=input_space, c_reg=c_reg)
    }
    for cat in ordered_cats:
        if cat_raw[cat.value]:
            heads[cat.value] = _fit_single_platt_head(
                cat_raw[cat.value],
                cat_y[cat.value],
                input_space=input_space,
                c_reg=c_reg,
            )

    return CalibrationModel(
        calibration_version=calibration_version,
        classifier_version=classifier_version,
        builder_version=builder_version,
        prefilter_version=prefilter_version,
        mode=mode,
        heads=heads,
    )


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """Equal-mass reliability bin (§5.7)."""

    bin_index: int
    count: int
    mean_pred: float
    empirical_rate: float
    min_pred: float
    max_pred: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bin_index": self.bin_index,
            "count": self.count,
            "mean_pred": self.mean_pred,
            "empirical_rate": self.empirical_rate,
            "min_pred": self.min_pred,
            "max_pred": self.max_pred,
        }


@dataclass(frozen=True, slots=True)
class HeadCalibrationMetrics:
    """Calibration diagnostics for a single head (opportunity or category)."""

    head: str
    n_samples: int
    n_positives: int
    prevalence: float
    brier_score: float
    brier_ci_95: tuple[float, float] | None
    bins: tuple[ReliabilityBin, ...]
    ece: float | None
    low_n_caveat: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": self.head,
            "n_samples": self.n_samples,
            "n_positives": self.n_positives,
            "prevalence": self.prevalence,
            "brier_score": self.brier_score,
            "brier_ci_95": list(self.brier_ci_95) if self.brier_ci_95 is not None else None,
            "bins": [b.to_dict() for b in self.bins],
            "ece": self.ece,
            "low_n_caveat": self.low_n_caveat,
        }


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Full reliability & Brier report across heads (§5.7, §12.7)."""

    calibration_version: str | None
    n_samples: int
    n_conversations: int
    heads: dict[str, HeadCalibrationMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_version": self.calibration_version,
            "n_samples": self.n_samples,
            "n_conversations": self.n_conversations,
            "heads": {k: v.to_dict() for k, v in self.heads.items()},
        }


def _bootstrap_brier_ci(
    by_session: Mapping[str, list[tuple[float, int]]],
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float] | None:
    session_ids = sorted(by_session)
    if len(session_ids) < 2 or n_resamples < 1:
        return None
    rng = random.Random(seed)
    session_briers: dict[str, tuple[float, int]] = {}
    for sid in session_ids:
        pairs = by_session[sid]
        sq_sum = sum((p - float(y)) ** 2 for p, y in pairs)
        session_briers[sid] = (sq_sum, len(pairs))

    boot_vals: list[float] = []
    n_conv = len(session_ids)
    for _ in range(n_resamples):
        total_sq = 0.0
        total_cnt = 0
        for _ in range(n_conv):
            chosen = session_ids[rng.randrange(n_conv)]
            sq_sum, cnt = session_briers[chosen]
            total_sq += sq_sum
            total_cnt += cnt
        if total_cnt > 0:
            boot_vals.append(total_sq / float(total_cnt))
    if not boot_vals:
        return None
    arr = np.asarray(boot_vals, dtype=np.float64)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return (float(lo), float(hi))


def _evaluate_single_head(
    head: str,
    preds: Sequence[float],
    ys: Sequence[int],
    session_ids: Sequence[str],
    *,
    n_bins: int,
    min_positives_for_ece: int,
    n_bootstrap: int,
    seed: int,
) -> HeadCalibrationMetrics:
    n_samples = len(preds)
    n_pos = sum(ys)
    prevalence = float(n_pos) / float(n_samples)
    p_arr = np.asarray(preds, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    brier = float(np.mean((p_arr - y_arr) ** 2))

    by_session: dict[str, list[tuple[float, int]]] = {}
    for sid, p, y in zip(session_ids, preds, ys, strict=True):
        by_session.setdefault(sid, []).append((p, y))
    brier_ci = _bootstrap_brier_ci(by_session, n_resamples=n_bootstrap, seed=seed)

    eff_bins = max(1, min(int(n_bins), n_samples))
    order = np.argsort(p_arr, kind="mergesort")
    chunks = np.array_split(order, eff_bins)
    bins_out: list[ReliabilityBin] = []
    ece_acc = 0.0
    for idx, ch in enumerate(chunks):
        if ch.size == 0:
            continue
        p_sub = p_arr[ch]
        y_sub = y_arr[ch]
        mean_p = float(np.mean(p_sub))
        emp = float(np.mean(y_sub))
        cnt = int(ch.size)
        ece_acc += (float(cnt) / float(n_samples)) * abs(mean_p - emp)
        bins_out.append(
            ReliabilityBin(
                bin_index=idx,
                count=cnt,
                mean_pred=mean_p,
                empirical_rate=emp,
                min_pred=float(np.min(p_sub)),
                max_pred=float(np.max(p_sub)),
            )
        )

    if n_pos < min_positives_for_ece:
        ece_val: float | None = None
        caveat: str | None = (
            f"Only {n_pos} positive examples (< {min_positives_for_ece}); "
            "ECE suppressed as uninformative (§5.7)."
        )
    else:
        ece_val = float(ece_acc)
        caveat = None

    return HeadCalibrationMetrics(
        head=head,
        n_samples=n_samples,
        n_positives=n_pos,
        prevalence=prevalence,
        brier_score=brier,
        brier_ci_95=brier_ci,
        bins=tuple(bins_out),
        ece=ece_val,
        low_n_caveat=caveat,
    )


def evaluate_calibration(
    rows: Sequence[tuple[ClassifierResult, LabelRecord]],
    *,
    n_bins: int = 10,
    min_positives_for_ece: int = 20,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> CalibrationReport:
    """Compute Brier score, equal-mass reliability bins, and conv-level bootstrap CIs (§5.7, §12.4)."""
    if not rows:
        raise CalibrationDataError("Calibration evaluation requires non-empty rows")
    _require_random_labels([lbl for _, lbl in rows])

    opp_p: list[float] = []
    opp_y: list[int] = []
    opp_sids: list[str] = []
    cat_p: dict[str, list[float]] = {}
    cat_y: dict[str, list[int]] = {}
    cat_sids: dict[str, list[str]] = {}
    cal_versions: set[str] = set()

    for res, lbl in rows:
        if isinstance(res, ClassifierErrorResult) or res.calibrated_scores is None:
            raise CalibrationDataError(
                f"Cannot evaluate calibration on uncalibrated/error result ({res.candidate_id})"
            )
        if res.calibration_version is not None:
            cal_versions.add(res.calibration_version)
        opp_p.append(float(res.calibrated_scores.opportunity))
        opp_y.append(1 if lbl.worthwhile else 0)
        opp_sids.append(lbl.session_id)

        if lbl.worthwhile and not lbl.categories:
            continue
        for cat_name, score in res.calibrated_scores.categories.items():
            if cat_name == Category.OTHER.value:
                continue
            cat_enum = Category(cat_name)
            cat_p.setdefault(cat_name, []).append(float(score))
            cat_y.setdefault(cat_name, []).append(
                1 if (lbl.worthwhile and cat_enum in lbl.categories) else 0
            )
            cat_sids.setdefault(cat_name, []).append(lbl.session_id)

    heads: dict[str, HeadCalibrationMetrics] = {
        "opportunity": _evaluate_single_head(
            "opportunity",
            opp_p,
            opp_y,
            opp_sids,
            n_bins=n_bins,
            min_positives_for_ece=min_positives_for_ece,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    }
    for idx, cat_name in enumerate(sorted(cat_p), start=1):
        heads[cat_name] = _evaluate_single_head(
            cat_name,
            cat_p[cat_name],
            cat_y[cat_name],
            cat_sids[cat_name],
            n_bins=n_bins,
            min_positives_for_ece=min_positives_for_ece,
            n_bootstrap=n_bootstrap,
            seed=seed + idx,
        )

    cal_ver = next(iter(cal_versions)) if len(cal_versions) == 1 else None
    return CalibrationReport(
        calibration_version=cal_ver,
        n_samples=len(rows),
        n_conversations=len(set(opp_sids)),
        heads=heads,
    )


def _builtin_heads_v0() -> dict[str, PlattHeadParams]:
    # Monotonic bootstrap identity calibration in probability space (not fitted on human labels).
    heads: dict[str, PlattHeadParams] = {
        "opportunity": PlattHeadParams(
            a=1.0,
            b=0.0,
            input_space="probability",
            n_samples=0,
            n_positives=0,
            degenerate_prior=False,
        )
    }
    for cat in sorted(PREDICTABLE_CATEGORIES, key=lambda c: c.value):
        heads[cat.value] = PlattHeadParams(
            a=1.0,
            b=0.0,
            input_space="probability",
            n_samples=0,
            n_positives=0,
            degenerate_prior=False,
        )
    return heads


BUILTIN_CAL_V0_TECHNICAL = CalibrationModel(
    calibration_version="cal-v0-technical",
    classifier_version="opportunity-baseline-v0",
    builder_version=BUILDER_VERSION,
    prefilter_version=PREFILTER_VERSION,
    mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
    heads=_builtin_heads_v0(),
)

BUILTIN_CAL_V0_RULES = CalibrationModel(
    calibration_version="cal-v0-technical-rules",
    classifier_version="rules_v0",
    builder_version=BUILDER_VERSION,
    prefilter_version=PREFILTER_VERSION,
    mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
    heads=_builtin_heads_v0(),
)

BUILTIN_CAL_V0_RULES_PLATT = CalibrationModel(
    calibration_version="rules_v0-platt-v0-a",
    classifier_version="rules_v0",
    builder_version=BUILDER_VERSION,
    prefilter_version=PREFILTER_VERSION,
    mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
    heads=_builtin_heads_v0(),
)

BUILTIN_CAL_V0_READOUT_PLATT = CalibrationModel(
    calibration_version="readout_v0-platt-v0-a",
    classifier_version="readout_v0",
    builder_version=BUILDER_VERSION,
    prefilter_version=PREFILTER_VERSION,
    mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
    heads=_builtin_heads_v0(),
)

BUILTIN_CAL_V0_LLM_PLATT_V0A = CalibrationModel(
    calibration_version="llm_baseline_v0-platt-v0-a",
    classifier_version="llm_baseline_v0",
    builder_version=BUILDER_VERSION,
    prefilter_version=PREFILTER_VERSION,
    mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
    heads=_builtin_heads_v0(),
)

BUILTIN_CAL_V0_LLM_PLATT = CalibrationModel(
    calibration_version="llm_baseline_v0-platt",
    classifier_version="llm_baseline_v0",
    builder_version=BUILDER_VERSION,
    prefilter_version=PREFILTER_VERSION,
    mode=SessionMode.TECHNICAL_WORKING_DISCUSSION,
    heads=_builtin_heads_v0(),
)


def default_calibration_registry() -> CalibrationRegistry:
    """Return built-in CalibrationRegistry matching shipped policy-v0-a.yaml."""
    return CalibrationRegistry(
        [
            BUILTIN_CAL_V0_TECHNICAL,
            BUILTIN_CAL_V0_RULES,
            BUILTIN_CAL_V0_RULES_PLATT,
            BUILTIN_CAL_V0_READOUT_PLATT,
            BUILTIN_CAL_V0_LLM_PLATT_V0A,
            BUILTIN_CAL_V0_LLM_PLATT,
        ]
    )


def default_calibration_metadata() -> dict[str, dict[str, str]]:
    """Return calibration metadata dict compatible with context_builder.load_policy()."""
    return default_calibration_registry().to_metadata()
