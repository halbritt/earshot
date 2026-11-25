"""Template loading, grounded slot rendering, and generic fallback (§9.1, §15 level 7)."""

from __future__ import annotations

import hashlib
import string
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from coach.canonical_state import SegmentRecord, StateSnapshot
from coach.context_builder import (
    SPAN_EXTRACTOR_VERSION,
    PolicyConfig,
    check_generic_referent,
    extract_slots_span_rules_v0,
    verify_grounded_slot,
)
from coach.contracts import (
    PREDICTABLE_CATEGORIES,
    Category,
    ContractValidationError,
    DecisionContext,
    GroundedSlot,
    Origin,
    ReasonCode,
    Route,
    SuggestionCandidate,
)

DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "policies" / "templates"


class TemplateConfigError(ContractValidationError):
    """Raised when a template YAML file violates schema constraints."""


@dataclass(frozen=True, slots=True)
class SlotSpec:
    extractor: str
    max_words: int
    must_be_substring: bool = True


@dataclass(frozen=True, slots=True)
class GroundedTemplateSpec:
    text: str
    slots: dict[str, SlotSpec]


@dataclass(frozen=True, slots=True)
class GenericAllowedWhen:
    anchor_is_latest_other_speech: bool = True
    max_anchor_age_ms: int = 6000


@dataclass(frozen=True, slots=True)
class GenericTemplateSpec:
    text: str
    allowed_when: GenericAllowedWhen


@dataclass(frozen=True, slots=True)
class TemplateSpec:
    id: str
    category: Category
    max_words: int
    grounded: GroundedTemplateSpec
    generic: GenericTemplateSpec
    resolution_cues: tuple[str, ...]
    invalidate_on: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TemplateRenderResult:
    candidate: SuggestionCandidate | None
    unrendered_reason: ReasonCode | None
    grounded_failure_reason: ReasonCode | None = None
    is_generic: bool = False


def _t_str(val: Any, name: str) -> str:
    if not isinstance(val, str) or not val.strip():
        raise TemplateConfigError(f"{name} must be a non-empty string")
    return val


def _t_int(val: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
        raise TemplateConfigError(f"{name} must be an int >= {minimum}, got {val!r}")
    return val


def _t_bool(val: Any, name: str) -> bool:
    if not isinstance(val, bool):
        raise TemplateConfigError(f"{name} must be a bool, got {val!r}")
    return val


def _t_map(val: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(val, Mapping):
        raise TemplateConfigError(f"{name} must be a mapping")
    return val


def _t_str_tuple(val: Any, name: str) -> tuple[str, ...]:
    if not isinstance(val, (list, tuple)):
        raise TemplateConfigError(f"{name} must be a list")
    return tuple(_t_str(item, f"{name}[]") for item in val)


def _placeholders(fmt_text: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, _, _ in string.Formatter().parse(fmt_text):
        if field_name is not None:
            if not field_name:
                raise TemplateConfigError("Unnamed format placeholder in grounded.text")
            fields.add(field_name)
    return fields


def load_template(path: str | Path) -> TemplateSpec:
    """Load and validate a single §9.1 YAML template."""
    raw = Path(path).read_text(encoding="utf-8")
    m = _t_map(yaml.safe_load(raw), "template")
    tpl_id = _t_str(m.get("id"), "template.id")
    try:
        cat = Category(_t_str(m.get("category"), "template.category"))
    except ValueError as exc:
        raise TemplateConfigError(f"Unknown template.category: {m.get('category')!r}") from exc
    if cat not in PREDICTABLE_CATEGORIES:
        raise TemplateConfigError(f"Template category cannot be {cat!r}")
    max_words = _t_int(m.get("max_words"), "template.max_words", minimum=1)

    grd_m = _t_map(m.get("grounded"), "template.grounded")
    grd_text = _t_str(grd_m.get("text"), "template.grounded.text")
    slots_m = _t_map(grd_m.get("slots"), "template.grounded.slots")
    slots: dict[str, SlotSpec] = {}
    for sk, sv in slots_m.items():
        slot_name = _t_str(sk, "template.grounded.slots.key")
        sm = _t_map(sv, f"template.grounded.slots.{slot_name}")
        slots[slot_name] = SlotSpec(
            extractor=_t_str(sm.get("extractor"), f"slots.{slot_name}.extractor"),
            max_words=_t_int(sm.get("max_words"), f"slots.{slot_name}.max_words", minimum=1),
            must_be_substring=_t_bool(
                sm.get("must_be_substring", True), f"slots.{slot_name}.must_be_substring"
            ),
        )
    if _placeholders(grd_text) != set(slots):
        raise TemplateConfigError(
            f"Template {tpl_id!r} grounded.text placeholders {_placeholders(grd_text)} "
            f"do not match slots {set(slots)}"
        )

    gen_m = _t_map(m.get("generic"), "template.generic")
    gen_text = _t_str(gen_m.get("text"), "template.generic.text")
    if len(gen_text.split()) > max_words:
        raise TemplateConfigError(f"Template {tpl_id!r} generic.text exceeds max_words={max_words}")
    aw_m = _t_map(gen_m.get("allowed_when"), "template.generic.allowed_when")
    allowed_when = GenericAllowedWhen(
        anchor_is_latest_other_speech=_t_bool(
            aw_m.get("anchor_is_latest_other_speech", True),
            "generic.allowed_when.anchor_is_latest_other_speech",
        ),
        max_anchor_age_ms=_t_int(
            aw_m.get("max_anchor_age_ms", 6000),
            "generic.allowed_when.max_anchor_age_ms",
            minimum=1,
        ),
    )

    return TemplateSpec(
        id=tpl_id,
        category=cat,
        max_words=max_words,
        grounded=GroundedTemplateSpec(text=grd_text, slots=slots),
        generic=GenericTemplateSpec(text=gen_text, allowed_when=allowed_when),
        resolution_cues=_t_str_tuple(m.get("resolution_cues", []), "template.resolution_cues"),
        invalidate_on=_t_str_tuple(m.get("invalidate_on", []), "template.invalidate_on"),
    )


def load_templates_dir(directory: str | Path = DEFAULT_TEMPLATES_DIR) -> dict[str, TemplateSpec]:
    """Load all YAML templates in a directory keyed by template.id."""
    root = Path(directory)
    out: dict[str, TemplateSpec] = {}
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix in {".yaml", ".yml"}:
            spec = load_template(p)
            if spec.id in out:
                raise TemplateConfigError(f"Duplicate template id {spec.id!r}")
            out[spec.id] = spec
    return out


def _suggestion_id(
    candidate_id: str,
    category: Category,
    template_id: str,
    state_version: int,
    variant: str,
) -> str:
    raw = f"{candidate_id}|{category.value}|{template_id}|{state_version}|{variant}".encode()
    return f"sug-{hashlib.sha256(raw).hexdigest()[:24]}"


def _resolve_anchor_segments(
    context: DecisionContext,
    snapshot: StateSnapshot,
) -> list[SegmentRecord] | None:
    segs: list[SegmentRecord] = []
    for ref in context.anchor_refs:
        seg = snapshot.segments_by_id.get(ref.segment_id)
        if seg is None or seg.revision != ref.revision:
            return None
        segs.append(seg)
    return segs if segs else None


def _resolve_template_for_fallback(
    category: Category,
    template_or_policy: TemplateSpec | PolicyConfig,
    templates: Mapping[str, TemplateSpec] | None,
) -> TemplateSpec | None:
    if isinstance(template_or_policy, TemplateSpec):
        return template_or_policy
    cat_pol = template_or_policy.categories.get(category)
    if cat_pol is None or cat_pol.template is None:
        return None
    tpl_map = templates if templates is not None else load_templates_dir()
    return tpl_map.get(cat_pol.template)


def render_generic_fallback(
    context: DecisionContext,
    category: Category,
    snapshot: StateSnapshot,
    template_or_policy: TemplateSpec | PolicyConfig,
    *,
    now_ms: int,
    templates: Mapping[str, TemplateSpec] | None = None,
    ready_ms: int | None = None,
    display_ttl_ms: int | None = None,
    origin: Origin | None = None,
    grounded_failure_reason: ReasonCode | None = None,
) -> TemplateRenderResult:
    """Render generic template iff anchor_is_latest_other_speech & age <= cap (§9.1)."""
    tpl = _resolve_template_for_fallback(category, template_or_policy, templates)
    if tpl is None:
        return TemplateRenderResult(
            candidate=None,
            unrendered_reason=ReasonCode.GENERIC_REFERENT_UNCLEAR,
            grounded_failure_reason=grounded_failure_reason,
            is_generic=False,
        )
    if tpl.generic.allowed_when.anchor_is_latest_other_speech:
        ok, fail_code = check_generic_referent(
            snapshot,
            context.anchor_refs,
            now_ms=now_ms,
            max_anchor_age_ms=tpl.generic.allowed_when.max_anchor_age_ms,
        )
        if not ok:
            return TemplateRenderResult(
                candidate=None,
                unrendered_reason=fail_code or ReasonCode.GENERIC_REFERENT_UNCLEAR,
                grounded_failure_reason=grounded_failure_reason,
                is_generic=False,
            )
    elif (
        now_ms < context.anchor_received_ms
        or (now_ms - context.anchor_received_ms) > tpl.generic.allowed_when.max_anchor_age_ms
    ):
        return TemplateRenderResult(
            candidate=None,
            unrendered_reason=ReasonCode.GENERIC_REFERENT_UNCLEAR,
            grounded_failure_reason=grounded_failure_reason,
            is_generic=False,
        )

    words = tpl.generic.text.split()
    if len(words) > tpl.max_words:
        return TemplateRenderResult(
            candidate=None,
            unrendered_reason=ReasonCode.SLOT_TOO_LONG,
            grounded_failure_reason=grounded_failure_reason,
            is_generic=False,
        )
    ttl = (
        display_ttl_ms
        if display_ttl_ms is not None
        else (
            template_or_policy.display.display_ttl_ms
            if isinstance(template_or_policy, PolicyConfig)
            else 15000
        )
    )
    cand = SuggestionCandidate(
        schema_version=1,
        suggestion_id=_suggestion_id(
            context.candidate_id, category, tpl.id, context.based_on_state_version, "generic"
        ),
        candidate_id=context.candidate_id,
        origin=origin or context.origin,
        category=category,
        route=Route.TEMPLATE,
        text=tpl.generic.text,
        word_count=len(words),
        based_on_state_version=context.based_on_state_version,
        evidence_refs=context.anchor_refs,
        template_id=tpl.id,
        grounded_slots={},
        ready_ms=ready_ms if ready_ms is not None else now_ms,
        display_by_ms=context.display_by_ms,
        display_ttl_ms=ttl,
        invalidate_on=tpl.invalidate_on,
    )
    return TemplateRenderResult(
        candidate=cand,
        unrendered_reason=None,
        grounded_failure_reason=grounded_failure_reason,
        is_generic=True,
    )


def render_template(
    context: DecisionContext,
    category: Category,
    snapshot: StateSnapshot,
    template: TemplateSpec,
    *,
    now_ms: int,
    ready_ms: int | None = None,
    display_ttl_ms: int = 15000,
    origin: Origin | None = None,
    slots_override: Mapping[str, GroundedSlot] | None = None,
) -> TemplateRenderResult:
    """Render a §9.1 template: grounded form first, generic fallback only if referent clear."""
    if template.category is not category:
        raise TemplateConfigError(
            f"Template {template.id!r} category {template.category} != {category}"
        )
    anchor_segs = _resolve_anchor_segments(context, snapshot)
    grounded_fail: ReasonCode | None = None

    if anchor_segs is None:
        grounded_fail = ReasonCode.SLOT_UNAVAILABLE
    else:
        for spec in template.grounded.slots.values():
            if spec.extractor != SPAN_EXTRACTOR_VERSION:
                grounded_fail = ReasonCode.SLOT_UNAVAILABLE
                break
        if grounded_fail is None:
            extracted = (
                dict(slots_override)
                if slots_override is not None
                else extract_slots_span_rules_v0(anchor_segs, template.category)
            )
            required_keys = set(template.grounded.slots)
            if not required_keys.issubset(extracted):
                grounded_fail = ReasonCode.SLOT_UNAVAILABLE
            else:
                used_slots = {k: extracted[k] for k in template.grounded.slots}
                speakers: set[str] = set()
                for k, slot in used_slots.items():
                    seg = snapshot.segments_by_id.get(slot.segment_id)
                    if seg is None or seg.revision != slot.revision:
                        grounded_fail = ReasonCode.SLOT_UNAVAILABLE
                        break
                    speakers.add(seg.speaker_id)
                    slot_spec = template.grounded.slots[k]
                    if len(slot.text.split()) > slot_spec.max_words:
                        grounded_fail = ReasonCode.SLOT_TOO_LONG
                        break
                    if slot_spec.must_be_substring and not verify_grounded_slot(slot, seg):
                        grounded_fail = ReasonCode.SLOT_UNAVAILABLE
                        break
                if grounded_fail is None and len(speakers) != 1:
                    # §9.1: never combine slots from different speakers' segments
                    grounded_fail = ReasonCode.SLOT_UNAVAILABLE

                if grounded_fail is None:
                    rendered_text = template.grounded.text.format(
                        **{k: v.text for k, v in used_slots.items()}
                    )
                    word_count = len(rendered_text.split())
                    if word_count > template.max_words:
                        grounded_fail = ReasonCode.SLOT_TOO_LONG
                    else:
                        # §9.1: re-check grounding invariant after substitution
                        post_ok = all(
                            verify_grounded_slot(slot, snapshot.segments_by_id[slot.segment_id])
                            and slot.text in rendered_text
                            for slot in used_slots.values()
                        )
                        if not post_ok:
                            grounded_fail = ReasonCode.SLOT_UNAVAILABLE
                        else:
                            cand = SuggestionCandidate(
                                schema_version=1,
                                suggestion_id=_suggestion_id(
                                    context.candidate_id,
                                    category,
                                    template.id,
                                    context.based_on_state_version,
                                    "grounded",
                                ),
                                candidate_id=context.candidate_id,
                                origin=origin or context.origin,
                                category=category,
                                route=Route.TEMPLATE,
                                text=rendered_text,
                                word_count=word_count,
                                based_on_state_version=context.based_on_state_version,
                                evidence_refs=context.anchor_refs,
                                template_id=template.id,
                                grounded_slots=used_slots,
                                ready_ms=ready_ms if ready_ms is not None else now_ms,
                                display_by_ms=context.display_by_ms,
                                display_ttl_ms=display_ttl_ms,
                                invalidate_on=template.invalidate_on,
                            )
                            return TemplateRenderResult(
                                candidate=cand,
                                unrendered_reason=None,
                                grounded_failure_reason=None,
                                is_generic=False,
                            )

    return render_generic_fallback(
        context,
        category,
        snapshot,
        template,
        now_ms=now_ms,
        ready_ms=ready_ms,
        display_ttl_ms=display_ttl_ms,
        origin=origin,
        grounded_failure_reason=grounded_fail,
    )
