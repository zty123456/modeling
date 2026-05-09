"""ModuleFusionRule v2 (rich format): semantic-aware fusion rules.

Contract: see ``docs/fusion_v2_rich_rules_zh.md``.

Key extensions over the legacy schema:
  - ``MatchPattern`` replaces strict ``op_sequences`` tuple-equality with
    three flexible matching tiers — class_only / ordered_regex / dag_signature.
  - ``IORole`` (alias ``IOSpec``) adds symbolic ``shape_role`` so
    downstream can derive batch / seq / hidden / heads / dtype.
  - ``ShapeDerivation`` declares how to extract semantic dimensions.
  - ``flops_formula`` / ``memory_formula`` (string) are AST-evaluated
    against the derived shape namespace; ``flops_callable`` /
    ``memory_callable`` are the Python registration counterparts.

Backward compatibility:
  - Legacy ``op_sequences`` + ``inputs`` / ``outputs`` fields still work;
    they are auto-translated to ``pattern`` and ``io_roles`` at
    construction time.
  - ``IOSpec`` is preserved as an alias of ``IORole``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Match patterns
# ─────────────────────────────────────────────────────────────────────────────

MatchKind = Literal["class_only", "ordered_regex", "dag_signature"]


# ATen ops that may interleave between matched ops without breaking
# an ``ordered_regex`` match.  Shape / dtype-cast / no-op rewires.
DEFAULT_SKIP_OPS: frozenset[str] = frozenset({
    "aten.view.default",
    "aten.reshape.default",
    "aten.permute.default",
    "aten.transpose.int",
    "aten.expand.default",
    "aten.squeeze.dim",
    "aten.unsqueeze.default",
    "aten.flatten.using_ints",
    "aten.unflatten.int",
    "aten.contiguous.memory_format",
    "aten._to_copy.default",
    "aten.to.dtype",
    "aten.to.dtype_layout",
    "aten.clone.default",
    "aten.detach.default",
    "aten.alias.default",
})


@dataclass(frozen=True)
class MatchPattern:
    """Pattern definition for matching a fusion group.

    See ``docs/fusion_v2_rich_rules_zh.md`` §2.1 for full semantics.
    """

    kind: MatchKind = "ordered_regex"
    op_regexes: tuple[str, ...] = ()
    op_multiset: tuple[tuple[str, int], ...] = ()
    skip_ops: frozenset[str] = field(default_factory=lambda: DEFAULT_SKIP_OPS)
    min_ops: int = 1
    max_ops: int = 1024

    def __post_init__(self) -> None:
        if self.kind == "ordered_regex" and not self.op_regexes:
            # ordered_regex with no regexes is invalid
            raise ValueError("ordered_regex requires non-empty op_regexes")
        if self.kind == "dag_signature" and not self.op_multiset:
            raise ValueError("dag_signature requires non-empty op_multiset")
        if self.min_ops < 1:
            raise ValueError("min_ops must be >= 1")
        if self.max_ops < self.min_ops:
            raise ValueError("max_ops must be >= min_ops")


# ─────────────────────────────────────────────────────────────────────────────
# IO role
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IORole:
    """How to extract one semantic IO tensor from a fusion group.

    See ``docs/fusion_v2_rich_rules_zh.md`` §2.2.
    """

    role: str
    source_kind: Literal["input", "output"] = "input"
    source_op_index: int = -1
    source_arg_index: int = -1
    shape_role: Optional[str] = None
    expected_dtype: Optional[str] = None


# Backward-compat alias for code paths that still import IOSpec.
IOSpec = IORole


# ─────────────────────────────────────────────────────────────────────────────
# Shape derivation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ShapeDerivation:
    """Symbolic axis derivation expressions.

    Each value is a string expression evaluated by ``_safe_eval`` over
    ``{role: TensorView, ...}`` plus previously-derived axes (so later
    axes can reference earlier ones).
    """

    batch_size:       Optional[str] = None
    seq_len:          Optional[str] = None
    hidden_in:        Optional[str] = None
    hidden_out:       Optional[str] = None
    num_heads:        Optional[str] = None
    head_dim:         Optional[str] = None
    intermediate_dim: Optional[str] = None
    vocab_size:       Optional[str] = None
    num_experts:      Optional[str] = None
    topk:             Optional[str] = None
    extra: tuple[tuple[str, str], ...] = ()

    def items(self) -> list[tuple[str, str]]:
        """Return ``(name, expr)`` pairs in declaration order.

        Standard axes come first, then ``extra`` in declared order.
        Earlier axes are visible in later ones' namespace.
        """
        out: list[tuple[str, str]] = []
        for name in ("batch_size", "seq_len", "hidden_in", "hidden_out",
                     "num_heads", "head_dim", "intermediate_dim",
                     "vocab_size", "num_experts", "topk"):
            v = getattr(self, name)
            if v is not None:
                out.append((name, v))
        out.extend(self.extra)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Module fusion rule
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModuleFusionRule:
    """One fusion rule keyed on an nn.Module class (or class-name regex).

    See ``docs/fusion_v2_rich_rules_zh.md`` §2.4 for the full schema.
    """

    target_class: type | str | tuple[str, ...]

    op_type: Optional[str] = None

    # Matching
    pattern: Optional[MatchPattern] = None
    op_sequences: tuple[tuple[str, ...], ...] = ()

    # IO roles
    io_roles: tuple[IORole, ...] = ()
    inputs:  tuple[IORole, ...] = ()
    outputs: tuple[IORole, ...] = ()

    # Semantic derivation
    shape_derivation: Optional[ShapeDerivation] = None
    flops_formula:  Optional[str] = None
    memory_formula: Optional[str] = None
    flops_callable:  Optional[Callable[[dict], float]] = None
    memory_callable: Optional[Callable[[dict], float]] = None
    flops_kind: Literal["sum_children", "from_io", "formula", "custom"] = "sum_children"
    custom_resolver: Optional[Callable] = None

    # Metadata
    annotations: dict = field(default_factory=dict)
    priority: int = 10

    def __post_init__(self) -> None:
        # Auto-merge legacy inputs/outputs into io_roles
        if self.inputs or self.outputs:
            merged = list(self.io_roles) + list(self.inputs) + list(self.outputs)
            object.__setattr__(self, "io_roles", tuple(merged))
            object.__setattr__(self, "inputs", ())
            object.__setattr__(self, "outputs", ())

        # Auto-build pattern when only op_sequences is provided.
        if self.pattern is None:
            if not self.op_sequences:
                # No matching info → match on class only.
                object.__setattr__(self, "pattern",
                                   MatchPattern(kind="class_only",
                                                op_regexes=(),
                                                op_multiset=()))
            elif len(self.op_sequences) == 1:
                seq = self.op_sequences[0]
                if not seq:
                    # Empty sequence → class_only.
                    object.__setattr__(self, "pattern",
                                       MatchPattern(kind="class_only",
                                                    op_regexes=(),
                                                    op_multiset=()))
                else:
                    object.__setattr__(self, "pattern",
                                       MatchPattern(
                                           kind="ordered_regex",
                                           op_regexes=tuple(re.escape(o) for o in seq),
                                           min_ops=len(seq),
                                       ))
            else:
                # Multiple alternatives must be expanded at YAML-load
                # time.  Reject construction here to surface bugs early.
                raise ValueError(
                    "multiple op_sequences must be expanded into multiple "
                    "rules by the loader; ModuleFusionRule allows at most one"
                )

        # Promote flops_kind to "formula" automatically when a formula or
        # callable is supplied but flops_kind was left at the default.
        if self.flops_kind == "sum_children" and (
            self.flops_formula or self.flops_callable
        ):
            object.__setattr__(self, "flops_kind", "formula")

    # ── Backward-compatible YAML constructor ────────────────────────────

    @classmethod
    def from_yaml_dict(cls, d: dict) -> "ModuleFusionRule":
        """Construct a rule from a parsed YAML dict (new or legacy schema).

        New schema (preferred):
            target_class, op_type, priority, enabled, match,
            io_roles, shape_derivation, flops_formula, memory_formula,
            annotations.

        Legacy schema (auto-translated):
            target_class, op_type, observed_op_seqs, inputs, outputs,
            priority, annotations.

        For multi-sequence ``observed_op_seqs``, the loader (yaml_loader.py)
        is responsible for splitting into multiple rules; this method
        accepts only single-sequence input.
        """
        from .resolver import resolve_short_name, resolve_short_names

        # ── Pattern (new schema) ──
        pattern: Optional[MatchPattern] = None
        match_d = d.get("match")
        if match_d is not None:
            pattern = _parse_match_dict(match_d)

        # ── Legacy op_sequences ──
        raw_seqs = d.get("observed_op_seqs", []) or []
        resolved_seqs: tuple[tuple[str, ...], ...] = tuple(
            resolve_short_names(seq) for seq in raw_seqs
        )

        # ── IO roles ──
        roles_list: list[IORole] = []
        for entry in (d.get("io_roles") or []):
            roles_list.append(_io_role_from_dict(entry))
        for entry in (d.get("inputs") or []):
            entry = {**entry, "source_kind": entry.get("source_kind", "input")}
            roles_list.append(_io_role_from_dict(entry))
        for entry in (d.get("outputs") or []):
            entry = {**entry, "source_kind": entry.get("source_kind", "output")}
            roles_list.append(_io_role_from_dict(entry))

        # ── Shape derivation ──
        sd: Optional[ShapeDerivation] = None
        sd_d = d.get("shape_derivation")
        if sd_d:
            sd = _parse_shape_derivation(sd_d)

        # target_class — list (regex alternatives) → tuple of str
        tc_raw = d["target_class"]
        if isinstance(tc_raw, list):
            target_class: type | str | tuple[str, ...] = tuple(tc_raw)
        else:
            target_class = tc_raw

        return cls(
            target_class=target_class,
            op_type=d.get("op_type"),
            pattern=pattern,
            op_sequences=resolved_seqs,
            io_roles=tuple(roles_list),
            shape_derivation=sd,
            flops_formula=d.get("flops_formula"),
            memory_formula=d.get("memory_formula"),
            flops_kind=d.get("flops_kind", "sum_children"),
            annotations=d.get("annotations", {}) or {},
            priority=d.get("priority", 10),
        )


# ─────────────────────────────────────────────────────────────────────────────
# YAML helpers (private)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_match_dict(d: dict) -> MatchPattern:
    """Build a MatchPattern from the YAML ``match:`` block."""
    from .resolver import resolve_short_name

    kind = d.get("kind", "ordered_regex")

    # ordered_regex
    op_regexes: tuple[str, ...] = ()
    if d.get("op_regexes"):
        op_regexes = tuple(d["op_regexes"])

    # dag_signature
    op_multiset: tuple[tuple[str, int], ...] = ()
    raw_ms = d.get("op_multiset")
    if raw_ms:
        if isinstance(raw_ms, dict):
            op_multiset = tuple((k, int(v)) for k, v in raw_ms.items())
        else:
            op_multiset = tuple((entry[0], int(entry[1])) for entry in raw_ms)

    # skip_ops_extra: short or full names; merged with DEFAULT_SKIP_OPS
    skip = set(DEFAULT_SKIP_OPS)
    for raw in (d.get("skip_ops_extra") or []):
        try:
            skip.add(resolve_short_name(raw))
        except KeyError:
            skip.add(raw)
    # skip_ops_only: replace, not merge
    if d.get("skip_ops_only"):
        skip = set()
        for raw in d["skip_ops_only"]:
            try:
                skip.add(resolve_short_name(raw))
            except KeyError:
                skip.add(raw)

    return MatchPattern(
        kind=kind,
        op_regexes=op_regexes,
        op_multiset=op_multiset,
        skip_ops=frozenset(skip),
        min_ops=int(d.get("min_ops", 1)),
        max_ops=int(d.get("max_ops", 1024)),
    )


def _io_role_from_dict(d: dict) -> IORole:
    return IORole(
        role=d["role"],
        source_kind=d.get("source_kind", "input"),
        source_op_index=int(d.get("source_op_index", -1)),
        source_arg_index=int(d.get("source_arg_index", -1)),
        shape_role=d.get("shape_role"),
        expected_dtype=d.get("expected_dtype"),
    )


def _parse_shape_derivation(d: dict) -> ShapeDerivation:
    standard = ("batch_size", "seq_len", "hidden_in", "hidden_out",
                "num_heads", "head_dim", "intermediate_dim",
                "vocab_size", "num_experts", "topk")
    kwargs: dict[str, Any] = {k: d[k] for k in standard if k in d}
    extra_raw = d.get("extra") or []
    extra: tuple[tuple[str, str], ...] = tuple(
        (entry[0], entry[1]) if isinstance(entry, (list, tuple))
        else (entry["name"], entry["expr"])
        for entry in extra_raw
    )
    if extra:
        kwargs["extra"] = extra
    return ShapeDerivation(**kwargs)
