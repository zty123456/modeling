"""Three-tier fusion-rule matcher.

Contract: see ``docs/fusion_v2_rich_rules_zh.md`` §2.1.

Three matching tiers, all gated by ``target_class`` ↔ ``module_class``:

1. ``class_only``:
       Validates ``module_class`` matches ``target_class`` and the group
       size lies in ``[min_ops, max_ops]``.  Op sequence is ignored.

2. ``ordered_regex``:
       Walks ``op_regexes`` and ``op_types`` with two pointers; ops in
       ``skip_ops`` may be skipped.  Trailing ops past the last matched
       regex must all be in ``skip_ops``.

3. ``dag_signature``:
       Order-agnostic histogram check: each ``(regex, min_count)`` must
       fullmatch at least ``min_count`` ops in ``op_types``.  ``skip_ops``
       does not exempt anything from the histogram.

``best_rule`` selects the highest-priority matching rule from a list,
breaking ties by list order (i.e. earliest-registered wins).
"""
from __future__ import annotations

import re
from typing import Optional

from .rule import IORole, MatchPattern, ModuleFusionRule  # noqa: F401  (re-export hint)


# ─────────────────────────────────────────────────────────────────────────────
# target_class ↔ module_class
# ─────────────────────────────────────────────────────────────────────────────

def _target_candidates(target_class: type | str | tuple[str, ...]) -> tuple[str, ...]:
    """Normalize ``target_class`` to a tuple of candidate strings/regexes."""
    if isinstance(target_class, type):
        return (target_class.__name__,)
    if isinstance(target_class, tuple):
        return tuple(str(c) for c in target_class)
    return (str(target_class),)


def _class_matches(module_class: str, target_class: type | str | tuple[str, ...]) -> bool:
    """Return True iff ``module_class`` is matched by ``target_class``.

    Strategy per candidate:
        - literal ``==`` first (fast path for plain class names)
        - then ``re.fullmatch`` (regex fallback)
    """
    candidates = _target_candidates(target_class)
    for cand in candidates:
        if module_class == cand:
            return True
        try:
            if re.fullmatch(cand, module_class) is not None:
                return True
        except re.error:
            # Malformed regex → treat as literal-only; already failed above.
            continue
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Per-tier matchers
# ─────────────────────────────────────────────────────────────────────────────

def _check_size(op_types: tuple[str, ...], pattern: MatchPattern) -> bool:
    n = len(op_types)
    return pattern.min_ops <= n <= pattern.max_ops


def _match_class_only(op_types: tuple[str, ...], pattern: MatchPattern) -> bool:
    return _check_size(op_types, pattern)


def _match_ordered_regex(op_types: tuple[str, ...], pattern: MatchPattern) -> bool:
    if not _check_size(op_types, pattern):
        return False
    regexes = pattern.op_regexes
    skip = pattern.skip_ops
    i = 0  # index into regexes
    j = 0  # index into op_types
    n_re = len(regexes)
    n_op = len(op_types)
    while i < n_re and j < n_op:
        op = op_types[j]
        try:
            hit = re.fullmatch(regexes[i], op) is not None
        except re.error:
            hit = False
        if hit:
            i += 1
            j += 1
        elif op in skip:
            j += 1
        else:
            return False
    if i != n_re:
        return False
    # Trailing ops past last regex must all be skip-ops.
    while j < n_op:
        if op_types[j] not in skip:
            return False
        j += 1
    return True


def _match_dag_signature(op_types: tuple[str, ...], pattern: MatchPattern) -> bool:
    if not _check_size(op_types, pattern):
        return False
    for regex, min_count in pattern.op_multiset:
        try:
            count = sum(1 for op in op_types if re.fullmatch(regex, op) is not None)
        except re.error:
            count = 0
        if count < min_count:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def match_group(
    op_types: tuple[str, ...],
    module_class: str,
    *,
    pattern: MatchPattern,
    target_class: type | str | tuple[str, ...],
) -> bool:
    """Return True iff the group matches ``pattern`` under ``target_class``."""
    if not _class_matches(module_class, target_class):
        return False

    kind = pattern.kind
    if kind == "class_only":
        return _match_class_only(op_types, pattern)
    if kind == "ordered_regex":
        return _match_ordered_regex(op_types, pattern)
    if kind == "dag_signature":
        return _match_dag_signature(op_types, pattern)
    raise ValueError(f"unknown MatchPattern.kind: {kind!r}")


def best_rule(
    op_types: tuple[str, ...],
    module_class: str,
    rules: list[ModuleFusionRule],
) -> Optional[ModuleFusionRule]:
    """Pick the highest-priority matching rule; ties broken by list order."""
    best: Optional[ModuleFusionRule] = None
    best_idx: int = -1
    for idx, rule in enumerate(rules):
        pat = rule.pattern
        if pat is None:
            continue
        if not match_group(
            op_types,
            module_class,
            pattern=pat,
            target_class=rule.target_class,
        ):
            continue
        if best is None or rule.priority > best.priority:
            best = rule
            best_idx = idx
        # ties: keep earlier (lower idx) — already satisfied because we
        # only replace on strictly greater priority.
        _ = best_idx
    return best
