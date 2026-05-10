"""Backward-compat shim for ``python.zrt.transform.fusion.match``.

The implementation moved to ``matching/matcher.py``.  This module
re-exports the public surface so existing imports keep working.
"""
from __future__ import annotations

from .matching.matcher import (  # noqa: F401
    RuleMatcher,
    _check_size,
    _class_matches,
    _match_class_only,
    _match_dag_signature,
    _match_ordered_regex,
    _target_candidates,
    best_rule,
    match_group,
)

__all__ = [
    "RuleMatcher",
    "best_rule",
    "match_group",
]
