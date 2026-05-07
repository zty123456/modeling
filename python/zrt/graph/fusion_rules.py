"""Backward-compatibility shim.

Fusion rules have moved to ``python.zrt.transform.fusion.rules``.
This module re-exports everything so existing imports continue to work.
"""
from python.zrt.transform.fusion.rules import *  # noqa: F401,F403
from python.zrt.transform.fusion.rules import (
    ALWAYS_TRANSPARENT,
    SHAPE_OPS,
    INIT_OPS,
    LIFT_OPS,
    POTENTIAL_COPY_OPS,
    PATTERN_SKIP,
    SEMANTIC_LABELS,
    PLATFORM_SUBPATTERNS,
    PLATFORM_SETTINGS,
    CONTAINER_SEMANTICS,
    SubPattern,
    get_semantic_label,
    get_subpatterns,
    get_platform_settings,
    match_subsequence,
)

__all__ = [
    "ALWAYS_TRANSPARENT",
    "SHAPE_OPS",
    "INIT_OPS",
    "LIFT_OPS",
    "POTENTIAL_COPY_OPS",
    "PATTERN_SKIP",
    "SEMANTIC_LABELS",
    "PLATFORM_SUBPATTERNS",
    "PLATFORM_SETTINGS",
    "CONTAINER_SEMANTICS",
    "SubPattern",
    "get_semantic_label",
    "get_subpatterns",
    "get_platform_settings",
    "match_subsequence",
]
