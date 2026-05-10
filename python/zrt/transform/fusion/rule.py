"""Backward-compat shim for ``python.zrt.transform.fusion.rule``.

The dataclasses moved to ``core/`` and the YAML helpers to
``loading/yaml_rule_loader.py``.  This module re-exports the public
surface so existing imports keep working.
"""
from __future__ import annotations

from .core.io_role import IORole, IOSpec, ShapeDerivation  # noqa: F401
from .core.pattern import (  # noqa: F401
    DEFAULT_SKIP_OPS,
    MatchKind,
    MatchPattern,
)
from .core.rule import ModuleFusionRule  # noqa: F401
from .loading.yaml_rule_loader import (  # noqa: F401  (legacy private helpers)
    _io_role_from_dict,
    _parse_match_dict,
    _parse_shape_derivation,
    rule_from_yaml_dict,
)

__all__ = [
    "DEFAULT_SKIP_OPS",
    "IORole",
    "IOSpec",
    "MatchKind",
    "MatchPattern",
    "ModuleFusionRule",
    "ShapeDerivation",
]
