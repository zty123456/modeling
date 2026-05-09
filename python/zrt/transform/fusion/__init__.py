"""Operator fusion module.

See ``docs/fusion_v2_rich_rules_zh.md`` for the rule contract.

Public API:
- ``FusionPass`` — GraphPass for the TransformPipeline.
- ``ModuleFusionRule`` — fusion rule (rich format).
- ``IORole``, ``IOSpec`` — IO role declaration (``IOSpec`` is alias).
- ``MatchPattern``, ``MatchKind``, ``DEFAULT_SKIP_OPS`` — match patterns.
- ``ShapeDerivation`` — symbolic axis derivation.
- ``register_rule``, ``clear_rules``, ``all_rules`` — registry.
- ``load_yaml_rules``, ``add_yaml_search_dir`` — YAML loading.
"""
from .api import FusionPass
from .registry import all_rules, clear_rules, register_rule
from .rule import (
    DEFAULT_SKIP_OPS,
    IORole,
    IOSpec,
    MatchKind,
    MatchPattern,
    ModuleFusionRule,
    ShapeDerivation,
)
from .yaml_loader import add_yaml_search_dir, load_yaml_rules

__all__ = [
    "FusionPass",
    "ModuleFusionRule",
    "IORole",
    "IOSpec",
    "MatchPattern",
    "MatchKind",
    "ShapeDerivation",
    "DEFAULT_SKIP_OPS",
    "register_rule",
    "clear_rules",
    "all_rules",
    "load_yaml_rules",
    "add_yaml_search_dir",
]
