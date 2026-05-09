"""Global rule registry: register and look up ModuleFusionRules.

Lookup goes through the three-tier matcher in ``match.py``:
    ``lookup_rule(op_types, module_class)`` returns the highest-priority
    rule whose ``MatchPattern`` matches the given group, with ties
    broken by registration order.

Storage is a single ordered list (insertion order preserved) so the
matcher sees rules in the order they were registered.  An auxiliary
class-name index narrows the candidate pool when ``module_class`` is
non-empty — this is purely a performance optimisation; correctness is
guaranteed by the matcher's own ``target_class`` check.
"""
from __future__ import annotations

import logging
from typing import Optional

from .rule import ModuleFusionRule

logger = logging.getLogger(__name__)


# All rules in registration order.
_ALL_RULES: list[ModuleFusionRule] = []

# Index: target_class candidate string → list of rules whose target_class
# contains that string literal.  Used as a fast filter.
_CLASS_INDEX: dict[str, list[ModuleFusionRule]] = {}


def _index_keys(rule: ModuleFusionRule) -> list[str]:
    """Return the literal class-name keys this rule should be indexed under."""
    tc = rule.target_class
    if isinstance(tc, type):
        return [tc.__name__]
    if isinstance(tc, tuple):
        return [str(c) for c in tc]
    return [str(tc)]


def register_rule(rule: ModuleFusionRule) -> None:
    """Add a rule to the global registry."""
    _ALL_RULES.append(rule)
    for key in _index_keys(rule):
        _CLASS_INDEX.setdefault(key, []).append(rule)


def lookup_rule(
    op_types: tuple[str, ...],
    *,
    module_class: str | None = None,
) -> Optional[ModuleFusionRule]:
    """Find the best-matching rule via the three-tier matcher.

    When ``module_class`` is non-empty, narrows to rules indexed under
    that exact class name first — and only falls back to the global
    list if no match is found there.  This preserves the v1 narrow-then-
    broaden strategy while routing through the new matcher.
    """
    from .match import best_rule

    if module_class:
        narrowed = _CLASS_INDEX.get(module_class, [])
        rule = best_rule(op_types, module_class, narrowed)
        if rule is not None:
            return rule

    return best_rule(op_types, module_class or "", _ALL_RULES)


# Legacy alias — old call sites pass ``op_types`` of full aten names.
lookup_rule_by_op_sequence = lookup_rule


def clear_rules() -> None:
    """Remove all registered rules (for test isolation)."""
    _ALL_RULES.clear()
    _CLASS_INDEX.clear()


def all_rules() -> list[ModuleFusionRule]:
    """Return every registered rule (flat list, registration order)."""
    return list(_ALL_RULES)
