"""Mixtral fusion rules.

Mixtral uses standard ``nn.Linear`` plus ``MixtralBLockSparseTop2MLP``
and ``MixtralSparseMoeBlock``.  Specific op sequences for these
modules should be provided via YAML rules.
"""
from __future__ import annotations


def import_mixtral_rules() -> None:
    """Register Mixtral-specific rules.

    Mixtral-specific modules require verified op sequences which
    should be provided via YAML rules discovered via ``discover.py``.
    """
    pass
