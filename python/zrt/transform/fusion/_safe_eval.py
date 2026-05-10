"""Backward-compat shim for ``python.zrt.transform.fusion._safe_eval``.

The implementation moved to ``semantics/safe_eval.py``.  This module
re-exports the public surface so existing imports keep working.
"""
from __future__ import annotations

from .semantics.safe_eval import (  # noqa: F401
    FormulaError,
    safe_eval,
)

__all__ = ["FormulaError", "safe_eval"]
