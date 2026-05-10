"""Backward-compat shim for ``python.zrt.transform.fusion.resolver``.

The implementation moved to ``loading/op_name_resolver.py``.  This
module re-exports the public surface so existing imports keep working.
"""
from __future__ import annotations

from .loading.op_name_resolver import (  # noqa: F401
    resolve_short_name,
    resolve_short_names,
)

__all__ = ["resolve_short_name", "resolve_short_names"]
