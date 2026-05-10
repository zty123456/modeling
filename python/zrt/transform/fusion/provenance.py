"""Backward-compat shim for ``python.zrt.transform.fusion.provenance``.

The implementation moved to ``building/io_resolver.py``.  This module
re-exports the public surface so existing imports keep working.
"""
from __future__ import annotations

from .building.io_resolver import FusedIOPort, resolve_io  # noqa: F401

__all__ = ["FusedIOPort", "resolve_io"]
