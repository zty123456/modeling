"""Common rules — re-exported from builtins.py.

This file is kept for backward compatibility.
Use ``import_builtin_rules()`` from ``builtins.py`` directly.
"""
from __future__ import annotations

from ..builtins import import_builtin_rules as import_common_rules

__all__ = ["import_common_rules"]
