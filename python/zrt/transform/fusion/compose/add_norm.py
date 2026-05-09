"""Add+Norm compositor — re-exported from algorithm._compose_add_norm.

The compositor lives in ``algorithm.py`` for simplicity; this module
exists for future compositor expansion (e.g. fused QKV).
"""
from ..algorithm import _compose_add_norm as compose_add_norm

__all__ = ["compose_add_norm"]
