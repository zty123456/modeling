"""Protocol for quant profile objects usable by the shared dtype dispatch.

Both ``ModelSpec`` (spec path) and ``GraphQuantProfile`` (graph path) satisfy
this protocol structurally — they expose the same dtype accessor fields and
``effective_*_dtype()`` fallback helpers.
"""
from __future__ import annotations

from typing import Protocol

from zrt.training.spec.dtype import Dtype


class QuantProfileLike(Protocol):
    """Structural type for objects providing per-component dtype accessors."""

    # Global dtypes
    param_dtype: Dtype
    act_dtype: Dtype

    # Per-component compute dtype
    attn_compute_dtype: Dtype
    shared_expert_compute_dtype: Dtype
    routed_expert_compute_dtype: Dtype

    # Per-component weight dtype
    attn_weight_dtype: Dtype
    shared_expert_weight_dtype: Dtype
    routed_expert_weight_dtype: Dtype

    # Effective dtype helpers (fallback to act_dtype when per-region is unset)
    def effective_attn_act_dtype(self) -> Dtype: ...
    def effective_moe_act_dtype(self) -> Dtype: ...
    def effective_residual_dtype(self) -> Dtype: ...
