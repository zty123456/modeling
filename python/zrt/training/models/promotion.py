"""Shared FP32-promotion input-byte multiplier for LN/RMSNorm and softmax ops.

Both the spec-side ``_promote_aware_elementwise_cost`` (``flops.py``) and the
graph-side ``FlopsPass`` (``transform/analysis/passes.py``) model the same
extra-read penalty when quantized input (FP8/FP4) must be promoted to FP32
for numerically stable LN/RMSNorm or softmax. This module is the single
source of truth for the multiplier.
"""
from __future__ import annotations


def ln_softmax_input_byte_multiplier(kind_or_type: str) -> float:
    """Return the input-byte penalty multiplier for FP32 promotion.

    LN/RMSNorm: +1x (single reduce pass for mean/var)
    Softmax:    +2x (max pass + sum-of-exp pass)
    Other:       0  (no promotion)

    Accepts either spec ``op.kind`` strings ("softmax", "ln", "rmsnorm") or
    graph ``op_type`` strings ("aten.softmax.int", "rms_norm", etc.) — uses
    substring matching that covers both.
    """
    s = kind_or_type.lower()
    if "softmax" in s:
        return 2.0
    if any(t in s for t in (
        "rms_norm", "rmsnorm", "layer_norm", "ln",
        "add_rms_norm", "add_layer_norm",
        "npu_add_rms", "gemma_norm", "rms_gated",
    )):
        return 1.0
    return 0.0
