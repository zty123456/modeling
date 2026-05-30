"""Shared component classifier — maps OpNode → component category string.

Used by both ``transform/analysis/quant.py`` (dtype dispatch) and
``ir/param_count.py`` (per-component parameter counting) so they classify
nodes identically.

Drift contract: new component buckets MUST be added here and in the dispatch
chain in ``training/models/quant_dispatch.py::dispatch``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.node import OpNode


def classify(node: "OpNode") -> str:
    """Map a graph OpNode's component/scope to a spec-style component name.

    Returns one of: "attention", "routed_expert", "shared_expert",
    "embedding", "norm", or "default".
    """
    comp = node.component.lower()
    scope = (node.scope or "").lower()
    op_type = (node.op_type or "").lower()

    # Embedding
    if comp == "embedding" or comp == "lm_head" or "embed" in comp:
        return "embedding"

    # Norm layers
    if comp in ("attn_norm", "ffn_norm", "final_norm"):
        return "norm"
    # Fused norm ops
    if any(t in op_type for t in ("rms_norm", "layer_norm", "add_rms_norm",
                                   "npu_add_rms", "gemma_norm", "rms_gated")):
        return "norm"

    # Shared expert (check BEFORE routed expert)
    if comp.startswith("moe.shared.") or "shared_expert" in scope:
        return "shared_expert"

    # Routed expert
    if comp.startswith("moe.experts.") or (
        "experts" in scope and "shared_expert" not in scope
    ):
        return "routed_expert"
    # Fused grouped MM that replaced expert ops
    fused = node.annotations.get("fused_by", "")
    if fused == "expert_grouped_mm":
        return "routed_expert"

    # Attention
    if comp.startswith("attn.") or "self_attn" in scope or "attention" in scope:
        return "attention"
    # Fused attention ops
    if any(t in op_type for t in ("sdpa", "flash_attn", "scaled_dot_product",
                                   "paged_attention", "sparse_attn",
                                   "mla_attn", "attention")):
        return "attention"

    return "default"
