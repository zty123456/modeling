"""Optimization passes (Stage 3): quantization, EPLB, MTP stubs."""
from __future__ import annotations

from typing import TYPE_CHECKING

from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


class QuantizationPass(GraphPass):
    """Annotate nodes with quantization dtype info."""

    name = "quantization"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        if ctx.quant is None:
            return graph
        g = graph.clone()
        for node in g.nodes.values():
            if node.category == "compute":
                node.annotations["quant_weight"] = ctx.quant.weight
                node.annotations["quant_act"]    = ctx.quant.activation_for_component(node.component)
                # KV cache dtype only relevant for attention ops
                if node.component.startswith("attn."):
                    node.annotations["quant_kv"] = ctx.quant.kv_cache
        return g


class EPLBPass(GraphPass):
    """Expert-level load balancing annotation (stub)."""

    name = "eplb"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        return graph


class SharedExpertPass(GraphPass):
    """Mark shared expert ops as externalized (parallel with routed experts)."""

    name = "shared_expert"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()
        for node in g.nodes.values():
            if "shared_expert" in node.scope.lower():
                node.annotations["shared_expert_external"] = True
        return g


class MTPPass(GraphPass):
    """Multi-Token Prediction annotation (stub)."""

    name = "mtp"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        return graph
