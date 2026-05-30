"""Optimization passes (Stage 3): quantization, EPLB, MTP stubs."""
from __future__ import annotations

from typing import TYPE_CHECKING

from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


class QuantizationPass(GraphPass):
    """Annotate nodes with quantization dtype info.

    When ``ctx.quant_profile`` is set (structured per-component profile),
    writes per-operand dtype annotations using the graph dtype resolver.
    Also writes legacy ``quant_weight``/``quant_act``/``quant_kv`` strings
    for backward compat with the roofline inference path.

    When only ``ctx.quant`` is set (legacy QuantConfig), falls back to the
    original string-based annotation path.
    """

    name = "quantization"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        # Structured profile path
        if ctx.quant_profile is not None:
            return self._run_with_profile(graph, ctx)
        # Legacy QuantConfig path
        if ctx.quant is not None:
            return self._run_legacy(graph, ctx)
        return graph

    def _run_with_profile(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        from python.zrt.transform.analysis.quant import graph_resolve_op_dtypes
        from zrt.training.spec.dtype import Dtype

        profile = ctx.quant_profile
        g = graph.clone()

        for node in g.nodes.values():
            if node.category != "compute":
                continue

            bundle = graph_resolve_op_dtypes(node, profile)

            # Per-operand dtype annotations (Dtype enum values)
            node.annotations["quant.in_act"] = bundle.in_act
            node.annotations["quant.weight"] = bundle.weight
            node.annotations["quant.out_act"] = bundle.out_act
            node.annotations["quant.compute"] = bundle.compute
            node.annotations["quant.grad_in"] = bundle.grad_in
            node.annotations["quant.grad_weight"] = bundle.grad_weight
            node.annotations["quant.grad_act"] = bundle.grad_act

            # KV cache dtype only on attention nodes
            if node.component.startswith("attn."):
                node.annotations["quant.kv_cache"] = profile.kv_cache_dtype

        # ── Comm payload dtype resolution ────────────────────────────────────
        # Map comm node op_types to the modeled payload dtype from the profile.
        self._annotate_comm_payloads(g, profile)

        return g

    def _annotate_comm_payloads(self, g: "OpGraph", profile) -> None:
        """Annotate comm nodes with payload_dtype from the quant profile."""
        from zrt.training.spec.dtype import Dtype

        for node in g.nodes.values():
            if node.category != "communication":
                continue
            # Idempotency: skip if already resolved by a prior run
            if node.attrs.get("payload_dtype_resolved"):
                continue

            ot = node.op_type.lower()
            if "send" in ot or "recv" in ot or "p2p" in ot:
                # PP P2P (including ring_p2p) → residual dtype
                payload = profile.effective_residual_dtype()
            elif "all_to_all" in ot:
                # EP A2A → MoE activation dtype
                payload = profile.effective_moe_act_dtype()
            elif "all_reduce" in ot:
                # DP grad reduce → grad dtype
                payload = profile.grad_dtype
            elif "all_gather" in ot or "reduce_scatter" in ot:
                # TP SP → act dtype (component-specific would need edge tracing)
                payload = profile.act_dtype
            else:
                payload = profile.act_dtype

            node.annotations["payload_dtype"] = payload
            # Adjust bucket_bytes to modeled dtype if current bytes are BF16-based
            bucket = node.attrs.get("bucket_bytes")
            if bucket and payload != Dtype.BF16:
                import math
                ratio = payload.bytes / Dtype.BF16.bytes
                node.attrs["bucket_bytes"] = math.ceil(bucket * ratio)
            node.attrs["payload_dtype_resolved"] = True

    def _run_legacy(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        from zrt.training.spec.dtype import Dtype
        g = graph.clone()
        for node in g.nodes.values():
            if node.category != "compute":
                continue
            act_str = ctx.quant.activation_for_component(node.component)
            w_str = ctx.quant.weight
            # Structured Dtype annotations (best-effort — legacy QuantConfig
            # may use inference-only dtypes like int8 that aren't in spec Dtype)
            try:
                node.annotations["quant.in_act"] = Dtype.parse(act_str)
                node.annotations["quant.weight"] = Dtype.parse(w_str)
                node.annotations["quant.compute"] = Dtype.parse(act_str)
            except ValueError:
                pass  # legacy string annotations still available below
            # Legacy string annotations (backward compat)
            node.annotations["quant_weight"] = w_str
            node.annotations["quant_act"] = act_str
            if node.component.startswith("attn."):
                try:
                    node.annotations["quant.kv_cache"] = Dtype.parse(ctx.quant.kv_cache)
                except ValueError:
                    pass
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
