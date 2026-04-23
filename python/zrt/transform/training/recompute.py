from __future__ import annotations

from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.transform.base import GraphPass
from zrt.transform.context import TransformContext


class RecomputePass(GraphPass):
    """Recompute pass for activation recomputation."""
    name = "recompute"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        """Run recompute pass on the graph.
        
        Args:
            graph: Input OpGraph
            ctx: TransformContext with training config
            
        Returns:
            New OpGraph with recompute annotations
        """
        if graph.phase != "train_forward":
            return graph  # recompute is a forward-graph annotation
        
        g = graph.clone()
        if not ctx.training:
            return g
        
        policy = ctx.training.recompute.per_layer_kind
        
        for node in g.nodes.values():
            layer_kind = self._layer_kind_of(node)
            tiers = policy.get(layer_kind, set())
            
            if self._matches_any_tier(node, tiers):
                node.annotations["recompute"] = True
                node.annotations["recompute_tier"] = self._matching_tier(node, tiers)
        
        return g

    def _layer_kind_of(self, node: OpNode) -> str:
        """Get the layer kind of a node.
        
        Args:
            node: OpNode to check
            
        Returns:
            Layer kind: "dense", "moe", or "mtp"
        """
        scope = node.scope.lower()
        
        if "moe" in scope:
            return "moe"
        elif "mtp" in scope:
            return "mtp"
        else:
            return "dense"

    def _matches_any_tier(self, node: OpNode, tiers: set[str]) -> bool:
        """Check if a node matches any recompute tier.
        
        Args:
            node: OpNode to check
            tiers: Set of recompute tiers
            
        Returns:
            True if node matches any tier, False otherwise
        """
        for tier in tiers:
            if self._matches_tier(node, tier):
                return True
        return False

    def _matches_tier(self, node: OpNode, tier: str) -> bool:
        """Check if a node matches a specific recompute tier.
        
        Args:
            node: OpNode to check
            tier: Recompute tier
            
        Returns:
            True if node matches the tier, False otherwise
        """
        if tier == "full":
            return True  # All ops in layer
        
        op_type = node.op_type
        scope = node.scope.lower()
        
        if tier == "attn":
            # softmax + attn_core + O-proj backward inputs
            return (
                "softmax" in op_type.lower() or
                "attention" in op_type.lower() or
                "attn" in scope
            )
        
        elif tier == "attn_upscale":
            # Just softmax
            return "softmax" in op_type.lower()
        
        elif tier == "ffn_swiglu":
            # Swiglu activation only
            return "swiglu" in op_type.lower() or "ffn" in scope
        
        elif tier == "ln":
            # Layernorm
            return "layer_norm" in op_type.lower() or "ln" in scope
        
        return False

    def _matching_tier(self, node: OpNode, tiers: set[str]) -> str:
        """Get the matching recompute tier for a node.
        
        Args:
            node: OpNode to check
            tiers: Set of recompute tiers
            
        Returns:
            Matching tier
        """
        # Check tiers in order of specificity
        for tier in ["attn_upscale", "ffn_swiglu", "ln", "attn", "full"]:
            if tier in tiers and self._matches_tier(node, tier):
                return tier
        return "full"
