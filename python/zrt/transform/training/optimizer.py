from __future__ import annotations

from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.edge import Edge
from zrt.transform.base import GraphPass
from zrt.transform.context import TransformContext


class OptimizerPass(GraphPass):
    """OptimizerPass for optimizer step annotation."""
    name = "optimizer"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        """Run OptimizerPass on the graph.
        
        Args:
            graph: Input OpGraph
            ctx: TransformContext with training config
            
        Returns:
            New OpGraph with optimizer step node
        """
        if graph.phase != "train_backward":
            return graph
        
        g = graph.clone()
        if not ctx.training:
            return g
        
        # Calculate total parameters on rank
        params = self._total_params_on_rank(g, ctx)
        opt = ctx.training.optimizer
        
        # Create optimizer step node
        step_node = OpNode(
            id="optimizer_step",
            op_type=f"optimizer.{opt}",
            inputs=[],
            outputs=[],
            attrs={
                "optimizer": opt,
                "params": params,
                "state_bytes": self._opt_state_bytes(opt, params),
                "step_flops": self._opt_step_flops(opt, params),
            },
            scope="optimizer.step",
            category="compute"
        )
        
        # Append the optimizer step node at the end of the graph
        self._append_at_end(g, step_node)
        
        return g

    def _total_params_on_rank(self, graph: OpGraph, ctx: TransformContext) -> int:
        """Calculate total parameters on rank.
        
        Args:
            graph: OpGraph to analyze
            ctx: TransformContext with parallel config
            
        Returns:
            Total parameters on rank
        """
        # This is a simplified implementation
        # In practice, you would need to analyze the graph to count parameters
        
        # Get model profile from context if available
        if hasattr(ctx, 'profile') and ctx.profile:
            total_params = ctx.profile.param_count()
        else:
            # Estimate based on common model sizes
            total_params = 70e9  # 70B parameters as default
        
        # Apply parallelism sharding
        tp = ctx.parallel.tp
        pp = ctx.parallel.pp
        dp = ctx.parallel.dp
        cp = ctx.parallel.cp
        
        # Calculate params per rank
        params_per_rank = total_params / (tp * pp * dp * cp)
        
        return int(params_per_rank)

    def _opt_state_bytes(self, optimizer: str, params: int) -> int:
        """Calculate optimizer state bytes.
        
        Args:
            optimizer: Optimizer name
            params: Number of parameters
            
        Returns:
            Optimizer state bytes
        """
        if optimizer == "adam":
            # Adam: 8 bytes per parameter (2 * 4 bytes for momentums)
            return params * 8
        elif optimizer == "muon":
            # Muon: 4 bytes per parameter + scratch space
            return params * 4 + params * 2  # 2 bytes for scratch
        else:
            # Default: 4 bytes per parameter
            return params * 4

    def _opt_step_flops(self, optimizer: str, params: int) -> int:
        """Calculate optimizer step FLOPs.
        
        Args:
            optimizer: Optimizer name
            params: Number of parameters
            
        Returns:
            Optimizer step FLOPs
        """
        if optimizer == "adam":
            # Adam: ~12 FLOPs per parameter
            return params * 12
        elif optimizer == "muon":
            # Muon: ~16 FLOPs per parameter (includes Newton-Schulz)
            return params * 16
        else:
            # Default: ~8 FLOPs per parameter
            return params * 8

    def _append_at_end(self, graph: OpGraph, new_node: OpNode):
        """Append a node at the end of the graph.
        
        Args:
            graph: OpGraph to modify
            new_node: OpNode to append
        """
        # Find all nodes with no successors
        end_nodes = []
        for node_id, node in graph.nodes.items():
            has_successors = False
            for edge in graph.edges:
                if edge.src == node_id:
                    has_successors = True
                    break
            if not has_successors:
                end_nodes.append(node)
        
        # If there are end nodes, connect them to the new node
        if end_nodes:
            # Create edges from all end nodes to the new node
            for end_node in end_nodes:
                # Use the first output of the end node as input to the new node
                if end_node.outputs:
                    edge = Edge(
                        src=end_node.id,
                        src_idx=0,
                        dst=new_node.id,
                        dst_idx=0,
                        tensor=end_node.outputs[0]
                    )
                    graph.edges.append(edge)
        
        # Add the new node to the graph
        graph.nodes[new_node.id] = new_node
