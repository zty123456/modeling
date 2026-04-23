from __future__ import annotations

from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.edge import Edge
from zrt.transform.base import GraphPass
from zrt.transform.context import TransformContext


class ZeroFSDPPass(GraphPass):
    """ZeroFSDPPass for ZeRO/FSDP sharding annotations."""
    name = "zero_fsdp"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        """Run ZeroFSDPPass on the graph.
        
        Args:
            graph: Input OpGraph
            ctx: TransformContext with training config
            
        Returns:
            New OpGraph with Zero/FSDP annotations
        """
        g = graph.clone()
        if not ctx.training:
            return g
        
        z = ctx.training.zero_stage
        dp = ctx.parallel.dp
        
        # shard factors per memory bucket
        g.metadata["zero"] = {
            "stage": z,
            "weight_shard": dp if z >= 3 else 1,
            "grad_shard": dp if z >= 2 else 1,
            "optstate_shard": dp if z >= 1 else 1,
        }
        
        # FSDP-3 also adds a per-layer AG during fwd/bwd — insert those collectives
        if z >= 3:
            self._insert_fsdp_ag(g, ctx)
        
        return g

    def _insert_fsdp_ag(self, graph: OpGraph, ctx: TransformContext):
        """Insert FSDP all-gather collectives for ZeRO-3.
        
        Args:
            graph: OpGraph to modify
            ctx: TransformContext with training config
        """
        dp = ctx.parallel.dp
        
        # For each layer, insert all-gather before the layer and reduce-scatter after
        layers = self._get_layers(graph)
        
        for layer in layers:
            # Find first and last nodes in the layer
            first_node = None
            last_node = None
            
            for node in graph.topo_sort():
                if self._node_in_layer(node, layer):
                    if first_node is None:
                        first_node = node
                    last_node = node
            
            if first_node and last_node:
                # Insert all-gather before the first node
                ag_id = f"comm_fsdp_ag_{first_node.id}"
                ag_node = OpNode(
                    id=ag_id,
                    op_type="comm.all_gather",
                    inputs=first_node.inputs.copy(),
                    outputs=first_node.inputs.copy(),
                    attrs={
                        "group_size": dp,
                        "collective": "all_gather",
                        "role": "fsdp_ag"
                    },
                    scope=first_node.scope,
                    category="communication"
                )
                ag_node.annotations["inserted_by"] = "zero_fsdp_pass"
                self._prepend_comm(graph, first_node.id, ag_node)
                
                # Insert reduce-scatter after the last node
                rs_id = f"comm_fsdp_rs_{last_node.id}"
                rs_node = OpNode(
                    id=rs_id,
                    op_type="comm.reduce_scatter",
                    inputs=last_node.outputs.copy(),
                    outputs=last_node.outputs.copy(),
                    attrs={
                        "group_size": dp,
                        "collective": "reduce_scatter",
                        "role": "fsdp_rs"
                    },
                    scope=last_node.scope,
                    category="communication"
                )
                rs_node.annotations["inserted_by"] = "zero_fsdp_pass"
                self._rewire(graph, last_node.id, rs_node)

    def _get_layers(self, graph: OpGraph) -> list[str]:
        """Get layer scopes from the graph.
        
        Args:
            graph: OpGraph to analyze
            
        Returns:
            List of layer scopes
        """
        layers = set()
        for node in graph.nodes.values():
            scope = node.scope
            # Extract layer scope (assuming format like "model.layers.0.self_attn")
            parts = scope.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    layer_scope = ".".join(parts[:i+2])
                    layers.add(layer_scope)
                    break
        return list(layers)

    def _node_in_layer(self, node: OpNode, layer_scope: str) -> bool:
        """Check if a node is in a layer.
        
        Args:
            node: OpNode to check
            layer_scope: Layer scope to check against
            
        Returns:
            True if node is in the layer, False otherwise
        """
        return node.scope.startswith(layer_scope)

    def _prepend_comm(self, g: OpGraph, dst_id: str, comm_node: OpNode):
        """Insert comm_node between all predecessors of dst_id and dst_id itself.
        
        Args:
            g: OpGraph to modify
            dst_id: ID of the destination node
            comm_node: Communication node to insert
        """
        in_edges = [e for e in g.edges if e.dst == dst_id]
        g.edges = [e for e in g.edges if e.dst != dst_id]

        # Add the comm node
        g.nodes[comm_node.id] = comm_node

        # predecessors → comm
        for e in in_edges:
            g.edges.append(Edge(
                src=e.src, src_idx=e.src_idx,
                dst=comm_node.id, dst_idx=e.dst_idx,
                tensor=e.tensor,
            ))

        # comm → dst
        for i, out_tensor in enumerate(comm_node.outputs):
            g.edges.append(Edge(
                src=comm_node.id, src_idx=i,
                dst=dst_id, dst_idx=i,
                tensor=out_tensor,
            ))

        g._rebuild_adjacency()

    def _rewire(self, g: OpGraph, src_id: str, comm_node: OpNode):
        """Insert comm_node between src_id and all its current successors.
        
        Args:
            g: OpGraph to modify
            src_id: ID of the source node
            comm_node: Communication node to insert
        """
        # Collect out-edges of src that we need to reroute
        old_out = [e for e in g.edges if e.src == src_id]

        # Remove those edges from the graph
        g.edges = [e for e in g.edges if e.src != src_id]

        # Add the comm node
        g.nodes[comm_node.id] = comm_node

        # src → comm (one edge per output slot of src)
        src_node = g.nodes[src_id]
        for i, out_tensor in enumerate(src_node.outputs):
            g.edges.append(Edge(
                src=src_id, src_idx=i,
                dst=comm_node.id, dst_idx=i,
                tensor=out_tensor,
            ))

        # comm → old successors
        for e in old_out:
            g.edges.append(Edge(
                src=comm_node.id, src_idx=e.src_idx,
                dst=e.dst, dst_idx=e.dst_idx,
                tensor=e.tensor,
            ))

        g._rebuild_adjacency()
