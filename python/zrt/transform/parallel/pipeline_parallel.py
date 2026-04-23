from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Set

from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode, Edge
from zrt.ir.types import TensorMeta
from zrt.transform.base import GraphPass
from zrt.transform.context import TransformContext


@dataclass
class LayerInfo:
    """Layer information for pipeline parallel partitioning."""
    layer_id: int
    layer_kind: str
    total_flops: float
    node_ids: Set[str]


class PipelineParallelPass(GraphPass):
    """Pipeline Parallel pass for layer assignment and P2P insertion."""
    name = "pipeline_parallel"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        """Run pipeline parallel pass on the graph.
        
        Args:
            graph: Input OpGraph
            ctx: TransformContext with parallel config and training config
            
        Returns:
            New OpGraph with pipeline parallel annotations and P2P nodes
        """
        if ctx.parallel.pp <= 1:
            return graph
        
        g = graph.clone()
        pp = ctx.parallel.pp
        pp_layer_assignment = ctx.training.pp_layer_assignment if ctx.training else None

        # 1. Collect layer information from graph hierarchy
        layers = self._collect_layers(g)
        
        # 2. Partition layers into pipeline stages
        stages = self._partition_layers(layers, pp, pp_layer_assignment)
        
        # 3. Annotate each node with its pipeline stage
        for node in g.nodes.values():
            node.annotations["pp_stage"] = self._get_node_stage(node, stages)
        
        # 4. Insert P2P communication nodes at stage boundaries
        self._insert_p2p_communication(g, stages)
        
        # 5. Warn if stage balance is poor
        self._check_stage_balance(stages)
        
        return g

    def _collect_layers(self, graph: OpGraph) -> List[LayerInfo]:
        """Collect layer information from the graph hierarchy.
        
        Args:
            graph: OpGraph to collect layers from
            
        Returns:
            List of LayerInfo objects
        """
        layers = []
        layer_id = 0
        
        # Get layers from hierarchy at depth 2 (assuming depth 0 is full graph, 1 is major components)
        for layer_node in graph.hierarchy.at_depth(2):
            if "layer" in layer_node.scope.lower():
                node_ids = set(layer_node.leaf_node_ids)
                total_flops = sum(
                    node.annotations.get("flops", 0) 
                    for node_id in node_ids 
                    if node_id in graph.nodes
                )
                
                layer_kind = "dense"
                if "moe" in layer_node.scope.lower():
                    layer_kind = "moe"
                elif "mtp" in layer_node.scope.lower():
                    layer_kind = "mtp"
                
                layers.append(LayerInfo(
                    layer_id=layer_id,
                    layer_kind=layer_kind,
                    total_flops=total_flops,
                    node_ids=node_ids
                ))
                layer_id += 1
        
        return layers

    def _partition_layers(self, layers: List[LayerInfo], pp: int, 
                         pp_layer_assignment: List[int] | None) -> List[List[LayerInfo]]:
        """Partition layers into pipeline stages.
        
        Args:
            layers: List of LayerInfo objects
            pp: Number of pipeline stages
            pp_layer_assignment: Explicit layer assignment or None for automatic
            
        Returns:
            List of stages, each containing a list of LayerInfo objects
        """
        stages = [[] for _ in range(pp)]
        
        if pp_layer_assignment:
            # Use explicit assignment
            for layer_idx, stage_idx in enumerate(pp_layer_assignment):
                if layer_idx < len(layers) and stage_idx < pp:
                    stages[stage_idx].append(layers[layer_idx])
        else:
            # Use greedy bin-packing to balance stages
            stage_flops = [0.0] * pp
            
            for layer in layers:
                # Find stage with minimum flops
                min_stage_idx = stage_flops.index(min(stage_flops))
                stages[min_stage_idx].append(layer)
                stage_flops[min_stage_idx] += layer.total_flops
        
        return stages

    def _get_node_stage(self, node: OpNode, stages: List[List[LayerInfo]]) -> int:
        """Get the pipeline stage for a node.
        
        Args:
            node: OpNode to find stage for
            stages: List of stages
            
        Returns:
            Pipeline stage index
        """
        for stage_idx, stage_layers in enumerate(stages):
            for layer in stage_layers:
                if node.id in layer.node_ids:
                    return stage_idx
        return 0  # Default to stage 0 if not found

    def _insert_p2p_communication(self, graph: OpGraph, stages: List[List[LayerInfo]]):
        """Insert P2P communication nodes at stage boundaries.
        
        Args:
            graph: OpGraph to modify
            stages: List of stages
        """
        # Get all nodes in each stage
        stage_nodes = []
        for stage in stages:
            stage_node_ids = set()
            for layer in stage:
                stage_node_ids.update(layer.node_ids)
            stage_nodes.append(stage_node_ids)
        
        # Insert P2P between stages
        for i in range(len(stages) - 1):
            current_stage_nodes = stage_nodes[i]
            next_stage_nodes = stage_nodes[i + 1]
            
            # Find last node in current stage and first node in next stage
            last_node = None
            first_node = None
            
            # Use topological order to find last node in current stage
            for node in graph.topo_sort():
                if node.id in current_stage_nodes:
                    last_node = node
                elif node.id in next_stage_nodes and first_node is None:
                    first_node = node
                    break
            
            if last_node and first_node:
                # Create P2P communication node
                p2p_node = OpNode(
                    id=f"comm_p2p_{i}_{i+1}",
                    op_type="comm.send_recv",
                    inputs=last_node.outputs.copy(),
                    outputs=first_node.inputs.copy(),
                    attrs={
                        "src_stage": i,
                        "dst_stage": i + 1,
                        "message_size": sum(t.memory_bytes for t in last_node.outputs)
                    },
                    scope=f"pipeline.p2p.{i}_{i+1}",
                    category="communication"
                )
                
                # Insert P2P node between last_node and first_node
                # This is a simplified implementation - in practice, you'd need to rewire edges
                graph.insert_after(last_node.id, p2p_node, [])

    def _check_stage_balance(self, stages: List[List[LayerInfo]]):
        """Check if stage balance is acceptable.
        
        Args:
            stages: List of stages
        """
        if not stages:
            return
        
        stage_flops = []
        for stage in stages:
            total_flops = sum(layer.total_flops for layer in stage)
            stage_flops.append(total_flops)
        
        if stage_flops:
            max_flops = max(stage_flops)
            min_flops = min(stage_flops)
            if min_flops > 0:
                balance_ratio = max_flops / min_flops
                if balance_ratio > 1.1:
                    import warnings
                    warnings.warn(f"Pipeline stage balance ratio is {balance_ratio:.2f} > 1.1, consider adjusting layer assignment")
