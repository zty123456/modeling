"""Parameter counting utilities for OpGraph IR.

Shared by inference reports and training analysis passes.
Three-tier strategy: metadata → name heuristic → structural fallback.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph

# Short op-name tokens that consume weight matrices.
# Mapped to the first input index that is a weight (inputs before it are activations/bias).
_MATMUL_WEIGHT_START: dict[str, int] = {
    "mm": 1, "matmul": 1, "linear": 1, "bmm": 1, "baddbmm": 2, "addmm": 2,
}
# Embedding ops: input[0] is the weight table, input[1] is the index tensor
_EMBED_OPS: frozenset[str] = frozenset({"embedding"})


def op_short(op_type: str) -> str:
    """Extract the short token from a qualified op name like 'aten.mm.default' -> 'mm'."""
    parts = op_type.split(".")
    return parts[1] if len(parts) >= 2 else parts[0]


def count_params(graph: OpGraph) -> int:
    """Count model parameters from an OpGraph.

    Three-tier strategy, tried in order:
    1. graph.metadata["total_params"] — authoritative when set by a model loader
    2. Name heuristic — tensor IDs containing "weight" or "param" (synthetic graphs)
    3. Structural fallback — external 2-D inputs to matmul/embedding ops, skipping
       activation positions that differ per op type (captured graphs use opaque IDs)
    """
    if graph.metadata.get("total_params", 0) > 0:
        return int(graph.metadata["total_params"])

    counted_ids: set[str] = set()
    name_total = 0
    for node in graph.nodes.values():
        if node.category == "compute":
            for inp in node.inputs:
                if inp.id in counted_ids:
                    continue
                if ("weight" in inp.id or "param" in inp.id) and inp.shape:
                    counted_ids.add(inp.id)
                    name_total += math.prod(inp.shape)
    if name_total > 0:
        return name_total

    produced_ids: set[str] = set()
    for node in graph.nodes.values():
        for out in node.outputs:
            produced_ids.add(out.id)

    counted_ids = set()
    struct_total = 0
    for node in graph.nodes.values():
        if node.category != "compute":
            continue
        short = op_short(node.op_type)
        weight_start = _MATMUL_WEIGHT_START.get(short)
        is_embed = short in _EMBED_OPS
        if weight_start is None and not is_embed:
            continue

        for i, inp in enumerate(node.inputs):
            if inp.id in produced_ids or inp.id in counted_ids:
                continue
            if weight_start is not None and i < weight_start:
                continue
            if is_embed and i > 0:
                continue  # only input[0] is the embedding table
            if inp.shape and len(inp.shape) == 2:
                counted_ids.add(inp.id)
                struct_total += math.prod(inp.shape)
    return struct_total
