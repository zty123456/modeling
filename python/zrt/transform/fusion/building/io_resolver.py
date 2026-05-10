"""FusedIOPort + IO resolution.

Step-1 note: dataclass + ``resolve_io`` literally copied from the
original ``python/zrt/transform/fusion/provenance.py``; ``_external_io``
literally copied from ``algorithm.py``.  No behaviour change.

Downstream consumers (simulator cost models, report generators) use
``FusedIOPort`` to trace fused-node tensors back to the original aten ops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from python.zrt.transform.fusion.core.io_role import IOSpec


@dataclass(frozen=True)
class FusedIOPort:
    """Provenance of one tensor in a fused node's inputs or outputs."""

    role: str                                    # "activation" / "weight" / "output" / ...
    origin_node_id: str                          # source child OpNode.id
    origin_op_type: str                          # e.g. "aten.mm.default"
    origin_arg_index: int                        # which tensor slot
    origin_kind: Literal["input", "output"]      # taken from input or output side
    origin_dtype_from: str = "explicit"          # how dtype was determined


def resolve_io(
    child_ops: list,
    spec: "IOSpec",
) -> Optional[FusedIOPort]:
    """Build a FusedIOPort from an IOSpec and the child ops of a fusion group.

    Returns ``None`` if indices are out of range.
    """
    if not child_ops:
        return None

    op_idx = spec.source_op_index
    if op_idx < 0:
        op_idx = len(child_ops) + op_idx
    if op_idx < 0 or op_idx >= len(child_ops):
        return None

    op = child_ops[op_idx]

    arg_idx = spec.source_arg_index
    tensors = op.inputs if spec.source_kind == "input" else op.outputs
    if arg_idx < 0:
        arg_idx = len(tensors) + arg_idx
    if arg_idx < 0 or arg_idx >= len(tensors):
        return None

    return FusedIOPort(
        role=spec.role,
        origin_node_id=op.id,
        origin_op_type=op.op_type,
        origin_arg_index=arg_idx,
        origin_kind=spec.source_kind,
    )


def _external_io(
    graph,
    group_ids: set,
) -> tuple[list, list]:
    """Return (ext_inputs, ext_outputs) for a node group.

    Step-1 note: body literally copied from
    ``algorithm._external_io``.
    """
    seen_in: set = set()
    seen_out: set = set()
    inputs: list = []
    outputs: list = []

    for e in graph.edges:
        if e.src not in group_ids and e.dst in group_ids:
            key = (e.tensor_id, e.dst_idx)
            if key not in seen_in and e.tensor is not None:
                seen_in.add(key)
                inputs.append(e.tensor)
        if e.src in group_ids and e.dst not in group_ids:
            key = (e.tensor_id, e.src_idx)
            if key not in seen_out and e.tensor is not None:
                seen_out.add(key)
                outputs.append(e.tensor)

    return inputs, outputs
