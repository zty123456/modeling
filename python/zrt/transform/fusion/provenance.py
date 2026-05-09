"""FusedIOPort: records where each IO tensor of a fused node comes from.

Downstream consumers (simulator cost models, report generators) use this
to trace fused-node tensors back to the original aten ops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from .rule import IOSpec


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
