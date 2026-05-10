"""Fused-node construction: builder, IO resolver, annotation propagator."""
from __future__ import annotations

from .annotation_propagator import _propagated_annotations
from .io_resolver import (
    FusedIOPort,
    _external_io,
    resolve_io,
)
from .node_builder import FusedNodeBuilder

__all__ = [
    "FusedIOPort",
    "FusedNodeBuilder",
    "_external_io",
    "_propagated_annotations",
    "resolve_io",
]
