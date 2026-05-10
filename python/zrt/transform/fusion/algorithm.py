"""Backward-compat shim for ``python.zrt.transform.fusion.algorithm``.

The implementation was split across:

* ``bucketing/call_id_bucketer.py`` — ``FusionGroup``, ``bucket_into_groups``
* ``building/node_builder.py``       — ``build_fused_node``, ``_build_collapsed_node``
* ``building/io_resolver.py``        — ``_external_io``
* ``building/annotation_propagator.py`` — ``_propagated_annotations``
* ``pipeline/fuser.py``              — ``fuse``
* ``pipeline/compositors.py``        — ``_compose_add_norm``

This module re-exports the public surface so existing imports keep
working.
"""
from __future__ import annotations

from .building.annotation_propagator import (  # noqa: F401
    _DICT_PROPAGATE_KEYS,
    _SCALAR_PROPAGATE_KEYS,
    _propagated_annotations,
)
from .building.io_resolver import _external_io  # noqa: F401
from .building.node_builder import (  # noqa: F401
    _build_collapsed_node,
    build_fused_node,
)
from .bucketing.call_id_bucketer import (  # noqa: F401
    FusionGroup,
    _extract_class_obj,
    _merge_parent_groups,
    _merge_two_groups,
    _parent,
    bucket_into_groups,
    bucket_nodes_by_leaf_module,
)
from .pipeline.compositors import _compose_add_norm  # noqa: F401
from .pipeline.fuser import fuse  # noqa: F401

# Module-level constant kept for legacy imports.
_ADD_NORM_RULE_NAME = "add_norm"

__all__ = [
    "FusionGroup",
    "bucket_into_groups",
    "bucket_nodes_by_leaf_module",
    "build_fused_node",
    "fuse",
]
