"""FusionPass: graph transform that applies MRO-based operator fusion.

Uses the v2 algorithm which matches fusion rules via ``type(module).__mro__``
instead of regex patterns.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


class FusionPass(GraphPass):
    """Apply MRO-based fusion to an OpGraph.

    Rules are loaded from ``platforms/`` based on the model classes found
    in the OpGraph's node metadata.
    """

    name = "fusion"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        from python.zrt.transform.fusion.algorithm import fuse
        from python.zrt.transform.fusion.platforms import load_platform_rules
        load_platform_rules(getattr(ctx, "model_id", "") or "")
        return fuse(graph, ctx)
