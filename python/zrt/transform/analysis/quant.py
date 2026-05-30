"""Graph-side op dtype resolution — maps OpNode + GraphQuantProfile → dtype bundle.

Delegates component classification to ``ir/component_classifier.py`` and
bundle construction to ``training/models/quant_dispatch.py``, so the graph
and spec paths share a single source of truth for both.

Drift contract: new component buckets added to ``dispatch`` in
``quant_dispatch.py`` MUST also be added to ``classify`` in
``component_classifier.py``. The parity test ``test_routed_expert_parity``
only covers existing buckets — new ones need explicit graph↔spec tests.
"""
from __future__ import annotations

from zrt.training.models.quant import OpDtypeBundle
from zrt.training.spec.dtype import Dtype
from python.zrt.transform.context import GraphQuantProfile
from python.zrt.ir.node import OpNode

# Re-export classify as _classify_graph_component for backward compat
from python.zrt.ir.component_classifier import classify as _classify_graph_component


def graph_resolve_op_dtypes(
    node: OpNode,
    profile: GraphQuantProfile | None,
) -> OpDtypeBundle:
    """Return the dtype bundle for one graph node.

    When ``profile`` is ``None``, returns an all-BF16 bundle matching
    current default behavior.
    """
    if profile is None:
        return OpDtypeBundle(
            in_act=Dtype.BF16, weight=Dtype.BF16, out_act=Dtype.BF16,
            compute=Dtype.BF16, grad_in=Dtype.BF16, grad_weight=Dtype.BF16,
            grad_act=Dtype.BF16,
        )

    from zrt.training.models.quant_dispatch import dispatch
    comp = _classify_graph_component(node)
    return dispatch(comp, profile)
