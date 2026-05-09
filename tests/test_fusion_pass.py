"""FusionPass OpGraph IR integration — graph-contract tests only.

These tests verify that ``FusionPass`` preserves OpGraph invariants on
inputs that no rule should match (empty / single-op / no-scope / comm /
cross-layer).  Rich-rule matching, semantic annotations, and DSv4 rule
behavior are covered separately under ``tests/transform/fusion/``.

(Rule-specific cases formerly in this file relied on a deprecated
"semantic label auto-fusion" code path and were removed during the
v2 rich-rules refactor — see ``docs/fusion_v2_rich_rules_zh.md``.)
"""
from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import DType, TensorMeta
from python.zrt.transform.context import (
    ParallelConfig,
    StreamConfig,
    TransformContext,
)
from python.zrt.transform.fusion.api import FusionPass
import python.zrt.hardware.registry as hw_registry


# ── helpers ───────────────────────────────────────────────────────────────────

def _t(tid, shape=(1, 128), dtype=DType.BF16):
    return TensorMeta.from_shape_dtype(tid, shape, dtype)


def _node(nid, op_type, scope="", layer="0", module_class="", category="compute"):
    return OpNode(
        id=nid, op_type=op_type,
        inputs=[_t(f"{nid}_in")], outputs=[_t(f"{nid}_out")],
        scope=scope, layer=layer, module_class=module_class, category=category,
    )


def _edge(src, dst):
    return Edge(src=src, src_idx=0, dst=dst, dst_idx=0, tensor=_t("e"))


def _graph(nodes, edges, name="test"):
    return OpGraph(name=name, phase="prefill",
                   nodes={n.id: n for n in nodes},
                   edges=edges)


def _ctx(hw_name="nvidia_h100_sxm"):
    hw = hw_registry.load(hw_name)
    return TransformContext(hw_spec=hw, parallel=ParallelConfig(tp=1),
                            stream_config=StreamConfig())


# ── pass-through cases ────────────────────────────────────────────────────────

def test_empty_graph_returns_empty():
    g = OpGraph(name="t", phase="prefill")
    out = FusionPass().run(g, _ctx())
    assert out.num_nodes() == 0


def test_single_node_no_scope_unchanged():
    n = _node("a", "aten.mm.default", scope="")
    g = _graph([n], [])
    out = FusionPass().run(g, _ctx())
    assert out.num_nodes() == 1
    assert out.nodes["a"].op_type == "aten.mm.default"


def test_does_not_mutate_input():
    n1 = _node("a", "aten.mm.default", scope="model.layers.0.mlp")
    n2 = _node("b", "aten.add.Tensor", scope="model.layers.0.mlp")
    g = _graph([n1, n2], [_edge("a", "b")])
    original_count = g.num_nodes()
    FusionPass().run(g, _ctx())
    assert g.num_nodes() == original_count


def test_different_layer_nodes_not_fused():
    """Same scope but different layers must not be merged."""
    scope = "model.layers.X.mlp"
    a = _node("a", "aten.mm.default", scope=scope, layer="0", module_class="MLP")
    b = _node("b", "aten.mm.default", scope=scope, layer="1", module_class="MLP")
    g = _graph([a, b], [])
    out = FusionPass().run(g, _ctx())
    assert out.num_nodes() == 2


def test_comm_node_breaks_group():
    """A communication node between same-scope nodes splits them."""
    scope = "model.layers.0.mlp"
    a    = _node("a", "aten.mm.default", scope=scope, module_class="MLP")
    comm = _node("c", "comm.all_reduce", scope="", category="communication")
    b    = _node("b", "aten.mm.default", scope=scope, module_class="MLP")
    g = _graph([a, comm, b], [_edge("a", "c"), _edge("c", "b")])

    out = FusionPass().run(g, _ctx())
    comm_nodes = [n for n in out.nodes.values() if n.category == "communication"]
    assert len(comm_nodes) == 1
    assert comm_nodes[0].op_type == "comm.all_reduce"


def test_different_scope_nodes_not_fused():
    """Different scopes never merge."""
    a = _node("a", "aten.mm.default", scope="model.layers.0.q_proj", module_class="Linear")
    b = _node("b", "aten.mm.default", scope="model.layers.0.k_proj", module_class="Linear")
    g = _graph([a, b], [])
    out = FusionPass().run(g, _ctx())
    # No rule fuses single-op groups across scopes; both survive.
    assert out.num_nodes() >= 2
