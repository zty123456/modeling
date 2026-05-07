"""Tests for SparseAttnSharedKV (npu_sas) fusion rules and annotation pass."""
import pytest
from python.zrt.ir.node import OpNode
from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.types import TensorMeta, DType
from python.zrt.transform.fusion.pass_ import FusionPass
from python.zrt.transform.fusion.sas_pass import SparseAttnSharedKVPass
from python.zrt.transform.context import TransformContext, ParallelConfig, StreamConfig
import python.zrt.hardware.registry as hw_registry


# ── shared helpers ────────────────────────────────────────────────────────────

def _t(tid, shape=(1, 128), dtype=DType.BF16):
    return TensorMeta.from_shape_dtype(tid, shape, dtype)


def _node(nid, op_type, scope="", layer="0", module_class=""):
    return OpNode(
        id=nid, op_type=op_type,
        inputs=[_t(f"{nid}_in")], outputs=[_t(f"{nid}_out")],
        scope=scope, layer=layer, module_class=module_class, category="compute",
    )


def _edge(src, dst):
    return Edge(src=src, src_idx=0, dst=dst, dst_idx=0, tensor=_t("e"))


def _graph(nodes, edges, name="test", metadata=None):
    return OpGraph(name=name, phase="prefill",
                   nodes={n.id: n for n in nodes},
                   edges=edges, metadata=metadata)


def _ctx(hw_name="ascend_910b"):
    hw = hw_registry.load(hw_name)
    return TransformContext(hw_spec=hw, parallel=ParallelConfig(tp=1),
                            stream_config=StreamConfig())


# ── npu_sas inference pattern ─────────────────────────────────────────────────

def test_inference_gather_pattern_ascend():
    """gather → bmm → softmax → bmm in Attention scope on ascend → 'npu_sas'."""
    scope = "transformer.layers.0.attn"
    mc = "DeepseekV4Attention"
    ops = [
        ("gather", "aten.gather.default"),
        ("bmm1",   "aten.bmm.default"),
        ("sfmx",   "aten.softmax.int"),
        ("bmm2",   "aten.bmm.default"),
    ]
    nodes = [_node(nid, op, scope=scope, layer="0", module_class=mc)
             for nid, op in ops]
    edges = [_edge(ops[i][0], ops[i+1][0]) for i in range(len(ops)-1)]
    g = _graph(nodes, edges)

    out = FusionPass().run(g, _ctx("ascend_910b"))
    fused = next(iter(out.nodes.values()))
    assert fused.op_type == "npu_sas", f"Expected 'npu_sas', got '{fused.op_type}'"


# ── npu_sas training pattern (Compressor gated-pooling) ──────────────────────

def test_training_compressor_class_ascend():
    """Any Compressor-class ops on ascend → 'npu_sas' (class-only match).

    The V4 Compressor performs gated KV pooling mapped to npu_sparse_attn_sharedkv.
    Its ops are split across multiple topo-sort groups due to interleaving with
    the norm sub-module, so class-name matching (empty op_seq) is used.
    """
    scope = "transformer.layers.0.attn.compressor"
    mc = "Compressor"
    ops = [
        ("add", "aten.add.Tensor"),
        ("sfmx", "aten._softmax.default"),
        ("mul",  "aten.mul.Tensor"),
    ]
    nodes = [_node(nid, op, scope=scope, layer="0", module_class=mc)
             for nid, op in ops]
    edges = [_edge(ops[i][0], ops[i+1][0]) for i in range(len(ops)-1)]
    g = _graph(nodes, edges)

    out = FusionPass().run(g, _ctx("ascend_910b"))
    fused = next(iter(out.nodes.values()))
    assert fused.op_type == "npu_sas", f"Expected 'npu_sas', got '{fused.op_type}'"


# ── npu_sas not triggered on CUDA ─────────────────────────────────────────────

def test_sas_not_triggered_on_cuda():
    """gather→bmm→softmax→bmm on H100 → 'v4_sparse_attn', not 'npu_sas'."""
    scope = "transformer.layers.0.attn"
    mc = "DeepseekV4Attention"
    ops = [
        ("gather", "aten.gather.default"),
        ("bmm1",   "aten.bmm.default"),
        ("sfmx",   "aten.softmax.int"),
        ("bmm2",   "aten.bmm.default"),
    ]
    nodes = [_node(nid, op, scope=scope, layer="0", module_class=mc)
             for nid, op in ops]
    edges = [_edge(ops[i][0], ops[i+1][0]) for i in range(len(ops)-1)]
    g = _graph(nodes, edges)

    out = FusionPass().run(g, _ctx("nvidia_h100_sxm"))
    fused = next(iter(out.nodes.values()))
    assert fused.op_type == "v4_sparse_attn", (
        f"Expected 'v4_sparse_attn' on CUDA, got '{fused.op_type}'"
    )


# ── SparseAttnSharedKVPass annotation ────────────────────────────────────────

def test_sas_pass_annotates_attn_type():
    """SparseAttnSharedKVPass sets attn_type/compress_ratio from graph.metadata."""
    # compress_ratios: layer 0 → 128 (HCA), layer 1 → 4 (CSA), layer 2 → 0 (SWA)
    compress_ratios = [128, 4, 0]

    nodes = []
    for layer_idx, (cr, expected_type) in enumerate(
            zip(compress_ratios, ["HCA", "CSA", "SWA"])):
        scope = f"transformer.layers.{layer_idx}.attn"
        n = _node(f"attn_{layer_idx}", "npu_sas",
                  scope=scope, layer=str(layer_idx),
                  module_class="DeepseekV4Attention")
        nodes.append(n)

    g = _graph(nodes, [], metadata={"compress_ratios": compress_ratios})

    ctx = _ctx("ascend_910b")
    out = SparseAttnSharedKVPass().run(g, ctx)

    expected = [("HCA", 128), ("CSA", 4), ("SWA", 0)]
    for (node_id, node) in out.nodes.items():
        layer_idx = int(node.scope.split("layers.")[1].split(".")[0])
        exp_type, exp_cr = expected[layer_idx]
        assert node.annotations.get("attn_type") == exp_type, (
            f"layer {layer_idx}: expected attn_type={exp_type!r}, "
            f"got {node.annotations.get('attn_type')!r}"
        )
        assert node.annotations.get("compress_ratio") == exp_cr


def test_sas_pass_skips_without_compress_ratios():
    """SparseAttnSharedKVPass is a no-op when compress_ratios not in metadata."""
    scope = "transformer.layers.0.attn"
    n = _node("a", "npu_sas", scope=scope, layer="0",
              module_class="DeepseekV4Attention")
    g = _graph([n], [])  # no compress_ratios in metadata

    ctx = _ctx("ascend_910b")
    out = SparseAttnSharedKVPass().run(g, ctx)
    node = next(iter(out.nodes.values()))
    assert "attn_type" not in node.annotations
