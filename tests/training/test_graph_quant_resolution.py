"""Tests for graph-side op dtype resolution — parity with spec resolve_op_dtypes."""
import pytest
from python.zrt.ir.node import OpNode
from python.zrt.transform.context import GraphQuantProfile
from python.zrt.transform.analysis.quant import (
    graph_resolve_op_dtypes, _classify_graph_component,
)
from zrt.training.spec.dtype import Dtype


def mk_node(component="", scope="", op_type="aten.mm.default"):
    return OpNode(id="test", op_type=op_type, component=component, scope=scope)


class TestComponentClassification:
    @pytest.mark.parametrize("comp,expected", [
        ("attn.q_proj", "attention"),
        ("attn.score", "attention"),
        ("attn.softmax", "attention"),
        ("attn.rope", "attention"),
        ("attn_norm", "norm"),
        ("ffn_norm", "norm"),
        ("final_norm", "norm"),
        ("moe.experts.gate_proj", "routed_expert"),
        ("moe.experts.up_proj", "routed_expert"),
        ("moe.shared.gate_proj", "shared_expert"),
        ("moe.shared.down_proj", "shared_expert"),
        ("embedding", "embedding"),
        ("lm_head", "embedding"),
        ("ffn.gate_proj", "default"),
        ("moe.gate.topk", "default"),
        ("hc.pre_attn", "default"),
    ])
    def test_component_classification(self, comp, expected):
        assert _classify_graph_component(mk_node(comp)) == expected

    def test_scope_based_attention(self):
        n = mk_node("", scope="model.layers.0.self_attn.q_proj")
        assert _classify_graph_component(n) == "attention"

    def test_scope_based_expert(self):
        n = mk_node("", scope="model.layers.1.block_sparse_moe.experts.0.gate_proj")
        assert _classify_graph_component(n) == "routed_expert"

    def test_fused_norm_op_type(self):
        for ot in ("rms_norm", "layer_norm", "add_rms_norm", "gemma_norm", "rms_gated"):
            assert _classify_graph_component(mk_node("unknown", op_type=ot)) == "norm"

    def test_fused_attn_op_type(self):
        for ot in ("scaled_dot_product_attention", "flash_attn", "mla_attn"):
            assert _classify_graph_component(mk_node("unknown", op_type=ot)) == "attention"

    def test_expert_grouped_mm_annotation(self):
        n = mk_node("ffn.gate_proj")
        n.annotations["fused_by"] = "expert_grouped_mm"
        assert _classify_graph_component(n) == "routed_expert"


class TestDtypeResolution:
    def test_none_profile_all_bf16(self):
        for comp in ("attn.q_proj", "moe.experts.gate_proj", "embedding"):
            b = graph_resolve_op_dtypes(mk_node(comp), None)
            assert b.in_act == Dtype.BF16
            assert b.compute == Dtype.BF16

    def test_bf16_profile(self):
        p = GraphQuantProfile()
        b = graph_resolve_op_dtypes(mk_node("attn.q_proj"), p)
        assert b.compute == Dtype.BF16

    def test_fp8_mixed_attention_stays_bf16(self):
        p = GraphQuantProfile.from_scalar("fp8")
        b = graph_resolve_op_dtypes(mk_node("attn.q_proj"), p)
        assert b.compute == Dtype.BF16
        assert b.in_act == Dtype.BF16

    def test_fp8_mixed_expert_fp8(self):
        p = GraphQuantProfile.from_scalar("fp8")
        b = graph_resolve_op_dtypes(mk_node("moe.experts.gate_proj"), p)
        assert b.compute == Dtype.FP8_E4M3
        assert b.in_act == Dtype.FP8_E4M3
        assert b.weight == Dtype.BF16

    def test_dsv4_fp4_weight(self):
        p = GraphQuantProfile.from_preset("deepseek_v4_fp8_fp4")
        b = graph_resolve_op_dtypes(mk_node("moe.experts.gate_proj"), p)
        assert b.weight == Dtype.FP4
        assert b.weight.stored_bytes == 0.5625

    def test_shared_expert_stays_bf16_in_fp8_mixed(self):
        p = GraphQuantProfile.from_scalar("fp8")
        b = graph_resolve_op_dtypes(mk_node("moe.shared.gate_proj"), p)
        assert b.compute == Dtype.BF16

    def test_embedding_always_bf16(self):
        p = GraphQuantProfile.from_scalar("fp4")
        b = graph_resolve_op_dtypes(mk_node("embedding"), p)
        assert b.compute == Dtype.BF16
        assert b.weight == Dtype.BF16

    def test_norm_always_bf16(self):
        p = GraphQuantProfile.from_scalar("fp4")
        b = graph_resolve_op_dtypes(mk_node("attn_norm"), p)
        assert b.compute == Dtype.BF16


class TestSpecParity:
    """Verify graph and spec resolve_op_dtypes produce matching bundles."""

    def _spec_op(self, component):
        return type("Op", (), {
            "component": component, "kind": "matmul",
            "inputs": [], "outputs": [], "meta": {},
        })()

    def _assert_bundles_match(self, spec_bundle, graph_bundle, label):
        for slot in ("in_act", "weight", "out_act", "compute", "grad_in", "grad_weight", "grad_act"):
            assert getattr(spec_bundle, slot) == getattr(graph_bundle, slot), \
                f"{label}: mismatch on {slot}: spec={getattr(spec_bundle, slot)} graph={getattr(graph_bundle, slot)}"

    def test_routed_expert_parity(self):
        from zrt.training.models.quant import resolve_op_dtypes
        from zrt.training.spec.model import ModelSpec, LayerKind

        model = ModelSpec(
            hidden=7168, ffn=2048, num_heads=32, num_kv_heads=32, head_dim=128,
            vocab=128256, seq_len=128, layers=[LayerKind.MOE],
            num_experts=8, moe_ffn=2048, top_k=2,
            routed_expert_compute_dtype=Dtype.FP8_E4M3,
            moe_act_dtype=Dtype.FP8_E4M3,
        )
        graph_profile = GraphQuantProfile.from_scalar("fp8")

        spec_bundle = resolve_op_dtypes(self._spec_op("routed_expert"), model)
        graph_bundle = graph_resolve_op_dtypes(mk_node("moe.experts.gate_proj"), graph_profile)

        assert spec_bundle.in_act == graph_bundle.in_act == Dtype.FP8_E4M3
        assert spec_bundle.compute == graph_bundle.compute == Dtype.FP8_E4M3
        assert spec_bundle.weight == graph_bundle.weight == Dtype.BF16

    def test_attention_parity_bf16(self):
        from zrt.training.models.quant import resolve_op_dtypes
        from zrt.training.spec.model import ModelSpec, LayerKind

        model = ModelSpec(
            hidden=4096, ffn=11008, num_heads=32, num_kv_heads=32, head_dim=128,
            vocab=32000, seq_len=128, layers=[LayerKind.DENSE],
        )
        graph_profile = GraphQuantProfile()

        spec_bundle = resolve_op_dtypes(self._spec_op("attention"), model)
        graph_bundle = graph_resolve_op_dtypes(mk_node("attn.q_proj"), graph_profile)

        self._assert_bundles_match(spec_bundle, graph_bundle, "attention_bf16")

    def test_shared_expert_parity(self):
        from zrt.training.models.quant import resolve_op_dtypes
        from zrt.training.spec.model import ModelSpec, LayerKind

        model = ModelSpec(
            hidden=7168, ffn=2048, num_heads=32, num_kv_heads=32, head_dim=128,
            vocab=128256, seq_len=128, layers=[LayerKind.MOE],
            num_experts=8, moe_ffn=2048, top_k=2, n_shared_experts=1,
            shared_expert_compute_dtype=Dtype.BF16,
            moe_act_dtype=Dtype.FP8_E4M3,
        )
        graph_profile = GraphQuantProfile.from_scalar("fp8")

        spec_bundle = resolve_op_dtypes(self._spec_op("shared_expert"), model)
        graph_bundle = graph_resolve_op_dtypes(mk_node("moe.shared.gate_proj"), graph_profile)

        self._assert_bundles_match(spec_bundle, graph_bundle, "shared_expert_fp8")

    def test_embedding_parity(self):
        from zrt.training.models.quant import resolve_op_dtypes
        from zrt.training.spec.model import ModelSpec, LayerKind

        model = ModelSpec(
            hidden=4096, ffn=11008, num_heads=32, num_kv_heads=32, head_dim=128,
            vocab=32000, seq_len=128, layers=[LayerKind.DENSE],
        )
        graph_profile = GraphQuantProfile()

        spec_bundle = resolve_op_dtypes(self._spec_op("embedding"), model)
        graph_bundle = graph_resolve_op_dtypes(mk_node("embedding"), graph_profile)

        self._assert_bundles_match(spec_bundle, graph_bundle, "embedding")
        # Embedding must always be BF16
        assert spec_bundle.compute == Dtype.BF16

    def test_norm_parity(self):
        from zrt.training.models.quant import resolve_op_dtypes
        from zrt.training.spec.model import ModelSpec, LayerKind

        model = ModelSpec(
            hidden=4096, ffn=11008, num_heads=32, num_kv_heads=32, head_dim=128,
            vocab=32000, seq_len=128, layers=[LayerKind.DENSE],
        )
        graph_profile = GraphQuantProfile()

        spec_bundle = resolve_op_dtypes(self._spec_op("norm"), model)
        graph_bundle = graph_resolve_op_dtypes(mk_node("attn_norm"), graph_profile)

        self._assert_bundles_match(spec_bundle, graph_bundle, "norm")
        assert spec_bundle.compute == Dtype.BF16

    def test_default_parity(self):
        from zrt.training.models.quant import resolve_op_dtypes
        from zrt.training.spec.model import ModelSpec, LayerKind

        model = ModelSpec(
            hidden=4096, ffn=11008, num_heads=32, num_kv_heads=32, head_dim=128,
            vocab=32000, seq_len=128, layers=[LayerKind.DENSE],
        )
        graph_profile = GraphQuantProfile()

        spec_bundle = resolve_op_dtypes(self._spec_op("ffn.gate_proj"), model)
        graph_bundle = graph_resolve_op_dtypes(mk_node("ffn.gate_proj"), graph_profile)

        self._assert_bundles_match(spec_bundle, graph_bundle, "default")


# ── Regression tests for post-review bug fixes ────────────────────────────────

class TestRingP2PClassification:
    """Bug #4: comm.ring_p2p must resolve to residual dtype, not act_dtype."""

    def test_ring_p2p_payload_is_residual_dtype(self):
        from python.zrt.transform.optim.passes import QuantizationPass
        from python.zrt.ir.graph import OpGraph
        from python.zrt.ir.node import OpNode
        from python.zrt.ir.types import TensorMeta, DType as IRDType

        p = GraphQuantProfile.from_scalar("fp8")
        # Set residual_dtype different from act_dtype
        p.residual_dtype = Dtype.BF16  # act is BF16, residual is BF16 too
        # Make moe_act FP8 so they differ
        p.moe_act_dtype = Dtype.FP8_E4M3

        g = OpGraph(name="test", phase="prefill")
        node = OpNode(id="comm1", op_type="comm.ring_p2p", category="communication")
        g.add_node(node)

        qpass = QuantizationPass()
        qpass._annotate_comm_payloads(g, p)

        assert node.annotations["payload_dtype"] == p.effective_residual_dtype()

    def test_p2p_in_op_type_matches_residual(self):
        from python.zrt.transform.optim.passes import QuantizationPass
        from python.zrt.ir.graph import OpGraph
        from python.zrt.ir.node import OpNode

        p = GraphQuantProfile.from_scalar("fp8")
        g = OpGraph(name="test", phase="prefill")
        node = OpNode(id="comm1", op_type="comm.p2p_send", category="communication")
        g.add_node(node)

        qpass = QuantizationPass()
        qpass._annotate_comm_payloads(g, p)
        assert node.annotations["payload_dtype"] == p.effective_residual_dtype()


class TestIdempotencyGuard:
    """Bug #5: bucket_bytes must not be rescaled on second pass invocation."""

    def test_double_run_same_bucket_bytes(self):
        from python.zrt.transform.optim.passes import QuantizationPass
        from python.zrt.ir.graph import OpGraph
        from python.zrt.ir.node import OpNode

        p = GraphQuantProfile.from_scalar("fp8")
        g = OpGraph(name="test", phase="prefill")
        node = OpNode(id="comm1", op_type="comm.all_reduce", category="communication")
        node.attrs["bucket_bytes"] = 1000
        g.add_node(node)

        qpass = QuantizationPass()
        qpass._annotate_comm_payloads(g, p)
        first_bucket = node.attrs["bucket_bytes"]

        # Second invocation should be a no-op
        qpass._annotate_comm_payloads(g, p)
        assert node.attrs["bucket_bytes"] == first_bucket


class TestUnifiedClassification:
    """Bug #9: param_count and quant.py must use the same classifier."""

    def test_shared_expert_in_scope_name(self):
        """Scope 'moe.shared.experts.0.gate_proj' must classify as shared_expert."""
        from python.zrt.ir.param_count import _classify_node_component
        n = mk_node("ffn.gate_proj", scope="model.layers.1.block_sparse_moe.shared_experts.experts.0.gate_proj")
        assert _classify_node_component(n) == "shared_expert"

    def test_routed_expert_consistency(self):
        from python.zrt.ir.param_count import _classify_node_component
        n = mk_node("ffn.gate_proj", scope="model.layers.1.block_sparse_moe.experts.0.gate_proj")
        assert _classify_node_component(n) == "routed_expert"
        assert _classify_graph_component(n) == "routed_expert"


class TestClassifyDispatchParity:
    """Verify classify() outputs cover all dispatch() buckets — drift guard."""

    def test_all_classify_outputs_routed_by_dispatch(self):
        """Every bucket returned by classify() must be handled by dispatch()."""
        from zrt.training.models.quant_dispatch import dispatch
        from python.zrt.transform.context import GraphQuantProfile

        profile = GraphQuantProfile()
        # Nodes exercising every classification branch
        cases = [
            ("attn.q_proj", "", "attention"),
            ("moe.experts.gate_proj", "", "routed_expert"),
            ("moe.shared.gate_proj", "", "shared_expert"),
            ("embedding", "", "embedding"),
            ("attn_norm", "", "norm"),
            ("ffn.gate_proj", "", "default"),
            ("moe.gate.topk", "", "default"),
        ]
        for comp, scope, expected_class in cases:
            node = mk_node(comp, scope=scope)
            classified = _classify_graph_component(node)
            assert classified == expected_class, f"{comp}: classify={classified}, expected={expected_class}"
            # dispatch must accept this without error
            bundle = dispatch(classified, profile)
            assert bundle.compute == Dtype.BF16

    def test_dispatch_accepts_all_known_buckets(self):
        """dispatch() must handle every bucket name that classify() can return."""
        from zrt.training.models.quant_dispatch import dispatch
        from python.zrt.transform.context import GraphQuantProfile

        profile = GraphQuantProfile()
        known_buckets = {"attention", "routed_expert", "shared_expert", "embedding", "norm", "default"}
        for bucket in sorted(known_buckets):
            bundle = dispatch(bucket, profile)
            assert bundle is not None, f"dispatch({bucket!r}) returned None"
