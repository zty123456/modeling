"""Integration test: training modelling driven by a captured-style OpGraph.

Captured graphs have opaque tensor IDs (t0, t1, ...) unlike synthetic test
graphs that use descriptive names like 'weight_0'.  This file verifies that
estimate_training() returns meaningful results in both cases.
"""
from __future__ import annotations

import math
from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.edge import Edge
from zrt.ir.types import TensorMeta, DType
from zrt.transform.context import TransformContext, ParallelConfig, TrainingConfig
from zrt.transform.analysis import estimate_training


def _hw():
    from zrt.hardware.spec import HardwareSpec, ComputeSpec, MemorySpec, InterconnectSpec, LinkSpec
    return HardwareSpec(
        name="test_gpu", vendor="test", device_type="gpu",
        compute=ComputeSpec(bf16_tflops=312),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=2000),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="nvlink", num_devices=8, bandwidth_gbps=600, latency_us=1.0),
            inter_node=LinkSpec(type="ib",     num_devices=128, bandwidth_gbps=400, latency_us=5.0),
        ),
    )


def _captured_style_graph(seq_len=2048, hidden=4096, ffn=16384, num_layers=32) -> OpGraph:
    """Minimal transformer forward graph with captured-style tensor IDs (t0, t1, ...).

    Structure per layer: linear(t_act, t_weight) → linear(t_mid, t_weight2) → add
    Two layers total to keep the test fast.
    """
    nodes: dict[str, OpNode] = {}
    edges: list[Edge] = []
    tid = 0

    def _tm(shape, dtype=DType.BF16):
        nonlocal tid
        t = TensorMeta(
            id=f"t{tid}",
            shape=shape,
            dtype=dtype,
            mem_bytes=int(math.prod(shape) * 2),
        )
        tid += 1
        return t

    # Input activation (t0) — external, not produced by any node
    t_input = _tm((seq_len, hidden))

    prev_out = t_input
    for layer_id in range(2):
        scope_base = f"model.layers.{layer_id}"

        # QKV projection: mm(activation, weight)
        t_w_qkv = _tm((hidden, hidden))   # weight — external
        t_qkv   = _tm((seq_len, hidden))  # output
        n_qkv = OpNode(
            id=f"L{layer_id}_qkv",
            op_type="aten.mm.default",
            inputs=[prev_out, t_w_qkv],
            outputs=[t_qkv],
            scope=f"{scope_base}.self_attn.qkv_proj",
            category="compute",
        )
        nodes[n_qkv.id] = n_qkv

        # FFN up-projection
        t_w_up = _tm((hidden, ffn))
        t_up   = _tm((seq_len, ffn))
        n_up = OpNode(
            id=f"L{layer_id}_up",
            op_type="aten.mm.default",
            inputs=[t_qkv, t_w_up],
            outputs=[t_up],
            scope=f"{scope_base}.mlp.up_proj",
            category="compute",
        )
        nodes[n_up.id] = n_up
        edges.append(Edge(src=n_qkv.id, dst=n_up.id, tensor=t_qkv, src_idx=0, dst_idx=0))

        # FFN down-projection
        t_w_down = _tm((ffn, hidden))
        t_down   = _tm((seq_len, hidden))
        n_down = OpNode(
            id=f"L{layer_id}_down",
            op_type="aten.mm.default",
            inputs=[t_up, t_w_down],
            outputs=[t_down],
            scope=f"{scope_base}.mlp.down_proj",
            category="compute",
        )
        nodes[n_down.id] = n_down
        edges.append(Edge(src=n_up.id, dst=n_down.id, tensor=t_up, src_idx=0, dst_idx=0))

        prev_out = t_down

    return OpGraph(
        name="captured_llama",
        phase="forward",
        nodes=nodes,
        edges=edges,
        metadata={
            "seq_len": seq_len,
            "hidden": hidden,
            "num_layers": num_layers,
            "batch_size": 1,
        },
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_captured_graph_tensor_ids_are_opaque():
    """Verify the test graph actually uses captured-style (opaque) tensor IDs."""
    g = _captured_style_graph()
    all_input_ids = {
        inp.id
        for node in g.nodes.values()
        for inp in node.inputs
    }
    # No "weight" or "param" in any tensor ID — these are captured-style IDs
    assert not any("weight" in tid or "param" in tid for tid in all_input_ids), (
        "Test graph should use opaque tensor IDs (t0, t1, ...) not named ones"
    )


def test_estimate_training_total_params_nonzero():
    """TrainingFlopsPass must count params accurately from a captured-style graph.

    For 2 layers with hidden=4096, ffn=16384:
      params = 2 × (QKV: hidden² + FFN-up: hidden×ffn + FFN-down: ffn×hidden)
             = 2 × (16.7M + 67.1M + 67.1M) ≈ 302M
    """
    hidden, ffn = 4096, 16384
    expected = 2 * (hidden * hidden + hidden * ffn + ffn * hidden)

    g = _captured_style_graph(seq_len=2048, hidden=hidden, ffn=ffn)
    hw = _hw()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )
    report = estimate_training(g, ctx)
    assert report.total_params > 0, "Expected non-zero total_params from captured-style graph"
    # Allow ±5% tolerance for any minor heuristic variance
    assert abs(report.total_params - expected) / expected < 0.05, (
        f"Param count {report.total_params:,} deviates >5% from expected {expected:,}"
    )


def test_estimate_training_step_time_nonzero():
    """estimate_training() on a captured graph with hw_spec must return step_time_ms > 0."""
    g = _captured_style_graph()
    hw = _hw()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )
    report = estimate_training(g, ctx)
    assert report.step_time_ms > 0, (
        f"Expected step_time_ms > 0 but got {report.step_time_ms}. "
        "RooflinePass may not be wired into estimate_training()."
    )


def test_estimate_training_flops_nonzero():
    """estimate_training() must return non-zero training_flops."""
    g = _captured_style_graph()
    hw = _hw()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )
    report = estimate_training(g, ctx)
    assert report.training_flops > 0, "Expected non-zero training_flops"
    assert report.backward_flops == 2 * report.forward_flops


def test_estimate_training_mfu_in_range():
    """MFU must be in [0, 1]."""
    g = _captured_style_graph()
    hw = _hw()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )
    report = estimate_training(g, ctx)
    assert 0 <= report.mfu <= 1, f"MFU out of range: {report.mfu}"


def test_estimate_training_metadata_total_params_overrides():
    """graph.metadata['total_params'] takes precedence over structural counting."""
    g = _captured_style_graph()
    known_params = 70_000_000_000  # 70B
    g.metadata["total_params"] = known_params

    hw = _hw()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )
    report = estimate_training(g, ctx)
    assert report.total_params == known_params


def test_estimate_training_parallel_config():
    """estimate_training() produces sensible results with TP+DP parallelism."""
    g = _captured_style_graph()
    hw = _hw()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=2, pp=1, dp=4),
        training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=1),
    )
    report = estimate_training(g, ctx)
    assert "TP2" in report.config_summary
    assert "DP4" in report.config_summary
    assert "ZeRO-1" in report.config_summary
    assert report.total_params > 0
    assert report.step_time_ms > 0


def test_pipeline_routing_runs_roofline_and_stream_assign():
    """estimate_training() via build_training_pipeline must run RooflinePass + StreamAssignPass.

    Verifies that nodes carry latency_us (from RooflinePass) and stream_id
    (from StreamAssignPass), proving the full pipeline ran, not just training passes.
    """
    g = _captured_style_graph()
    hw = _hw()
    ctx = TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )

    from python.zrt.transform.pipeline import build_training_pipeline
    pipe = build_training_pipeline()
    result = pipe.run(g, ctx)

    # All compute nodes must have latency_us (from RooflinePass)
    for nid, node in result.nodes.items():
        assert "latency_us" in node.annotations, (
            f"Node {nid} missing latency_us — RooflinePass did not run"
        )
        assert "stream_id" in node.annotations, (
            f"Node {nid} missing stream_id — StreamAssignPass did not run"
        )

    # Graph must have training_flops metadata (from TrainingFlopsPass)
    assert "training_flops" in result.metadata
    assert result.metadata["training_flops"] > 0


def test_backward_fusion_rules_fire_on_backward_graph():
    """Verify that backward fusion rules from fusion_rules.py:195-311 match
    on a synthetic backward graph when run through build_training_pipeline().

    Creates a backward OpGraph with ops matching norm_backward and
    gated_mlp_backward sub-patterns, then asserts at least one node gets
    relabeled with a backward fusion label.
    """
    import math
    from python.zrt.transform.pipeline import build_training_pipeline

    # Build a synthetic backward graph with backward-style ops in matching scopes
    nodes: dict[str, OpNode] = {}
    edges: list[Edge] = []
    tid = 0

    def _tm(shape, dtype=DType.BF16):
        nonlocal tid
        t = TensorMeta(
            id=f"t{tid}", shape=shape, dtype=dtype,
            mem_bytes=int(math.prod(shape) * 2),
        )
        tid += 1
        return t

    # Group 1: norm_backward — native_layer_norm_backward in RMSNorm scope
    t_norm_in = _tm((2048, 4096))
    t_norm_out = _tm((2048, 4096))
    n_norm = OpNode(
        id="bwd_layernorm",
        op_type="aten.native_layer_norm_backward.default",
        inputs=[t_norm_in],
        outputs=[t_norm_out],
        scope="model.layers.0.input_layernorm",
        module_class="RMSNorm",
        category="compute",
    )
    nodes[n_norm.id] = n_norm

    # Group 2: gated_mlp_backward — silu_backward + mul + mm in MLP scope
    t_mlp_in = _tm((2048, 4096))
    t_silu_out = _tm((2048, 4096))
    n_silu = OpNode(
        id="bwd_silu",
        op_type="aten.silu_backward.default",
        inputs=[t_mlp_in],
        outputs=[t_silu_out],
        scope="model.layers.0.mlp",
        module_class="MLP",
        category="compute",
    )
    nodes[n_silu.id] = n_silu

    t_mul_out = _tm((2048, 4096))
    n_mul = OpNode(
        id="bwd_mul",
        op_type="aten.mul.default",
        inputs=[t_silu_out],
        outputs=[t_mul_out],
        scope="model.layers.0.mlp",
        module_class="MLP",
        category="compute",
    )
    nodes[n_mul.id] = n_mul

    t_w = _tm((4096, 4096))
    t_mm_out = _tm((2048, 4096))
    n_mm = OpNode(
        id="bwd_mm",
        op_type="aten.mm.default",
        inputs=[t_mul_out, t_w],
        outputs=[t_mm_out],
        scope="model.layers.0.mlp",
        module_class="MLP",
        category="compute",
    )
    nodes[n_mm.id] = n_mm

    edges.append(Edge(src=n_silu.id, dst=n_mul.id, tensor=t_silu_out, src_idx=0, dst_idx=0))
    edges.append(Edge(src=n_mul.id, dst=n_mm.id, tensor=t_mul_out, src_idx=0, dst_idx=0))

    graph = OpGraph(
        name="synthetic_backward",
        phase="train_backward",
        nodes=nodes,
        edges=edges,
        metadata={
            "seq_len": 2048,
            "hidden": 4096,
            "num_layers": 32,
            "batch_size": 1,
        },
    )

    from zrt.hardware.spec import HardwareSpec, ComputeSpec, MemorySpec, InterconnectSpec, LinkSpec
    hw_nvidia = HardwareSpec(
        name="nvidia_gpu", vendor="nvidia", device_type="gpu",
        compute=ComputeSpec(bf16_tflops=312),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=2000),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="nvlink", num_devices=8, bandwidth_gbps=600, latency_us=1.0),
            inter_node=LinkSpec(type="ib",     num_devices=128, bandwidth_gbps=400, latency_us=5.0),
        ),
    )
    ctx = TransformContext(
        hw_spec=hw_nvidia,
        parallel=ParallelConfig(tp=1, pp=1, dp=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )

    pipe = build_training_pipeline()
    result = pipe.run(graph, ctx)

    # Collect all op_types (including fused nodes)
    all_op_types = {n.op_type for n in result.nodes.values()}

    # At least one backward fusion label must be present
    backward_labels = {
        "norm_backward", "gated_mlp_backward", "mlp_backward",
        "sdpa_backward", "attn_grad", "embedding_backward",
    }
    matched = all_op_types & backward_labels
    assert matched, (
        f"No backward fusion labels found. Got op_types: {all_op_types}. "
        f"Expected one of: {backward_labels}"
    )
