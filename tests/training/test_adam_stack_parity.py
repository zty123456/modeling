"""Stack A / Stack B Adam optimizer parity tests (US-007).

Verifies that the shared helpers produce the same results as Stack A's
_compute_optimizer_time for various parallelism configurations.
"""
import pytest
from dataclasses import dataclass

from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import GPU, SystemSpec
from zrt.training.spec.model import LayerKind
from zrt.training.spec.strategy import OptKind
from zrt.hardware.spec import InterconnectSpec, LinkSpec

from zrt.training.compose.schedules import _compute_optimizer_time
from zrt.training.models.optimizer import adam_step_time_s
from zrt.training.models.memory import adam_params_on_rank


def _make_system() -> SystemSpec:
    h100 = GPU(
        name="H100", flops_bf16=990.0, flops_fp8=1979.0,
        hbm_gb=80.0, hbm_bw_gbps=3350.0,
    )
    intra = LinkSpec(type="NVLink", bandwidth_gbps=900.0, latency_us=0.5, num_devices=8)
    inter = LinkSpec(type="RoCE", bandwidth_gbps=100.0, latency_us=10.0)
    return SystemSpec(
        gpu=h100, host_mem_gb=512.0,
        interconnect=InterconnectSpec(intra_node=intra, inter_node=inter),
        nodes=1, gpus_per_node=8,
    )


def _dense_model() -> ModelSpec:
    """Roughly GPT-3 175B geometry (dense)."""
    return ModelSpec(
        hidden=12288, ffn=49152, num_heads=96, num_kv_heads=96, head_dim=128,
        vocab=50257, seq_len=2048,
        layers=[LayerKind.DENSE] * 96,
        num_experts=0, moe_ffn=0,
    )


def _moe_model() -> ModelSpec:
    """Roughly DeepSeek-V3 geometry (MoE)."""
    return ModelSpec(
        hidden=7168, ffn=18432, num_heads=128, num_kv_heads=128, head_dim=128,
        vocab=129280, seq_len=4096,
        layers=[LayerKind.DENSE] * 3 + [LayerKind.MOE] * 58 + [LayerKind.DENSE],
        num_experts=256, moe_ffn=18432, n_shared_experts=1, top_k=8,
    )


@pytest.fixture
def system():
    return _make_system()


class TestAdamStackParity:
    """Verify Stack A and Stack B Adam time match for key configurations."""

    @pytest.mark.parametrize("tp,pp,ep,dp,zero_stage", [
        (1, 1, 1, 8, 1),   # dense_zero1
        (2, 1, 1, 4, 3),   # dense_zero3
    ])
    def test_dense_parity(self, system, tp, pp, ep, dp, zero_stage):
        model = _dense_model()
        strategy = Strategy(
            tp=tp, pp=pp, ep=ep, dp=dp,
            zero_stage=zero_stage,
            optimizer=OptKind.ADAM,
        )
        # Stack A reference
        time_a = _compute_optimizer_time(model, system, strategy)

        # Stack B via shared helpers
        total_params = model.total_params()
        n_layers = len(model.layers)
        embed_params = model.vocab * model.hidden * 2
        n_moe = sum(1 for lk in model.layers if lk == LayerKind.MOE)
        expert_params_full = (
            n_moe * 3 * model.hidden * model.moe_ffn * model.num_experts
            if n_moe and model.moe_ffn and model.num_experts else 0
        )
        P_step = adam_params_on_rank(
            total_params=total_params, n_layers=n_layers,
            embed_params=embed_params, expert_params_full=expert_params_full,
            tp=tp, pp=pp, ep=ep, dp=dp,
            zero_stage=zero_stage, apply_dp_for_zero=1,
        )
        hbm_bw_bps = system.gpu.hbm_bw_gbps * 1e9
        time_b = adam_step_time_s(P_step, hbm_bw_bps, gpu_name=system.gpu.name)

        assert time_a == pytest.approx(time_b, rel=0.01), (
            f"Stack A={time_a:.6e}s vs Stack B={time_b:.6e}s "
            f"(tp={tp} pp={pp} ep={ep} dp={dp} zero={zero_stage})"
        )

    @pytest.mark.parametrize("tp,pp,ep,dp,zero_stage", [
        (2, 2, 8, 8, 1),   # moe_ep_zero1
        (2, 2, 8, 8, 3),   # moe_ep_zero3
    ])
    def test_moe_parity(self, system, tp, pp, ep, dp, zero_stage):
        model = _moe_model()
        strategy = Strategy(
            tp=tp, pp=pp, ep=ep, dp=dp,
            zero_stage=zero_stage,
            optimizer=OptKind.ADAM,
        )
        # Stack A reference
        time_a = _compute_optimizer_time(model, system, strategy)

        # Stack B via shared helpers
        total_params = model.total_params()
        n_layers = len(model.layers)
        embed_params = model.vocab * model.hidden * 2
        n_moe = sum(1 for lk in model.layers if lk == LayerKind.MOE)
        expert_params_full = (
            n_moe * 3 * model.hidden * model.moe_ffn * model.num_experts
            if n_moe and model.moe_ffn and model.num_experts else 0
        )
        P_step = adam_params_on_rank(
            total_params=total_params, n_layers=n_layers,
            embed_params=embed_params, expert_params_full=expert_params_full,
            tp=tp, pp=pp, ep=ep, dp=dp,
            zero_stage=zero_stage, apply_dp_for_zero=1,
        )
        hbm_bw_bps = system.gpu.hbm_bw_gbps * 1e9
        time_b = adam_step_time_s(P_step, hbm_bw_bps, gpu_name=system.gpu.name)

        assert time_a == pytest.approx(time_b, rel=0.01), (
            f"Stack A={time_a:.6e}s vs Stack B={time_b:.6e}s "
            f"(tp={tp} pp={pp} ep={ep} dp={dp} zero={zero_stage})"
        )

    def test_adam_comm_is_zero(self, system):
        """Adam optimizer comm is always 0 in both stacks."""
        model = _dense_model()
        strategy = Strategy(tp=1, pp=1, ep=1, dp=8, zero_stage=1, optimizer=OptKind.ADAM)
        from zrt.training.compose.schedules import _compute_optimizer_comm_time
        comm = _compute_optimizer_comm_time(model, system, strategy)
        assert comm == pytest.approx(0.0)


class TestEndToEndParity:
    """End-to-end parity: OptimizerPass → TrainingPipelinePass vs Stack A."""

    def test_dense_e2e_optimizer_time_matches_stack_a(self, system):
        """Run OptimizerPass on a minimal graph, then TrainingPipelinePass timing,
        and compare against Stack A's _compute_optimizer_time."""
        from python.zrt.ir.graph import OpGraph
        from python.zrt.ir.node import OpNode
        from python.zrt.ir.types import TensorMeta, DType
        from python.zrt.transform.training.optimizer import OptimizerPass
        from python.zrt.transform.analysis.training import TrainingPipelinePass
        from python.zrt.transform.context import TransformContext, TrainingConfig, ParallelConfig
        from python.zrt.hardware.registry import load as load_hw

        model = _dense_model()
        tp, dp, zero_stage = 1, 8, 1
        total_params = model.total_params()

        # Build minimal backward graph with geometry metadata
        bwd_node = OpNode(
            id="bwd_0",
            op_type="aten.mm",
            inputs=[TensorMeta.from_shape_dtype("x", (1, 128), DType.BF16)],
            outputs=[TensorMeta.from_shape_dtype("y", (1, 128), DType.BF16)],
            attrs={"is_param": True, "param_count": total_params},
            category="compute",
        )
        bwd_node.annotations["phase"] = "bwd"
        graph = OpGraph(
            name="test_e2e",
            phase="train_backward",
            nodes={"bwd_0": bwd_node},
            edges=[],
            metadata={
                "total_params": total_params,
                "num_layers": len(model.layers),
                "num_layers_traced": len(model.layers),
                "hidden": model.hidden,
                "vocab_size": model.vocab,
                "model_type": "gpt",
            },
        )

        hw = load_hw("nvidia_h100_sxm")
        ctx = TransformContext(
            hw_spec=hw,
            parallel=ParallelConfig(tp=tp, dp=dp),
            training=TrainingConfig(optimizer="adam", zero_stage=zero_stage),
        )

        # Run OptimizerPass → populates step_bytes, state_bytes, etc.
        opt_pass = OptimizerPass()
        graph = opt_pass.run(graph, ctx)

        # Verify step_bytes was populated
        opt_node = graph.nodes["optimizer_step"]
        assert opt_node.attrs["step_bytes"] > 0, "OptimizerPass must populate step_bytes"

        # Run TrainingPipelinePass timing
        compute_us, ag_us, rs_us, _ = TrainingPipelinePass._compute_optimizer_step_time(
            graph, hw, ctx,
        )

        # Stack A reference
        strategy = Strategy(tp=tp, pp=1, ep=1, dp=dp, zero_stage=zero_stage, optimizer=OptKind.ADAM)
        time_a_s = _compute_optimizer_time(model, system, strategy)
        time_a_us = time_a_s * 1e6

        assert compute_us == pytest.approx(time_a_us, rel=0.05), (
            f"E2E Stack B={compute_us:.2f}µs vs Stack A={time_a_us:.2f}µs"
        )
