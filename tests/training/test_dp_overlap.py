"""Tests for unified DP overlap window across all composers + dualbatch path."""

import pytest
from zrt.training.compose.schedules import (
    OneF1BComposer, Interleaved1F1BComposer, ZeroBubbleComposer,
    DualPipeComposer, DualPipeVComposer, pipeline_step_time,
)
from zrt.training.compose.stage import StageTime
from zrt.training.ir.builders import build_graph
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.spec.strategy import Strategy, PPSched
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.spec.system import SystemSpec, GPU


def _make_system():
    return SystemSpec(
        gpu=GPU(name="h100", flops_bf16=989, flops_fp8=1979, hbm_gb=80, hbm_bw_gbps=3350),
        host_mem_gb=256,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0,
                                topology="all_to_all", num_devices=8),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=400, latency_us=5.0,
                                topology="fat_tree"),
        ),
        nodes=1, gpus_per_node=8,
    )


class TestSteadyBwdOverlap:
    """dp_steady_overlap_ratio lets DP hide in steady-state backward compute
    (FSDP / Megatron behavior), not only in the explicit cooldown phase."""

    def test_default_ratio_is_0p5(self):
        s = Strategy(tp=1, pp=1, dp=8, micro_batch=1, global_batch=8)
        assert s.dp_steady_overlap_ratio == 0.5

    def test_ratio_extends_hide_window_in_dualpipe(self):
        """DualPipe cooldown is small; with ratio>0, steady_bwd contributes
        and dp_exposed shrinks."""
        # Two equal stages, fwd=10, bwd=20 → t_stage=30
        # pp=2 → cooldown = (pp-1)/2 * t_stage = 15
        # M=4 → steady_bwd_per_mb = bwd = 20, steady_bwd = M*bwd = 80
        stages = [StageTime(fwd=10.0, bwd=20.0), StageTime(fwd=10.0, bwd=20.0)]
        s_no = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=4,
                        pp_schedule=PPSched.DUALPIPE, dp_steady_overlap_ratio=0.0)
        s_half = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=4,
                          pp_schedule=PPSched.DUALPIPE, dp_steady_overlap_ratio=0.5)

        r_no = DualPipeComposer().compose(stages, M=4, pp=2, dp_ar_time=100.0, strategy=s_no)
        r_half = DualPipeComposer().compose(stages, M=4, pp=2, dp_ar_time=100.0, strategy=s_half)

        # ratio=0: hide = min(cooldown=15, 100) = 15 → exposed = 85
        # ratio=0.5: hide = min(cooldown=15 + 0.5*80=55, 100) = 55 → exposed = 45
        assert r_no.dp_exposed == pytest.approx(85.0)
        assert r_half.dp_exposed == pytest.approx(45.0)

    def test_ratio_zero_matches_legacy_behavior(self):
        """ratio=0 should reproduce the previous cooldown-only window exactly,
        keeping backward-compat with anchor fixtures."""
        stages = [StageTime(fwd=5.0, bwd=10.0), StageTime(fwd=5.0, bwd=10.0)]
        s = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=4,
                     pp_schedule=PPSched.ONE_F_ONE_B, dp_steady_overlap_ratio=0.0)
        # pp=2 1F1B: cooldown = (pp-1)*max(bwd) = 10
        r = OneF1BComposer().compose(stages, M=4, pp=2, dp_ar_time=20.0, strategy=s)
        # hide = min(10, 20) = 10 → exposed = 10
        assert r.dp_exposed == pytest.approx(10.0)


class TestDualbatchRecoversWithSteadyOverlap:
    """When dualbatch=True eats cooldown, DP can still overlap with steady BWD."""

    def test_dualbatch_with_ratio_zero_keeps_dp_fully_exposed(self):
        """Backward-compat: ratio=0 → existing dualbatch behavior."""
        model = ModelSpec(hidden=2048, ffn=8192, num_heads=16, num_kv_heads=16,
                          head_dim=128, vocab=32000, seq_len=1024,
                          layers=[LayerKind.DENSE]*4)
        s = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=4,
                     pp_schedule=PPSched.DUALPIPE, dualbatch=True,
                     dp_steady_overlap_ratio=0.0)
        graph = build_graph(model, s)
        result = pipeline_step_time(graph, model, _make_system(), s)
        # After dualbatch eats cooldown → new_cooldown=0 → exposed=dp_total
        assert result.dp_hidden == pytest.approx(0.0, abs=1e-9)

    def test_dualbatch_with_ratio_half_recovers_dp_hide(self):
        model = ModelSpec(hidden=2048, ffn=8192, num_heads=16, num_kv_heads=16,
                          head_dim=128, vocab=32000, seq_len=1024,
                          layers=[LayerKind.DENSE]*4)
        s = Strategy(tp=1, pp=2, dp=4, micro_batch=1, global_batch=4,
                     pp_schedule=PPSched.DUALPIPE, dualbatch=True,
                     dp_steady_overlap_ratio=0.5)
        graph = build_graph(model, s)
        result = pipeline_step_time(graph, model, _make_system(), s)
        # With ratio>0, dp_hidden > 0 even though dualbatch zeroed cooldown
        assert result.dp_hidden > 0.0, \
            f"steady-bwd window should recover some DP hide, got dp_hidden={result.dp_hidden}"
