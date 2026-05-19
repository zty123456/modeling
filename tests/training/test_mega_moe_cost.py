from __future__ import annotations

import pytest

from zrt.training.models.mega_moe import (
    infer_quant_variant,
    resolve_mega_moe_waves,
    simulate_wave_pipeline,
)
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec


def _moe_model(**kwargs) -> ModelSpec:
    base = dict(
        hidden=1024,
        ffn=4096,
        num_heads=16,
        num_kv_heads=16,
        head_dim=64,
        vocab=32000,
        seq_len=128,
        layers=[LayerKind.MOE],
        num_experts=8,
        moe_ffn=2048,
        top_k=2,
        n_shared_experts=1,
    )
    base.update(kwargs)
    return ModelSpec(**base)


def test_infer_quant_variant_standard_for_bf16():
    model = _moe_model()

    assert infer_quant_variant(model) == "standard"


def test_infer_quant_variant_w4a8_for_fp4_weights_and_fp8_moe_acts():
    model = _moe_model(
        routed_expert_compute_dtype=Dtype.FP8_E4M3,
        routed_expert_weight_dtype=Dtype.FP4,
        moe_act_dtype=Dtype.FP8_E4M3,
    )

    assert infer_quant_variant(model) == "w4a8"


def test_resolve_waves_prefers_valid_divisor_not_above_target():
    assert resolve_mega_moe_waves(requested=6, hardware_waves=0, experts_per_rank=8) == 4
    assert resolve_mega_moe_waves(requested=0, hardware_waves=4, experts_per_rank=8) == 4
    assert resolve_mega_moe_waves(requested=0, hardware_waves=0, experts_per_rank=3) == 3


def test_wave_pipeline_one_wave_is_serial():
    result = simulate_wave_pipeline(waves=1, dispatch_s=1.0, compute_s=3.0, combine_s=2.0)

    assert result.total_s == pytest.approx(6.0)
    assert result.exposed_comm_s == pytest.approx(3.0)
    assert result.hidden_comm_s == pytest.approx(0.0)


def test_wave_pipeline_more_waves_hide_comm_when_compute_dominates():
    one = simulate_wave_pipeline(waves=1, dispatch_s=1.0, compute_s=6.0, combine_s=1.0)
    four = simulate_wave_pipeline(waves=4, dispatch_s=0.25, compute_s=1.5, combine_s=0.25)

    assert four.total_s < one.total_s
    assert four.hidden_comm_s > 0.0
