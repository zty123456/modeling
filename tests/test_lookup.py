"""Tests for python.zrt.simulator.backends.lookup helper functions.

Run:
    pytest tests/test_lookup.py -v
"""
from __future__ import annotations

import math

import pytest

from python.zrt.ir.node import OpNode
from python.zrt.ir.types import DType, TensorMeta
from python.zrt.hardware import load as load_hw
from python.zrt.simulator.backends.lookup import (
    _convert_tensor,
    _get_cost_model_op_type,
    _get_primary_dtype,
    _calculate_hw_util,
    _build_sim_result,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _tm(tid: str, shape: tuple[int, ...], dtype: DType = DType.BF16) -> TensorMeta:
    return TensorMeta.from_shape_dtype(tid, shape, dtype)


def _node(
    op_type: str,
    inputs: list[TensorMeta] | None = None,
    outputs: list[TensorMeta] | None = None,
    op_short: str = "",
    category: str = "compute",
    annotations: dict | None = None,
) -> OpNode:
    return OpNode(
        id="test_node",
        op_type=op_type,
        op_short=op_short,
        inputs=inputs or [],
        outputs=outputs or [],
        category=category,
        annotations=annotations or {},
    )


@pytest.fixture(scope="module")
def hw_910b():
    return load_hw("ascend_910b")


# ═════════════════════════════════════════════════════════════════════════════
# _convert_tensor
# ═════════════════════════════════════════════════════════════════════════════

class TestConvertTensor:

    def test_basic_bf16_2d(self):
        t = _tm("a", (64, 128), DType.BF16)
        d = _convert_tensor(t)
        assert d["dtype"] == "torch.bfloat16"
        assert d["shape"] == [64, 128]
        assert d["origin_shape"] == [64, 128]
        assert d["size"] == 64 * 128
        assert d["format"] == ""
        assert d["name"] == ""

    def test_fp16_dtype(self):
        t = _tm("a", (32,), DType.FP16)
        d = _convert_tensor(t)
        assert d["dtype"] == "torch.float16"
        assert d["origin_dtype"] == "torch.float16"
        assert d["size"] == 32

    def test_fp32_dtype(self):
        t = _tm("a", (8, 16), DType.FP32)
        d = _convert_tensor(t)
        assert d["dtype"] == "torch.float32"

    def test_int8_dtype(self):
        t = _tm("a", (100,), DType.INT8)
        d = _convert_tensor(t)
        assert d["dtype"] == "torch.int8"

    def test_int4_maps_to_torch_int8_quant(self):
        t = _tm("a", (64, 64), DType.INT4)
        d = _convert_tensor(t)
        assert d["dtype"] == "torch.int8_quant"

    def test_int64_dtype(self):
        t = _tm("idx", (1, 128), DType.INT64)
        d = _convert_tensor(t)
        assert d["dtype"] == "torch.int64"

    def test_bool_dtype(self):
        t = _tm("mask", (1, 1, 128, 128), DType.BOOL)
        d = _convert_tensor(t)
        assert d["dtype"] == "torch.bool"

    def test_unknown_dtype_falls_back_to_bf16(self):
        t = _tm("a", (10,), DType.UNKNOWN)
        d = _convert_tensor(t)
        assert d["dtype"] == "torch.bfloat16"

    def test_3d_tensor(self):
        t = _tm("a", (2, 64, 128))
        d = _convert_tensor(t)
        assert d["shape"] == [2, 64, 128]
        assert d["size"] == 2 * 64 * 128

    def test_scalar_tensor(self):
        t = _tm("a", ())
        d = _convert_tensor(t)
        assert d["shape"] == []
        assert d["size"] == 1  # math.prod(()) = 1

    def test_fp8_e4m3(self):
        t = _tm("a", (128, 256), DType.FP8_E4M3)
        d = _convert_tensor(t)
        assert d["dtype"] == "torch.float8_e4m3fn"

    def test_fp8_e5m2(self):
        t = _tm("a", (64,), DType.FP8_E5M2)
        d = _convert_tensor(t)
        assert d["dtype"] == "torch.float8_e5m2fn"


# ═════════════════════════════════════════════════════════════════════════════
# _get_cost_model_op_type
# ═════════════════════════════════════════════════════════════════════════════

class TestGetCostModelOpType:

    def test_mm_maps_to_matmul(self):
        node = _node("aten.mm.default", op_short="mm")
        assert _get_cost_model_op_type(node) == "Matmul"

    def test_bmm_maps_to_batch_mat_mul(self):
        node = _node("aten.bmm.default", op_short="bmm")
        assert _get_cost_model_op_type(node) == "BatchMatMul"

    def test_flash_attn_maps_to_flash_attention(self):
        node = _node("flash_attn", op_short="flash_attn")
        assert _get_cost_model_op_type(node) == "FlashAttention"

    def test_sdpa_maps_to_flash_attention(self):
        node = _node("aten._scaled_dot_product_flash_attention.default", op_short="sdpa")
        assert _get_cost_model_op_type(node) == "FlashAttention"

    def test_floor_divide_maps_to_floor_div(self):
        node = _node("aten.floor_divide.default", op_short="floor_divide")
        assert _get_cost_model_op_type(node) == "FloorDiv"

    def test_unknown_op_converts_to_pascal_case(self):
        node = _node("aten.layer_norm.default", op_short="layer_norm")
        assert _get_cost_model_op_type(node) == "LayerNorm"

    def test_multi_underscore_op_to_pascal_case(self):
        node = _node("aten.scaled_dot_product_attention.default", op_short="scaled_dot_product_attention")
        assert _get_cost_model_op_type(node) == "ScaledDotProductAttention"

    def test_falls_back_to_op_type_when_op_short_empty(self):
        node = _node("aten.addmm.default", op_short="")
        # "aten.addmm.default" → "Aten.Addmm.Default"
        result = _get_cost_model_op_type(node)
        assert result == "Aten.Addmm.Default"

    def test_falls_back_to_op_type_when_op_short_none_equivalent(self):
        # op_short is "" by default, and the 'not op_short' check catches it
        node = _node("aten.relu.default")
        node.op_short = ""  # explicitly empty
        result = _get_cost_model_op_type(node)
        assert result == "Aten.Relu.Default"

    def test_single_word_op_short(self):
        node = _node("aten.silu.default", op_short="silu")
        assert _get_cost_model_op_type(node) == "Silu"


# ═════════════════════════════════════════════════════════════════════════════
# _get_primary_dtype
# ═════════════════════════════════════════════════════════════════════════════

class TestGetPrimaryDtype:

    def test_returns_first_input_dtype_for_compute_node(self):
        node = _node(
            "aten.mm.default",
            inputs=[_tm("a", (64, 128), DType.BF16), _tm("b", (128, 64), DType.FP16)],
            outputs=[_tm("o", (64, 64), DType.BF16)],
        )
        assert _get_primary_dtype(node) == DType.BF16

    def test_returns_output_dtype_for_communication_node(self):
        node = _node(
            "comm.all_reduce",
            inputs=[_tm("a", (128,), DType.BF16)],
            outputs=[_tm("o", (128,), DType.FP16)],
            category="communication",
        )
        assert _get_primary_dtype(node) == DType.FP16

    def test_returns_output_dtype_when_no_inputs(self):
        node = _node(
            "aten.constant.default",
            inputs=[],
            outputs=[_tm("o", (64,), DType.FP32)],
        )
        assert _get_primary_dtype(node) == DType.FP32

    def test_returns_bf16_when_no_inputs_and_no_outputs(self):
        node = _node("aten.noop.default", inputs=[], outputs=[])
        assert _get_primary_dtype(node) == DType.BF16

    def test_returns_bf16_when_category_not_compute_and_no_outputs(self):
        node = _node(
            "comm.all_reduce",
            inputs=[_tm("a", (128,), DType.FP16)],
            outputs=[],
            category="communication",
        )
        assert _get_primary_dtype(node) == DType.BF16

    def test_quant_act_fp8_normalizes_to_fp8_e4m3(self):
        node = _node(
            "aten.mm.default",
            inputs=[_tm("a", (64, 128), DType.FP8_E4M3)],
            outputs=[_tm("o", (64, 64), DType.FP8_E4M3)],
            annotations={"quant_act": "fp8"},
        )
        assert _get_primary_dtype(node) == DType.FP8_E4M3

    def test_quant_act_bf16_is_ignored(self):
        """quant_act values in ('bf16', 'fp16', 'fp32') are skipped; falls back to input dtype."""
        node = _node(
            "aten.mm.default",
            inputs=[_tm("a", (64, 128), DType.FP16)],
            outputs=[_tm("o", (64, 64), DType.FP16)],
            annotations={"quant_act": "bf16"},
        )
        assert _get_primary_dtype(node) == DType.FP16

    def test_quant_act_fp16_is_ignored(self):
        node = _node(
            "aten.mm.default",
            inputs=[_tm("a", (64, 128), DType.BF16)],
            outputs=[_tm("o", (64, 64), DType.BF16)],
            annotations={"quant_act": "fp16"},
        )
        assert _get_primary_dtype(node) == DType.BF16

    def test_quant_act_fp32_is_ignored(self):
        node = _node(
            "aten.mm.default",
            inputs=[_tm("a", (64, 128), DType.BF16)],
            outputs=[_tm("o", (64, 64), DType.BF16)],
            annotations={"quant_act": "fp32"},
        )
        assert _get_primary_dtype(node) == DType.BF16

    def test_quant_act_invalid_falls_back_to_input(self):
        node = _node(
            "aten.mm.default",
            inputs=[_tm("a", (64, 128), DType.BF16)],
            outputs=[_tm("o", (64, 64))],
            annotations={"quant_act": "int4"},
        )
        # "int4" is not in ("bf16", "fp16", "fp32"), normalized as "int4",
        # DType("int4") succeeds → DType.INT4. But if it failed ValueError, falls back.
        # Actually DType("int4") should succeed.
        result = _get_primary_dtype(node)
        assert result == DType.INT4

    def test_quant_act_unknown_value_raises_valueerror_falls_back(self):
        node = _node(
            "aten.mm.default",
            inputs=[_tm("a", (64, 128), DType.BF16)],
            outputs=[_tm("o", (64, 64))],
            annotations={"quant_act": "nonexistent_dtype"},
        )
        # "nonexistent_dtype" not in ("bf16", "fp16", "fp32"), normalized stays same,
        # DType("nonexistent_dtype") raises ValueError, caught → falls back to input dtype
        assert _get_primary_dtype(node) == DType.BF16


# ═════════════════════════════════════════════════════════════════════════════
# _calculate_hw_util
# ═════════════════════════════════════════════════════════════════════════════

class TestCalculateHwUtil:

    def test_normal_case_compute_bound(self, hw_910b):
        # 4096^3 matmul: flops ≈ 2 * 4096^3 = 137,438,953,472
        M, K, N = 4096, 4096, 4096
        flops = 2 * M * N * K
        node = _node(
            "aten.mm.default",
            inputs=[_tm("a", (M, K)), _tm("b", (K, N))],
            outputs=[_tm("o", (M, N))],
            annotations={"flops": flops},
        )
        latency_us = 500.0
        util = _calculate_hw_util(node, hw_910b, latency_us)
        assert 0.0 < util <= 1.0

    def test_latency_zero_returns_zero(self, hw_910b):
        node = _node(
            "aten.mm.default",
            inputs=[_tm("a", (64, 64)), _tm("b", (64, 64))],
            outputs=[_tm("o", (64, 64))],
            annotations={"flops": 1000},
        )
        assert _calculate_hw_util(node, hw_910b, 0.0) == 0.0

    def test_flops_zero_returns_zero(self, hw_910b):
        node = _node(
            "aten.view.default",
            inputs=[_tm("a", (64, 64))],
            outputs=[_tm("o", (32, 128))],
            annotations={"flops": 0},
        )
        assert _calculate_hw_util(node, hw_910b, 100.0) == 0.0

    def test_peak_flops_zero_when_no_annotations(self, hw_910b):
        """When annotations are missing flops, util should be 0 regardless."""
        node = _node(
            "aten.mm.default",
            inputs=[_tm("a", (64, 128)), _tm("b", (128, 64))],
            outputs=[_tm("o", (64, 64))],
        )
        # flops defaults to 0 from annotations.get("flops", 0)
        assert _calculate_hw_util(node, hw_910b, 100.0) == 0.0

    def test_util_clamped_to_1_0(self, hw_910b):
        """hw_util should not exceed 1.0 even if actual_rate > peak."""
        # Tiny FLOPs but infinitesimal latency → impossibly high rate
        node = _node(
            "aten.mm.default",
            inputs=[_tm("a", (64, 64)), _tm("b", (64, 64))],
            outputs=[_tm("o", (64, 64))],
            annotations={"flops": 1_000_000_000},
        )
        util = _calculate_hw_util(node, hw_910b, 0.001)  # 1ns → absurd rate
        assert util == 1.0

    def test_small_decode_style_matmul(self, hw_910b):
        """Single-token decode matmul: low utilization due to memory-bound."""
        M, K, N = 1, 7168, 7168
        flops = 2 * M * N * K
        node = _node(
            "aten.mm.default",
            inputs=[_tm("a", (M, K)), _tm("b", (K, N))],
            outputs=[_tm("o", (M, N))],
            annotations={"flops": flops},
        )
        # Assume a reasonable latency for a small matmul
        util = _calculate_hw_util(node, hw_910b, 10.0)
        assert 0.0 < util < 1.0
        assert util < 0.5  # decode matmul is heavily memory-bound


# ═════════════════════════════════════════════════════════════════════════════
# _build_sim_result
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildSimResult:

    def test_confidence_zeroed_when_hw_util_zero(self, hw_910b):
        """When no flops annotation, hw_util=0 → confidence should be 0."""
        node = _node(
            "aten.view.default",
            inputs=[_tm("a", (64, 64))],
            outputs=[_tm("o", (32, 128))],
        )
        r = _build_sim_result(node, hw_910b, 0.0, "lookup", 0.8)
        assert r.confidence == 0.0

    def test_confidence_preserved_when_hw_util_nonzero(self, hw_910b):
        M, K, N = 1024, 4096, 7168
        flops = 2 * M * N * K
        node = _node(
            "aten.mm.default",
            inputs=[_tm("a", (M, K)), _tm("b", (K, N))],
            outputs=[_tm("o", (M, N))],
            annotations={"flops": flops},
        )
        r = _build_sim_result(node, hw_910b, 200.0, "lookup", 0.8)
        assert r.confidence == 0.8
        assert r.backend == "lookup"
        assert r.latency_us == 200.0
        assert r.hw_utilization > 0.0

    def test_backend_field_reflects_passed_value(self, hw_910b):
        node = _node(
            "aten.mm.default",
            inputs=[_tm("a", (64, 64)), _tm("b", (64, 64))],
            outputs=[_tm("o", (64, 64))],
            annotations={"flops": 1000},
        )
        r = _build_sim_result(node, hw_910b, 50.0, "tilesim", 0.9)
        assert r.backend == "tilesim"

    def test_op_node_id_preserved(self, hw_910b):
        node = _node(
            "aten.silu.default",
            inputs=[_tm("x", (128,))],
            outputs=[_tm("o", (128,))],
        )
        node.id = "op_42"
        r = _build_sim_result(node, hw_910b, 5.0, "lookup", 0.3)
        assert r.op_node_id == "op_42"

    def test_annotations_flow_through_to_result(self, hw_910b):
        node = _node(
            "aten.mm.default",
            inputs=[_tm("a", (64, 128)), _tm("b", (128, 64))],
            outputs=[_tm("o", (64, 64))],
            annotations={
                "flops": 1_048_576,
                "compute_us": 12.5,
                "memory_us": 3.2,
                "read_bytes": 65536,
                "write_bytes": 16384,
                "arithmetic_intensity": 12.8,
                "bound": "compute",
            },
        )
        r = _build_sim_result(node, hw_910b, 100.0, "lookup", 0.8)
        assert r.flops == 1_048_576
        assert r.compute_us == 12.5
        assert r.memory_us == 3.2
        assert r.read_bytes == 65536
        assert r.write_bytes == 16384
        assert r.arithmetic_intensity == 12.8
        assert r.bound == "compute"
