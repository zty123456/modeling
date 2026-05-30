"""Tests for the shared FP32-promotion multiplier table."""
import pytest
from zrt.training.models.promotion import ln_softmax_input_byte_multiplier


class TestPromotionMultiplier:
    # Softmax variants — spec kind and graph op_type
    @pytest.mark.parametrize("s", ["softmax", "aten.softmax.int"])
    def test_softmax_returns_2(self, s):
        assert ln_softmax_input_byte_multiplier(s) == 2.0

    # LN/RMSNorm variants — graph op_type strings
    @pytest.mark.parametrize("s", [
        "rms_norm", "aten.native_layer_norm.default", "add_rms_norm",
        "npu_add_rms", "gemma_norm", "rms_gated", "add_layer_norm",
    ])
    def test_norm_graph_ops_return_1(self, s):
        assert ln_softmax_input_byte_multiplier(s) == 1.0

    # Spec-side op.kind strings
    @pytest.mark.parametrize("s", ["ln", "rmsnorm"])
    def test_norm_spec_kinds_return_1(self, s):
        assert ln_softmax_input_byte_multiplier(s) == 1.0

    # Non-matching ops
    @pytest.mark.parametrize("s", [
        "aten.add.Tensor", "aten.mm.default", "aten.mul.Tensor",
        "matmul", "relu", "silu", "swiglu",
    ])
    def test_other_returns_0(self, s):
        assert ln_softmax_input_byte_multiplier(s) == 0.0
