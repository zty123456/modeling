"""Tests for GraphQuantProfile: creation, normalization, presets, invalid inputs."""
import pytest
from python.zrt.transform.context import GraphQuantProfile
from zrt.training.spec.dtype import Dtype


class TestScalarNormalization:
    def test_none_returns_none(self):
        assert GraphQuantProfile.from_scalar(None) is None

    def test_bf16_returns_bf16_profile(self):
        p = GraphQuantProfile.from_scalar("bf16")
        assert p.param_dtype == Dtype.BF16
        assert p.act_dtype == Dtype.BF16

    def test_fp8_produces_fp8_mixed(self):
        p = GraphQuantProfile.from_scalar("fp8")
        assert p.routed_expert_compute_dtype == Dtype.FP8_E4M3
        assert p.moe_act_dtype == Dtype.FP8_E4M3
        assert p.routed_expert_weight_dtype == Dtype.BF16  # weights stay BF16
        assert p.attn_compute_dtype == Dtype.BF16

    def test_fp8_e5m2(self):
        p = GraphQuantProfile.from_scalar("fp8_e5m2")
        assert p.routed_expert_compute_dtype == Dtype.FP8_E5M2
        assert p.moe_act_dtype == Dtype.FP8_E5M2

    def test_fp4_produces_dsv4_paper(self):
        p = GraphQuantProfile.from_scalar("fp4")
        assert p.routed_expert_weight_dtype == Dtype.FP4
        assert p.routed_expert_compute_dtype == Dtype.FP4

    def test_int8_raises(self):
        with pytest.raises(ValueError, match="inference-only"):
            GraphQuantProfile.from_scalar("int8")

    def test_int4_raises(self):
        with pytest.raises(ValueError, match="inference-only"):
            GraphQuantProfile.from_scalar("int4")

    def test_w8a8_raises(self):
        with pytest.raises(ValueError, match="inference-only"):
            GraphQuantProfile.from_scalar("w8a8")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            GraphQuantProfile.from_scalar("xyz123")


class TestPresetExpansion:
    def test_dsv4_fp8_fp4(self):
        p = GraphQuantProfile.from_preset("deepseek_v4_fp8_fp4")
        assert p.routed_expert_weight_dtype == Dtype.FP4
        assert p.routed_expert_compute_dtype == Dtype.FP8_E4M3
        assert p.moe_act_dtype == Dtype.FP8_E4M3
        assert p.attn_compute_dtype == Dtype.BF16
        assert p.shared_expert_compute_dtype == Dtype.BF16

    def test_dsv4_paper_fp4(self):
        p = GraphQuantProfile.from_preset("deepseek_v4_paper_fp4")
        assert p.routed_expert_compute_dtype == Dtype.FP4
        assert p.routed_expert_weight_dtype == Dtype.FP4

    def test_fp8_mixed(self):
        p = GraphQuantProfile.from_preset("fp8_mixed")
        assert p.routed_expert_compute_dtype == Dtype.FP8_E4M3
        assert p.moe_act_dtype == Dtype.FP8_E4M3
        assert p.routed_expert_weight_dtype == Dtype.BF16

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="unknown"):
            GraphQuantProfile.from_preset("nonexistent_preset")


class TestFromDict:
    def test_basic(self):
        p = GraphQuantProfile.from_dict({
            "routed_expert_compute_dtype": "fp8_e4m3",
            "moe_act_dtype": "fp8_e4m3",
        })
        assert p.routed_expert_compute_dtype == Dtype.FP8_E4M3
        assert p.effective_moe_act_dtype() == Dtype.FP8_E4M3

    def test_with_preset(self):
        p = GraphQuantProfile.from_dict({
            "quant_preset": "deepseek_v4_fp8_fp4",
            "moe_act_dtype": "fp8_e5m2",  # override
        })
        assert p.routed_expert_weight_dtype == Dtype.FP4  # from preset
        assert p.moe_act_dtype == Dtype.FP8_E5M2  # override wins

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="unknown dtype"):
            GraphQuantProfile.from_dict({"routed_expert_compute_dtype": "bad_dtype"})


class TestEffectiveDtypes:
    def test_effective_moe_act_fallback(self):
        p = GraphQuantProfile()
        assert p.effective_moe_act_dtype() == Dtype.BF16  # falls back to act_dtype

    def test_effective_attn_act_fallback(self):
        p = GraphQuantProfile()
        assert p.effective_attn_act_dtype() == Dtype.BF16

    def test_effective_residual_fallback(self):
        p = GraphQuantProfile()
        assert p.effective_residual_dtype() == Dtype.BF16

    def test_fp4_stored_bytes(self):
        assert Dtype.FP4.stored_bytes == 0.5625  # 0.5 + 2/32

    def test_bf16_stored_bytes(self):
        assert Dtype.BF16.stored_bytes == 2.0


# ── Regression tests for post-review bug fixes ────────────────────────────────

class TestBoolFlagParsing:
    """Bug #2: bool('false') == True was wrong — str-to-bool must handle strings."""

    def test_string_false_disables_flag(self):
        p = GraphQuantProfile.from_dict({"ln_softmax_promote_fp32": "false"})
        assert p.ln_softmax_promote_fp32 is False

    def test_string_true_enables_flag(self):
        p = GraphQuantProfile.from_dict({"ln_softmax_promote_fp32": "true"})
        assert p.ln_softmax_promote_fp32 is True

    def test_string_zero_disables_flag(self):
        p = GraphQuantProfile.from_dict({"assume_all_casts_fused": "0"})
        assert p.assume_all_casts_fused is False


class TestKvCacheDtype:
    """Bug #3: kv_cache_dtype was missing from _dtype_fields (now uses fields())."""

    def test_kv_cache_dtype_from_dict(self):
        p = GraphQuantProfile.from_dict({"kv_cache_dtype": "fp8_e4m3"})
        assert p.kv_cache_dtype == Dtype.FP8_E4M3

    def test_kv_cache_dtype_defaults_to_bf16(self):
        p = GraphQuantProfile.from_dict({})
        assert p.kv_cache_dtype == Dtype.BF16


class TestParamDtypeIndependence:
    """Bug #6: param_dtype was derived from act_dtype — must be independent."""

    def test_preset_param_dtype_is_bf16(self):
        for name in ("fp8_mixed", "deepseek_v4_fp8_fp4", "deepseek_v4_paper_fp4"):
            p = GraphQuantProfile.from_preset(name)
            assert p.param_dtype == Dtype.BF16, f"{name}: param_dtype should be BF16"

    def test_dict_act_dtype_does_not_affect_param_dtype(self):
        p = GraphQuantProfile.from_dict({"act_dtype": "fp8_e4m3"})
        assert p.act_dtype == Dtype.FP8_E4M3
        assert p.param_dtype == Dtype.BF16


class TestStrictParsePostValidation:
    """Bug #7: validate after parsing, not before."""

    def test_unknown_dtype_not_silently_bf16(self):
        with pytest.raises(ValueError, match="unknown dtype"):
            GraphQuantProfile.from_dict({"routed_expert_compute_dtype": "tf32"})


class TestInferenceOnlyExpansion:
    """Bug #8: expanded inference-only rejection set."""

    @pytest.mark.parametrize("q", ["w4a8", "w4a8kv4", "w8a8kv8", "w8a4"])
    def test_inference_only_rejection(self, q):
        with pytest.raises(ValueError, match="inference-only"):
            GraphQuantProfile.from_scalar(q)
