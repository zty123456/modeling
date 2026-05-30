"""Tests for Dtype.parse — the unified dtype string parser."""
import pytest
from zrt.training.spec.dtype import Dtype


class TestParseKnownNames:
    @pytest.mark.parametrize("name,expected", [
        ("fp32", Dtype.FP32),
        ("bf16", Dtype.BF16),
        ("fp16", Dtype.FP16),
        ("fp8_e4m3", Dtype.FP8_E4M3),
        ("fp8_e5m2", Dtype.FP8_E5M2),
        ("fp4", Dtype.FP4),
    ])
    def test_canonical_names(self, name, expected):
        assert Dtype.parse(name) == expected


class TestParseAliases:
    @pytest.mark.parametrize("alias,expected", [
        ("float32", Dtype.FP32),
        ("bfloat16", Dtype.BF16),
        ("float16", Dtype.FP16),
        ("fp8", Dtype.FP8_E4M3),
        ("float8", Dtype.FP8_E4M3),
        ("mxfp4", Dtype.FP4),
        ("nvfp4", Dtype.FP4),
        ("BF16", Dtype.BF16),
        ("  fp8_e4m3  ", Dtype.FP8_E4M3),
    ])
    def test_aliases(self, alias, expected):
        assert Dtype.parse(alias) == expected


class TestParseUnknown:
    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown dtype"):
            Dtype.parse("tf32")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="unknown dtype"):
            Dtype.parse("")

    def test_int8_raises(self):
        with pytest.raises(ValueError, match="unknown dtype"):
            Dtype.parse("int8")
