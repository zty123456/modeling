from enum import Enum


class Dtype(str, Enum):
    """Unified element dtype — covers training (FP) and IR (INT/BOOL) domains.

    Note: ``.bytes`` and ``.stored_bytes`` return ``float`` (FP4 = 0.5).
    Callers that multiply by element counts and need an int byte total
    must round explicitly.

    The ``str, Enum`` mixin means ``Dtype.BF16 == "bf16"`` is True.
    """
    # Floating-point (training domain)
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FP4 = "fp4"

    # Integer / boolean / sentinel (IR domain)
    INT8 = "int8"
    INT4 = "int4"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    BOOL = "bool"
    UNKNOWN = "unknown"

    @property
    def bytes(self) -> float:
        return _BYTES[self.value]

    @property
    def block_overhead_bytes_per_elem(self) -> float:
        # FP4 uses MXFP-style block=32 with one BF16 (2B) scale per block.
        return 2.0 / 32.0 if self is Dtype.FP4 else 0.0

    @property
    def stored_bytes(self) -> float:
        return self.bytes + self.block_overhead_bytes_per_elem

    @property
    def itemsize(self) -> float:
        """Backward-compat alias for ``.bytes`` (mirrors IR DType API)."""
        return self.bytes

    @property
    def bits(self) -> float:
        return self.bytes * 8

    def is_floating(self) -> bool:
        return self in _FLOATING

    def is_integer(self) -> bool:
        return self in _INTEGER

    def is_quantized(self) -> bool:
        """True for sub-BF16 dtypes that trigger FP32 promotion penalties."""
        return self in (Dtype.FP8_E4M3, Dtype.FP8_E5M2, Dtype.FP4)

    @classmethod
    def parse(cls, s: str) -> "Dtype":
        """Parse a dtype string, raising ``ValueError`` on unknown names.

        Supports aliases: ``bfloat16`` → BF16, ``mxfp4``/``nvfp4`` → FP4,
        ``float8``/``fp8`` → FP8_E4M3.
        """
        key = s.lower().strip()
        result = _PARSE_ALIASES.get(key)
        if result is None:
            raise ValueError(
                f"unknown dtype {s!r}; valid: fp32, bf16, fp16, fp8_e4m3, fp8_e5m2, fp4"
            )
        return result


_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "fp4": 0.5,
    "int8": 1.0,
    "int4": 0.5,
    "int32": 4.0,
    "int64": 8.0,
    "uint8": 1.0,
    "bool": 1.0,
    "unknown": 2.0,
}

_FLOATING = frozenset({
    Dtype.FP32, Dtype.BF16, Dtype.FP16,
    Dtype.FP8_E4M3, Dtype.FP8_E5M2, Dtype.FP4,
})

_INTEGER = frozenset({
    Dtype.INT8, Dtype.INT4, Dtype.INT32, Dtype.INT64, Dtype.UINT8,
})

_PARSE_ALIASES: dict[str, Dtype] = {
    "fp32": Dtype.FP32, "float32": Dtype.FP32,
    "bf16": Dtype.BF16, "bfloat16": Dtype.BF16,
    "fp16": Dtype.FP16, "float16": Dtype.FP16,
    "fp8": Dtype.FP8_E4M3, "float8": Dtype.FP8_E4M3,
    "fp8_e4m3": Dtype.FP8_E4M3,
    "fp8_e5m2": Dtype.FP8_E5M2,
    "fp4": Dtype.FP4, "mxfp4": Dtype.FP4, "nvfp4": Dtype.FP4,
}

# Back-compat alias: callers that use ``Dtype.FP8`` get E4M3 (V4 forward GEMM).
Dtype.FP8 = Dtype.FP8_E4M3  # type: ignore[attr-defined]
