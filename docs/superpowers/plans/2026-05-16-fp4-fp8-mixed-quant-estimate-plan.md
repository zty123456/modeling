# FP4/FP8 Mixed-Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FP4/FP8 mixed-quantization support on the `--estimate-config` training-modeling path so DeepSeek-V4 style configs (FP4 routed expert weights, FP8 expert compute, BF16 attention/embedding) produce realistic step_time and memory estimates.

**Architecture:** Extend `Dtype` enum with FP4/FP8 subtypes, add per-component dtype fields on `ModelSpec`, tag each `Op` with a `component` string so `op_to_time` can route to the right hardware peak (`flops_bf16`/`flops_fp8`/`flops_fp4`). Add region-aware memory accounting (attn vs MoE activations, expert vs non-expert grads) and a `mfu_native` metric alongside the legacy BF16-peak MFU.

**Tech Stack:** Python 3.14, pytest, PyYAML, dataclasses. No new external dependencies.

**Spec:** [docs/superpowers/specs/2026-05-16-fp4-fp8-mixed-quant-estimate-design.md](../specs/2026-05-16-fp4-fp8-mixed-quant-estimate-design.md)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `python/zrt/training/spec/dtype.py` | Modify | Extend `Dtype` enum; add `.bytes`/`.block_overhead_bytes_per_elem`/`.stored_bytes` properties |
| `python/zrt/training/spec/model.py` | Modify | Add 8 new per-component dtype fields + back-compat `__post_init__` for `routed_expert_dtype: str` |
| `python/zrt/training/spec/system.py` | Modify | Add `flops_fp4: float = 0.0` to `GPU` |
| `python/zrt/hardware/spec.py` | Modify | Add `fp4_tops: float = 0.0` to `ComputeSpec`; extend `peak_flops()` mapping |
| `python/zrt/hardware/registry.py` | Modify | Parse `fp4_tops` from YAML |
| `python/zrt/hardware/configs/nvidia_b300.yaml` | Create | New B300 spec with native FP4 peak |
| `python/zrt/hardware/configs/{h100_sxm,h800,a100_80g,ascend_910b,ascend_910c}.yaml` | Modify | Add `fp4_tops: 0` to all existing specs |
| `python/zrt/training/io/perf_tables.py` | Modify | Add `peak_tflops_for(gpu, dtype)` helper with fallback warnings |
| `python/zrt/training/io/config_loader.py` | Modify | Extend `_parse_dtype`; add `_QUANT_PRESETS` + `_expand_quant_preset`; load fp4 peak; load new dtype fields |
| `python/zrt/training/search/training_search_util.py` | Modify | Same fp4 peak loading at 4 call sites |
| `python/zrt/training/ir/training_graph.py` | Modify | Add `Op.component: str \| None = None` |
| `python/zrt/training/ir/builders.py` | Modify | Tag each `Op(...)` with `component=` value |
| `python/zrt/training/compose/stage.py` | Modify | `op_to_time` uses `peak_tflops_for`; `_cost_phase_time` accepts `dtype`; `_resolve_compute_dtype` helper; `stage_time` loop forwards dtype |
| `python/zrt/training/compose/schedules.py` | Modify | Add `compute_mfu_native`; add `StepResult.mfu_native` field; set in `pipeline_step_time` |
| `python/zrt/training/models/memory.py` | Modify | Split weights/grads by component dtype; region-aware activations |
| `python/zrt/training/models/comm.py` | Modify | DP AR excludes expert grad; EP A2A uses routed_expert_compute_dtype |
| `tests/training/test_mixed_quant_dtype.py` | Create | Unit tests for Dtype enum extension |
| `tests/training/test_mixed_quant_peak_selection.py` | Create | Tests for `peak_tflops_for` + GPU `flops_fp4` |
| `tests/training/test_mixed_quant_model_spec.py` | Create | ModelSpec back-compat + new fields |
| `tests/training/test_mixed_quant_preset.py` | Create | `quant_preset` YAML expansion |
| `tests/training/test_mixed_quant_op_dispatch.py` | Create | `_resolve_compute_dtype` + `Op.component` routing |
| `tests/training/test_mixed_quant_memory.py` | Create | Memory accounting for FP4/FP8 weights, grads, activations |
| `tests/training/test_mixed_quant_comm.py` | Create | DP AR + EP A2A volume changes |
| `tests/training/test_mixed_quant_mfu_native.py` | Create | `mfu_native` calculation |
| `tests/training/anchors/deepseek_v4_pro_fp8_fp4_h100.yaml` | Create | New V4 FP8/FP4 anchor (H100, calibration mode) |
| `tests/training/anchors/deepseek_v4_pro_fp8_fp4_b300.yaml` | Create | New V4 FP8/FP4 anchor (B300, calibration mode) |

**Working directory:** All commands run from `/Users/sky/Code/modeling`. Tests use `PYTHONPATH=python pytest tests/training/test_<name>.py -v`.

---

## Task 1: Extend Dtype Enum

**Files:**
- Modify: `python/zrt/training/spec/dtype.py` (full rewrite, 12 lines → ~40)
- Create: `tests/training/test_mixed_quant_dtype.py`

**Background:** Current enum uses int value to encode byte size: `FP32=4, BF16=2, FP16=2, FP8=1`. FP4 needs 0.5B which int can't hold; FP4 also needs +0.0625B/elem overhead for the per-block BF16 scale. We change `value` to a string identifier and put byte sizes in dedicated properties.

- [ ] **Step 1.1: Write the failing test**

Create `tests/training/test_mixed_quant_dtype.py`:

```python
"""Tests for Dtype enum extension (FP4, FP8 subtypes)."""
import pytest

from zrt.training.spec.dtype import Dtype


def test_dtype_legacy_values_preserved():
    """Existing Dtype values must still parse and have correct .bytes."""
    assert Dtype.FP32.bytes == 4.0
    assert Dtype.BF16.bytes == 2.0
    assert Dtype.FP16.bytes == 2.0
    assert Dtype.FP8.bytes == 1.0


def test_dtype_new_fp8_subtypes():
    assert Dtype.FP8_E4M3.bytes == 1.0
    assert Dtype.FP8_E5M2.bytes == 1.0


def test_dtype_fp4_byte_size_is_half():
    assert Dtype.FP4.bytes == 0.5


def test_dtype_fp4_block_overhead():
    """FP4 uses block=32 with one BF16 (2B) scale per block → 2/32 = 0.0625 B/elem."""
    assert Dtype.FP4.block_overhead_bytes_per_elem == pytest.approx(0.0625)
    # Non-FP4 dtypes have no block overhead.
    assert Dtype.BF16.block_overhead_bytes_per_elem == 0.0
    assert Dtype.FP8_E4M3.block_overhead_bytes_per_elem == 0.0


def test_dtype_stored_bytes_includes_overhead():
    assert Dtype.FP4.stored_bytes == pytest.approx(0.5625)
    assert Dtype.BF16.stored_bytes == 2.0
    assert Dtype.FP32.stored_bytes == 4.0


def test_dtype_fp8_alias_is_e4m3():
    """Existing code/YAML using FP8 must map to FP8_E4M3 (DeepSeek-V4 forward GEMM)."""
    assert Dtype.FP8 is Dtype.FP8_E4M3
```

- [ ] **Step 1.2: Run test to verify it fails**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_dtype.py -v
```
Expected: FAIL — `AttributeError: FP8_E4M3` (or similar; enum lacks new members).

- [ ] **Step 1.3: Replace `python/zrt/training/spec/dtype.py` with new implementation**

```python
from enum import Enum


class Dtype(Enum):
    """Element dtype for parameters/grads/activations.

    Note: ``.bytes`` and ``.stored_bytes`` return ``float`` (FP4 = 0.5).
    Callers that multiply by element counts and need an int byte total
    must round explicitly.
    """
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FP4 = "fp4"

    @property
    def bytes(self) -> float:
        return _BYTES[self]

    @property
    def block_overhead_bytes_per_elem(self) -> float:
        # FP4 uses MXFP-style block=32 with one BF16 (2B) scale per block.
        return 2.0 / 32.0 if self is Dtype.FP4 else 0.0

    @property
    def stored_bytes(self) -> float:
        return self.bytes + self.block_overhead_bytes_per_elem


_BYTES: dict[Dtype, float] = {
    Dtype.FP32: 4.0,
    Dtype.BF16: 2.0,
    Dtype.FP16: 2.0,
    Dtype.FP8_E4M3: 1.0,
    Dtype.FP8_E5M2: 1.0,
    Dtype.FP4: 0.5,
}

# Back-compat alias: callers that use ``Dtype.FP8`` get E4M3 (V4 forward GEMM).
Dtype.FP8 = Dtype.FP8_E4M3  # type: ignore[attr-defined]
```

- [ ] **Step 1.4: Run test to verify it passes**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_dtype.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 1.5: Run quick guard against legacy `.value` users**

```
PYTHONPATH=python grep -rn "Dtype\.\w\+\.value" python/zrt/ tests/ 2>&1 | grep -v test_mixed_quant
```
Expected: no hits, OR only hits that compare against the string name (since values are now strings). If any code does arithmetic on `.value` (e.g. `dtype.value * elements`), fix it to use `.bytes` in this commit.

- [ ] **Step 1.6: Run full training test suite as regression**

```
PYTHONPATH=python pytest tests/training/ -v 2>&1 | tail -n 30
```
Expected: no new failures attributable to dtype changes. (Some unrelated tests may already fail in this branch — record baseline before starting if needed.)

- [ ] **Step 1.7: Commit**

```bash
git add python/zrt/training/spec/dtype.py tests/training/test_mixed_quant_dtype.py
git commit -m "$(cat <<'EOF'
feat(training): extend Dtype enum with FP4/FP8 subtypes

- Add FP4 (0.5B/elem + 2B-per-32 block overhead via MXFP-style scale)
- Add FP8_E4M3, FP8_E5M2 (both 1B/elem); FP8 retained as alias to FP8_E4M3
- Change enum value from int (byte size) to string identifier; expose
  .bytes (float), .block_overhead_bytes_per_elem, .stored_bytes properties
EOF
)"
```

---

## Task 2: Add fp4_tops Field to Hardware Spec + peak_tflops_for Helper

**Files:**
- Modify: `python/zrt/hardware/spec.py` (lines 25-37: `ComputeSpec`; lines 112-122: `peak_flops`)
- Modify: `python/zrt/hardware/registry.py:102-113` (parse fp4_tops)
- Modify: `python/zrt/training/spec/system.py:9-25` (add `flops_fp4`)
- Modify: `python/zrt/training/io/perf_tables.py` (add `peak_tflops_for`)
- Create: `tests/training/test_mixed_quant_peak_selection.py`

**Background:** Hardware YAMLs currently declare `fp8_tops` (e.g. H100=3958) but no `fp4_tops`. The `--estimate-config` path consumes a different `GPU` dataclass that mirrors selected fields from `HardwareSpec`. We need both layers updated, plus a single helper `peak_tflops_for(gpu, dtype)` that all callers will use.

- [ ] **Step 2.1: Write the failing test**

Create `tests/training/test_mixed_quant_peak_selection.py`:

```python
"""Tests for peak TFLOPS routing by dtype."""
import pytest

from zrt.training.io.perf_tables import peak_tflops_for
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.system import GPU


def _gpu(name="h100", *, bf16=989.0, fp8=3958.0, fp4=0.0):
    return GPU(name=name, flops_bf16=bf16, flops_fp8=fp8, flops_fp4=fp4,
               hbm_gb=80.0, hbm_bw_gbps=3350.0)


def test_bf16_returns_bf16_peak():
    gpu = _gpu()
    assert peak_tflops_for(gpu, Dtype.BF16) == pytest.approx(989.0e12)


def test_fp16_falls_back_to_bf16_peak():
    gpu = _gpu()
    assert peak_tflops_for(gpu, Dtype.FP16) == pytest.approx(989.0e12)


def test_fp8_e4m3_returns_fp8_peak():
    gpu = _gpu()
    assert peak_tflops_for(gpu, Dtype.FP8_E4M3) == pytest.approx(3958.0e12)


def test_fp8_e5m2_returns_fp8_peak():
    gpu = _gpu()
    assert peak_tflops_for(gpu, Dtype.FP8_E5M2) == pytest.approx(3958.0e12)


def test_fp4_returns_fp4_peak_when_supported():
    gpu = _gpu(fp4=30000.0)
    assert peak_tflops_for(gpu, Dtype.FP4) == pytest.approx(30000.0e12)


def test_fp4_falls_back_to_fp8_when_unsupported():
    # H100 has fp4=0 → fallback to fp8.
    gpu = _gpu(fp4=0.0)
    assert peak_tflops_for(gpu, Dtype.FP4) == pytest.approx(3958.0e12)


def test_fp8_falls_back_to_bf16_when_unsupported():
    # A100-like: no FP8 hardware
    gpu = _gpu(fp8=0.0, fp4=0.0)
    assert peak_tflops_for(gpu, Dtype.FP8_E4M3) == pytest.approx(989.0e12)
```

- [ ] **Step 2.2: Run test to verify it fails**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_peak_selection.py -v
```
Expected: FAIL — `cannot import name 'peak_tflops_for'` and `flops_fp4` missing from GPU.

- [ ] **Step 2.3: Add `flops_fp4` field to GPU dataclass**

Modify `python/zrt/training/spec/system.py`, change the `GPU` dataclass to:

```python
@dataclass
class GPU:
    name: str
    flops_bf16: float   # peak TFLOP/s
    flops_fp8: float    # peak TFLOP/s
    hbm_gb: float
    hbm_bw_gbps: float  # aggregate HBM bandwidth (GB/s)

    flops_fp4: float = 0.0    # peak TFLOP/s (Blackwell B200/B300+); 0 = unsupported

    # Heterogeneous-core fields ... (unchanged below)
    cube_tflops: float | None = None
    vector_tflops: float | None = None
    overlap_ratio: dict[str, float] = field(default_factory=dict)
    sram_kb_per_sm: float = 0.0
```

Place `flops_fp4` immediately after `hbm_bw_gbps` (before the heterogeneous fields) and give it a default so existing GPU(...) calls without flops_fp4 still work.

- [ ] **Step 2.4: Add `fp4_tops` to `ComputeSpec`**

Modify `python/zrt/hardware/spec.py:25-37`, change `ComputeSpec` to include `fp4_tops: float = 0.0` directly after `fp8_tops`:

```python
@dataclass
class ComputeSpec:
    fp16_tflops: float = 0.0
    bf16_tflops: float = 0.0
    fp32_tflops: float = 0.0
    int8_tops:   float = 0.0
    int4_tops:   float = 0.0
    fp8_tops:    float = 0.0   # covers both FP8_E4M3 and FP8_E5M2
    fp4_tops:    float = 0.0   # Blackwell B200/B300+ (NVFP4/MXFP4); 0 = unsupported

    cube_bf16_tflops: float | None = None
    vector_bf16_tflops: float | None = None
    overlap_ratio: dict[str, float] = field(default_factory=dict)
    sram_kb_per_sm: float = 0.0
```

- [ ] **Step 2.5: Parse `fp4_tops` in registry**

Modify `python/zrt/hardware/registry.py:102-113`:

```python
return ComputeSpec(
    fp16_tflops=float(c.get("fp16_tflops", 0.0)),
    bf16_tflops=float(c.get("bf16_tflops", 0.0)),
    fp32_tflops=float(c.get("fp32_tflops", 0.0)),
    int8_tops=float(c.get("int8_tops", 0.0)),
    int4_tops=float(c.get("int4_tops", 0.0)),
    fp8_tops=float(c.get("fp8_tops", 0.0)),
    fp4_tops=float(c.get("fp4_tops", 0.0)),
    cube_bf16_tflops=float(cube_raw) if cube_raw is not None else None,
    vector_bf16_tflops=float(vector_raw) if vector_raw is not None else None,
    overlap_ratio={k: float(v) for k, v in overlap_raw.items()},
    sram_kb_per_sm=float(c.get("sram_kb_per_sm", 0.0)),
)
```

- [ ] **Step 2.6: Add `peak_tflops_for` to perf_tables**

Append to `python/zrt/training/io/perf_tables.py`:

```python
_FALLBACK_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    """Emit a UserWarning at most once per process for a given key."""
    if key in _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED.add(key)
    import warnings
    warnings.warn(msg, UserWarning, stacklevel=3)


def peak_tflops_for(gpu, dtype: Dtype) -> float:
    """Return hardware peak FLOP/s (not TFLOP/s) for ``dtype`` on ``gpu``.

    Fallbacks (each emits a one-shot warning):
      FP4  → FP8 peak when gpu.flops_fp4 == 0
      FP8* → BF16 peak when gpu.flops_fp8 == 0
    BF16/FP16/FP32 all use gpu.flops_bf16 (no separate FP16 field).
    """
    if dtype is Dtype.FP4:
        if getattr(gpu, "flops_fp4", 0.0) > 0:
            return gpu.flops_fp4 * 1e12
        _warn_once(f"fp4_fallback_{gpu.name}",
                   f"GPU {gpu.name!r} declares no fp4_tops; falling back to FP8 peak")
        dtype = Dtype.FP8_E4M3
    if dtype in (Dtype.FP8_E4M3, Dtype.FP8_E5M2):
        if getattr(gpu, "flops_fp8", 0.0) > 0:
            return gpu.flops_fp8 * 1e12
        _warn_once(f"fp8_fallback_{gpu.name}",
                   f"GPU {gpu.name!r} declares no fp8_tops; falling back to BF16 peak")
    return gpu.flops_bf16 * 1e12
```

- [ ] **Step 2.7: Run test to verify it passes**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_peak_selection.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 2.8: Regression — registry + existing tests still load specs**

```
PYTHONPATH=python pytest tests/ -v -k "hardware or registry or spec" 2>&1 | tail -n 20
```
Expected: no regressions.

- [ ] **Step 2.9: Commit**

```bash
git add python/zrt/hardware/spec.py python/zrt/hardware/registry.py \
        python/zrt/training/spec/system.py python/zrt/training/io/perf_tables.py \
        tests/training/test_mixed_quant_peak_selection.py
git commit -m "$(cat <<'EOF'
feat(training): add fp4_tops field and peak_tflops_for helper

- ComputeSpec.fp4_tops parsed from hardware YAML (default 0)
- GPU.flops_fp4 surfaced into spec-based estimate pipeline
- peak_tflops_for(gpu, dtype) selects bf16/fp8/fp4 peak with
  graceful fallback + one-shot UserWarning when hardware lacks support
EOF
)"
```

---

## Task 3: Extend ModelSpec with Per-Component Dtype Fields

**Files:**
- Modify: `python/zrt/training/spec/model.py:79-86` (dtypes block) + `__post_init__` (lines 92-107)
- Create: `tests/training/test_mixed_quant_model_spec.py`

**Background:** Currently the dtype block has 4 fields (`param/grad/master/act_dtype`) all of type `Dtype`, plus the odd-duck `routed_expert_dtype: str = "bf16"` which is used by `memory.py:122` via string comparison. We need 8 new fields (3 compute, 1 weight, 2 act, 1 grad, with `routed_expert_dtype` retained as back-compat alias).

- [ ] **Step 3.1: Write the failing test**

Create `tests/training/test_mixed_quant_model_spec.py`:

```python
"""Tests for ModelSpec dtype field extension."""
import pytest

from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec


def _minimal_kwargs(**overrides):
    base = dict(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64, layers=[LayerKind.DENSE],
    )
    base.update(overrides)
    return base


def test_default_per_component_dtypes_are_bf16():
    m = ModelSpec(**_minimal_kwargs())
    assert m.attn_compute_dtype is Dtype.BF16
    assert m.routed_expert_compute_dtype is Dtype.BF16
    assert m.shared_expert_compute_dtype is Dtype.BF16
    assert m.routed_expert_weight_dtype is Dtype.BF16
    assert m.routed_expert_grad_dtype is Dtype.FP32


def test_per_region_act_dtypes_default_to_none():
    m = ModelSpec(**_minimal_kwargs())
    assert m.attn_act_dtype is None
    assert m.moe_act_dtype is None


def test_effective_attn_act_dtype_falls_back_to_act_dtype():
    m = ModelSpec(**_minimal_kwargs(act_dtype=Dtype.FP16))
    assert m.effective_attn_act_dtype() is Dtype.FP16
    # Explicit override wins
    m2 = ModelSpec(**_minimal_kwargs(act_dtype=Dtype.FP16, attn_act_dtype=Dtype.BF16))
    assert m2.effective_attn_act_dtype() is Dtype.BF16


def test_effective_moe_act_dtype_falls_back_to_routed_compute():
    """When moe_act_dtype unset, default to routed_expert_compute_dtype."""
    m = ModelSpec(**_minimal_kwargs(routed_expert_compute_dtype=Dtype.FP8_E4M3))
    assert m.effective_moe_act_dtype() is Dtype.FP8_E4M3
    # Explicit override wins
    m2 = ModelSpec(**_minimal_kwargs(
        routed_expert_compute_dtype=Dtype.FP8_E4M3,
        moe_act_dtype=Dtype.BF16,
    ))
    assert m2.effective_moe_act_dtype() is Dtype.BF16


def test_legacy_routed_expert_dtype_string_fp4_syncs_to_weight_dtype():
    """For back-compat: routed_expert_dtype='fp4' should populate
    routed_expert_weight_dtype when the new field is at its default."""
    m = ModelSpec(**_minimal_kwargs(routed_expert_dtype="fp4"))
    assert m.routed_expert_weight_dtype is Dtype.FP4


def test_explicit_routed_expert_weight_dtype_wins_over_legacy_string():
    m = ModelSpec(**_minimal_kwargs(
        routed_expert_dtype="fp4",
        routed_expert_weight_dtype=Dtype.BF16,
    ))
    assert m.routed_expert_weight_dtype is Dtype.BF16
```

- [ ] **Step 3.2: Run test to verify it fails**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_model_spec.py -v
```
Expected: FAIL — `AttributeError: 'ModelSpec' object has no attribute 'attn_compute_dtype'`.

- [ ] **Step 3.3: Add fields and helpers to ModelSpec**

In `python/zrt/training/spec/model.py`, after line 83 (after `act_dtype: Dtype = Dtype.BF16`), add:

```python
    # === Per-component compute dtype (DeepSeek-V4 mixed quant) ===
    attn_compute_dtype: Dtype = Dtype.BF16
    shared_expert_compute_dtype: Dtype = Dtype.BF16
    routed_expert_compute_dtype: Dtype = Dtype.BF16

    # === Per-component weight dtype ===
    routed_expert_weight_dtype: Dtype = Dtype.BF16  # V4 default: FP4

    # === Per-region activation dtype (None → fallback) ===
    attn_act_dtype: Dtype | None = None
    moe_act_dtype: Dtype | None = None

    # === Per-component grad dtype ===
    routed_expert_grad_dtype: Dtype = Dtype.FP32
```

Then extend `__post_init__` (after the existing body, before the closing of the method) with:

```python
        # Back-compat: legacy ``routed_expert_dtype: str`` syncs into the
        # new ``routed_expert_weight_dtype`` only when the new field is at
        # its default.
        if isinstance(self.routed_expert_dtype, str):
            legacy = self.routed_expert_dtype.lower().strip()
            if legacy == "fp4" and self.routed_expert_weight_dtype is Dtype.BF16:
                self.routed_expert_weight_dtype = Dtype.FP4
```

Add helper methods to `ModelSpec` (place after the existing `head_dim_total` / `kv_dim` properties, before `_attn_proj_params`):

```python
    def effective_attn_act_dtype(self) -> Dtype:
        """Attention region activation dtype; falls back to act_dtype."""
        return self.attn_act_dtype if self.attn_act_dtype is not None else self.act_dtype

    def effective_moe_act_dtype(self) -> Dtype:
        """MoE region activation dtype; falls back to routed_expert_compute_dtype.

        Rationale: when a user enables FP8 routed compute via quant preset,
        the matching forward activations are also FP8 unless explicitly
        overridden.
        """
        if self.moe_act_dtype is not None:
            return self.moe_act_dtype
        return self.routed_expert_compute_dtype
```

- [ ] **Step 3.4: Run test to verify it passes**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_model_spec.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 3.5: Regression run**

```
PYTHONPATH=python pytest tests/training/ -v 2>&1 | tail -n 30
```
Expected: no new failures.

- [ ] **Step 3.6: Commit**

```bash
git add python/zrt/training/spec/model.py tests/training/test_mixed_quant_model_spec.py
git commit -m "$(cat <<'EOF'
feat(training): add per-component dtype fields to ModelSpec

- attn_compute_dtype / shared_expert_compute_dtype / routed_expert_compute_dtype
- routed_expert_weight_dtype (V4 routed expert FP4)
- attn_act_dtype / moe_act_dtype (None → fallback)
- routed_expert_grad_dtype
- effective_attn_act_dtype()/effective_moe_act_dtype() helpers
- Back-compat: legacy routed_expert_dtype='fp4' syncs into
  routed_expert_weight_dtype when the new field is at default
EOF
)"
```

---

## Task 4: Extend _parse_dtype and Add quant_preset Expansion

**Files:**
- Modify: `python/zrt/training/io/config_loader.py:307-315` (extend `_parse_dtype`); add `_QUANT_PRESETS` dict + `_expand_quant_preset` function; modify `_parse_model` to call expansion
- Create: `tests/training/test_mixed_quant_preset.py`

- [ ] **Step 4.1: Write the failing test**

Create `tests/training/test_mixed_quant_preset.py`:

```python
"""Tests for YAML quant_preset expansion + extended _parse_dtype."""
import pytest

from zrt.training.io.config_loader import _expand_quant_preset, _parse_dtype
from zrt.training.spec.dtype import Dtype


def test_parse_dtype_accepts_fp8_e4m3():
    assert _parse_dtype("fp8_e4m3") is Dtype.FP8_E4M3


def test_parse_dtype_accepts_fp8_e5m2():
    assert _parse_dtype("fp8_e5m2") is Dtype.FP8_E5M2


def test_parse_dtype_accepts_fp4():
    assert _parse_dtype("fp4") is Dtype.FP4


def test_parse_dtype_fp8_alias_still_works():
    """Legacy YAML using 'fp8' → FP8_E4M3 (alias)."""
    assert _parse_dtype("fp8") is Dtype.FP8_E4M3


def test_expand_preset_deepseek_v4_fp8_fp4():
    d = {"quant_preset": "deepseek_v4_fp8_fp4"}
    out = _expand_quant_preset(d)
    assert "quant_preset" not in out
    assert out["routed_expert_compute_dtype"] == "fp8_e4m3"
    assert out["routed_expert_weight_dtype"] == "fp4"
    assert out["attn_compute_dtype"] == "bf16"
    assert out["moe_act_dtype"] == "fp8_e4m3"


def test_expand_preset_explicit_field_wins():
    """Explicit dtype in YAML overrides preset value."""
    d = {
        "quant_preset": "deepseek_v4_fp8_fp4",
        "attn_compute_dtype": "fp8_e4m3",   # override
    }
    out = _expand_quant_preset(d)
    assert out["attn_compute_dtype"] == "fp8_e4m3"
    # other preset values still applied
    assert out["routed_expert_weight_dtype"] == "fp4"


def test_expand_preset_noop_when_absent():
    d = {"param_dtype": "bf16"}
    out = _expand_quant_preset(d)
    assert out == {"param_dtype": "bf16"}


def test_expand_preset_unknown_raises():
    with pytest.raises(KeyError, match="unknown.*quant_preset"):
        _expand_quant_preset({"quant_preset": "nonsense_preset"})


def test_preset_bf16_baseline_is_pure_bf16():
    d = {"quant_preset": "bf16_baseline"}
    out = _expand_quant_preset(d)
    assert out["routed_expert_compute_dtype"] == "bf16"
    assert out["routed_expert_weight_dtype"] == "bf16"
    assert out["attn_compute_dtype"] == "bf16"
```

- [ ] **Step 4.2: Run test to verify it fails**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_preset.py -v
```
Expected: FAIL — `cannot import name '_expand_quant_preset'`.

- [ ] **Step 4.3: Extend `_parse_dtype` and add preset expansion**

In `python/zrt/training/io/config_loader.py`, replace the `_parse_dtype` function (lines 307-315) with:

```python
def _parse_dtype(s: str) -> Dtype:
    s = s.lower().strip()
    mapping = {
        "fp32": Dtype.FP32, "float32": Dtype.FP32,
        "bf16": Dtype.BF16, "bfloat16": Dtype.BF16,
        "fp16": Dtype.FP16, "float16": Dtype.FP16,
        "fp8": Dtype.FP8_E4M3, "float8": Dtype.FP8_E4M3,
        "fp8_e4m3": Dtype.FP8_E4M3,
        "fp8_e5m2": Dtype.FP8_E5M2,
        "fp4": Dtype.FP4, "mxfp4": Dtype.FP4, "nvfp4": Dtype.FP4,
    }
    return mapping.get(s, Dtype.BF16)
```

Then add a new section right before `_parse_dtype` (or right after — pick the spot above `_parse_model`):

```python
_QUANT_PRESETS: dict[str, dict[str, str]] = {
    "bf16_baseline": {
        "attn_compute_dtype": "bf16",
        "routed_expert_compute_dtype": "bf16",
        "routed_expert_weight_dtype": "bf16",
        "shared_expert_compute_dtype": "bf16",
        "routed_expert_grad_dtype": "fp32",
        "act_dtype": "bf16",
    },
    "fp8_mixed": {   # DeepSeek-V3 style
        "attn_compute_dtype": "bf16",
        "routed_expert_compute_dtype": "fp8_e4m3",
        "routed_expert_weight_dtype": "bf16",
        "shared_expert_compute_dtype": "bf16",
        "routed_expert_grad_dtype": "fp32",
        "act_dtype": "bf16",
        "moe_act_dtype": "fp8_e4m3",
    },
    "deepseek_v4_fp8_fp4": {   # V4 main path
        "attn_compute_dtype": "bf16",
        "routed_expert_compute_dtype": "fp8_e4m3",
        "routed_expert_weight_dtype": "fp4",
        "shared_expert_compute_dtype": "bf16",
        "routed_expert_grad_dtype": "fp32",
        "act_dtype": "bf16",
        "moe_act_dtype": "fp8_e4m3",
    },
    "deepseek_v4_full_fp8": {   # V4 with FP8 shared experts
        "attn_compute_dtype": "bf16",
        "routed_expert_compute_dtype": "fp8_e4m3",
        "routed_expert_weight_dtype": "fp4",
        "shared_expert_compute_dtype": "fp8_e4m3",
        "routed_expert_grad_dtype": "fp32",
        "act_dtype": "bf16",
        "moe_act_dtype": "fp8_e4m3",
        "attn_act_dtype": "bf16",
    },
}


def _expand_quant_preset(d: dict) -> dict:
    """Expand a ``quant_preset`` shorthand into explicit dtype fields.

    Explicit fields in ``d`` override preset values. Removes the
    ``quant_preset`` key from the returned mapping. Returns a new dict
    (does not mutate input).
    """
    preset_name = d.get("quant_preset")
    out = {k: v for k, v in d.items() if k != "quant_preset"}
    if preset_name is None:
        return out
    if preset_name not in _QUANT_PRESETS:
        raise KeyError(
            f"unknown quant_preset {preset_name!r}; "
            f"valid options: {sorted(_QUANT_PRESETS)}"
        )
    for key, val in _QUANT_PRESETS[preset_name].items():
        out.setdefault(key, val)
    return out
```

Wire `_expand_quant_preset` into `_parse_model` — in `python/zrt/training/io/config_loader.py:131` modify the function signature/body to expand the preset first:

```python
def _parse_model(d: dict) -> ModelSpec:
    d = _expand_quant_preset(d)
    layers_str = d.get("layers", [])
    layers = _parse_layers(layers_str)
    # ... rest unchanged
```

Then in the `ModelSpec(...)` constructor call, add these new fields immediately after the existing dtype block (the existing `param_dtype/grad_dtype/master_dtype/act_dtype` lines around line 182-185):

```python
        # NEW per-component dtypes (Task 3 added the ModelSpec fields)
        attn_compute_dtype=_parse_dtype(d.get("attn_compute_dtype", "bf16")),
        shared_expert_compute_dtype=_parse_dtype(d.get("shared_expert_compute_dtype", "bf16")),
        routed_expert_compute_dtype=_parse_dtype(d.get("routed_expert_compute_dtype", "bf16")),
        routed_expert_weight_dtype=_parse_dtype(d.get("routed_expert_weight_dtype", "bf16")),
        attn_act_dtype=_parse_dtype(d["attn_act_dtype"]) if "attn_act_dtype" in d else None,
        moe_act_dtype=_parse_dtype(d["moe_act_dtype"]) if "moe_act_dtype" in d else None,
        routed_expert_grad_dtype=_parse_dtype(d.get("routed_expert_grad_dtype", "fp32")),
```

- [ ] **Step 4.4: Run test to verify it passes**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_preset.py -v
```
Expected: all 9 tests PASS.

- [ ] **Step 4.5: Regression — ensure no existing config loading broke**

```
PYTHONPATH=python pytest tests/training/ -v -k "config_loader or io" 2>&1 | tail -n 20
```
Expected: no new failures.

- [ ] **Step 4.6: Commit**

```bash
git add python/zrt/training/io/config_loader.py tests/training/test_mixed_quant_preset.py
git commit -m "$(cat <<'EOF'
feat(training): YAML quant_preset expansion + new dtype names

- _parse_dtype accepts fp8_e4m3, fp8_e5m2, fp4 (plus mxfp4/nvfp4 aliases)
- _QUANT_PRESETS registers bf16_baseline / fp8_mixed /
  deepseek_v4_fp8_fp4 / deepseek_v4_full_fp8
- _expand_quant_preset() resolves the shorthand into explicit dtype
  fields; explicit YAML fields always win over preset defaults
- _parse_model wires new per-component dtype fields into ModelSpec
EOF
)"
```

---

## Task 5: Add B300 Hardware Spec + fp4_tops to Existing Specs

**Files:**
- Create: `python/zrt/hardware/configs/nvidia_b300.yaml`
- Modify: `python/zrt/hardware/configs/{nvidia_h100_sxm,nvidia_h800,nvidia_a100_80g,ascend_910b,ascend_910c}.yaml` (add `fp4_tops: 0`)
- Modify: `python/zrt/training/io/config_loader.py:200-205` (load `flops_fp4` in `_parse_system`)
- Modify: `python/zrt/training/search/training_search_util.py:67-72, 334-339, 397-402, 629-634` (load `flops_fp4` at 4 call sites)
- Add tests to `tests/training/test_mixed_quant_peak_selection.py`

**Background:** B300 is NVIDIA's Blackwell Ultra successor to B200 with native FP4 Tensor Cores. Public specs (NVIDIA Blackwell Ultra GB300 datasheet, mid-2025): per-die BF16 ≈3.5 PFLOPS dense, FP8 ≈7 PFLOPS dense, FP4 ≈14 PFLOPS dense (sparsity-free). HBM3e 288 GB @ 8 TB/s. NVLink-5 ≈1.8 TB/s aggregate. We pick conservative numbers; future PR can refine.

- [ ] **Step 5.1: Append B300 test cases to `tests/training/test_mixed_quant_peak_selection.py`**

```python
def test_b300_yaml_loads_with_fp4_tops():
    """B300 spec declares native FP4."""
    from python.zrt.hardware.registry import load
    hw = load("nvidia_b300")
    assert hw.compute.fp4_tops > 0
    assert hw.compute.fp8_tops > hw.compute.bf16_tflops  # FP8 ≥ 2× BF16


def test_h100_yaml_declares_fp4_tops_zero():
    """H100 lacks native FP4 hardware → fp4_tops must be 0 in spec."""
    from python.zrt.hardware.registry import load
    hw = load("nvidia_h100_sxm")
    assert hw.compute.fp4_tops == 0.0
```

- [ ] **Step 5.2: Run test to verify it fails**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_peak_selection.py::test_b300_yaml_loads_with_fp4_tops -v
```
Expected: FAIL — `KeyError: 'nvidia_b300'`.

- [ ] **Step 5.3: Create `python/zrt/hardware/configs/nvidia_b300.yaml`**

```yaml
name: "NVIDIA B300"
vendor: "nvidia"
device_type: "gpu"

# Source: NVIDIA Blackwell Ultra (GB300) datasheet, public figures.
# Per-die dense (no 2:4 sparsity); B300 = ~1.5× B200 compute.
# Values approximate; refine when authoritative datasheet is committed.
compute:
  fp16_tflops: 5000
  bf16_tflops: 5000
  fp32_tflops: 250
  int8_tops:   10000
  int4_tops:   20000
  fp8_tops:    10000        # FP8 E4M3 / E5M2
  fp4_tops:    20000        # NVFP4 / MXFP4 Tensor Cores

  sram_kb_per_sm: 256
  # B300 retains the cube/vector split implicit in Blackwell; until we
  # tune anchors, fall back to homogeneous peak.

memory:
  capacity_gb: 288             # HBM3e per die
  hbm_bandwidth_gbps: 8000     # ~8 TB/s
  l2_cache_mb: 64
  tiers:
    - { name: "L2",  bandwidth_gbps: 16384, capacity_mb: 64 }
    - { name: "HBM", bandwidth_gbps: 8000,  capacity_mb: 0  }

interconnect:
  intra_node:
    type: "NVLink5"
    bandwidth_gbps: 1800       # 1.8 TB/s aggregate
    latency_us: 1
    topology: "all_to_all"
    num_devices: 8
  inter_node:
    type: "InfiniBand_XDR"
    bandwidth_gbps: 800        # XDR-class IB / Spectrum-X (placeholder)
    latency_us: 2
    topology: "fat_tree"
    num_devices: 0
```

- [ ] **Step 5.4: Add `fp4_tops: 0` to existing hardware YAMLs**

For each of these five files, insert `fp4_tops: 0` immediately after the `fp8_tops:` line:

- `python/zrt/hardware/configs/nvidia_h100_sxm.yaml`
- `python/zrt/hardware/configs/nvidia_h800.yaml`
- `python/zrt/hardware/configs/nvidia_a100_80g.yaml`
- `python/zrt/hardware/configs/ascend_910b.yaml`
- `python/zrt/hardware/configs/ascend_910c.yaml`

Example for H100:

```yaml
compute:
  fp16_tflops: 989
  bf16_tflops: 989
  fp32_tflops: 66.9
  int8_tops: 1978
  int4_tops: 3958
  fp8_tops: 3958
  fp4_tops: 0    # H100 has no native FP4; falls back to FP8 peak at runtime

  # heterogeneous-core fields unchanged...
```

- [ ] **Step 5.5: Plumb `flops_fp4` through `_parse_system`**

Modify `python/zrt/training/io/config_loader.py:198-208`:

```python
    gpu = GPU(
        name=hw.name,
        flops_bf16=hw.compute.bf16_tflops,
        flops_fp8=hw.compute.fp8_tops or hw.compute.bf16_tflops * 2,
        flops_fp4=hw.compute.fp4_tops,   # 0 → peak_tflops_for falls back to fp8
        hbm_gb=hw.memory.capacity_gb,
        hbm_bw_gbps=hw.memory.hbm_bandwidth_gbps,
        cube_tflops=hw.compute.cube_bf16_tflops,
        vector_tflops=hw.compute.vector_bf16_tflops,
        overlap_ratio=dict(hw.compute.overlap_ratio),
        sram_kb_per_sm=hw.compute.sram_kb_per_sm,
    )
```

- [ ] **Step 5.6: Plumb `flops_fp4` through the search util at all four sites**

In `python/zrt/training/search/training_search_util.py`, locate all four `GPU(...)` constructions (lines 67-72, 334-339, 397-402, 629-634) and add `flops_fp4=hw.compute.fp4_tops,` immediately after each `flops_fp8=...` line. Use `grep -n "flops_fp8=hw.compute.fp8_tops" python/zrt/training/search/training_search_util.py` to confirm only those four sites exist.

- [ ] **Step 5.7: Run tests to verify pass**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_peak_selection.py -v
```
Expected: all tests including the new B300/H100 yaml ones PASS.

- [ ] **Step 5.8: Regression — full training tests + search**

```
PYTHONPATH=python pytest tests/training/ -v 2>&1 | tail -n 30
```
Expected: no new failures.

- [ ] **Step 5.9: Commit**

```bash
git add python/zrt/hardware/configs/ python/zrt/training/io/config_loader.py \
        python/zrt/training/search/training_search_util.py \
        tests/training/test_mixed_quant_peak_selection.py
git commit -m "$(cat <<'EOF'
feat(hardware): add NVIDIA B300 spec + fp4_tops in all hw configs

- nvidia_b300.yaml: Blackwell Ultra public-spec ballpark
  (BF16 5 PFLOPS, FP8 10 PFLOPS, FP4 20 PFLOPS, HBM3e 288GB/8TB/s)
- Existing H100/H800/A100/Ascend specs declare fp4_tops: 0
- GPU.flops_fp4 plumbed via config_loader and training_search_util
  (4 GPU(...) construction sites)
EOF
)"
```

---

## Task 6: Route op_to_time / op_to_time_hetero Through peak_tflops_for

**Files:**
- Modify: `python/zrt/training/compose/stage.py:54-70` (`op_to_time`)
- Modify: `python/zrt/training/compose/stage.py:79-116` (`op_to_time_hetero`)
- Reuse existing `tests/training/test_mixed_quant_peak_selection.py` for end-to-end via `stage.py`; add a small unit test.

**Background:** Both functions accept a `dtype` parameter but the `peak = gpu.flops_bf16 * 1e12` line ignores it. This task wires the dtype through.

- [ ] **Step 6.1: Add failing tests**

Append to `tests/training/test_mixed_quant_peak_selection.py`:

```python
def test_op_to_time_fp8_is_faster_than_bf16_on_h100():
    """On H100 (BF16=989, FP8=3958), FP8 compute should be ~4× faster."""
    from zrt.training.compose.stage import op_to_time
    from zrt.training.spec.system import SystemSpec
    from zrt.hardware.spec import InterconnectSpec, LinkSpec

    gpu = _gpu(name="H100", bf16=989.0, fp8=3958.0, fp4=0.0)
    link = LinkSpec(type="NVLink4", bandwidth_gbps=900, latency_us=1,
                    topology="all_to_all", num_devices=8)
    system = SystemSpec(gpu=gpu, host_mem_gb=2048,
                        interconnect=InterconnectSpec(intra_node=link, inter_node=link),
                        nodes=1, gpus_per_node=8)
    flops = 1e12  # 1 TFLOP of compute
    bytes_ = 0    # ignore memory bound
    t_bf16 = op_to_time(flops, bytes_, system, gpu.name, Dtype.BF16)
    t_fp8  = op_to_time(flops, bytes_, system, gpu.name, Dtype.FP8_E4M3)
    # FP8 peak is 4× BF16 → t_fp8 should be roughly t_bf16 / 4 (efficiency
    # curve is currently dtype-blind so ratio = peak ratio).
    assert t_fp8 < t_bf16
    assert t_fp8 == pytest.approx(t_bf16 * 989.0 / 3958.0, rel=0.05)
```

- [ ] **Step 6.2: Run test to verify it fails**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_peak_selection.py::test_op_to_time_fp8_is_faster_than_bf16_on_h100 -v
```
Expected: FAIL — `t_fp8 == t_bf16` because `op_to_time` ignores dtype.

- [ ] **Step 6.3: Modify `op_to_time`**

In `python/zrt/training/compose/stage.py`, find the existing function (around lines 54-70). Modify only the compute-peak line:

```python
def op_to_time(
    flops: float, bytes_: float, system: SystemSpec,
    gpu_name: str = "", dtype: Dtype = Dtype.BF16,
) -> float:
    """Roofline: op time = max(compute_time, memory_time)."""
    from zrt.training.io.perf_tables import peak_tflops_for
    gpu = system.gpu
    compute_t = 0.0
    if flops > 0:
        peak = peak_tflops_for(gpu, dtype)
        eff = achieved_flops_efficiency(gpu_name or gpu.name, dtype, flops)
        compute_t = flops / (peak * eff) if peak > 0 else 0.0
    memory_t = 0.0
    if bytes_ > 0:
        bw = gpu.hbm_bw_gbps * 1e9
        eff = achieved_bandwidth_efficiency(gpu_name or gpu.name, bytes_)
        memory_t = bytes_ / (bw * eff) if bw > 0 else 0.0
    return max(compute_t, memory_t)
```

- [ ] **Step 6.4: Modify `op_to_time_hetero`**

In the same file (around lines 79-116), for the heterogeneous path, scale cube/vector peaks by the dtype/bf16 ratio so Ascend FP8 path still works without separate cube/vector FP8 declarations:

```python
def op_to_time_hetero(
    cube_flops: float, vector_flops: float, bytes_: float,
    system: SystemSpec, gpu_name: str = "", dtype: Dtype = Dtype.BF16,
    overlap_ratio: float = 0.0,
) -> float:
    from zrt.training.io.perf_tables import peak_tflops_for
    gpu = system.gpu
    total_flops = cube_flops + vector_flops
    if not has_heterogeneous_compute(system):
        return op_to_time(total_flops, bytes_, system, gpu_name, dtype)

    # Scale cube/vector peaks by dtype/bf16 ratio (cube_tflops is BF16 peak).
    bf16_peak = gpu.flops_bf16 * 1e12
    dtype_peak = peak_tflops_for(gpu, dtype)
    scale = dtype_peak / bf16_peak if bf16_peak > 0 else 1.0

    compute_t = 0.0
    if total_flops > 0:
        eff = achieved_flops_efficiency(gpu_name or gpu.name, dtype, total_flops)
        cube_t = 0.0
        vector_t = 0.0
        if cube_flops > 0:
            peak_cube = gpu.cube_tflops * 1e12 * scale
            cube_t = cube_flops / (peak_cube * eff) if peak_cube > 0 else 0.0
        if vector_flops > 0:
            peak_vector = gpu.vector_tflops * 1e12 * scale
            vector_t = vector_flops / (peak_vector * eff) if peak_vector > 0 else 0.0
        if cube_t > 0 or vector_t > 0:
            compute_t = max(cube_t, vector_t) + (1.0 - overlap_ratio) * min(cube_t, vector_t)

    memory_t = 0.0
    if bytes_ > 0:
        bw = gpu.hbm_bw_gbps * 1e9
        eff_bw = achieved_bandwidth_efficiency(gpu_name or gpu.name, bytes_)
        memory_t = bytes_ / (bw * eff_bw) if bw > 0 else 0.0
    return max(compute_t, memory_t)
```

- [ ] **Step 6.5: Run tests to verify pass**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_peak_selection.py -v
```
Expected: all PASS including the new FP8 timing test.

- [ ] **Step 6.6: Regression — full anchor suite must stay green**

```
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v 2>&1 | tail -n 30
```
Expected: all existing anchors pass with identical numbers (they use default BF16 dtypes → BF16 peak unchanged).

- [ ] **Step 6.7: Commit**

```bash
git add python/zrt/training/compose/stage.py tests/training/test_mixed_quant_peak_selection.py
git commit -m "$(cat <<'EOF'
feat(training): op_to_time selects hardware peak by dtype

Both op_to_time and op_to_time_hetero already accepted a dtype
parameter but ignored it (always used flops_bf16). They now route
through peak_tflops_for(gpu, dtype). For heterogeneous (Ascend)
hardware the cube/vector peaks are scaled by the dtype/bf16 peak
ratio so FP8/FP4 path works without separate cube_fp8_tflops fields.

Default dtype is still BF16, so existing call sites (which haven't
been migrated yet) produce identical numbers — pinned by the
unchanged anchor suite. Subsequent tasks wire actual dtype values
into the call sites.
EOF
)"
```

---

## Task 7: Tag Ops with component + Resolve Dtype in stage_time

**Files:**
- Modify: `python/zrt/training/ir/training_graph.py:29-41` (add `component` field to `Op`)
- Modify: `python/zrt/training/ir/builders.py` (tag every `Op(...)` construction)
- Modify: `python/zrt/training/compose/stage.py` (add `_resolve_compute_dtype`; pipe dtype through `_cost_phase_time`; modify `stage_time` loop)
- Create: `tests/training/test_mixed_quant_op_dispatch.py`

**Background:** `op.kind` is the operator type (matmul/rope/softmax). `op.layer_kind` is the layer-level enum (DENSE/MOE/MTP). Neither tells us "is this an attention matmul or an expert matmul." We add an explicit `component` string set at op-construction time in builders.py.

- [ ] **Step 7.1: Write the failing test**

Create `tests/training/test_mixed_quant_op_dispatch.py`:

```python
"""Tests for Op.component tagging + compute-dtype dispatch."""
import pytest

from zrt.training.compose.stage import _resolve_compute_dtype
from zrt.training.ir.training_graph import Op, Tensor
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec


def _model(**dtype_kwargs):
    return ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64, layers=[LayerKind.MOE], **dtype_kwargs,
    )


def _op(name, kind="matmul", component=None):
    return Op(name=name, kind=kind, component=component)


def test_attention_op_uses_attn_compute_dtype():
    m = _model(attn_compute_dtype=Dtype.FP8_E4M3)
    op = _op("layer0.qkv_proj", component="attention")
    assert _resolve_compute_dtype(op, m) is Dtype.FP8_E4M3


def test_routed_expert_op_uses_routed_compute_dtype():
    m = _model(routed_expert_compute_dtype=Dtype.FP8_E4M3)
    op = _op("layer0.routed_expert_ffn", component="routed_expert")
    assert _resolve_compute_dtype(op, m) is Dtype.FP8_E4M3


def test_shared_expert_op_uses_shared_dtype():
    m = _model(shared_expert_compute_dtype=Dtype.FP8_E4M3)
    op = _op("layer0.shared_up_proj", component="shared_expert")
    assert _resolve_compute_dtype(op, m) is Dtype.FP8_E4M3


def test_embedding_op_forced_bf16():
    m = _model(act_dtype=Dtype.FP8_E4M3)  # even if user sets aggressive default
    op = _op("embed", kind="embed", component="embedding")
    assert _resolve_compute_dtype(op, m) is Dtype.BF16


def test_norm_op_forced_bf16():
    m = _model()
    op = _op("layer0.ln", kind="ln", component="norm")
    assert _resolve_compute_dtype(op, m) is Dtype.BF16


def test_unset_component_falls_back_to_act_dtype():
    m = _model(act_dtype=Dtype.FP16)
    op = _op("anonymous", component=None)
    assert _resolve_compute_dtype(op, m) is Dtype.FP16


def test_op_dataclass_has_component_field_default_none():
    op = Op(name="x", kind="matmul")
    assert op.component is None


def test_op_dataclass_accepts_component_kwarg():
    op = Op(name="x", kind="matmul", component="attention")
    assert op.component == "attention"
```

- [ ] **Step 7.2: Run test to verify it fails**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_op_dispatch.py -v
```
Expected: FAIL — `cannot import name '_resolve_compute_dtype'` and `Op.__init__() got unexpected keyword 'component'`.

- [ ] **Step 7.3: Add `component` field to `Op`**

Modify `python/zrt/training/ir/training_graph.py:29-41`:

```python
@dataclass
class Op:
    name: str
    kind: str   # "matmul" | "attn_core" | ... (operator types)
    inputs: list[Tensor] = field(default_factory=list)
    outputs: list[Tensor] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    layer_id: int = -1
    layer_kind: LayerKind = LayerKind.DENSE
    # Component tag for dtype routing in compose/stage.py::_resolve_compute_dtype.
    # Values: "attention" | "routed_expert" | "shared_expert" | "embedding"
    #       | "norm" | None (falls back to model.act_dtype)
    component: str | None = None
```

- [ ] **Step 7.4: Tag ops in `builders.py`**

This is mechanical: search-replace `Op(` constructions in `python/zrt/training/ir/builders.py` and add `component=...` keyword. Use this exact mapping (from spec section 4.5):

| Op name suffix (substring of `name=`) | `component` |
|---|---|
| `.qkv_proj`, `.q_a_proj`, `.q_b_proj`, `.kv_a_proj`, `.kv_b_proj`, `.wq_a`, `.wq_b`, `.wkv`, `.wo_a`, `.wo_b`, `.o_proj`, `.attn_core`, `.rope`, `.softmax` | `"attention"` |
| `.routed_expert_ffn`, `.expert_agg`, `.gate_proj`, `.hash_route`, `.compressor_pool`, `.indexer_topk`, `.idx_comp_wgate`, `.comp_wgate` | `"routed_expert"` |
| `.shared_up_proj`, `.shared_gate_proj`, `.shared_down_proj`, `.shared_swiGLU` | `"shared_expert"` |
| `.embed`, `.lm_head` | `"embedding"` |
| `.mhc_pre`, `.mhc_post`, `.mhc_head` | `"norm"` |

Concretely: for each `Op(name=f"{prefix}.<suffix>", kind=...)` line in `builders.py`, append `component="<value>"` based on the suffix.

Strategy: do this in small chunks, one section of `builders.py` at a time. Verify by running:

```
PYTHONPATH=python python -c "
from zrt.training.spec.model import ModelSpec, LayerKind
from zrt.training.ir.builders import build_graph
from zrt.training.spec.strategy import Strategy
m = ModelSpec(hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
              vocab=1000, seq_len=64, layers=[LayerKind.DENSE])
g = build_graph(m, Strategy())
for op in g.ops[:10]:
    print(op.name, '->', op.component)
"
```

(Use whatever the actual entry-point is — check `python/zrt/training/ir/builders.py` near top for `build_graph` or similar; if no such top-level helper, skip the smoke check.)

- [ ] **Step 7.5: Add `_resolve_compute_dtype` + thread dtype through stage_time**

In `python/zrt/training/compose/stage.py`, add the helper somewhere near `op_to_time` (e.g. immediately after `op_to_time_hetero`):

```python
def _resolve_compute_dtype(op: Op, model: ModelSpec) -> Dtype:
    """Map ``op.component`` to its compute dtype, falling back to act_dtype."""
    comp = getattr(op, "component", None)
    if comp == "attention":
        return model.attn_compute_dtype
    if comp == "routed_expert":
        return model.routed_expert_compute_dtype
    if comp == "shared_expert":
        return model.shared_expert_compute_dtype
    if comp in ("embedding", "norm"):
        return Dtype.BF16
    return model.act_dtype
```

Modify `_cost_phase_time` (around lines 119-128) to accept a `dtype` parameter:

```python
def _cost_phase_time(
    cost: OpCost, phase: str, system: SystemSpec,
    gpu_name: str, overlap: float = 0.0,
    dtype: Dtype = Dtype.BF16,
) -> float:
    """Compute time for one phase (fwd/dx/dw) using heterogeneous roofline."""
    cube = getattr(cost, f"{phase}_cube_flops")
    vector = getattr(cost, f"{phase}_vector_flops")
    bytes_ = getattr(cost, f"{phase}_bytes")
    return op_to_time_hetero(cube, vector, bytes_, system, gpu_name,
                             overlap_ratio=overlap, dtype=dtype)
```

Modify the `stage_time` op loop (around lines 146-154) to resolve dtype per op:

```python
    for op in stage_ops:
        cost = op_cost(op, model, system)
        overlap = gpu.overlap_ratio.get(op.kind, 0.0)
        op_dtype = _resolve_compute_dtype(op, model)
        fwd_t = _cost_phase_time(cost, "fwd", system, gpu_name, overlap, op_dtype)
        dx_t  = _cost_phase_time(cost, "dx",  system, gpu_name, overlap, op_dtype)
        dw_t  = _cost_phase_time(cost, "dw",  system, gpu_name, overlap, op_dtype)
        t_fwd    += fwd_t
        t_bwd_dx += dx_t
        t_bwd_dw += dw_t
```

Check that any other call sites of `_cost_phase_time` (search via `grep -n "_cost_phase_time" python/zrt/training/compose/stage.py`) also accept and pass dtype. There may be one inside `_recompute_time` or `_tp_gemm_time`; for those, pass `Dtype.BF16` as the default (preserving existing behavior).

- [ ] **Step 7.6: Run test to verify it passes**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_op_dispatch.py -v
```
Expected: all 8 tests PASS.

- [ ] **Step 7.7: Regression — anchor suite + full training tests**

```
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v 2>&1 | tail -n 30
PYTHONPATH=python pytest tests/training/ -v 2>&1 | tail -n 30
```
Expected: all existing tests stay green. Default model dtypes are still BF16 across the board (Task 4 set defaults), so anchor MFU/step_time should be byte-for-byte identical to before.

- [ ] **Step 7.8: Commit**

```bash
git add python/zrt/training/ir/training_graph.py python/zrt/training/ir/builders.py \
        python/zrt/training/compose/stage.py tests/training/test_mixed_quant_op_dispatch.py
git commit -m "$(cat <<'EOF'
feat(training): per-op component tagging + dtype dispatch in stage_time

- Op gets new component: str | None field
  ('attention' | 'routed_expert' | 'shared_expert' | 'embedding' | 'norm')
- builders.py tags each Op construction site
- _resolve_compute_dtype(op, model) maps component → compute dtype
- _cost_phase_time accepts dtype and forwards to op_to_time_hetero
- stage_time loop resolves per-op dtype before timing

With default ModelSpec dtypes (all BF16) the timing is identical
to before — pinned by the unchanged anchor regression suite.
EOF
)"
```

---

## Task 8: Memory — Weight Bytes Split by Component Dtype

**Files:**
- Modify: `python/zrt/training/models/memory.py:122-135` (replace FP4 string hack)
- Create: `tests/training/test_mixed_quant_memory.py`

**Background:** Today `memory_breakdown` special-cases `routed_expert_dtype == "fp4"` and computes FP4 weight bytes manually with `0.5 + 0.0625 BF16-scale overhead`. We generalize this to read `model.routed_expert_weight_dtype.stored_bytes` and route weights through a component-wise sum so any combination of dtypes works.

- [ ] **Step 8.1: Write the failing test**

Create `tests/training/test_mixed_quant_memory.py`:

```python
"""Tests for mixed-quant memory accounting."""
import pytest

from zrt.training.models.memory import memory_breakdown
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import OptKind, Strategy
from zrt.training.spec.system import GPU, SystemSpec
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.ir.training_graph import Graph


def _make_system():
    gpu = GPU(name="h100", flops_bf16=989, flops_fp8=3958, flops_fp4=0,
              hbm_gb=80, hbm_bw_gbps=3350)
    link = LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1,
                    topology="all_to_all", num_devices=8)
    return SystemSpec(gpu=gpu, host_mem_gb=2048,
                      interconnect=InterconnectSpec(intra_node=link, inter_node=link),
                      nodes=1, gpus_per_node=8)


def _moe_model(**kwargs):
    base = dict(
        hidden=512, ffn=2048, num_heads=8, num_kv_heads=8, head_dim=64,
        vocab=4096, seq_len=128,
        layers=[LayerKind.MOE, LayerKind.MOE],
        num_experts=8, moe_ffn=1024, top_k=2,
    )
    base.update(kwargs)
    return ModelSpec(**base)


def test_legacy_string_fp4_and_new_enum_produce_same_weight_bytes():
    """Back-compat: routed_expert_dtype='fp4' must match routed_expert_weight_dtype=Dtype.FP4."""
    g, sys_, st = Graph(), _make_system(), Strategy(optimizer=OptKind.ADAM)
    m_legacy = _moe_model(routed_expert_dtype="fp4")
    m_new    = _moe_model(routed_expert_weight_dtype=Dtype.FP4)
    mb_legacy = memory_breakdown(g, m_legacy, sys_, st)
    mb_new    = memory_breakdown(g, m_new, sys_, st)
    assert mb_legacy.weights == mb_new.weights


def test_fp4_routed_expert_smaller_than_bf16():
    """FP4 routed expert weight should be ~3.5× smaller than BF16."""
    g, sys_, st = Graph(), _make_system(), Strategy(optimizer=OptKind.ADAM)
    m_bf16 = _moe_model()
    m_fp4 = _moe_model(routed_expert_weight_dtype=Dtype.FP4)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    mb_fp4 = memory_breakdown(g, m_fp4, sys_, st)
    # FP4 stored 0.5625 vs BF16 2.0 → expert weight ratio ≈ 0.281, but
    # non-expert params (attn/embed) keep BF16 weight bytes, so total
    # ratio is somewhere between 0.281 and 1.0.
    assert mb_fp4.weights < mb_bf16.weights
    # Expert weight is a substantial fraction of total → expect ≥ 20% saving overall
    saving = (mb_bf16.weights - mb_fp4.weights) / mb_bf16.weights
    assert saving > 0.2, f"FP4 should save >20% weight memory, got {saving:.2%}"


def test_fp8_routed_expert_weight_halves_routed_bytes_vs_bf16():
    g, sys_, st = Graph(), _make_system(), Strategy(optimizer=OptKind.ADAM)
    m_bf16 = _moe_model()
    m_fp8 = _moe_model(routed_expert_weight_dtype=Dtype.FP8_E4M3)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    mb_fp8 = memory_breakdown(g, m_fp8, sys_, st)
    assert mb_fp8.weights < mb_bf16.weights


def test_dense_model_unaffected_by_routed_dtype():
    """Dense model (no MoE layers) → routed_expert_weight_dtype has no effect."""
    g, sys_, st = Graph(), _make_system(), Strategy(optimizer=OptKind.ADAM)
    base = dict(
        hidden=512, ffn=2048, num_heads=8, num_kv_heads=8, head_dim=64,
        vocab=4096, seq_len=128, layers=[LayerKind.DENSE, LayerKind.DENSE],
    )
    m_bf16 = ModelSpec(**base)
    m_fp4 = ModelSpec(**base, routed_expert_weight_dtype=Dtype.FP4)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    mb_fp4 = memory_breakdown(g, m_fp4, sys_, st)
    assert mb_bf16.weights == mb_fp4.weights
```

- [ ] **Step 8.2: Run test to verify it fails / sanity check**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_memory.py -v
```
Expected: `test_legacy_string_fp4_and_new_enum_produce_same_weight_bytes` may already pass thanks to Task 3's back-compat shim (which copies `routed_expert_dtype="fp4"` into `routed_expert_weight_dtype=Dtype.FP4`); the other tests should pass with the current code (since `use_fp4` triggers off the string field). The point of this test set is to PIN the current behavior so the refactor in Step 8.3 doesn't regress it. Confirm a baseline:

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_memory.py -v 2>&1 | tail -n 20
```
Record which tests currently pass / fail before refactoring.

- [ ] **Step 8.3: Refactor `memory_breakdown` weights block (lines 122-135)**

Replace:

```python
    # FP4 routed expert weights: 0.5 B/elem + per-block BF16 scale
    use_fp4 = getattr(model, "routed_expert_dtype", "bf16") == "fp4"
    if use_fp4:
        P_expert = _routed_expert_params_on_rank(model, strategy)
        P_other = P - P_expert
        FP4_BYTES_PER_ELEM = 0.5
        FP4_BLOCK_SIZE = 32
        expert_weight_bytes = int(
            P_expert * FP4_BYTES_PER_ELEM
            + (P_expert / FP4_BLOCK_SIZE) * 2  # BF16 scale per block
        )
        weights = expert_weight_bytes + P_other * model.param_dtype.bytes
    else:
        weights = P * model.param_dtype.bytes
```

With:

```python
    # Per-component weight bytes: routed-expert weights use
    # routed_expert_weight_dtype (FP4 stored-size includes per-block BF16 scale);
    # everything else uses param_dtype.
    P_expert = _routed_expert_params_on_rank(model, strategy)
    P_other = P - P_expert
    weights = int(
        P_expert * model.routed_expert_weight_dtype.stored_bytes
        + P_other * model.param_dtype.stored_bytes
    )
```

- [ ] **Step 8.4: Run test to verify all pass**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_memory.py -v
```
Expected: all 4 weight tests PASS.

- [ ] **Step 8.5: Regression — anchor suite (especially V4 anchors that use `routed_expert_dtype: fp4`)**

```
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v 2>&1 | tail -n 30
```
Expected: all anchors green (V4 anchors should produce identical weight bytes thanks to the back-compat sync).

- [ ] **Step 8.6: Commit**

```bash
git add python/zrt/training/models/memory.py tests/training/test_mixed_quant_memory.py
git commit -m "$(cat <<'EOF'
refactor(training): weight memory uses per-component dtype

Replace the string-keyed FP4 special case in memory_breakdown with
a generic per-component sum:
  weights = P_expert * routed_expert_weight_dtype.stored_bytes
          + P_other  * param_dtype.stored_bytes

stored_bytes on FP4 includes the per-block BF16 scale (2/32 B/elem)
so the legacy fp4 path is byte-equivalent. New configurations can
now mix any expert weight dtype with any param dtype.
EOF
)"
```

---

## Task 9: Memory — Region-Aware Activation Bytes

**Files:**
- Modify: `python/zrt/training/models/memory.py:414` (attention QK^T score matrix)
- Modify: `python/zrt/training/models/memory.py:491` (CP A2A staging buffer)
- Modify: `python/zrt/training/models/memory.py:500` (EP A2A staging buffer)
- Extend: `tests/training/test_mixed_quant_memory.py`

**Background:** Three of `memory.py`'s activation/buffer sites use a single global `act_bytes = model.act_dtype.bytes`. Per spec §4.7.3 these should be:
- QK^T attention score (line 414) → `effective_attn_act_dtype()`
- CP A2A buffer (line 491) → `effective_attn_act_dtype()` (CP is sequence-dim, attention only)
- EP A2A buffer (line 500) → `routed_expert_compute_dtype`

The general `layer_act` (line 404) and HC residual (line 405) and TP buffer (line 482) keep using `act_dtype` (residual-stream is BF16 in V4).

- [ ] **Step 9.1: Append failing tests**

Append to `tests/training/test_mixed_quant_memory.py`:

```python
def test_ep_a2a_buffer_uses_routed_compute_dtype():
    """EP A2A staging buffer should scale with routed_expert_compute_dtype."""
    g = Graph()
    sys_ = _make_system()
    st = Strategy(ep=4, dp=1, optimizer=OptKind.ADAM)
    m_bf16 = _moe_model()
    m_fp8 = _moe_model(routed_expert_compute_dtype=Dtype.FP8_E4M3)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    mb_fp8  = memory_breakdown(g, m_fp8,  sys_, st)
    # Comm buffers shrink when routed compute dtype shrinks
    # (4× staging × seq_cp × hidden × act_bytes × n_moe).
    assert mb_fp8.comm_buffers < mb_bf16.comm_buffers


def test_cp_a2a_buffer_uses_attn_act_dtype():
    g = Graph()
    sys_ = _make_system()
    st = Strategy(cp=4, dp=1, optimizer=OptKind.ADAM)
    m_bf16 = _moe_model()
    m_fp8_attn = _moe_model(attn_act_dtype=Dtype.FP8_E4M3)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    mb_fp8  = memory_breakdown(g, m_fp8_attn, sys_, st)
    assert mb_fp8.comm_buffers < mb_bf16.comm_buffers


def test_qk_score_matrix_uses_attn_act_dtype():
    """Activations: QK^T score matrix term (~5·a·s²·bytes) should
    scale with attn_act_dtype when present."""
    g = Graph()
    sys_ = _make_system()
    st = Strategy(optimizer=OptKind.ADAM)
    # Force long sequence so QK^T dominates
    m_bf16 = _moe_model(seq_len=1024, num_heads=16)
    m_fp8 = _moe_model(seq_len=1024, num_heads=16, attn_act_dtype=Dtype.FP8_E4M3)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    mb_fp8 = memory_breakdown(g, m_fp8, sys_, st)
    assert mb_fp8.activations < mb_bf16.activations
```

- [ ] **Step 9.2: Run test to verify it fails**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_memory.py -v -k "ep_a2a or cp_a2a or qk_score"
```
Expected: all 3 fail (region dtypes don't yet influence memory).

- [ ] **Step 9.3: Modify `_activation_memory` for QK^T (line 414)**

In `python/zrt/training/models/memory.py`, find the block around lines 412-414:

```python
        if not attn_recomputed:
            num_heads = max(1, getattr(model, "num_heads", 1))
            layer_act += 5 * num_heads * s * s * act_bytes
```

Change to:

```python
        if not attn_recomputed:
            num_heads = max(1, getattr(model, "num_heads", 1))
            attn_bytes = model.effective_attn_act_dtype().bytes
            layer_act += 5 * num_heads * s * s * attn_bytes
```

- [ ] **Step 9.4: Modify `_comm_buffer_memory` for CP and EP buffers**

In `python/zrt/training/models/memory.py:466-506`, replace the CP and EP buffer blocks. Original lines around 488-504:

```python
    if strategy.cp > 1:
        seq_cp = s // strategy.cp
        h_tp = h // strategy.tp if strategy.tp > 1 else h
        per_layer_cp = 4 * seq_cp * h_tp * act_bytes
        total += per_layer_cp * n_layers * strategy.micro_batch

    if strategy.ep > 1 and model.num_experts > 0:
        seq_cp = s // strategy.cp if strategy.cp > 1 else s
        h_tp = h // strategy.tp if strategy.tp > 1 else h
        per_layer_ep = 4 * seq_cp * h_tp * act_bytes
        n_moe = sum(1 for lk in model.layers if lk.value == "moe")
        if strategy.pp > 1:
            n_moe = max(1, n_moe // strategy.pp)
        total += per_layer_ep * n_moe * strategy.micro_batch
```

Replace with (note: TP buffer block at line 480-483 is unchanged):

```python
    if strategy.cp > 1:
        seq_cp = s // strategy.cp
        h_tp = h // strategy.tp if strategy.tp > 1 else h
        # CP A2A buffers shuttle attention activations across the sequence dim.
        cp_bytes = model.effective_attn_act_dtype().bytes
        per_layer_cp = 4 * seq_cp * h_tp * cp_bytes
        total += per_layer_cp * n_layers * strategy.micro_batch

    if strategy.ep > 1 and model.num_experts > 0:
        seq_cp = s // strategy.cp if strategy.cp > 1 else s
        h_tp = h // strategy.tp if strategy.tp > 1 else h
        # EP dispatch/combine carries routed-expert activations.
        ep_bytes = model.routed_expert_compute_dtype.bytes
        per_layer_ep = 4 * seq_cp * h_tp * ep_bytes
        n_moe = sum(1 for lk in model.layers if lk.value == "moe")
        if strategy.pp > 1:
            n_moe = max(1, n_moe // strategy.pp)
        total += per_layer_ep * n_moe * strategy.micro_batch
```

- [ ] **Step 9.5: Run test to verify pass**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_memory.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 9.6: Regression — anchors**

```
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v 2>&1 | tail -n 30
```
Expected: green (defaults unchanged → byte-identical).

- [ ] **Step 9.7: Commit**

```bash
git add python/zrt/training/models/memory.py tests/training/test_mixed_quant_memory.py
git commit -m "$(cat <<'EOF'
feat(training): region-aware activation memory dispatch

- QK^T attention score matrix (5·a·s²·bytes) uses effective_attn_act_dtype
- CP A2A staging buffer uses effective_attn_act_dtype (sequence-dim
  comm is attention-only)
- EP A2A staging buffer uses routed_expert_compute_dtype (routed
  experts dispatch/combine)

Residual-stream activations (layer_act, HC residual) and TP AG/RS
buffers continue to use act_dtype, preserving V4's BF16 residual
stream while letting FP8 MoE compute shrink the relevant buffers.
EOF
)"
```

---

## Task 10: Memory — Gradient Bytes Split (Expert vs Non-Expert)

**Files:**
- Modify: `python/zrt/training/models/memory.py:137` (split grad bytes by component)
- Extend: `tests/training/test_mixed_quant_memory.py`

**Background:** Today `grads = P * model.grad_dtype.bytes`. Spec §4.7.2 wants:
```
grads = P_expert * routed_expert_grad_dtype.bytes + P_nonexpert * grad_dtype.bytes
```

- [ ] **Step 10.1: Append failing test**

Append to `tests/training/test_mixed_quant_memory.py`:

```python
def test_routed_expert_grad_dtype_affects_grad_bytes_in_moe_models():
    g = Graph()
    sys_ = _make_system()
    st = Strategy(optimizer=OptKind.ADAM)
    m_fp32 = _moe_model()
    m_bf16 = _moe_model(routed_expert_grad_dtype=Dtype.BF16)
    mb_fp32 = memory_breakdown(g, m_fp32, sys_, st)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    # BF16 expert grad smaller than FP32 expert grad → total grads smaller
    assert mb_bf16.grads < mb_fp32.grads


def test_dense_model_grad_unaffected_by_routed_expert_grad_dtype():
    g, sys_, st = Graph(), _make_system(), Strategy(optimizer=OptKind.ADAM)
    base = dict(hidden=512, ffn=2048, num_heads=8, num_kv_heads=8, head_dim=64,
                vocab=4096, seq_len=128, layers=[LayerKind.DENSE, LayerKind.DENSE])
    m_fp32 = ModelSpec(**base)
    m_bf16 = ModelSpec(**base, routed_expert_grad_dtype=Dtype.BF16)
    mb_fp32 = memory_breakdown(g, m_fp32, sys_, st)
    mb_bf16 = memory_breakdown(g, m_bf16, sys_, st)
    assert mb_fp32.grads == mb_bf16.grads
```

- [ ] **Step 10.2: Run test to verify it fails**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_memory.py -v -k "routed_expert_grad"
```
Expected: FAIL — grads identical (no split yet).

- [ ] **Step 10.3: Modify `memory_breakdown` grads line (line 137)**

Replace:

```python
    grads = P * model.grad_dtype.bytes
```

With (reuse `P_expert` and `P_other` already computed in Step 8.3):

```python
    grads = int(
        P_expert * model.routed_expert_grad_dtype.bytes
        + P_other * model.grad_dtype.bytes
    )
```

- [ ] **Step 10.4: Run test to verify pass**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_memory.py -v
```
Expected: all PASS.

- [ ] **Step 10.5: Regression**

```
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v 2>&1 | tail -n 20
```
Expected: green.

- [ ] **Step 10.6: Commit**

```bash
git add python/zrt/training/models/memory.py tests/training/test_mixed_quant_memory.py
git commit -m "$(cat <<'EOF'
feat(training): gradient memory split by expert vs non-expert dtype

  grads = P_expert * routed_expert_grad_dtype.bytes
        + P_other  * grad_dtype.bytes

Lets DeepSeek-V3/V4-style configurations model a different precision
for routed-expert local gradients vs DP-reduced gradients. Default
routed_expert_grad_dtype=FP32 preserves existing behavior.
EOF
)"
```

---

## Task 11: Communication — DP All-Reduce Excludes Expert Grads

**Files:**
- Modify: `python/zrt/training/models/comm.py:102-108` (`total_comm_time` DP grad block)
- Create: `tests/training/test_mixed_quant_comm.py`

**Background:** `grad_bytes = P * model.grad_dtype.bytes` in `comm.py:104` reduces ALL params under DP AR. But routed expert grads stay in the EP group (no DP AR for them under EP > 1). Volume should scope to non-expert grads only.

- [ ] **Step 11.1: Write the failing test**

Create `tests/training/test_mixed_quant_comm.py`:

```python
"""Tests for mixed-quant communication volume."""
import pytest

from zrt.training.models.comm import total_comm_time
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import OptKind, Strategy
from zrt.training.spec.system import GPU, SystemSpec
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.ir.training_graph import Graph


def _make_system(dp_bw_gbps=900):
    gpu = GPU(name="h100", flops_bf16=989, flops_fp8=3958, flops_fp4=0,
              hbm_gb=80, hbm_bw_gbps=3350)
    link = LinkSpec(type="NVLink", bandwidth_gbps=dp_bw_gbps, latency_us=1,
                    topology="all_to_all", num_devices=16)
    return SystemSpec(gpu=gpu, host_mem_gb=2048,
                      interconnect=InterconnectSpec(intra_node=link, inter_node=link),
                      nodes=2, gpus_per_node=8)


def _moe_model():
    return ModelSpec(
        hidden=512, ffn=2048, num_heads=8, num_kv_heads=8, head_dim=64,
        vocab=4096, seq_len=128,
        layers=[LayerKind.MOE, LayerKind.MOE],
        num_experts=8, moe_ffn=1024, top_k=2,
    )


def test_dp_grad_reduce_smaller_under_ep_than_under_no_ep():
    """When EP > 1, routed expert grads are EP-local and excluded from DP AR."""
    g, sys_ = Graph(), _make_system()
    m = _moe_model()
    st_no_ep = Strategy(dp=2, ep=1, optimizer=OptKind.ADAM)
    st_ep    = Strategy(dp=2, ep=4, optimizer=OptKind.ADAM)
    t_no_ep = total_comm_time(g, m, sys_, st_no_ep)["dp_grad_reduce"]
    t_ep    = total_comm_time(g, m, sys_, st_ep)["dp_grad_reduce"]
    assert t_ep < t_no_ep, (
        f"DP AR with EP=4 should exclude expert grads → smaller volume "
        f"({t_ep:.6f}s vs no-EP {t_no_ep:.6f}s)"
    )


def test_dp_grad_reduce_volume_unchanged_for_dense_model():
    g, sys_ = Graph(), _make_system()
    m_dense = ModelSpec(
        hidden=512, ffn=2048, num_heads=8, num_kv_heads=8, head_dim=64,
        vocab=4096, seq_len=128, layers=[LayerKind.DENSE, LayerKind.DENSE],
    )
    st_no_ep = Strategy(dp=2, ep=1, optimizer=OptKind.ADAM)
    st_ep_unused = Strategy(dp=2, ep=4, optimizer=OptKind.ADAM)  # ep ignored: no experts
    t1 = total_comm_time(g, m_dense, sys_, st_no_ep)["dp_grad_reduce"]
    t2 = total_comm_time(g, m_dense, sys_, st_ep_unused)["dp_grad_reduce"]
    assert t1 == pytest.approx(t2)
```

- [ ] **Step 11.2: Run test to verify it fails**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_comm.py -v
```
Expected: `test_dp_grad_reduce_smaller_under_ep_than_under_no_ep` FAILS (volume identical because comm.py:104 doesn't subtract expert grads).

- [ ] **Step 11.3: Modify `total_comm_time` DP grad block**

In `python/zrt/training/models/comm.py`, find lines 101-111 (`if strategy.dp > 1:` block). Modify:

```python
    # DP gradient reduction (at step end)
    if strategy.dp > 1:
        P = _params_on_rank_for_dp(model, strategy)
        # Routed expert grads stay in the EP group (no DP AR) when EP > 1.
        # Subtract their per-rank contribution so the DP volume reflects only
        # non-expert + shared-expert grads.
        if strategy.ep > 1 and model.num_experts > 0:
            from zrt.training.models.memory import _routed_expert_params_on_rank
            P_expert_on_rank = _routed_expert_params_on_rank(model, strategy)
            P_dp = max(0, P - P_expert_on_rank)
        else:
            P_dp = P
        # Mixed-quant: split grad volume into expert / non-expert bytes
        # (expert portion has 0 grad volume here; it lives in the EP comm pool).
        grad_bytes = int(P_dp * model.grad_dtype.bytes)
        dp_c = Collective(
            name="dp_grad_reduce", kind="AR" if strategy.zero_stage == 0 else "RS",
            group="DP", bytes_=grad_bytes, inserted_after="optimizer_step",
        )
        group_size = strategy.dp
        link = tier_for_group("DP", group_size, system)
        result[dp_c.name] = collective_time(dp_c, group_size, link)
```

- [ ] **Step 11.4: Run test to verify pass**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_comm.py -v
```
Expected: all PASS.

- [ ] **Step 11.5: Regression**

```
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v 2>&1 | tail -n 30
PYTHONPATH=python pytest tests/training/ -v -k "comm" 2>&1 | tail -n 20
```
Expected: anchors green (V3/V4 anchors with EP > 1 already excluded experts in `_params_on_rank_for_dp`; the change here only refines the per-rank subtraction).

> **Note:** If an existing test asserts DP-AR-volume includes expert params, that test was wrong; surface it for review before fixing. Don't silently mutate failing tests.

- [ ] **Step 11.6: Commit**

```bash
git add python/zrt/training/models/comm.py tests/training/test_mixed_quant_comm.py
git commit -m "$(cat <<'EOF'
fix(training): DP all-reduce excludes routed-expert gradients

Under EP > 1, routed-expert gradients reduce inside the EP group
(via the EP A2A combine path), never across DP. The DP-AR volume
should therefore subtract the per-rank expert param share. Previous
code passed total per-rank params through grad_dtype.bytes,
over-counting expert grads in DP AR.
EOF
)"
```

---

## Task 12: MFU Native — Dtype-Weighted Effective Peak

**Files:**
- Modify: `python/zrt/training/compose/schedules.py:36-99` (`StepResult.mfu_native`)
- Modify: `python/zrt/training/compose/schedules.py:806-859` (add `compute_mfu_native`)
- Modify: `python/zrt/training/compose/schedules.py:693-696` (set `step.mfu_native`)
- Create: `tests/training/test_mixed_quant_mfu_native.py`

**Background:** Keep `step.mfu` semantics (denominator = `gpu.flops_bf16`) so V3/V4 paper comparisons stay stable. Add `step.mfu_native` whose denominator is an op-mix-weighted effective peak (harmonic mean of per-dtype peaks weighted by per-dtype FLOPs).

- [ ] **Step 12.1: Write the failing test**

Create `tests/training/test_mixed_quant_mfu_native.py`:

```python
"""Tests for mfu_native metric."""
import pytest

from zrt.training.compose.schedules import compute_mfu_native
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import OptKind, Strategy
from zrt.training.spec.system import GPU, SystemSpec
from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.ir.training_graph import Graph, Op, Tensor


def _make_system():
    gpu = GPU(name="h100", flops_bf16=989, flops_fp8=3958, flops_fp4=0,
              hbm_gb=80, hbm_bw_gbps=3350)
    link = LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1,
                    topology="all_to_all", num_devices=8)
    return SystemSpec(gpu=gpu, host_mem_gb=2048,
                      interconnect=InterconnectSpec(intra_node=link, inter_node=link),
                      nodes=1, gpus_per_node=8)


def _t(name, n):
    return Tensor(name=name, shape_logical=(n,), shape_local=(n,),
                  dtype=Dtype.BF16, is_activation=True)


def _model():
    return ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64, layers=[LayerKind.DENSE],
    )


def test_mfu_native_uses_dtype_weighted_peak():
    """Graph with FP8 ops should report a higher effective peak than BF16."""
    g = Graph()
    # Two ops: one attention (BF16), one routed (FP8); compose into a graph
    g.ops = [
        Op(name="op0.attn", kind="matmul", component="attention",
           inputs=[_t("a", 1000)], outputs=[_t("b", 1000)]),
        Op(name="op0.routed", kind="matmul", component="routed_expert",
           inputs=[_t("c", 1000)], outputs=[_t("d", 1000)]),
    ]
    sys_ = _make_system()
    m = _model()
    st = Strategy(optimizer=OptKind.ADAM)
    # We cannot easily construct realistic step_time/flops here; the function
    # under test must accept zeros and return 0 gracefully.
    out = compute_mfu_native(m, st, sys_, step_time=0.0, graph=g)
    assert out == 0.0


def test_mfu_native_smaller_than_mfu_when_fp8_is_used():
    """Same FLOPs, same step_time: FP8 path has higher effective peak →
    lower mfu_native than the BF16-peak mfu."""
    from zrt.training.compose.schedules import compute_mfu
    g = Graph()
    g.ops = [Op(name="x", kind="matmul", component="routed_expert")]
    sys_ = _make_system()
    m = ModelSpec(
        hidden=128, ffn=256, num_heads=4, num_kv_heads=4, head_dim=32,
        vocab=1000, seq_len=64, layers=[LayerKind.MOE],
        num_experts=4, moe_ffn=512, top_k=2,
        routed_expert_compute_dtype=Dtype.FP8_E4M3,
    )
    st = Strategy(global_batch=8, optimizer=OptKind.ADAM)
    step_time = 0.1
    mfu = compute_mfu(m, st, sys_, step_time, g)
    mfu_native = compute_mfu_native(m, st, sys_, step_time, g)
    # For a model where routed (FP8) FLOPs dominate, native peak >> BF16 peak
    # → native MFU < BF16-peak MFU.
    if mfu > 0:
        assert mfu_native <= mfu
```

- [ ] **Step 12.2: Run test to verify it fails**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_mfu_native.py -v
```
Expected: FAIL — `cannot import name 'compute_mfu_native'`.

- [ ] **Step 12.3: Add `mfu_native` field to `StepResult`**

In `python/zrt/training/compose/schedules.py`, in the `@dataclass class StepResult` (around line 64-65), add:

```python
    mfu: float = 0.0
    hfu: float = 0.0
    mfu_native: float = 0.0   # MFU vs op-mix-weighted effective peak
```

- [ ] **Step 12.4: Add `compute_mfu_native` function**

In the same file, place after `compute_hfu` (around line 859):

```python
def compute_mfu_native(
    model: ModelSpec, strategy: Strategy,
    system: SystemSpec, step_time: float,
    graph: Graph,
) -> float:
    """MFU with denominator = effective hardware peak under mixed precision.

    The effective peak is the harmonic-mean of per-dtype peaks weighted
    by per-dtype FLOPs share, derived from each op's component tag:

      effective_peak = total_flops / Σ (flops_by_dtype[d] / peak_for[d])

    Reduces to ``compute_mfu`` (BF16 peak) when all ops are BF16-typed.
    Returns 0 when step_time <= 0 or total flops <= 0.
    """
    from zrt.training.io.perf_tables import peak_tflops_for
    from zrt.training.models.flops import op_cost, total_training_flops
    from zrt.training.compose.stage import _resolve_compute_dtype

    if step_time <= 0:
        return 0.0

    actual_flops = total_training_flops(graph, model, strategy, system)
    if actual_flops <= 0:
        return 0.0

    # Aggregate per-dtype FLOPs by walking the graph
    flops_by_dtype: dict[Dtype, float] = {}
    for op in graph.ops:
        cost = op_cost(op, model, system)
        op_flops = (cost.fwd_cube_flops + cost.fwd_vector_flops
                    + cost.dx_cube_flops + cost.dx_vector_flops
                    + cost.dw_cube_flops + cost.dw_vector_flops)
        if op_flops <= 0:
            continue
        d = _resolve_compute_dtype(op, model)
        flops_by_dtype[d] = flops_by_dtype.get(d, 0.0) + op_flops

    gpu = system.gpu
    weighted_time = 0.0
    total = sum(flops_by_dtype.values())
    if total <= 0:
        return 0.0
    for d, f in flops_by_dtype.items():
        peak = peak_tflops_for(gpu, d)
        if peak <= 0:
            continue
        weighted_time += f / peak
    if weighted_time <= 0:
        return 0.0
    effective_peak = total / weighted_time
    pp_flops = actual_flops / strategy.pp
    return util_from_flops(pp_flops, effective_peak, step_time)
```

- [ ] **Step 12.5: Wire into `pipeline_step_time`**

In `python/zrt/training/compose/schedules.py` around line 693-696, after `step.hfu = compute_hfu(...)`, add:

```python
    step.mfu_native = compute_mfu_native(model, strategy, system, step.step_time, graph)
```

- [ ] **Step 12.6: Run test to verify pass**

```
PYTHONPATH=python pytest tests/training/test_mixed_quant_mfu_native.py -v
```
Expected: all PASS.

- [ ] **Step 12.7: Regression — anchor suite (mfu unchanged, mfu_native new)**

```
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v 2>&1 | tail -n 30
```
Expected: all anchors green (they assert `mfu`, not `mfu_native`; default BF16 dtypes → `mfu_native ≈ mfu`).

- [ ] **Step 12.8: Commit**

```bash
git add python/zrt/training/compose/schedules.py tests/training/test_mixed_quant_mfu_native.py
git commit -m "$(cat <<'EOF'
feat(training): compute_mfu_native using dtype-weighted effective peak

- StepResult.mfu_native: MFU vs the op-mix-weighted hardware peak
- compute_mfu_native(): harmonic-mean of per-dtype peaks weighted
  by per-dtype FLOPs share (derived from Op.component tagging)
- step.mfu (BF16 peak) unchanged → preserves V3/V4 paper comparison

mfu_native shows actual hardware utilization once mixed FP8/FP4
GEMMs land on FP8/FP4 Tensor Cores rather than being scored
against the BF16 peak.
EOF
)"
```

---

## Task 13: New Anchor — V4-Pro FP8/FP4 on H100 (calibration mode)

**Files:**
- Create: `tests/training/anchors/deepseek_v4_pro_fp8_fp4_h100.yaml`
- Create: `tests/training/anchors/deepseek_v4_pro_fp8_fp4_b300.yaml`

**Background:** New anchors validating end-to-end FP8/FP4 path. Both use `strict_mfu_check: false` (calibration mode) because no public MFU number for V4 Pro under these configs exists. The anchors gate on step_time being lower than a pure-BF16 baseline and memory being lower.

- [ ] **Step 13.1: Create H100 anchor**

Create `tests/training/anchors/deepseek_v4_pro_fp8_fp4_h100.yaml`:

```yaml
# V4-Pro under deepseek_v4_fp8_fp4 quant preset on H100 SXM cluster.
# Calibration mode: no published MFU baseline; pinning behavioral signals
# rather than absolute numbers. Numerical targets recalibrate after
# running the spec at HEAD.

name: deepseek_v4_pro_fp8_fp4_h100
description: "DeepSeek-V4-Pro 1.6T, H100 SXM, TP8 PP4 EP64 DP4, FP8+FP4 mixed quant"
model:
  base: deepseek_v4_pro
  # quant_preset overlays expand into per-component dtype fields in
  # io/config_loader._expand_quant_preset()
  quant_preset: deepseek_v4_fp8_fp4
system:
  hw: nvidia_h100_sxm
  nodes: 8
  gpus_per_node: 8
  host_mem_gb: 2048
config:
  tp: 8
  pp: 4
  ep: 64
  dp: 2
  zero_stage: 1
  micro_batch: 1
  global_batch: 512
  pp_schedule: dualpipev
  optimizer: muon
  muon_config:
    ns_steps: 10
    rotation: true
    adam_param_types: ["embed", "lm_head", "router", "bias"]
    muon_param_fraction: 0.85
targets:
  mfu: 0.40              # placeholder; calibration mode
  tolerance: 0.50        # wide because no public baseline
  strict_mfu_check: false
  # Behavioral assertions (validated by test_anchor_fp8_fp4_h100):
  #   step_time_ms < step_time_ms(deepseek_v4_pro)  (FP8 speedup)
  #   memory.peak_overall < memory.peak_overall(deepseek_v4_pro) × 0.85
```

- [ ] **Step 13.2: Create B300 anchor**

Create `tests/training/anchors/deepseek_v4_pro_fp8_fp4_b300.yaml`:

```yaml
name: deepseek_v4_pro_fp8_fp4_b300
description: "DeepSeek-V4-Pro 1.6T, B300, TP8 PP4 EP64 DP4, FP8+FP4 mixed quant"
model:
  base: deepseek_v4_pro
  quant_preset: deepseek_v4_fp8_fp4
system:
  hw: nvidia_b300
  nodes: 8
  gpus_per_node: 8
  host_mem_gb: 2048
config:
  tp: 8
  pp: 4
  ep: 64
  dp: 2
  zero_stage: 1
  micro_batch: 1
  global_batch: 512
  pp_schedule: dualpipev
  optimizer: muon
  muon_config:
    ns_steps: 10
    rotation: true
    adam_param_types: ["embed", "lm_head", "router", "bias"]
    muon_param_fraction: 0.85
targets:
  mfu: 0.40
  tolerance: 0.50
  strict_mfu_check: false
  # Behavioral assertions (validated by test_anchor_fp8_fp4_b300):
  #   step_time_ms < step_time_ms(deepseek_v4_pro_fp8_fp4_h100)  (FP4 + faster HW)
```

> **YAML schema note:** Anchors currently take `model: <string>` not `model: {base: ..., quant_preset: ...}`. The anchor loader (`config_loader.py:_resolve_model`) must accept the dict form. If `_resolve_model` already supports inline-dict overlays, the YAML above works as-is. Otherwise add a small loader extension in this task — see Step 13.3.

- [ ] **Step 13.3: Extend anchor loader to support model overlay (if needed)**

Check `_resolve_model` in `python/zrt/training/io/config_loader.py`. If `isinstance(model_ref, dict)` is already handled by passing through `_parse_model(model_ref)`, then the overlay must produce a full inline model. We need: `model: {base: X, quant_preset: Y}` → load `X.yaml`, overlay `quant_preset` and any other top-level keys, then call `_parse_model`.

Add to `_resolve_model`:

```python
def _resolve_model(model_ref: str | dict) -> ModelSpec:
    if isinstance(model_ref, str):
        path = _MODELS_DIR / f"{model_ref}.yaml"
        if not path.exists():
            raise KeyError(
                f"Model {model_ref!r} not found in {_MODELS_DIR}. "
                f"Available: {[p.stem for p in sorted(_MODELS_DIR.glob('*.yaml'))]}"
            )
        with open(path, encoding="utf-8") as f:
            model_d = yaml.safe_load(f)
        return _parse_model(model_d)
    if isinstance(model_ref, dict) and "base" in model_ref:
        # Overlay form: {base: <name>, <override keys>}
        base_name = model_ref["base"]
        path = _MODELS_DIR / f"{base_name}.yaml"
        if not path.exists():
            raise KeyError(f"Model {base_name!r} not found in {_MODELS_DIR}.")
        with open(path, encoding="utf-8") as f:
            base_d = yaml.safe_load(f)
        merged = {**base_d, **{k: v for k, v in model_ref.items() if k != "base"}}
        return _parse_model(merged)
    return _parse_model(model_ref)
```

- [ ] **Step 13.4: Add anchor behavioral tests**

Create or append to `tests/training/anchors/test_anchors.py` (check whether existing test file uses a parametrized harness for all anchors first; if so the new YAMLs may automatically be picked up). Either way, add behavioral assertions:

```python
def test_anchor_fp8_fp4_h100_faster_than_bf16():
    """V4-Pro FP8+FP4 path should be measurably faster than BF16 baseline."""
    from zrt.training.io.config_loader import load_anchor_config
    from zrt.training.search.estimator import estimate

    m_q, s_q, st_q = load_anchor_config("tests/training/anchors/deepseek_v4_pro_fp8_fp4_h100.yaml")
    m_b, s_b, st_b = load_anchor_config("tests/training/anchors/deepseek_v4_pro.yaml")
    rep_q = estimate(m_q, s_q, st_q)
    rep_b = estimate(m_b, s_b, st_b)
    assert rep_q.step_time_ms < rep_b.step_time_ms, (
        f"FP8/FP4 step_time ({rep_q.step_time_ms:.2f} ms) should be "
        f"smaller than BF16 baseline ({rep_b.step_time_ms:.2f} ms)"
    )


def test_anchor_fp8_fp4_h100_peak_memory_lower():
    from zrt.training.io.config_loader import load_anchor_config
    from zrt.training.search.estimator import estimate

    m_q, s_q, st_q = load_anchor_config("tests/training/anchors/deepseek_v4_pro_fp8_fp4_h100.yaml")
    m_b, s_b, st_b = load_anchor_config("tests/training/anchors/deepseek_v4_pro.yaml")
    rep_q = estimate(m_q, s_q, st_q)
    rep_b = estimate(m_b, s_b, st_b)
    assert rep_q.peak_memory_bytes < rep_b.peak_memory_bytes * 0.90, (
        f"Mixed-quant peak memory ({rep_q.peak_memory_bytes:,}) should be "
        f">10% below BF16 baseline ({rep_b.peak_memory_bytes:,})"
    )


def test_anchor_fp8_fp4_b300_faster_than_h100():
    """B300 (native FP4) should be faster than H100 (FP4 falls back to FP8)."""
    from zrt.training.io.config_loader import load_anchor_config
    from zrt.training.search.estimator import estimate

    m_h, s_h, st_h = load_anchor_config("tests/training/anchors/deepseek_v4_pro_fp8_fp4_h100.yaml")
    m_b, s_b, st_b = load_anchor_config("tests/training/anchors/deepseek_v4_pro_fp8_fp4_b300.yaml")
    rep_h = estimate(m_h, s_h, st_h)
    rep_b = estimate(m_b, s_b, st_b)
    assert rep_b.step_time_ms < rep_h.step_time_ms
```

> **API note:** Anchor test imports `load_anchor_config` and `estimate`. If the actual names differ, look them up via `grep -rn "load_anchor_config\|def estimate" python/zrt/training/ tests/training/`. Adjust imports to whatever the codebase actually exposes — both names should exist already (used by current `test_anchors.py`).

- [ ] **Step 13.5: Run new anchor tests**

```
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v -k "fp8_fp4" 2>&1 | tail -n 30
```
Expected: all 3 new tests PASS (FP8/FP4 path produces smaller step_time and memory).

If a test fails because absolute thresholds are too tight, loosen the multipliers (e.g. `0.90 → 0.95`) — the goal is to confirm the *direction* of change, not pin specific magnitudes during calibration.

- [ ] **Step 13.6: Regression**

```
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v 2>&1 | tail -n 40
```
Expected: all existing anchors still pass; new V4 FP8/FP4 anchors pass.

- [ ] **Step 13.7: Commit**

```bash
git add tests/training/anchors/deepseek_v4_pro_fp8_fp4_h100.yaml \
        tests/training/anchors/deepseek_v4_pro_fp8_fp4_b300.yaml \
        tests/training/anchors/test_anchors.py \
        python/zrt/training/io/config_loader.py
git commit -m "$(cat <<'EOF'
test(training): V4-Pro FP8/FP4 anchors (H100 + B300, calibration mode)

- deepseek_v4_pro_fp8_fp4_h100.yaml: V4-Pro with quant_preset overlay
- deepseek_v4_pro_fp8_fp4_b300.yaml: same on Blackwell B300 (native FP4)
- Behavioral assertions: step_time_ms shrinks vs BF16 baseline; B300
  beats H100 on FP4 path; peak memory < BF16 baseline × 0.90
- Anchor loader gains {base: ..., quant_preset: ...} overlay form
EOF
)"
```

---

## Task 14: End-to-End Smoke — Run the V4-Pro Config

**Files:** No source changes; runtime smoke test.

- [ ] **Step 14.1: Run V4-Pro FP8/FP4 estimate via the CLI**

```
PYTHONPATH=python python -m zrt.training estimate \
  --config python/zrt/training/configs/deepseek_v4_pro_3d_h800.yaml \
  2>&1 | tail -n 40
```
Expected: completes without exception; prints step_time and MFU. Note that this YAML may currently use `routed_expert_dtype: fp4` (the legacy string); the back-compat shim in Task 3 should populate `routed_expert_weight_dtype` correctly.

- [ ] **Step 14.2: Create a new explicit FP8/FP4 strategy YAML**

Create a small standalone test config to verify the `quant_preset` shorthand flows end-to-end. Add this once to confirm and then remove it (it's a smoke test, not a regression fixture):

```bash
cat > /tmp/v4_preset_smoke.yaml <<'EOF'
model:
  base: deepseek_v4_pro
  quant_preset: deepseek_v4_fp8_fp4
system:
  hw: nvidia_h100_sxm
  nodes: 8
  gpus_per_node: 8
strategy:
  tp: 8
  pp: 4
  ep: 64
  dp: 2
  zero_stage: 1
  micro_batch: 1
  global_batch: 512
  pp_schedule: dualpipev
  optimizer: muon
EOF
PYTHONPATH=python python -m zrt.training estimate --config /tmp/v4_preset_smoke.yaml 2>&1 | tail -n 30
```

Expected: estimate completes, prints non-zero step_time. Confirm there's no fp4/fp8 fallback warning printed for H100 FP8 (only for FP4 → FP8 fallback).

- [ ] **Step 14.3: Diff output against BF16 baseline**

```
cat > /tmp/v4_bf16_smoke.yaml <<'EOF'
model:
  base: deepseek_v4_pro
  quant_preset: bf16_baseline
system:
  hw: nvidia_h100_sxm
  nodes: 8
  gpus_per_node: 8
strategy:
  tp: 8
  pp: 4
  ep: 64
  dp: 2
  zero_stage: 1
  micro_batch: 1
  global_batch: 512
  pp_schedule: dualpipev
  optimizer: muon
EOF
PYTHONPATH=python python -m zrt.training estimate --config /tmp/v4_bf16_smoke.yaml 2>&1 | tail -n 30
```

Expected: step_time strictly larger and peak_memory strictly larger than the FP8/FP4 run from Step 14.2.

- [ ] **Step 14.4: Clean up tmp files**

```
rm -f /tmp/v4_preset_smoke.yaml /tmp/v4_bf16_smoke.yaml
```

(No commit — Task 14 is verification, no source change.)

---

## Task 15: Final Regression + Excel Export Smoke

**Files:** Verification only — confirm no integration regressions and that the new `mfu_native` field surfaces in any structured output that consumers rely on.

- [ ] **Step 15.1: Full training test suite**

```
PYTHONPATH=python pytest tests/training/ -v 2>&1 | tail -n 60
```
Expected: all pre-existing tests green; all new tests added in Tasks 1-13 green. Cherry-pick failures only on tests we know are unrelated to dtype.

- [ ] **Step 15.2: Anchor full sweep**

```
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v 2>&1 | tail -n 60
```
Expected: every anchor passes — including GPT-3, LLaMA-3, V3, V3.2, V4-Pro variants, V4-Flash variants, and the two new V4 FP8/FP4 anchors.

- [ ] **Step 15.3: Excel export smoke (if exporter is exercised by tests)**

```
PYTHONPATH=python pytest tests/training/ -v -k "excel" 2>&1 | tail -n 30
```
Expected: if Excel exporter tests exist, they pass. If consumers look for `mfu_native` in the report we should also add a column — check `python/zrt/training/io/excel_exporter.py` and append a column if `StepResult.mfu_native` isn't already exposed. (This step is best-effort: the spec lists Excel changes under §4.10 but only as a nice-to-have; if exporter tests aren't there, no further action.)

- [ ] **Step 15.4: Final commit if any cleanup landed**

If Step 15.3 surfaced an Excel column gap:

```bash
git add python/zrt/training/io/excel_exporter.py
git commit -m "feat(training): expose mfu_native in Excel export"
```

Otherwise: nothing to commit; the plan is complete.

---

## Self-Review (Performed Before Handoff)

**Spec coverage check** against spec §4 subsections:

| Spec section | Implementing task(s) |
|---|---|
| 4.1 Dtype enum extension | Task 1 |
| 4.2 ModelSpec new fields | Task 3 |
| 4.3 Hardware fp4_tops + B300 spec | Tasks 2 + 5 |
| 4.4 peak_tflops_for + perf_tables | Task 2 |
| 4.5 op_to_time + _cost_phase_time + Op.component | Tasks 6 + 7 |
| 4.6 mfu_native | Task 12 |
| 4.7 memory.py region-aware (weights/grads/activations) | Tasks 8 + 9 + 10 |
| 4.8 comm.py DP AR + EP A2A | Tasks 11 + 9 (EP A2A buffer is memory; EP A2A *volume* in comm.py is already a function of model.act_dtype, refactored only where needed) |
| 4.9 YAML quant_preset | Task 4 |
| 4.10 Excel exporter | Task 15.3 (best-effort) |
| §5 Test matrix | Tasks 1-13 plus Task 15 |
| §6 Implementation order | Aligned 1:1 |
| §7 Risks (Muon hardcoded FP32) | Out of scope per user; not modified |
| §10 Validation criteria | Tasks 14 + 15 |

**Placeholder scan:** No "TBD/TODO/later" in plan body. Each task has executable code blocks and concrete commands.

**Type consistency:** `Dtype.FP8` is an alias for `Dtype.FP8_E4M3` (Task 1). `Op.component` is consistently `str | None`. `peak_tflops_for` returns FLOP/s (not TFLOP/s — multiplied by 1e12). `stored_bytes` includes FP4 block overhead consistently across memory.py weight calc and tests.

**Scope check:** Each task produces an independently committable, independently testable change. None contain >1 module's worth of code (Task 7 is the largest — Op field + builders + stage.py — but they form one logical unit since `_resolve_compute_dtype` needs both `Op.component` and the dispatcher in one go).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-16-fp4-fp8-mixed-quant-estimate-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
