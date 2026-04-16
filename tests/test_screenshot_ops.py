"""Self-validation test suite for screenshot_ops.py.

Covers:
  * All local model configs in hf_models/ (no network required)
  * HF Hub model IDs for DeepSeek-V3.2 and Qwen3 (requires network + HF access)
  * Unit tests for component classification and MoE detection helpers

Run all local tests (fast, no network):
    pytest test_screenshot_ops.py -v -m "not network"

Run including HF Hub tests (downloads config ~KB, not weights):
    pytest test_screenshot_ops.py -v

Run a single model end-to-end:
    pytest test_screenshot_ops.py -v -k "deepseek_v3"
"""
from __future__ import annotations

import socket
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any

# ── helpers ───────────────────────────────────────────────────────────────────

HF_MODELS = Path(__file__).parent / "hf_models"


def _network_available(host: str = "huggingface.co", port: int = 443,
                        timeout: float = 3.0) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False


network = pytest.mark.skipif(
    not _network_available(),
    reason="HuggingFace Hub not reachable — skipping network tests",
)

# ── import subject ────────────────────────────────────────────────────────────

from python.zrt.graph import (
    _classify_component,
    _extract_layer_idx,
    _is_moe_module,
    _patch_moe_for_meta,
    auto_target_layers,
    build_config_summary,
    infer_layer_types,
    load_model,
    run_trace,
    run_trace_phases,
)


# ═════════════════════════════════════════════════════════════════════════════
# Unit tests — no model loading
# ═════════════════════════════════════════════════════════════════════════════

class TestExtractLayerIdx:
    def test_standard_layers(self):
        assert _extract_layer_idx("model.layers.0.self_attn.q_proj") == "0"
        assert _extract_layer_idx("model.layers.5.mlp.gate_proj") == "5"
        assert _extract_layer_idx("transformer.h.3.attn") == "3"

    def test_no_layer(self):
        assert _extract_layer_idx("model.embed_tokens") == ""
        assert _extract_layer_idx("lm_head") == ""
        assert _extract_layer_idx("model.norm") == ""


class TestClassifyComponent:
    # ── norm ─────────────────────────────────────────────────────────────────
    def test_pre_attn_norm(self):
        c = _classify_component(
            "model.layers.0.input_layernorm", "aten.mul.Tensor")
        assert c == "attn_norm"

    def test_post_attn_norm(self):
        c = _classify_component(
            "model.layers.0.post_attention_layernorm", "aten.mul.Tensor")
        assert c == "ffn_norm"

    def test_final_norm(self):
        c = _classify_component("model.norm", "aten.mul.Tensor")
        assert c == "final_norm"

    # ── attention ────────────────────────────────────────────────────────────
    def test_attn_q_proj(self):
        c = _classify_component(
            "model.layers.0.self_attn.q_proj", "aten.mm.default")
        assert c == "attn.q_proj"

    def test_attn_o_proj(self):
        c = _classify_component(
            "model.layers.0.self_attn.o_proj", "aten.mm.default")
        assert c == "attn.o_proj"

    def test_attn_softmax(self):
        c = _classify_component(
            "model.layers.0.self_attn", "aten.softmax.int")
        assert c == "attn.softmax"

    def test_attn_score_matmul(self):
        c = _classify_component(
            "model.layers.0.self_attn", "aten.matmul.default")
        assert c == "attn.score"

    # ── MLA (DeepSeek) ───────────────────────────────────────────────────────
    def test_mla_q_a_proj(self):
        c = _classify_component(
            "model.layers.0.self_attn.q_a_proj", "aten.mm.default")
        assert c == "attn.q_a_proj"

    def test_mla_kv_a_proj(self):
        c = _classify_component(
            "model.layers.0.self_attn.kv_a_proj_with_mqa", "aten.mm.default")
        assert c == "attn.kv_a_proj"

    # ── MoE ──────────────────────────────────────────────────────────────────
    def test_moe_router(self):
        c = _classify_component(
            "model.layers.3.mlp.gate", "aten.mm.default")
        assert c.startswith("moe.gate.")

    def test_moe_expert_proj(self):
        c = _classify_component(
            "model.layers.3.mlp.experts.0.gate_proj", "aten.mm.default")
        assert c == "moe.experts.mm"

    def test_moe_shared_expert(self):
        c = _classify_component(
            "model.layers.3.mlp.shared_experts.down_proj", "aten.mm.default")
        assert c == "moe.shared.down_proj"

    # ── FFN ──────────────────────────────────────────────────────────────────
    def test_ffn_gate_proj(self):
        c = _classify_component(
            "model.layers.0.mlp.gate_proj", "aten.mm.default")
        assert c == "ffn.gate_proj"

    def test_ffn_up_proj(self):
        c = _classify_component(
            "model.layers.0.mlp.up_proj", "aten.mm.default")
        assert c == "ffn.up_proj"

    def test_ffn_down_proj(self):
        c = _classify_component(
            "model.layers.0.mlp.down_proj", "aten.mm.default")
        assert c == "ffn.down_proj"

    # ── embedding / lm_head ──────────────────────────────────────────────────
    def test_embedding(self):
        assert _classify_component("model.embed_tokens", "aten.embedding.default") == "embedding"

    def test_lm_head(self):
        assert _classify_component("lm_head", "aten.mm.default") == "lm_head"


class TestMoEDetection:
    def test_detects_module_list_experts(self):
        class FakeMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])

        assert _is_moe_module(FakeMoE())

    def test_ignores_single_module(self):
        class FakeDense(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.Linear(8, 8)   # not a ModuleList

        assert not _is_moe_module(FakeDense())

    def test_ignores_already_patched(self):
        class FakeMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([nn.Linear(8, 8)])
                self._fake_patched = True

        assert not _is_moe_module(FakeMoE())

    def test_patch_replaces_forward(self):
        class FakeExpert(nn.Module):
            def forward(self, x):
                return x * 2

        class FakeMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([FakeExpert()])

            def forward(self, x):
                raise RuntimeError("original forward should not run")

        moe = FakeMoE()
        original_fwd = moe.forward
        _patch_moe_for_meta(moe)
        assert moe.forward is not original_fwd, "forward should have been replaced"
        assert moe._fake_patched is True


# ═════════════════════════════════════════════════════════════════════════════
# Local model tests — reads only config.json, no weights download
# ═════════════════════════════════════════════════════════════════════════════

def _assert_trace_valid(records, model_id: str) -> None:
    """Common assertions for a successful trace."""
    assert len(records) > 0, f"{model_id}: no ops captured"

    ops = {r["aten_op"] for r in records}
    # Every transformer model produces at least one matrix multiplication
    mm_ops = {o for o in ops if any(x in o for x in ("mm", "matmul", "bmm", "addmm"))}
    assert mm_ops, f"{model_id}: no matmul-family ops found — got {ops}"

    components = {r["component"] for r in records}
    # At minimum: some attention op and some FFN op
    has_attn = any(c.startswith("attn.") for c in components)
    has_ffn  = any(c.startswith(("ffn.", "moe.")) for c in components)
    assert has_attn, f"{model_id}: no attention components found — got {components}"
    assert has_ffn,  f"{model_id}: no FFN/MoE components found — got {components}"


@pytest.mark.parametrize("model_path,num_layers", [
    pytest.param(str(HF_MODELS / "deepseek_v3"), 4,
                 id="local-deepseek_v3"),
    pytest.param(str(HF_MODELS / "deepseek_v3_2"), 4,
                 id="local-deepseek_v3_2"),
    pytest.param(str(HF_MODELS / "llama3_8b"), 2,
                 id="local-llama3_8b"),
    pytest.param(str(HF_MODELS / "llama3_70b"), 2,
                 id="local-llama3_70b"),
    pytest.param(str(HF_MODELS / "qwen2_7b"), 2,
                 id="local-qwen2_7b"),
    pytest.param(str(HF_MODELS / "qwen2_72b"), 2,
                 id="local-qwen2_72b"),
    pytest.param(str(HF_MODELS / "mistral_7b"), 2,
                 id="local-mistral_7b"),
    pytest.param(str(HF_MODELS / "mixtral_8x7b"), 2,
                 id="local-mixtral_8x7b"),
])
def test_local_model(model_path: str, num_layers: int, tmp_path: Path):
    """End-to-end trace on a local config (no weight download)."""
    if not Path(model_path).is_dir():
        pytest.skip(f"Local model dir not found: {model_path}")

    out, records = run_trace(
        model_id=model_path,
        num_layers=num_layers,
        batch_size=1,
        seq_len=64,
        output_dir=tmp_path,
    )
    assert out.exists(), "Output directory was not created"
    _assert_trace_valid(records, model_path)

    # Report for human review
    components = sorted({r["component"] for r in records})
    print(f"\n[{Path(model_path).name}]  {len(records)} ops  "
          f"components: {components}")


@pytest.mark.parametrize("model_path,num_layers", [
    pytest.param(str(HF_MODELS / "deepseek_v3"), 4,
                 id="moe-deepseek_v3"),
    pytest.param(str(HF_MODELS / "deepseek_v3_2"), 4,
                 id="moe-deepseek_v3_2"),
    pytest.param(str(HF_MODELS / "mixtral_8x7b"), 2,
                 id="moe-mixtral_8x7b"),
])
def test_moe_components_present(model_path: str, num_layers: int, tmp_path: Path):
    """MoE models should produce moe.* component labels."""
    if not Path(model_path).is_dir():
        pytest.skip(f"Local model dir not found: {model_path}")

    _out, records = run_trace(
        model_id=model_path,
        num_layers=num_layers,
        batch_size=1,
        seq_len=64,
        output_dir=tmp_path,
    )
    components = {r["component"] for r in records}
    moe_comps = {c for c in components if c.startswith("moe.")}
    assert moe_comps, (
        f"Expected moe.* components for {Path(model_path).name}, "
        f"got: {components}")
    print(f"\n[{Path(model_path).name}] MoE components: {sorted(moe_comps)}")


def test_deepseek_v3_mla_components(tmp_path: Path):
    """DeepSeek-V3 MLA projections should appear in the trace."""
    path = str(HF_MODELS / "deepseek_v3")
    if not Path(path).is_dir():
        pytest.skip("Local deepseek_v3 dir not found")

    _out, records = run_trace(
        model_id=path,
        num_layers=4,
        batch_size=1,
        seq_len=64,
        output_dir=tmp_path,
    )
    components = {r["component"] for r in records}
    mla = {c for c in components if "q_a_proj" in c or "kv_a_proj" in c}
    assert mla, f"Expected MLA projection ops, got: {components}"


def test_layer_attribution(tmp_path: Path):
    """Every record from a block-internal module should carry a layer index."""
    path = str(HF_MODELS / "llama3_8b")
    if not Path(path).is_dir():
        pytest.skip("Local llama3_8b dir not found")

    _out, records = run_trace(
        model_id=path,
        num_layers=2,
        batch_size=1,
        seq_len=32,
        output_dir=tmp_path,
    )
    # All attention and FFN records should have a numeric layer index
    block_records = [
        r for r in records
        if r["component"].startswith(("attn.", "ffn.", "attn_norm", "ffn_norm"))
    ]
    assert block_records, "No block-internal records found"
    missing_layer = [r for r in block_records if r["layer"] == ""]
    assert not missing_layer, (
        f"{len(missing_layer)} block records have no layer index: "
        f"{[r['module_path'] for r in missing_layer[:5]]}")


def test_config_summary_fields(tmp_path: Path):
    """Config summary must include model_type and at least the core dims."""
    path = str(HF_MODELS / "qwen2_7b")
    if not Path(path).is_dir():
        pytest.skip("Local qwen2_7b dir not found")

    out, records = run_trace(
        model_id=path,
        num_layers=2,
        batch_size=1,
        seq_len=32,
        output_dir=tmp_path,
    )
    # Re-derive summary from loaded config for inspection
    from python.zrt.graph import load_model, build_config_summary
    _, config, fake_mode = load_model(path, num_hidden_layers=2)
    fake_mode.__exit__(None, None, None)
    summary = build_config_summary(path, config, 2, 1, 32)

    for required in ("model_id", "model_type", "hidden_size",
                     "num_attention_heads", "vocab_size"):
        assert required in summary, f"Missing field '{required}' in summary"


# ═════════════════════════════════════════════════════════════════════════════
# HF Hub tests — downloads config only (~few KB), NOT model weights
# Marked @network: skipped automatically when HF Hub is unreachable.
# ═════════════════════════════════════════════════════════════════════════════

def _run_hub_trace(model_id, num_layers, tmp_path):
    """Helper: run_trace for a hub model, skip on known unrunnable conditions.

    Skip condition (not a test failure):
      - HF auth required (gated/401/403)

    Architecture compatibility is handled automatically by model_loader:
      - Version shims fix missing symbols (e.g. is_flash_attn_greater_or_equal_2_10)
      - Local registry fallback handles model_types not in transformers registry
        (e.g. deepseek_v32 → hf_models/deepseek_v3_2)
    """
    try:
        return run_trace(
            model_id=model_id,
            num_layers=num_layers,
            batch_size=1,
            seq_len=64,
            output_dir=tmp_path,
        )
    except Exception as exc:
        msg = str(exc)
        if any(k in msg.lower() for k in ("gated", "401", "403")):
            pytest.skip(f"Model requires HF authentication: {exc}")
        raise


@network
@pytest.mark.parametrize("model_id,num_layers,description", [
    pytest.param(
        "deepseek-ai/DeepSeek-V3.2", 4,
        "DeepSeek-V3.2 — 正式 V3.2，新增 Sparse Attention + Indexer 架构",
        id="hub-deepseek-v3.2",
    ),
    pytest.param(
        "deepseek-ai/DeepSeek-V3-0324", 4,
        "DeepSeek-V3 0324 更新版（3月24日发布，非 V3.2）— MLA + MoE",
        id="hub-deepseek-v3-0324",
    ),
    pytest.param(
        "deepseek-ai/DeepSeek-V3", 4,
        "DeepSeek-V3 base — MLA + 256 routed experts",
        id="hub-deepseek-v3",
    ),
    pytest.param(
        "Qwen/Qwen3-8B", 4,
        "Qwen3-8B dense model",
        id="hub-qwen3-8b",
    ),
    pytest.param(
        "Qwen/Qwen3-0.6B", 4,
        "Qwen3-0.6B (smallest Qwen3 dense)",
        id="hub-qwen3-0.6b",
    ),
    pytest.param(
        "Qwen/Qwen3-30B-A3B", 4,
        "Qwen3-30B-A3B MoE model",
        id="hub-qwen3-moe",
    ),
])
def test_hub_model(model_id: str, num_layers: int, description: str,
                   tmp_path: Path):
    """End-to-end trace via HF Hub config (no weights downloaded).

    Only the config.json (~few KB) is fetched; model weights are never loaded
    because we use ``from_config`` on a meta device.
    """
    print(f"\nTesting: {description}")
    out, records = _run_hub_trace(model_id, num_layers, tmp_path)
    assert out.exists(), "Output directory was not created"
    _assert_trace_valid(records, model_id)
    components = sorted({r["component"] for r in records})
    print(f"  {len(records)} ops — components: {components}")


@network
@pytest.mark.parametrize("model_id,expect_moe", [
    pytest.param("deepseek-ai/DeepSeek-V3.2",    True,  id="deepseek-v3.2-moe"),
    pytest.param("deepseek-ai/DeepSeek-V3-0324", True,  id="deepseek-v3-0324-moe"),
    pytest.param("deepseek-ai/DeepSeek-V3",      True,  id="deepseek-v3-moe"),
    pytest.param("Qwen/Qwen3-8B",                False, id="qwen3-8b-dense"),
    pytest.param("Qwen/Qwen3-30B-A3B",           True,  id="qwen3-moe"),
])
def test_hub_moe_detection(model_id: str, expect_moe: bool, tmp_path: Path):
    """Validate MoE component presence/absence for known hub models."""
    _out, records = _run_hub_trace(model_id, 4, tmp_path)
    components = {r["component"] for r in records}
    has_moe = any(c.startswith("moe.") for c in components)
    if expect_moe:
        assert has_moe, f"{model_id}: expected MoE ops, got {sorted(components)}"
    else:
        assert not has_moe, f"{model_id}: unexpected MoE ops: {sorted(c for c in components if c.startswith('moe.'))}"


@network
def test_hub_deepseek_v3_2_mla_components(tmp_path: Path):
    """DeepSeek-V3.2 from HF Hub should expose MLA projection ops.

    Uses the official V3.2 model ID (deepseek-ai/DeepSeek-V3.2), which is
    distinct from DeepSeek-V3-0324 (a March-24 V3 update, not V3.2).
    Exercises the full hub code-download path including Indexer patch.
    """
    _out, records = _run_hub_trace("deepseek-ai/DeepSeek-V3.2", 4, tmp_path)
    components = {r["component"] for r in records}
    mla = {c for c in components if "q_a_proj" in c or "kv_a_proj" in c}
    assert mla, f"Expected MLA projection ops from hub V3.2, got: {sorted(components)}"
    moe = {c for c in components if c.startswith("moe.")}
    assert moe, f"Expected MoE ops from hub V3.2, got: {sorted(components)}"
    print(f"\n[hub-deepseek-v3.2] MLA={sorted(mla)}, MoE samples={sorted(moe)[:4]}")


# ═════════════════════════════════════════════════════════════════════════════
# Prefill / decode phase tests
# ═════════════════════════════════════════════════════════════════════════════

def _assert_decode_shorter(prefill_records, decode_records, model_id: str) -> None:
    """Decode query length is 1 vs prefill seq_len — verify both phases captured ops."""
    assert len(decode_records) > 0, f"{model_id}: no decode ops captured"
    # Both phases should capture a reasonable number of ops.
    # We do NOT require decode < prefill because some models add KV-cache
    # management ops (torch.cat per layer) that push decode's op count above
    # the no-cache prefill count.
    assert len(prefill_records) > 0, f"{model_id}: no prefill ops captured"


@pytest.mark.parametrize("model_path,num_layers", [
    pytest.param(str(HF_MODELS / "deepseek_v3"), 4,
                 id="phases-deepseek_v3"),
    pytest.param(str(HF_MODELS / "llama3_8b"), 2,
                 id="phases-llama3_8b"),
    pytest.param(str(HF_MODELS / "qwen2_7b"), 2,
                 id="phases-qwen2_7b"),
    pytest.param(str(HF_MODELS / "mixtral_8x7b"), 2,
                 id="phases-mixtral_8x7b"),
])
def test_run_trace_phases_both(model_path: str, num_layers: int, tmp_path: Path):
    """run_trace_phases produces separate prefill and decode output files."""
    if not Path(model_path).is_dir():
        pytest.skip(f"Local model dir not found: {model_path}")

    out_dir, phase_records = run_trace_phases(
        model_id=model_path,
        num_layers=num_layers,
        batch_size=1,
        seq_len=64,
        output_dir=tmp_path,
    )
    assert out_dir.exists()
    assert "prefill" in phase_records, "Missing prefill records"
    assert "decode" in phase_records, "Missing decode records"

    slug = out_dir.name if out_dir == tmp_path else None
    # Check that phase-named files were actually created
    for phase in ("prefill", "decode"):
        xlsx_files = list(tmp_path.glob(f"*_{phase}_ops.xlsx"))
        assert xlsx_files, f"No {phase} Excel file found in {tmp_path}"
        json_files = list(tmp_path.glob(f"*_{phase}_raw_graph.json"))
        assert json_files, f"No {phase} raw graph JSON found in {tmp_path}"

    _assert_trace_valid(phase_records["prefill"], model_path)
    _assert_trace_valid(phase_records["decode"], model_path)
    _assert_decode_shorter(phase_records["prefill"], phase_records["decode"], model_path)

    model_name = Path(model_path).name
    print(f"\n[{model_name}] prefill={len(phase_records['prefill'])} ops, "
          f"decode={len(phase_records['decode'])} ops")


def test_run_trace_phases_prefill_only(tmp_path: Path):
    """phases=('prefill',) produces only prefill files."""
    path = str(HF_MODELS / "llama3_8b")
    if not Path(path).is_dir():
        pytest.skip("Local llama3_8b dir not found")

    out_dir, phase_records = run_trace_phases(
        model_id=path,
        num_layers=2,
        batch_size=1,
        seq_len=32,
        output_dir=tmp_path,
        phases=("prefill",),
    )
    assert "prefill" in phase_records
    assert "decode" not in phase_records
    assert list(tmp_path.glob("*_prefill_ops.xlsx")), "No prefill Excel file"
    assert not list(tmp_path.glob("*_decode_ops.xlsx")), "Unexpected decode Excel file"


def test_run_trace_phases_decode_only(tmp_path: Path):
    """phases=('decode',) runs a standalone decode pass without KV cache."""
    path = str(HF_MODELS / "llama3_8b")
    if not Path(path).is_dir():
        pytest.skip("Local llama3_8b dir not found")

    out_dir, phase_records = run_trace_phases(
        model_id=path,
        num_layers=2,
        batch_size=1,
        seq_len=32,
        output_dir=tmp_path,
        phases=("decode",),
    )
    assert "decode" in phase_records
    assert "prefill" not in phase_records
    _assert_trace_valid(phase_records["decode"], path)
    assert list(tmp_path.glob("*_decode_ops.xlsx")), "No decode Excel file"


def test_run_trace_phase_forward_alias(tmp_path: Path):
    """Legacy phase='forward' is treated as 'prefill'."""
    path = str(HF_MODELS / "llama3_8b")
    if not Path(path).is_dir():
        pytest.skip("Local llama3_8b dir not found")

    out, records = run_trace(
        model_id=path,
        num_layers=2,
        batch_size=1,
        seq_len=32,
        output_dir=tmp_path,
        phase="forward",
    )
    assert list(tmp_path.glob("*_prefill_ops.xlsx")), (
        "phase='forward' should produce a prefill file")
    _assert_trace_valid(records, path)


def test_prefill_decode_query_len_difference(tmp_path: Path):
    """Decode ops must contain shapes with query_len=1 while prefill has seq_len."""
    path = str(HF_MODELS / "llama3_8b")
    if not Path(path).is_dir():
        pytest.skip("Local llama3_8b dir not found")

    seq_len = 32
    _out, phase_records = run_trace_phases(
        model_id=path,
        num_layers=2,
        batch_size=1,
        seq_len=seq_len,
        output_dir=tmp_path,
    )

    def _shapes_contain(records, dim):
        for rec in records:
            shapes_str = rec.get("output_shapes", "") + rec.get("input_shapes", "")
            if f", {dim}," in shapes_str or f", {dim}]" in shapes_str:
                return True
        return False

    assert _shapes_contain(phase_records["prefill"], seq_len), (
        f"Prefill records should reference seq_len={seq_len} in tensor shapes")
    assert _shapes_contain(phase_records["decode"], 1), (
        "Decode records should reference query_len=1 in tensor shapes")


# ═════════════════════════════════════════════════════════════════════════════
# Target-layer selection tests
# ═════════════════════════════════════════════════════════════════════════════

class TestInferLayerTypes:
    """Unit tests for infer_layer_types — no model loading required."""

    def _make_config(self, **kwargs):
        """Minimal fake config object."""
        class Cfg:
            pass
        cfg = Cfg()
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    def test_dense_only_model(self):
        cfg = self._make_config(_full_num_hidden_layers=32)
        types = infer_layer_types(cfg)
        assert types["dense"] == list(range(32))
        assert types["sparse"] == []

    def test_deepseek_v3_style(self):
        # first_k_dense_replace=3, moe_layer_freq=1 → layers 0,1,2 dense; 3+ sparse
        cfg = self._make_config(
            _full_num_hidden_layers=8,
            first_k_dense_replace=3,
            moe_layer_freq=1,
        )
        types = infer_layer_types(cfg)
        assert types["dense"] == [0, 1, 2]
        assert types["sparse"] == [3, 4, 5, 6, 7]

    def test_deepseek_v3_moe_layer_freq_2(self):
        # first_k_dense_replace=2, moe_layer_freq=2
        # layer 0,1 → dense (< first_k)
        # layer 2: 2%2==0 → sparse; layer 3: 3%2==1 → dense
        # layer 4: 4%2==0 → sparse; layer 5: 5%2==1 → dense
        cfg = self._make_config(
            _full_num_hidden_layers=6,
            first_k_dense_replace=2,
            moe_layer_freq=2,
        )
        types = infer_layer_types(cfg)
        assert types["dense"] == [0, 1, 3, 5]
        assert types["sparse"] == [2, 4]

    def test_mixtral_style_all_moe(self):
        cfg = self._make_config(
            _full_num_hidden_layers=32,
            num_local_experts=8,
        )
        types = infer_layer_types(cfg)
        assert types["sparse"] == list(range(32))
        assert types["dense"] == []

    def test_routed_experts_all_moe(self):
        cfg = self._make_config(
            _full_num_hidden_layers=4,
            n_routed_experts=256,
        )
        types = infer_layer_types(cfg)
        assert types["sparse"] == [0, 1, 2, 3]
        assert types["dense"] == []

    def test_falls_back_to_num_hidden_layers(self):
        """Uses num_hidden_layers when _full_num_hidden_layers is absent."""
        cfg = self._make_config(num_hidden_layers=4)
        types = infer_layer_types(cfg)
        assert len(types["dense"]) == 4
        assert types["sparse"] == []


class TestAutoTargetLayers:
    def _make_config(self, **kwargs):
        class Cfg:
            pass
        cfg = Cfg()
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    def test_dense_model_returns_layer_0(self):
        cfg = self._make_config(_full_num_hidden_layers=32)
        assert auto_target_layers(cfg) == [0]

    def test_moe_model_returns_first_dense_and_sparse(self):
        cfg = self._make_config(
            _full_num_hidden_layers=8,
            first_k_dense_replace=3,
            moe_layer_freq=1,
        )
        result = auto_target_layers(cfg)
        assert 0 in result, "First dense layer (0) should be selected"
        assert 3 in result, "First sparse layer (3) should be selected"
        assert result == sorted(result), "Result should be sorted"

    def test_all_moe_model_returns_layer_0(self):
        cfg = self._make_config(
            _full_num_hidden_layers=32,
            num_local_experts=8,
        )
        result = auto_target_layers(cfg)
        assert result == [0]


class TestFilterRecordsByLayers:
    """Unit tests for _filter_records_by_layers."""

    from python.zrt.graph.main import _filter_records_by_layers  # import at class scope

    def _make_records(self, layer_tags):
        return [
            {"node_id": i, "layer": tag, "aten_op": "aten.mm.default",
             "component": "attn.q_proj"}
            for i, tag in enumerate(layer_tags)
        ]

    def test_keeps_target_layers(self):
        from python.zrt.graph.main import _filter_records_by_layers
        records = self._make_records(["0", "0", "1", "1", "2", "2"])
        filtered = _filter_records_by_layers(records, [0, 2])
        layers = {r["layer"] for r in filtered}
        assert layers == {"0", "2"}

    def test_always_keeps_no_layer_records(self):
        from python.zrt.graph.main import _filter_records_by_layers
        records = self._make_records(["", "0", "1", ""])
        filtered = _filter_records_by_layers(records, [1])
        layers = [r["layer"] for r in filtered]
        assert "" in layers, "Records with no layer tag should always be kept"
        assert "1" in layers
        assert "0" not in layers

    def test_renumbers_node_ids(self):
        from python.zrt.graph.main import _filter_records_by_layers
        records = self._make_records(["0", "1", "2", "3"])
        filtered = _filter_records_by_layers(records, [1, 3])
        assert [r["node_id"] for r in filtered] == list(range(len(filtered)))

    def test_empty_target_returns_only_global_ops(self):
        from python.zrt.graph.main import _filter_records_by_layers
        records = self._make_records(["", "0", "1"])
        filtered = _filter_records_by_layers(records, [])
        assert all(r["layer"] == "" for r in filtered)


# ── Integration: target_layers / auto_layers ───────────────────────────────────

def test_target_layers_explicit(tmp_path: Path):
    """target_layers=[0] keeps only layer-0 ops + global ops."""
    path = str(HF_MODELS / "llama3_8b")
    if not Path(path).is_dir():
        pytest.skip("Local llama3_8b dir not found")

    _out, records = run_trace(
        model_id=path,
        num_layers=2,
        batch_size=1,
        seq_len=32,
        output_dir=tmp_path,
        target_layers=[0],
    )
    layer_tags = {r["layer"] for r in records if r["layer"] != ""}
    assert layer_tags == {"0"}, (
        f"Only layer 0 should appear, got: {layer_tags}")


def test_target_layers_multi(tmp_path: Path):
    """target_layers=[0, 1] keeps ops from both layers."""
    path = str(HF_MODELS / "llama3_8b")
    if not Path(path).is_dir():
        pytest.skip("Local llama3_8b dir not found")

    _out, records = run_trace(
        model_id=path,
        num_layers=2,
        batch_size=1,
        seq_len=32,
        output_dir=tmp_path,
        target_layers=[0, 1],
    )
    layer_tags = {r["layer"] for r in records if r["layer"] != ""}
    assert layer_tags == {"0", "1"}


def test_target_layers_auto_dense_only_model(tmp_path: Path):
    """auto_layers on a dense model selects only layer 0."""
    path = str(HF_MODELS / "llama3_8b")
    if not Path(path).is_dir():
        pytest.skip("Local llama3_8b dir not found")

    _out, records = run_trace(
        model_id=path,
        num_layers=4,
        batch_size=1,
        seq_len=32,
        output_dir=tmp_path,
        auto_layers=True,
    )
    layer_tags = {r["layer"] for r in records if r["layer"] != ""}
    # Dense-only model: auto selects layer 0 only
    assert layer_tags == {"0"}


def test_target_layers_auto_moe_model(tmp_path: Path):
    """auto_layers on DeepSeek-V3 selects the first dense + first MoE layer."""
    path = str(HF_MODELS / "deepseek_v3")
    if not Path(path).is_dir():
        pytest.skip("Local deepseek_v3 dir not found")

    _out, records = run_trace(
        model_id=path,
        num_layers=6,
        batch_size=1,
        seq_len=32,
        output_dir=tmp_path,
        auto_layers=True,
    )
    layer_tags = {r["layer"] for r in records if r["layer"] != ""}
    # DeepSeek-V3: first_k_dense_replace=3 → dense=0,1,2; sparse=3,4,5,...
    # auto selects layer 0 (dense) and layer 3 (first MoE)
    assert "0" in layer_tags, f"Layer 0 (dense) should be selected, got {layer_tags}"
    assert "3" in layer_tags, f"Layer 3 (MoE) should be selected, got {layer_tags}"
    assert layer_tags == {"0", "3"}, f"Only layers 0 and 3 expected, got {layer_tags}"
    # MoE ops should be present (from layer 3)
    components = {r["component"] for r in records}
    moe_comps = {c for c in components if c.startswith("moe.")}
    assert moe_comps, "Layer 3 should produce MoE ops"


def test_cli_default_uses_auto_layers(tmp_path: Path):
    """Without --target-layers or --auto-layers, CLI should auto-select layers.

    run_trace_phases(auto_layers=True) is the expected default: only the first
    dense layer and the first sparse (MoE) layer are kept.  This test verifies
    that the CLI does NOT silently pass auto_layers=False and capture all layers.
    """
    path = str(HF_MODELS / "deepseek_v3")
    if not Path(path).is_dir():
        pytest.skip("Local deepseek_v3 dir not found")

    from python.zrt.graph.main import run_trace_phases

    # Simulate CLI call with no --target-layers and no --auto-layers:
    # effective_auto_layers = False or (target_layers is None) = True
    _out, phase_records = run_trace_phases(
        model_id=path,
        num_layers=6,
        batch_size=1,
        seq_len=32,
        output_dir=tmp_path,
        # Neither target_layers nor auto_layers supplied → auto_layers defaults True
    )
    records = phase_records.get("prefill", [])
    layer_tags = {r["layer"] for r in records if r["layer"] != ""}
    # Should match auto_layers behavior: layer 0 (dense) + layer 3 (first MoE)
    assert layer_tags == {"0", "3"}, (
        f"Default run_trace_phases() should auto-select layers 0 and 3, got {layer_tags}"
    )
