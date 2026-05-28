from __future__ import annotations

from types import SimpleNamespace

from zrt.training.io.operator_time_stats import build_operator_time_stats
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.report import TrainingReport


def _base_model(**overrides) -> ModelSpec:
    data = {
        "hidden": 128,
        "ffn": 256,
        "num_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 32,
        "vocab": 1000,
        "seq_len": 64,
        "layers": [LayerKind.DENSE],
    }
    data.update(overrides)
    return ModelSpec(**data)


def _by_label(rows: list[dict]) -> dict[str, dict]:
    return {row["label"]: row for row in rows}


def test_operator_time_stats_splits_matmul_family_and_lm_head_without_total():
    rows = build_operator_time_stats(
        model=_base_model(),
        report=TrainingReport(step_time_ms=100.0, compute_time_ms=40.0),
        op_dicts=[
            {"name": "layer.q_proj", "kind": "matmul", "component": "attention", "total_ms": 12.0},
            {"name": "layer.up_proj", "kind": "matmul", "component": "routed_expert", "total_ms": 10.0},
            {"name": "lm_head", "kind": "lm_head", "total_ms": 8.0},
            {"name": "layer.norm", "kind": "rmsnorm", "total_ms": 5.0},
        ],
    )

    by_label = _by_label(rows)
    assert "Matmul family total" not in by_label
    assert by_label["Attention matmul family"]["time_ms"] == 12.0
    assert by_label["Attention matmul family"]["pct_of_step"] == 0.12
    assert by_label["Attention matmul family"]["pct_of_useful_compute"] == 0.3
    assert by_label["Attention matmul family"]["op_count"] == 1
    assert by_label["MoE/FFN matmul family"]["time_ms"] == 10.0
    assert by_label["MoE/FFN matmul family"]["pct_of_step"] == 0.1
    assert by_label["MoE/FFN matmul family"]["pct_of_useful_compute"] == 0.25
    assert by_label["MoE/FFN matmul family"]["op_count"] == 1
    assert by_label["LM head matmul"]["time_ms"] == 8.0
    assert by_label["LM head matmul"]["pct_of_step"] == 0.08
    assert by_label["LM head matmul"]["pct_of_useful_compute"] == 0.2
    assert by_label["LM head matmul"]["op_count"] == 1


def test_operator_time_stats_counts_compressor_and_indexer_matmuls_as_attention():
    rows = build_operator_time_stats(
        model=_base_model(),
        report=TrainingReport(step_time_ms=100.0, compute_time_ms=60.0),
        op_dicts=[
            {"name": "L0.comp_wkv", "kind": "matmul", "component": "routed_expert", "total_ms": 3.0},
            {"name": "L0.comp_wgate", "kind": "matmul", "component": "routed_expert", "total_ms": 4.0},
            {"name": "L0.idx_wq_b", "kind": "matmul", "component": "routed_expert", "total_ms": 5.0},
            {"name": "L0.idx_weights", "kind": "matmul", "component": "routed_expert", "total_ms": 6.0},
            {"name": "L0.idx_comp_wkv", "kind": "matmul", "component": "routed_expert", "total_ms": 7.0},
            {"name": "L0.idx_comp_wgate", "kind": "matmul", "component": "routed_expert", "total_ms": 8.0},
            {"name": "L0.routed_expert_ffn", "kind": "matmul", "component": "routed_expert", "total_ms": 9.0},
        ],
    )

    by_label = _by_label(rows)
    assert by_label["Attention matmul family"]["time_ms"] == 33.0
    assert by_label["Attention matmul family"]["op_count"] == 6
    assert by_label["MoE/FFN matmul family"]["time_ms"] == 9.0
    assert by_label["MoE/FFN matmul family"]["op_count"] == 1


def test_operator_time_stats_uses_compute_time_for_schedule_aware_step_scale():
    rows = build_operator_time_stats(
        model=_base_model(),
        report=TrainingReport(step_time_ms=200.0, compute_time_ms=100.0),
        strategy=SimpleNamespace(pp=2, num_microbatches=lambda: 8),
        op_dicts=[
            {"name": "layer.q_proj", "kind": "matmul", "total_ms": 10.0},
            {"name": "layer.norm", "kind": "rmsnorm", "total_ms": 10.0},
        ],
    )

    matmul = _by_label(rows)["Attention matmul family"]
    assert matmul["time_ms"] == 50.0
    assert matmul["pct_of_step"] == 0.25
    assert matmul["pct_of_useful_compute"] == 0.5
    assert matmul["op_count"] == 1


def test_operator_time_stats_falls_back_to_microbatch_pp_scale_without_compute_time():
    rows = build_operator_time_stats(
        model=_base_model(),
        report=TrainingReport(step_time_ms=100.0, compute_time_ms=0.0),
        strategy=SimpleNamespace(pp=2, num_microbatches=lambda: 8),
        op_dicts=[
            {"name": "layer.q_proj", "kind": "matmul", "total_ms": 10.0},
        ],
    )

    matmul = _by_label(rows)["Attention matmul family"]
    assert matmul["time_ms"] == 40.0
    assert matmul["pct_of_step"] == 0.4
    assert matmul["pct_of_useful_compute"] == 0.0
    assert matmul["op_count"] == 1


def test_operator_time_stats_emits_dsv4_csa_hca_and_swa_rows():
    model = _base_model(
        model_type="deepseek_v4",
        num_kv_heads=1,
        o_lora_rank=16,
        o_groups=2,
        compress_ratios=[4, 128, 0],
        swa_window=128,
        index_topk=16,
        layers=[LayerKind.MOE, LayerKind.MOE, LayerKind.MOE],
        num_experts=8,
        moe_ffn=64,
        top_k=2,
    )

    rows = build_operator_time_stats(
        model=model,
        report=TrainingReport(step_time_ms=100.0),
        op_dicts=[
            {"name": "L0.wq_a", "kind": "matmul", "component": "attention", "layer_id": 0, "total_ms": 3.0},
            {
                "name": "L0.sparse_attn",
                "kind": "sparse_attn",
                "component": "attention",
                "layer_id": 0,
                "total_ms": 7.0,
                "meta": {"sparse_topk": 16, "swa_window": 16},
            },
            {"name": "L1.wq_a", "kind": "matmul", "component": "attention", "layer_id": 1, "total_ms": 4.0},
            {
                "name": "L1.hca_attn",
                "kind": "hca_attn",
                "component": "attention",
                "layer_id": 1,
                "total_ms": 6.0,
                "meta": {"s": 128, "compress_ratio": 128, "swa_window": 1},
            },
            {
                "name": "L2.swa_attn",
                "kind": "swa_attn",
                "component": "attention",
                "layer_id": 2,
                "total_ms": 5.0,
                "meta": {"swa_window": 128},
            },
        ],
    )

    by_label = _by_label(rows)
    assert by_label["CSA attention block"]["time_ms"] == 10.0
    assert by_label["CSA attention block"]["pct_of_step"] == 0.1
    assert by_label["HCA attention block"]["time_ms"] == 10.0
    assert by_label["SWA operator"]["time_ms"] == 18.0
    assert by_label["SWA operator"]["op_count"] == 3


def test_operator_time_stats_emits_dsv32_flashattention_and_mla_rows():
    model = _base_model(
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        index_topk=16,
    )

    rows = build_operator_time_stats(
        model=model,
        report=TrainingReport(step_time_ms=100.0),
        op_dicts=[
            {"name": "L0.q_a_proj", "kind": "matmul", "component": "attention", "layer_id": 0, "total_ms": 4.0},
            {"name": "L0.idx_score_topk", "kind": "indexer_topk", "component": "attention", "layer_id": 0, "total_ms": 3.0},
            {"name": "L0.attn_core", "kind": "attn_core", "component": "attention", "layer_id": 0, "total_ms": 8.0},
            {"name": "L0.ffn", "kind": "matmul", "component": "routed_expert", "layer_id": 0, "total_ms": 20.0},
        ],
    )

    by_label = _by_label(rows)
    assert by_label["FlashAttention"]["time_ms"] == 8.0
    assert by_label["FlashAttention"]["pct_of_step"] == 0.08
    assert by_label["MLA attention block"]["time_ms"] == 15.0


def test_operator_time_stats_handles_zero_step_time():
    rows = build_operator_time_stats(
        model=_base_model(),
        report=TrainingReport(step_time_ms=0.0, compute_time_ms=0.0),
        op_dicts=[{"name": "mm", "kind": "matmul", "component": "attention", "total_ms": 10.0}],
    )

    assert _by_label(rows)["Attention matmul family"]["pct_of_step"] == 0.0
    assert _by_label(rows)["Attention matmul family"]["pct_of_useful_compute"] == 0.0
