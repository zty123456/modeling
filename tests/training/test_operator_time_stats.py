from __future__ import annotations

from types import SimpleNamespace

from zrt.training.io.operator_time_stats import (
    build_operator_time_stats,
    classify_op_groups,
)
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


def test_operator_time_stats_includes_matmul_total_plus_family_breakdown():
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
    assert by_label["Matmul family total"]["time_ms"] == 30.0
    assert by_label["Matmul family total"]["pct_of_step"] == 0.3
    assert by_label["Matmul family total"]["pct_of_useful_compute"] == 0.75
    assert by_label["Matmul family total"]["op_count"] == 3
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


def test_operator_time_stats_reports_mtp_embed_matmul_not_global_embedding_lookup():
    rows = build_operator_time_stats(
        model=_base_model(layers=[LayerKind.MTP]),
        report=TrainingReport(step_time_ms=100.0, compute_time_ms=50.0),
        op_dicts=[
            {"name": "embed", "kind": "embed", "layer_id": -1, "total_ms": 2.5},
            {
                "name": "L0.mtp_embed_proj",
                "kind": "matmul",
                "layer_kind": "mtp",
                "layer_id": 0,
                "total_ms": 7.5,
            },
            {"name": "lm_head", "kind": "lm_head", "layer_id": -1, "total_ms": 5.0},
        ],
    )

    by_label = _by_label(rows)
    assert by_label["Matmul family total"]["time_ms"] == 12.5
    assert by_label["Matmul family total"]["op_count"] == 2
    assert by_label["MTP embed matmul"]["time_ms"] == 7.5
    assert by_label["MTP embed matmul"]["pct_of_step"] == 0.075
    assert by_label["MTP embed matmul"]["pct_of_useful_compute"] == 0.15
    assert by_label["MTP embed matmul"]["op_count"] == 1
    assert "Embedding lookup" not in by_label


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


def test_classify_op_groups_attention_matmul_is_both_matmul_and_attention():
    groups = classify_op_groups(
        {"name": "L0.q_proj", "kind": "matmul", "component": "attention"}
    )
    assert "matmul" in groups
    assert "attention" in groups
    assert "attention_matmul" in groups
    assert "ffn" not in groups


def test_classify_op_groups_ffn_matmul():
    groups = classify_op_groups(
        {"name": "L0.routed_expert_ffn", "kind": "matmul", "component": "routed_expert"}
    )
    assert "matmul" in groups
    assert "ffn" in groups
    assert "attention_matmul" not in groups


def test_classify_op_groups_lm_head_is_matmul_family():
    groups = classify_op_groups({"name": "lm_head", "kind": "lm_head"})
    assert "matmul" in groups
    assert "lm_head" in groups


def test_classify_op_groups_mtp_embed():
    groups = classify_op_groups(
        {"name": "L0.mtp_embed_proj", "kind": "matmul", "layer_kind": "mtp"}
    )
    assert "matmul" in groups
    assert "mtp_embed" in groups


def test_classify_op_groups_kind_based_indexer_and_sparse_fa():
    assert classify_op_groups({"name": "idx", "kind": "indexer_topk"}) == [
        "attention",
        "indexer",
    ]
    assert classify_op_groups({"name": "fa", "kind": "attn_core"}) == [
        "attention",
        "sparse_fa",
    ]


def test_classify_op_groups_non_compute_op_has_no_groups():
    assert classify_op_groups({"name": "L0.norm", "kind": "rmsnorm"}) == []


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
    composite_label = "CSA/HCA/SWA composite attention core"
    assert by_label[composite_label]["time_ms"] == 18.0
    assert by_label[composite_label]["op_count"] == 3


def test_operator_time_stats_emits_dsv32_sparse_fa_indexer_and_mla_rows():
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
            # MLA projections
            {"name": "L0.q_a_proj", "kind": "matmul", "component": "attention", "layer_id": 0, "total_ms": 4.0},
            {"name": "L0.q_b_proj", "kind": "matmul", "component": "attention", "layer_id": 0, "total_ms": 5.0},
            {"name": "L0.kv_a_proj", "kind": "matmul", "component": "attention", "layer_id": 0, "total_ms": 2.0},
            {"name": "L0.kv_b_proj", "kind": "matmul", "component": "attention", "layer_id": 0, "total_ms": 6.0},
            {"name": "L0.o_proj", "kind": "matmul", "component": "attention", "layer_id": 0, "total_ms": 7.0},
            # Indexer aux + scoring
            {"name": "L0.idx_wq_b", "kind": "matmul", "component": "routed_expert", "layer_id": 0, "total_ms": 1.0},
            {"name": "L0.idx_weights", "kind": "matmul", "component": "routed_expert", "layer_id": 0, "total_ms": 1.5},
            {"name": "L0.idx_comp_wkv", "kind": "matmul", "component": "routed_expert", "layer_id": 0, "total_ms": 1.5},
            {"name": "L0.idx_comp_wgate", "kind": "matmul", "component": "routed_expert", "layer_id": 0, "total_ms": 2.0},
            {"name": "L0.idx_score_topk", "kind": "indexer_topk", "component": "attention", "layer_id": 0, "total_ms": 3.0},
            # Attention core
            {"name": "L0.attn_core", "kind": "attn_core", "component": "attention", "layer_id": 0, "total_ms": 8.0},
            # MoE: router + shared + routed expert
            {"name": "L0.router", "kind": "matmul", "component": "routed_expert", "layer_id": 0, "total_ms": 0.5},
            {"name": "L0.shared_up_proj", "kind": "matmul", "component": "shared_expert", "layer_id": 0, "total_ms": 1.2},
            {"name": "L0.shared_gate_proj", "kind": "matmul", "component": "shared_expert", "layer_id": 0, "total_ms": 1.3},
            {"name": "L0.shared_down_proj", "kind": "matmul", "component": "shared_expert", "layer_id": 0, "total_ms": 1.5},
            {"name": "L0.routed_expert_ffn", "kind": "matmul", "component": "routed_expert", "layer_id": 0, "total_ms": 18.0},
            # Global non-layer ops
            {"name": "embed", "kind": "embed", "layer_id": -1, "total_ms": 2.5},
            {"name": "lm_head", "kind": "lm_head", "layer_id": -1, "total_ms": 6.0},
        ],
    )

    by_label = _by_label(rows)
    # V3.2 per-component matmul breakdown
    assert by_label["MLA proj matmul (Q/KV/O)"]["time_ms"] == 24.0
    assert by_label["MLA proj matmul (Q/KV/O)"]["op_count"] == 5
    assert by_label["Indexer aux matmul"]["time_ms"] == 6.0
    assert by_label["Indexer aux matmul"]["op_count"] == 4
    assert by_label["MoE router matmul"]["time_ms"] == 0.5
    assert by_label["MoE router matmul"]["op_count"] == 1
    assert by_label["MoE shared expert matmul"]["time_ms"] == 4.0
    assert by_label["MoE shared expert matmul"]["op_count"] == 3
    assert by_label["MoE routed expert matmul (gmm)"]["time_ms"] == 18.0
    assert by_label["MoE routed expert matmul (gmm)"]["op_count"] == 1
    # DSA / attention block rows
    assert by_label["Sparse FA core (DSA)"]["time_ms"] == 8.0
    assert by_label["Sparse FA core (DSA)"]["op_count"] == 1
    assert by_label["Lightning Indexer"]["time_ms"] == 3.0
    assert by_label["Lightning Indexer"]["op_count"] == 1
    assert by_label["DSA attention compute"]["time_ms"] == 11.0
    assert by_label["DSA attention compute"]["op_count"] == 2
    # MLA attention block = attention component + indexer_topk + rope/compressor by kind
    # Here: 5 MLA proj + idx_score_topk + attn_core = 24 + 3 + 8 = 35
    assert by_label["MLA attention block"]["time_ms"] == 35.0
    # Global non-layer rows
    assert by_label["LM head matmul"]["time_ms"] == 6.0
    assert "Embedding lookup" not in by_label
    assert "FlashAttention" not in by_label


def test_operator_time_stats_handles_zero_step_time():
    rows = build_operator_time_stats(
        model=_base_model(),
        report=TrainingReport(step_time_ms=0.0, compute_time_ms=0.0),
        op_dicts=[{"name": "mm", "kind": "matmul", "component": "attention", "total_ms": 10.0}],
    )

    assert _by_label(rows)["Attention matmul family"]["pct_of_step"] == 0.0
    assert _by_label(rows)["Attention matmul family"]["pct_of_useful_compute"] == 0.0
