from __future__ import annotations

from typing import Any


def _to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _to_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _pct(time_ms: float, step_time_ms: float) -> float:
    if step_time_ms <= 0:
        return 0.0
    return time_ms / step_time_ms


def _training_step_scale(strategy: Any | None) -> float:
    if strategy is None:
        return 1.0

    num_microbatches = getattr(strategy, "num_microbatches", None)
    try:
        microbatches = float(
            num_microbatches() if callable(num_microbatches) else num_microbatches
        )
    except Exception:
        microbatches = 1.0

    pp = _to_float(getattr(strategy, "pp", 1.0))
    if microbatches <= 0 or pp <= 0:
        return 1.0
    return microbatches / pp


def _operator_time_scale(report: Any, op_dicts: list[dict], strategy: Any | None) -> float:
    if strategy is None:
        return 1.0

    useful_compute_ms = _to_float(getattr(report, "compute_time_ms", 0.0))
    total_op_ms = sum(_to_float(op.get("total_ms", 0.0)) for op in op_dicts)
    if useful_compute_ms > 0 and total_op_ms > 0:
        return useful_compute_ms / total_op_ms

    return _training_step_scale(strategy)


def _is_attention_op(op: dict) -> bool:
    component = str(op.get("component", "") or "").lower()
    component_group = str(op.get("component_group", "") or "").lower()
    if component == "attention" or component_group == "attention":
        return True

    name = str(op.get("name", "") or "").lower()
    kind = str(op.get("kind", "") or "").lower()
    return (
        "attn" in name
        or "attention" in name
        or kind in {"attn_core", "sparse_attn", "hca_attn", "swa_attn"}
        or kind in {"compressor_pool", "indexer_topk", "rope"}
    )


def _is_matmul_op(op: dict) -> bool:
    return str(op.get("kind", "") or "").lower() == "matmul"


def _is_lm_head_op(op: dict) -> bool:
    kind = str(op.get("kind", "") or "").lower()
    name = str(op.get("name", "") or "").lower()
    return kind == "lm_head" or name == "lm_head" or name.endswith(".lm_head")


def _is_mtp_embed_matmul_op(op: dict) -> bool:
    if not _is_matmul_op(op):
        return False

    name = str(op.get("name", "") or "").lower()
    layer_kind = str(op.get("layer_kind", "") or "").lower()
    return (
        "mtp_embed_proj" in name
        or ("embed_proj" in name and "mtp" in name)
        or ("embed_proj" in name and "mtp" in layer_kind)
    )


def _has_any_name_marker(op: dict, markers: tuple[str, ...]) -> bool:
    name = str(op.get("name", "") or "").lower()
    return any(marker in name for marker in markers)


_ATTENTION_MATMUL_NAME_MARKERS = (
    "qkv",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj",
    "kv_b_proj",
    "wq",
    "wk",
    "wv",
    "wkv",
    "o_proj",
    "wo_a",
    "wo_b",
    "comp_wkv",
    "comp_wgate",
    "idx_wq_b",
    "idx_weights",
    "idx_comp_wkv",
    "idx_comp_wgate",
)


def _is_attention_matmul_op(op: dict) -> bool:
    if not _is_matmul_op(op):
        return False
    if _is_attention_op(op):
        return True

    return _has_any_name_marker(op, _ATTENTION_MATMUL_NAME_MARKERS)


def _is_ffn_matmul_op(op: dict) -> bool:
    if not _is_matmul_op(op):
        return False
    if _is_attention_matmul_op(op):
        return False

    component = str(op.get("component", "") or "").lower()
    component_group = str(op.get("component_group", "") or "").lower()
    if component in {"routed_expert", "shared_expert", "ffn", "moe"}:
        return True
    if component_group in {"routed_expert", "shared_expert", "ffn", "moe"}:
        return True

    name = str(op.get("name", "") or "").lower()
    ffn_name_markers = (
        "ffn",
        "mlp",
        "up_proj",
        "gate_proj",
        "down_proj",
        "router",
        "routed_expert",
        "shared_expert",
        "expert",
        "comp_wgate",
        "idx_comp_wgate",
    )
    return any(marker in name for marker in ffn_name_markers)


def _is_dsv4(model: Any) -> bool:
    model_type = str(getattr(model, "model_type", "") or "").lower()
    return model_type == "deepseek_v4" or bool(getattr(model, "use_v4_attn", False))


def _is_dsv32(model: Any) -> bool:
    model_type = str(getattr(model, "model_type", "") or "").lower()
    if model_type in {"deepseek_v3_2", "deepseek-v3-2", "dsv3.2", "dsv32"}:
        return True
    return (
        bool(getattr(model, "use_mla", False))
        and _to_int(getattr(model, "index_topk", 0), 0) > 0
        and not bool(getattr(model, "use_v4_attn", False))
    )


def _row(
    label: str,
    time_ms: float,
    op_count: int,
    step_time_ms: float,
    useful_compute_ms: float,
) -> dict:
    return {
        "label": label,
        "time_ms": time_ms,
        "pct_of_step": _pct(time_ms, step_time_ms),
        "pct_of_useful_compute": _pct(time_ms, useful_compute_ms),
        "op_count": op_count,
    }


def _append_if_present(
    rows: list[dict],
    label: str,
    ops: list[dict],
    step_time_ms: float,
    useful_compute_ms: float,
    time_scale: float,
) -> None:
    if not ops:
        return
    rows.append(
        _row(
            label,
            sum(_to_float(op.get("total_ms", 0.0)) for op in ops) * time_scale,
            len(ops),
            step_time_ms,
            useful_compute_ms,
        )
    )


def classify_op_groups(op: dict) -> list[str]:
    """Return the semantic group tags an op belongs to.

    These groups are the ones that cannot be derived from ``kind`` alone
    (attention vs matmul vs ffn, etc.). They reuse the same predicates that
    ``build_operator_time_stats`` uses so the static summary and the
    interactive report share a single source of truth. An op can belong to
    several groups at once (e.g. a ``q_proj`` matmul is both ``matmul`` and
    ``attention_matmul``).
    """
    groups: list[str] = []
    if _is_matmul_op(op) or _is_lm_head_op(op):
        groups.append("matmul")
    if _is_attention_op(op):
        groups.append("attention")
    if _is_attention_matmul_op(op):
        groups.append("attention_matmul")
    if _is_ffn_matmul_op(op):
        groups.append("ffn")
    if _is_lm_head_op(op):
        groups.append("lm_head")
    if _is_mtp_embed_matmul_op(op):
        groups.append("mtp_embed")

    kind = str(op.get("kind", "") or "").lower()
    if kind == "indexer_topk":
        groups.append("indexer")
    if kind == "attn_core":
        groups.append("sparse_fa")

    return groups


def _has_swa_window(op: dict) -> bool:
    kind = str(op.get("kind", "") or "").lower()
    if kind not in {"sparse_attn", "hca_attn", "swa_attn"}:
        return False
    meta = op.get("meta", {}) or {}
    return _to_float(meta.get("swa_window", 0.0)) > 0


def build_operator_time_stats(
    *,
    model: Any,
    report: Any,
    op_dicts: list[dict],
    strategy: Any | None = None,
) -> list[dict]:
    """Build estimate-report operator time-share rows.

    Percentages are relative to ``report.step_time_ms`` and
    ``report.compute_time_ms``. The input ``op_dicts`` use the same shape
    produced by ``html_exporter._op_to_dict``.
    """
    step_time_ms = _to_float(getattr(report, "step_time_ms", 0.0))
    useful_compute_ms = _to_float(getattr(report, "compute_time_ms", 0.0))
    time_scale = _operator_time_scale(report, op_dicts, strategy)
    rows: list[dict] = []

    _append_if_present(
        rows,
        "Matmul family total",
        [op for op in op_dicts if _is_matmul_op(op) or _is_lm_head_op(op)],
        step_time_ms,
        useful_compute_ms,
        time_scale,
    )
    _append_if_present(
        rows,
        "Attention matmul family",
        [op for op in op_dicts if _is_attention_matmul_op(op)],
        step_time_ms,
        useful_compute_ms,
        time_scale,
    )
    _append_if_present(
        rows,
        "MoE/FFN matmul family",
        [op for op in op_dicts if _is_ffn_matmul_op(op)],
        step_time_ms,
        useful_compute_ms,
        time_scale,
    )
    _append_if_present(
        rows,
        "LM head matmul",
        [op for op in op_dicts if _is_lm_head_op(op)],
        step_time_ms,
        useful_compute_ms,
        time_scale,
    )
    _append_if_present(
        rows,
        "MTP embed matmul",
        [op for op in op_dicts if _is_mtp_embed_matmul_op(op)],
        step_time_ms,
        useful_compute_ms,
        time_scale,
    )

    if _is_dsv4(model):
        csa_ops: list[dict] = []
        hca_ops: list[dict] = []
        for op in op_dicts:
            if not _is_attention_op(op):
                continue
            layer_id = _to_int(op.get("layer_id"), -1)
            if layer_id < 0 or not hasattr(model, "get_layer_cp_type"):
                continue
            cp_type = str(model.get_layer_cp_type(layer_id)).lower()
            if cp_type == "csa":
                csa_ops.append(op)
            elif cp_type == "hca":
                hca_ops.append(op)

        swa_component_ops = [
            op for op in op_dicts
            if _has_swa_window(op)
        ]

        _append_if_present(
            rows, "CSA attention block", csa_ops, step_time_ms, useful_compute_ms, time_scale
        )
        _append_if_present(
            rows, "HCA attention block", hca_ops, step_time_ms, useful_compute_ms, time_scale
        )
        _append_if_present(
            rows, "CSA/HCA/SWA composite attention core", swa_component_ops,
            step_time_ms, useful_compute_ms, time_scale
        )

    if _is_dsv32(model):
        # Per-component matmul breakdown (parallel to V4's per-variant attention rows).
        mla_proj_markers = ("q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj", "o_proj")
        mla_proj_ops = [
            op for op in op_dicts
            if _is_matmul_op(op) and _has_any_name_marker(op, mla_proj_markers)
        ]
        indexer_aux_markers = ("idx_wq_b", "idx_weights", "idx_comp_wkv", "idx_comp_wgate")
        indexer_aux_ops = [
            op for op in op_dicts
            if _is_matmul_op(op) and _has_any_name_marker(op, indexer_aux_markers)
        ]
        router_ops = [
            op for op in op_dicts
            if _is_matmul_op(op)
            and str(op.get("name", "") or "").lower().rsplit(".", 1)[-1] == "router"
        ]
        shared_expert_ops = [
            op for op in op_dicts
            if _is_matmul_op(op)
            and str(op.get("component", "") or "").lower() == "shared_expert"
        ]
        routed_expert_ops = [
            op for op in op_dicts
            if _is_matmul_op(op)
            and "routed_expert_ffn" in str(op.get("name", "") or "").lower()
        ]

        _append_if_present(
            rows, "MLA proj matmul (Q/KV/O)", mla_proj_ops, step_time_ms, useful_compute_ms, time_scale
        )
        _append_if_present(
            rows, "Indexer aux matmul", indexer_aux_ops, step_time_ms, useful_compute_ms, time_scale
        )
        _append_if_present(
            rows, "MoE router matmul", router_ops, step_time_ms, useful_compute_ms, time_scale
        )
        _append_if_present(
            rows, "MoE shared expert matmul", shared_expert_ops, step_time_ms, useful_compute_ms, time_scale
        )
        _append_if_present(
            rows, "MoE routed expert matmul (gmm)", routed_expert_ops, step_time_ms, useful_compute_ms, time_scale
        )

        sparse_fa_ops = [
            op for op in op_dicts
            if str(op.get("kind", "") or "").lower() == "attn_core"
        ]
        indexer_ops = [
            op for op in op_dicts
            if str(op.get("kind", "") or "").lower() == "indexer_topk"
        ]
        dsa_compute_ops = sparse_fa_ops + indexer_ops
        mla_ops = [op for op in op_dicts if _is_attention_op(op)]

        _append_if_present(
            rows, "Sparse FA core (DSA)", sparse_fa_ops, step_time_ms, useful_compute_ms, time_scale
        )
        _append_if_present(
            rows, "Lightning Indexer", indexer_ops, step_time_ms, useful_compute_ms, time_scale
        )
        _append_if_present(
            rows, "DSA attention compute", dsa_compute_ops, step_time_ms, useful_compute_ms, time_scale
        )
        _append_if_present(
            rows, "MLA attention block", mla_ops, step_time_ms, useful_compute_ms, time_scale
        )

    return rows
