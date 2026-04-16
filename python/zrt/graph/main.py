"""Entry point: load model, trace forward, write Excel + JSON + computation graph.

Public API::

    from python.zrt.graph import run_trace, run_trace_phases, build_config_summary, load_model

    # Trace both prefill and decode in one call (recommended)
    output_dir, phase_records = run_trace_phases(
        model_id="deepseek-ai/DeepSeek-V3-0324",
        num_layers=4,
        batch_size=1,
        seq_len=128,
        output_dir="output/graph/DeepSeek-V3-0324",  # optional
    )

    # Trace a single phase
    output_dir, records = run_trace(
        model_id="deepseek-ai/DeepSeek-V3-0324",
        num_layers=4,
        batch_size=1,
        seq_len=128,
        phase="prefill",
    )
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from python.zrt.graph.dispatch import RecordingDispatch, TensorTracker
from python.zrt.graph.excel_writer import ExcelWriter
from python.zrt.graph.graph_builder import build_op_graph, build_fused_op_graph
from python.zrt.graph.graph_exporter import export_all
from python.zrt.graph.model_loader import load_model
from python.zrt.graph.tracker import ModuleTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

logger = logging.getLogger(__name__)

# Backward-compat map for --model v3 / v3.2 shorthand
_MODEL_DIRS = {
    "v3":  "deepseek_v3",
    "v3.2": "deepseek_v3_2",
}

# Normalise legacy "forward" phase name
_PHASE_ALIASES = {"forward": "prefill"}


# ── Layer-type inference ───────────────────────────────────────────────────────

def infer_layer_types(config: Any) -> Dict[str, List[int]]:
    """Infer which transformer layers are dense vs. sparse (MoE) from config.

    Handles three architectures without hard-coding model names:

    * **DeepSeek-V3 / V3.2 style** — config has ``first_k_dense_replace`` and
      (optionally) ``moe_layer_freq``:
      layers ``[0, first_k_dense_replace)`` are dense; the remaining layers are
      MoE when ``layer_idx % moe_layer_freq == 0``, otherwise dense.
    * **Mixtral style** — config has ``num_local_experts`` (but no
      ``first_k_dense_replace``): all layers are treated as MoE.
    * **Standard dense models** — no MoE fields: all layers are dense.

    The layer count is taken from ``config._full_num_hidden_layers`` when
    available (set by :func:`load_model`), falling back to
    ``config.num_hidden_layers``.

    Returns
    -------
    {"dense": [layer_idx, ...], "sparse": [layer_idx, ...]}
        Each list is sorted; together they cover every layer index.
    """
    total: int = (
        getattr(config, "_full_num_hidden_layers", None)
        or getattr(config, "num_hidden_layers", 0)
    )

    first_k_dense: Optional[int] = getattr(config, "first_k_dense_replace", None)
    moe_layer_freq: int = getattr(config, "moe_layer_freq", 1) or 1
    has_local_experts: bool = getattr(config, "num_local_experts", None) is not None
    has_routed_experts: bool = getattr(config, "n_routed_experts", None) is not None

    dense: List[int] = []
    sparse: List[int] = []

    if first_k_dense is not None:
        # DeepSeek-V3 style: first_k_dense dense layers, then a mix depending
        # on moe_layer_freq (layer_idx % moe_layer_freq == 0 → MoE).
        for i in range(total):
            if i < first_k_dense:
                dense.append(i)
            elif i % moe_layer_freq == 0:
                sparse.append(i)
            else:
                dense.append(i)
    elif has_local_experts or has_routed_experts:
        # Mixtral / all-MoE architectures
        sparse = list(range(total))
    else:
        # Plain dense model (LLaMA, Qwen2, Mistral, etc.)
        dense = list(range(total))

    return {"dense": dense, "sparse": sparse}


def auto_target_layers(config: Any) -> List[int]:
    """Return the representative layer indices to trace for this config.

    Selects the **first dense layer** and the **first sparse (MoE) layer**
    (when the model has MoE layers).  For purely dense models returns ``[0]``.
    For all-MoE models returns ``[0]``.

    The returned list is sorted and deduplicated.
    """
    types = infer_layer_types(config)
    result: List[int] = []
    if types["dense"]:
        result.append(types["dense"][0])
    if types["sparse"]:
        result.append(types["sparse"][0])
    return sorted(set(result)) or [0]


# ── Record filtering ───────────────────────────────────────────────────────────

def _filter_records_by_layers(
    records: List[Dict[str, Any]],
    target_layers: List[int],
) -> List[Dict[str, Any]]:
    """Keep only op records belonging to the given layer indices.

    Records that have no layer tag (e.g. embedding lookup, final norm,
    lm_head) are *always* retained because they are model-level ops that
    appear exactly once regardless of which layers are traced.

    ``node_id`` is renumbered sequentially in the filtered list.
    """
    target_set: Set[str] = {str(i) for i in target_layers}
    filtered: List[Dict[str, Any]] = []
    for rec in records:
        layer_str = rec.get("layer", "")
        if layer_str == "" or layer_str in target_set:
            filtered.append(dict(rec))
    for i, rec in enumerate(filtered):
        rec["node_id"] = i
    return filtered


# ── Public helpers ─────────────────────────────────────────────────────────────

def build_config_summary(
    model_id: str,
    config: Any,
    num_hidden_layers: int,
    batch_size: int,
    seq_len: int,
    target_layers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Return a config dict suitable for the Excel Model Config sheet.

    Always includes ``model_id``, ``model_type``, ``hidden_size``,
    ``num_attention_heads``, and ``vocab_size``.  Architecture-specific
    fields (MLA ranks, MoE counts, etc.) are included when present.
    """
    def _get(attr: str) -> Any:
        return getattr(config, attr, None)

    summary: Dict[str, Any] = {
        "model_id": model_id,
        "model_type": _get("model_type") or "unknown",
        "hidden_size": _get("hidden_size"),
        "intermediate_size": _get("intermediate_size"),
        "num_hidden_layers (full)": getattr(
            config, "_full_num_hidden_layers", _get("num_hidden_layers")),
        "num_hidden_layers (traced)": num_hidden_layers,
        "num_attention_heads": _get("num_attention_heads"),
        "num_key_value_heads": _get("num_key_value_heads"),
        "vocab_size": _get("vocab_size"),
        "batch_size": batch_size,
        "seq_len": seq_len,
    }
    if target_layers is not None:
        summary["target_layers"] = str(target_layers)

    # Optional architecture-specific fields
    for field in (
        "moe_intermediate_size",
        "q_lora_rank", "kv_lora_rank",
        "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
        "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
        "first_k_dense_replace",
        "num_local_experts",
        "sliding_window", "head_dim", "rope_theta",
        "index_head_dim", "index_n_heads", "index_topk",
    ):
        val = _get(field)
        if val is not None:
            summary[field] = val

    return {k: v for k, v in summary.items() if v is not None}


def _make_model_slug(model_id: str) -> str:
    """Create a filesystem-safe slug from a model ID."""
    return re.sub(r"[^\w]+", "_", Path(model_id).name).strip("_")


# ── Internal phase helpers ─────────────────────────────────────────────────────

def _trace_phase(
    model: Any,
    config: Any,
    batch_size: int,
    seq_len: int,
    phase: str,
    past_key_values: Any = None,
) -> Tuple[List[Dict[str, Any]], "ModuleTracker", Any]:
    """Run one forward pass for *phase* and return ``(records, tracker, output)``.

    ``tracker.remove()`` is called before returning so forward hooks are
    cleaned up, but the tracker's metadata (``path_to_class``,
    ``path_to_children``) remains available for downstream writers.

    Parameters
    ----------
    phase:
        ``"prefill"`` — full-sequence pass (query_len == seq_len, no KV cache).
        ``"decode"``  — single-token pass (query_len == 1, with KV cache).
    past_key_values:
        KV cache returned by a prior prefill pass.  Should be the
        ``past_key_values`` attribute of the prefill model output.
        Ignored when *phase* is ``"prefill"``.
    """
    phase = _PHASE_ALIASES.get(phase, phase)

    if phase == "decode":
        query_len = 1
        pos_start = seq_len          # next token position after the prefill window
    else:
        query_len = seq_len
        pos_start = 0

    input_ids = torch.randint(0, config.vocab_size, (batch_size, query_len))
    position_ids = (
        torch.arange(pos_start, pos_start + query_len)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    if phase == "decode":
        if past_key_values is not None:
            # KV cache present: the model attends to all past + current token.
            # Mask shape (1, 1, 1, seq_len+1) covers the full context window.
            total_len = seq_len + 1
            mask = torch.zeros(1, 1, 1, total_len)
        else:
            # No KV cache: only the single current token is in K/V memory.
            # Use a (1, 1, 1, 1) mask so shapes are consistent.
            mask = torch.zeros(1, 1, 1, 1)
    else:
        # Standard causal mask for the full prefill window.
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)

    forward_kwargs: Dict[str, Any] = dict(
        input_ids=input_ids,
        attention_mask=mask,
        position_ids=position_ids,
        use_cache=True,
    )
    if past_key_values is not None:
        forward_kwargs["past_key_values"] = past_key_values

    tensor_tracker = TensorTracker()
    tracker = ModuleTracker(model)
    recorder = RecordingDispatch(
        tensor_tracker=tensor_tracker,
        module_tracker=tracker,
        skip_reshapes=True,
    )
    try:
        with recorder, torch.no_grad():
            output = model(**forward_kwargs)
    finally:
        tracker.remove()

    return recorder.records, tracker, output


def _save_phase_outputs(
    records: List[Dict[str, Any]],
    tracker: "ModuleTracker",
    phase: str,
    slug: str,
    output_dir: Path,
    config_summary: Dict[str, Any],
) -> None:
    """Write Excel + JSON + ONNX graph files for one phase."""
    from python.zrt.graph.fusion import FusionEngine

    excel_path = output_dir / f"{slug}_{phase}_ops.xlsx"
    writer = ExcelWriter(tracker)
    writer.write(records, excel_path, config_summary)
    logger.info("Excel saved to %s", excel_path)

    raw_graph = build_op_graph(records)
    fusion_engine = FusionEngine(tracker)
    fused_with_children = fusion_engine.fuse_keep_children(records)
    fused_graph = build_fused_op_graph(fused_with_children, records)

    graph_paths = export_all(
        raw_graph=raw_graph,
        fused_graph=fused_graph,
        raw_records=records,
        fused_records=fused_with_children,
        output_dir=output_dir,
        model_name=slug,
        phase=phase,
    )
    for artifact, path in graph_paths.items():
        logger.info("  %s: %s", artifact, path)


# ── Public API ─────────────────────────────────────────────────────────────────

def run_trace_phases(
    model_id: str,
    num_layers: int = 4,
    batch_size: int = 1,
    seq_len: int = 128,
    output_dir: Optional[Any] = None,
    phases: Tuple[str, ...] = ("prefill", "decode"),
    target_layers: Optional[List[int]] = None,
    auto_layers: bool = True,
) -> Tuple[Path, Dict[str, List[Dict[str, Any]]]]:
    """Load *model_id* once, trace each requested phase, write separate files.

    Prefill and decode phases share the same ``FakeTensorMode`` context so
    the KV cache produced by prefill (fake tensors) can be passed directly
    into the decode forward pass.

    Parameters
    ----------
    phases:
        Ordered sequence of phases to trace.  Each element must be
        ``"prefill"`` or ``"decode"``.  When both are given, prefill runs
        first and its ``past_key_values`` are fed to the decode pass so that
        attention shapes reflect the actual KV-cache layout.
    target_layers:
        Optional explicit list of layer indices to keep in the output.
        For example ``[0, 3]`` retains only ops from layer 0 and layer 3.
        Ops with no layer tag (embedding, lm_head, final norm) are always
        retained.  The model is loaded with ``max(target_layers)+1`` layers
        so all requested indices exist.
    auto_layers:
        When ``True`` and *target_layers* is ``None``, automatically infer
        the first dense and first sparse (MoE) layer indices from the model
        config and use those as *target_layers*.  Ignored when
        *target_layers* is already set.

    Returns
    -------
    (output_dir, {phase: records})
        *output_dir* — ``Path`` to the directory containing all artifacts.
        The dict maps each phase name to its list of op-record dicts.

    Output files (per phase)
    ------------------------
    ``<slug>_prefill_ops.xlsx``       — op table, Excel
    ``<slug>_prefill_raw_graph.json`` — raw aten graph, JSON
    ``<slug>_prefill_raw_graph.onnx`` — raw aten graph, ONNX (Netron-ready)
    ``<slug>_prefill_fused_graph.*``  — fused graph
    ``<slug>_decode_ops.xlsx``        — same set for decode
    …
    """
    # ── Resolve target_layers before loading the model ────────────────────
    if auto_layers and target_layers is None:
        # Quick config-only load to infer layer types (no model weights).
        from python.zrt.graph.model_loader import _load_config
        from python.zrt.graph.patches import apply_compat_patches
        apply_compat_patches()
        cfg_tmp, _ = _load_config(model_id)
        target_layers = auto_target_layers(cfg_tmp)
        logger.info("Auto-selected target layers: %s", target_layers)

    # When specific layers are requested, ensure we load enough layers.
    effective_num_layers = num_layers
    if target_layers:
        min_required = max(target_layers) + 1
        if min_required > effective_num_layers:
            logger.info(
                "Extending num_layers %d → %d to include target layer %d.",
                effective_num_layers, min_required, max(target_layers),
            )
            effective_num_layers = min_required

    logger.info("Loading model %s (%d layers) …", model_id, effective_num_layers)
    model, config, fake_mode = load_model(model_id, num_hidden_layers=effective_num_layers)

    slug = _make_model_slug(model_id)
    if output_dir is None:
        output_dir = Path("output") / "graph" / slug
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_summary = build_config_summary(
        model_id, config, effective_num_layers, batch_size, seq_len,
        target_layers=target_layers,
    )

    all_records: Dict[str, List[Dict[str, Any]]] = {}
    past_key_values: Any = None
    canonical_phases = [_PHASE_ALIASES.get(p, p) for p in phases]

    try:
        for phase in canonical_phases:
            query_len = 1 if phase == "decode" else seq_len
            logger.info(
                "Tracing %s phase (batch=%d, query_len=%d) …",
                phase, batch_size, query_len,
            )
            records, tracker, output = _trace_phase(
                model, config, batch_size, seq_len, phase, past_key_values)
            logger.info("  Captured %d ops.", len(records))

            # Pass KV cache from prefill into the subsequent decode pass.
            if phase == "prefill" and "decode" in canonical_phases:
                past_key_values = getattr(output, "past_key_values", None)
                if past_key_values is None:
                    logger.warning(
                        "Prefill output contains no past_key_values; "
                        "decode pass will run without KV cache "
                        "(attention shapes may differ from a real decode step)")

            # Filter to requested layers (if any).
            if target_layers is not None:
                before = len(records)
                records = _filter_records_by_layers(records, target_layers)
                logger.info(
                    "  Layer filter %s: %d → %d ops.",
                    target_layers, before, len(records),
                )

            _save_phase_outputs(
                records, tracker, phase, slug, output_dir, config_summary)
            all_records[phase] = records

    finally:
        fake_mode.__exit__(None, None, None)

    logger.info("All outputs saved to %s", output_dir)
    return output_dir, all_records


def run_trace(
    model_id: str,
    num_layers: int = 4,
    batch_size: int = 1,
    seq_len: int = 128,
    output_dir: Optional[Any] = None,
    phase: str = "prefill",
    target_layers: Optional[List[int]] = None,
    auto_layers: bool = False,
) -> Tuple[Path, List[Dict[str, Any]]]:
    """Load *model_id*, trace a single phase, write Excel + graph outputs.

    For tracing both prefill and decode in one call prefer
    :func:`run_trace_phases`.

    Parameters
    ----------
    phase:
        ``"prefill"`` (default) — full-sequence forward pass.
        ``"decode"``            — single-token forward pass (no KV cache fed
                                  in; use :func:`run_trace_phases` to get the
                                  correct decode shapes with KV cache).
        ``"forward"``           — legacy alias for ``"prefill"``.
    target_layers:
        Optional list of layer indices to retain in the output.
    auto_layers:
        Auto-infer first dense + first sparse layer when ``target_layers``
        is not supplied.

    Returns
    -------
    (output_dir, records)
    """
    output_dir, phase_records = run_trace_phases(
        model_id=model_id,
        num_layers=num_layers,
        batch_size=batch_size,
        seq_len=seq_len,
        output_dir=output_dir,
        phases=(_PHASE_ALIASES.get(phase, phase),),
        target_layers=target_layers,
        auto_layers=auto_layers,
    )
    canonical = _PHASE_ALIASES.get(phase, phase)
    return output_dir, phase_records[canonical]


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace LLM operator sequences and write Excel + computation graph.")
    parser.add_argument(
        "model_id", nargs="?",
        help="HF Hub model ID or local directory (e.g. deepseek-ai/DeepSeek-V3-0324)")
    parser.add_argument(
        "--model", choices=_MODEL_DIRS.keys(),
        help="Shorthand for local DeepSeek model: v3 or v3.2 (backward compat)")
    parser.add_argument("--layers", type=int, default=4,
                        help="Number of transformer layers to trace (default: 4)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Dummy input batch size (default: 1)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Prefill sequence length (default: 128)")
    parser.add_argument("--output-dir", "-o",
                        help="Output directory (default: output/graph/<model_slug>)")
    parser.add_argument(
        "--phases", nargs="+", default=["prefill", "decode"],
        choices=["prefill", "decode", "forward"],
        metavar="PHASE",
        help="Phases to trace: prefill, decode, or both (default: prefill decode). "
             "'forward' is an alias for 'prefill'.")
    # Legacy single-phase flag kept for backward compat
    parser.add_argument(
        "--phase", default=None,
        help="(legacy) Trace a single phase. Overrides --phases when set.")

    # Layer selection
    _layer_group = parser.add_mutually_exclusive_group()
    _layer_group.add_argument(
        "--target-layers",
        metavar="IDX",
        help="Comma-separated layer indices to trace, e.g. '0,3'.  "
             "The model is loaded with enough layers to reach the highest index.",
    )
    _layer_group.add_argument(
        "--auto-layers",
        action="store_true",
        default=False,
        help="Automatically select the first dense layer and the first sparse "
             "(MoE) layer based on the model config.  Mutually exclusive with "
             "--target-layers.",
    )
    args = parser.parse_args()

    # Resolve model_id: positional takes precedence over --model shorthand
    if args.model_id:
        model_id = args.model_id
    elif args.model:
        model_dir_name = _MODEL_DIRS[args.model]
        model_id = str(
            Path(__file__).parent.parent.parent / "hf_models" / model_dir_name)
    else:
        parser.error("Provide a model_id argument or --model v3/v3.2")

    output_dir = Path(args.output_dir) if args.output_dir else None

    # --phase (legacy) overrides --phases
    if args.phase is not None:
        phases = [args.phase]
    else:
        phases = args.phases

    # Parse --target-layers "0,3" → [0, 3]
    target_layers: Optional[List[int]] = None
    if args.target_layers:
        try:
            target_layers = [int(x.strip()) for x in args.target_layers.split(",")]
        except ValueError:
            parser.error(
                f"--target-layers must be comma-separated integers, "
                f"got: {args.target_layers!r}"
            )

    # When neither --target-layers nor --auto-layers is given, default to
    # auto_layers=True so the CLI behaves the same as the function default.
    effective_auto_layers = args.auto_layers or (target_layers is None)

    run_trace_phases(
        model_id=model_id,
        num_layers=args.layers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        output_dir=output_dir,
        phases=tuple(phases),
        target_layers=target_layers,
        auto_layers=effective_auto_layers,
    )


if __name__ == "__main__":
    main()
