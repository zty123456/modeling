"""Capture the operator sequence of any HuggingFace causal LM via
dispatch-level tracing and write the results to an Excel file,
computation graph (JSON/ONNX), and fusion rules.

Public API::

    from python.zrt.graph import run_trace, load_model, build_config_summary
    from python.zrt.graph import build_op_graph, build_fused_op_graph, export_graphs

Imports from submodules that require torch (main, model_loader, patches, classifier,
graph_builder, graph_exporter, compat) are lazy — they only load when accessed.

Fusion symbols (ALWAYS_TRANSPARENT, SHAPE_OPS, etc.) are lazy-re-exported from
python.zrt.transform.fusion.rules.  The old graph/fusion.py and
graph/fusion_rules.py shim modules have been removed.
"""

from python.zrt.ir.graph import OpGraph
from python.zrt.ir.adapter import records_to_opgraph, fused_records_to_opgraph

_MODULE_MAP = {
    # ── Graph builder / tracer (require torch) ──────────────────────────────
    "main": "python.zrt.graph.main",
    "run_trace": "python.zrt.graph.main",
    "run_trace_phases": "python.zrt.graph.main",
    "TraceResult": "python.zrt.graph.main",
    "TracePhaseResult": "python.zrt.graph.main",
    "build_config_summary": "python.zrt.graph.main",
    "infer_layer_types": "python.zrt.graph.main",
    "auto_target_layers": "python.zrt.graph.main",
    "load_model": "python.zrt.graph.model_loader",
    "_is_moe_module": "python.zrt.graph.patches",
    "_patch_moe_for_meta": "python.zrt.graph.patches",
    "_classify_component": "python.zrt.graph.classifier",
    "_extract_layer_idx": "python.zrt.graph.classifier",
    "build_op_graph": "python.zrt.graph.graph_builder",
    "build_fused_op_graph": "python.zrt.graph.graph_builder",
    "export_graphs": "python.zrt.report.onnx_exporter",
    "find_local_fallback": "python.zrt.graph.compat",
    "_LOCAL_REGISTRY": "python.zrt.graph.compat",
    # ── Fusion rules (moved to transform/fusion/rules.py) ───────────────────
    "ALWAYS_TRANSPARENT": "python.zrt.transform.fusion.rules",
    "SHAPE_OPS": "python.zrt.transform.fusion.rules",
    "INIT_OPS": "python.zrt.transform.fusion.rules",
    "LIFT_OPS": "python.zrt.transform.fusion.rules",
    "POTENTIAL_COPY_OPS": "python.zrt.transform.fusion.rules",
    "PATTERN_SKIP": "python.zrt.transform.fusion.rules",
    "SEMANTIC_LABELS": "python.zrt.transform.fusion.rules",
    "PLATFORM_SUBPATTERNS": "python.zrt.transform.fusion.rules",
    "PLATFORM_SETTINGS": "python.zrt.transform.fusion.rules",
    "CONTAINER_SEMANTICS": "python.zrt.transform.fusion.rules",
    "SubPattern": "python.zrt.transform.fusion.rules",
    "get_semantic_label": "python.zrt.transform.fusion.rules",
    "get_subpatterns": "python.zrt.transform.fusion.rules",
    "get_platform_settings": "python.zrt.transform.fusion.rules",
    "match_subsequence": "python.zrt.transform.fusion.rules",
}

_CACHE: dict = {}


def __getattr__(name: str):
    if name in _MODULE_MAP:
        mod_path = _MODULE_MAP[name]
        if mod_path not in _CACHE:
            import importlib
            _CACHE[mod_path] = importlib.import_module(mod_path)
        return getattr(_CACHE[mod_path], name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "main",
    "run_trace",
    "run_trace_phases",
    "TraceResult",
    "TracePhaseResult",
    "build_config_summary",
    "infer_layer_types",
    "auto_target_layers",
    "load_model",
    "OpGraph",
    "records_to_opgraph",
    "fused_records_to_opgraph",
    "build_op_graph",
    "build_fused_op_graph",
    "export_graphs",
    "_classify_component",
    "_extract_layer_idx",
    "_is_moe_module",
    "_patch_moe_for_meta",
    "find_local_fallback",
    "_LOCAL_REGISTRY",
    "ALWAYS_TRANSPARENT",
    "SHAPE_OPS",
    "INIT_OPS",
    "LIFT_OPS",
    "POTENTIAL_COPY_OPS",
    "PATTERN_SKIP",
    "SEMANTIC_LABELS",
    "PLATFORM_SUBPATTERNS",
    "PLATFORM_SETTINGS",
    "get_semantic_label",
    "get_subpatterns",
    "get_platform_settings",
    "match_subsequence",
    "SubPattern",
]
