"""Capture the operator sequence of any HuggingFace causal LM via
dispatch-level tracing and write the results to an Excel file,
computation graph (JSON/ONNX), and fusion rules.

Public API::

    from python.zrt.graph import run_trace, load_model, build_config_summary
    from python.zrt.graph import build_op_graph, build_fused_op_graph, export_graphs
"""
from python.zrt.graph.main import (
    main,
    run_trace,
    run_trace_phases,
    build_config_summary,
    infer_layer_types,
    auto_target_layers,
)
from python.zrt.graph.model_loader import load_model
from python.zrt.graph.patches import (
    _is_moe_module,
    patch_moe_for_meta as _patch_moe_for_meta,
)
from python.zrt.graph.classifier import (
    classify_component as _classify_component,
    extract_layer_idx as _extract_layer_idx,
)
from python.zrt.graph.graph_builder import build_op_graph, build_fused_op_graph
from python.zrt.graph.graph_exporter import export_all as export_graphs
from python.zrt.graph.compat import find_local_fallback, _LOCAL_REGISTRY

__all__ = [
    "main",
    "run_trace",
    "run_trace_phases",
    "build_config_summary",
    "infer_layer_types",
    "auto_target_layers",
    "load_model",
    "build_op_graph",
    "build_fused_op_graph",
    "export_graphs",
    "_classify_component",
    "_extract_layer_idx",
    "_is_moe_module",
    "_patch_moe_for_meta",
    "find_local_fallback",
    "_LOCAL_REGISTRY",
]
