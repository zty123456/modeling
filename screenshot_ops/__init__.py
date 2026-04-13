"""Capture the operator sequence of any HuggingFace causal LM via
dispatch-level tracing and write the results to an Excel file.

Public API::

    from screenshot_ops import run_trace, load_model, build_config_summary
    from screenshot_ops import (
        _classify_component, _extract_layer_idx,
        _is_moe_module, _patch_moe_for_meta,
    )
"""
from screenshot_ops.main import main, run_trace, build_config_summary
from screenshot_ops.model_loader import (
    load_model,
    _is_moe_module,
    patch_moe_for_meta as _patch_moe_for_meta,
)
from screenshot_ops.classifier import (
    classify_component as _classify_component,
    extract_layer_idx as _extract_layer_idx,
)

__all__ = [
    "main",
    "run_trace",
    "build_config_summary",
    "load_model",
    "_classify_component",
    "_extract_layer_idx",
    "_is_moe_module",
    "_patch_moe_for_meta",
]
