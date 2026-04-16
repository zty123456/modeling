"""Export computation graphs to JSON and ONNX format.

ONNX export leverages Netron's scope-nesting: each ONNX node's ``name``
uses ``/`` separators encoding the module hierarchy so that Netron renders
collapsible groups matching the nn.Module tree.

The ONNX ``op_type`` is set to a clean short name (e.g. ``mul``, ``mm``,
``softmax``), and rich metadata (module_class, component, layer, shapes)
is stored as ONNX node attributes visible in Netron's sidebar.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import onnx
from onnx import AttributeProto, TensorProto, helper

logger = logging.getLogger(__name__)

# ── Torch dtype → ONNX elem_type ─────────────────────────────────────────────

_DTYPE_MAP = {
    "torch.float16":  TensorProto.FLOAT16,
    "torch.float32":  TensorProto.FLOAT,
    "torch.float64":  TensorProto.DOUBLE,
    "torch.bfloat16": TensorProto.BFLOAT16,
    "torch.int8":     TensorProto.INT8,
    "torch.int16":    TensorProto.INT16,
    "torch.int32":    TensorProto.INT32,
    "torch.int64":    TensorProto.INT64,
    "torch.uint8":    TensorProto.UINT8,
    "torch.bool":     TensorProto.BOOL,
}


def _to_onnx_elem_type(dtype_str: str) -> int:
    return _DTYPE_MAP.get(dtype_str.strip(), TensorProto.FLOAT)


def _parse_shape(shape_str: str) -> List[int]:
    """Parse '[1, 128, 7168]' into [1, 128, 7168]."""
    s = shape_str.strip().strip("[]")
    if not s:
        return []
    try:
        return [int(x.strip()) for x in s.split(",")]
    except ValueError:
        return []


def _split_shape_list(s: str) -> List[str]:
    """Split '[1, 128], [64]' into ['[1, 128]', '[64]']."""
    if not s:
        return []
    result: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in s:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        result.append("".join(current).strip())
    return result


# ── aten op → clean short name ───────────────────────────────────────────────

def _aten_op_short_name(aten_op: str) -> str:
    """Extract the core operation name from an aten op string.

    Examples::

        aten.mul.Tensor       → mul
        aten.mm.default       → mm
        aten._to_copy.default → to_copy
        aten.pow.Tensor_Scalar→ pow
        aten.embedding.default→ embedding
        softmax               → softmax   (passthrough)
    """
    parts = aten_op.split(".")
    if len(parts) >= 2 and parts[0] == "aten":
        name = parts[1]
    elif len(parts) >= 2:
        name = parts[1]
    else:
        name = aten_op
    # Strip leading underscore from internal ops like _to_copy
    return name.lstrip("_") if name.startswith("_") else name


# ── Module path → ONNX scope ────────────────────────────────────────────────

def _module_path_to_scope(module_path: str) -> str:
    """Convert ``model.layers.0.self_attn.q_a_proj`` to
    ``model/layers.0/self_attn/q_a_proj`` for Netron grouping.

    Keeps container+index pairs together (``layers.0``, ``experts.3``)
    as a single scope level for readability.
    """
    if not module_path:
        return ""
    parts = module_path.split(".")
    scope_parts: List[str] = []
    _CONTAINERS = {"layers", "blocks", "h", "layer", "experts"}
    i = 0
    while i < len(parts):
        p = parts[i]
        if p in _CONTAINERS and i + 1 < len(parts):
            try:
                int(parts[i + 1])
                scope_parts.append(f"{p}.{parts[i + 1]}")
                i += 2
                continue
            except ValueError:
                pass
        scope_parts.append(p)
        i += 1
    return "/".join(scope_parts)


# ── JSON export ──────────────────────────────────────────────────────────────

def export_graph_json(G: nx.DiGraph, output_path: Path) -> Path:
    """Export graph as node-link JSON."""
    data = nx.node_link_data(G)

    for node in data.get("nodes", []):
        for k, v in list(node.items()):
            if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                node[k] = str(v)
    for key in ("links", "edges"):
        for edge in data.get(key, []):
            for k, v in list(edge.items()):
                if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    edge[k] = str(v)

    if "links" in data and "edges" not in data:
        data["edges"] = data.pop("links")
    data["metadata"] = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    logger.info("Exported graph JSON to %s", output_path)
    return output_path


# ── ONNX export ──────────────────────────────────────────────────────────────

def _build_onnx_from_records(
    records: List[Dict[str, Any]],
    model_name: str,
    is_fused: bool = False,
) -> onnx.ModelProto:
    """Build an ONNX ModelProto from traced operator records.

    - ``op_type``: clean short name (``mul``, ``mm``, ``softmax``, …)
    - ``name``: ``<module_scope>/<op_short>_<idx>`` for Netron grouping
    - Node attributes: module_class, component, layer, shapes, dtypes
    """
    onnx_nodes: List[onnx.NodeProto] = []
    tensor_info: Dict[str, Dict[str, Any]] = {}
    tid_to_name: Dict[int, str] = {}

    # ── Pass 1: assign tensor names from producers ────────────────────────
    for rec in records:
        idx = rec["node_id"]
        out_ids = rec.get("_output_ids", [])
        out_shapes = _split_shape_list(
            rec.get("fused_output_shapes", rec.get("output_shapes", ""))
            if is_fused else rec.get("output_shapes", ""))
        out_dtypes = (
            rec.get("fused_output_dtypes", rec.get("output_dtypes", ""))
            if is_fused else rec.get("output_dtypes", "")).split(", ")

        for slot, tid in enumerate(out_ids):
            if tid not in tid_to_name:
                tname = f"t{tid}"
                tid_to_name[tid] = tname
                shape = _parse_shape(out_shapes[slot]) if slot < len(out_shapes) else []
                dtype = out_dtypes[slot].strip() if slot < len(out_dtypes) else ""
                tensor_info[tname] = {"shape": shape, "dtype": dtype}

    # Register external-input tensors
    for rec in records:
        in_ids = rec.get("_input_ids", [])
        in_shapes = _split_shape_list(
            rec.get("fused_input_shapes", rec.get("input_shapes", ""))
            if is_fused else rec.get("input_shapes", ""))
        in_dtypes = (
            rec.get("fused_input_dtypes", rec.get("input_dtypes", ""))
            if is_fused else rec.get("input_dtypes", "")).split(", ")

        for slot, tid in enumerate(in_ids):
            if tid not in tid_to_name:
                tname = f"t{tid}"
                tid_to_name[tid] = tname
                shape = _parse_shape(in_shapes[slot]) if slot < len(in_shapes) else []
                dtype = in_dtypes[slot].strip() if slot < len(in_dtypes) else ""
                tensor_info[tname] = {"shape": shape, "dtype": dtype}

    # ── Pass 2: create ONNX nodes ────────────────────────────────────────
    all_produced: set = set()
    all_consumed: set = set()

    for rec in records:
        idx = rec["node_id"]
        module_path = rec.get("module_path", "")
        module_class = rec.get("module_class", "")
        component = rec.get("component", "")
        layer = rec.get("layer", "")

        if is_fused:
            raw_op_type = rec.get("fused_op", rec.get("aten_op", "op"))
            # For fused: use the fused_op as the display name,
            # but clean it up for op_type
            aten_ops_str = rec.get("aten_ops", "")
            # e.g. "self_attn.q_a_proj (Linear)" → "q_a_proj"
            # or "input_layernorm (DeepseekV3RMSNorm)" → "RMSNorm"
            op_display = _fused_op_display_name(raw_op_type, module_class)
        else:
            raw_op_type = rec.get("aten_op", "op")
            aten_ops_str = ""
            op_display = _aten_op_short_name(raw_op_type)

        # Build scope-based node name for Netron grouping
        scope = _module_path_to_scope(module_path)
        if scope:
            node_name = f"{scope}/{op_display}_{idx}"
        else:
            node_name = f"{op_display}_{idx}"

        in_ids = rec.get("_input_ids", [])
        out_ids = rec.get("_output_ids", [])
        input_names = [tid_to_name[tid] for tid in in_ids if tid in tid_to_name]
        output_names = [tid_to_name[tid] for tid in out_ids if tid in tid_to_name]

        for tid in in_ids:
            all_consumed.add(tid)
        for tid in out_ids:
            all_produced.add(tid)

        # Build ONNX node with attributes for Netron sidebar
        node = helper.make_node(
            op_type=op_display,
            inputs=input_names,
            outputs=output_names,
            name=node_name,
        )

        # Add metadata as ONNX attributes (visible in Netron properties)
        if module_path:
            node.attribute.append(
                helper.make_attribute("module_path", module_path))
        if module_class:
            node.attribute.append(
                helper.make_attribute("module_class", module_class))
        if component:
            node.attribute.append(
                helper.make_attribute("component", component))
        if layer:
            node.attribute.append(
                helper.make_attribute("layer", layer))

        # Raw aten op info
        node.attribute.append(
            helper.make_attribute("aten_op", raw_op_type))

        # Fused op extras
        if is_fused:
            if aten_ops_str:
                node.attribute.append(
                    helper.make_attribute("aten_ops", aten_ops_str))
            num_sub = rec.get("num_sub_ops", 0)
            if num_sub:
                node.attribute.append(
                    helper.make_attribute("num_sub_ops", num_sub))
            fl = rec.get("fusion_level", "")
            if fl:
                node.attribute.append(
                    helper.make_attribute("fusion_level", fl))

        # Shape/dtype as attributes too, for quick reference
        in_shapes_str = (rec.get("fused_input_shapes", rec.get("input_shapes", ""))
                         if is_fused else rec.get("input_shapes", ""))
        out_shapes_str = (rec.get("fused_output_shapes", rec.get("output_shapes", ""))
                          if is_fused else rec.get("output_shapes", ""))
        if in_shapes_str:
            node.attribute.append(
                helper.make_attribute("input_shapes", in_shapes_str))
        if out_shapes_str:
            node.attribute.append(
                helper.make_attribute("output_shapes", out_shapes_str))

        onnx_nodes.append(node)

    # ── Graph inputs / outputs / value_info ───────────────────────────────
    external_in_tids = all_consumed - all_produced
    external_out_tids = all_produced - all_consumed

    graph_inputs = []
    graph_outputs = []
    value_infos = []

    for tid, tname in sorted(tid_to_name.items(), key=lambda x: x[0]):
        info = tensor_info.get(tname, {})
        shape = info.get("shape", [])
        dtype_str = info.get("dtype", "")
        elem_type = _to_onnx_elem_type(dtype_str)

        type_proto = helper.make_tensor_type_proto(
            elem_type, shape if shape else None)
        vi = helper.make_value_info(tname, type_proto)

        if tid in external_in_tids:
            graph_inputs.append(vi)
        elif tid in external_out_tids:
            graph_outputs.append(vi)
        else:
            value_infos.append(vi)

    if not graph_inputs:
        graph_inputs.append(helper.make_value_info(
            "dummy_input",
            helper.make_tensor_type_proto(TensorProto.FLOAT, [1])))
    if not graph_outputs:
        graph_outputs.append(helper.make_value_info(
            "dummy_output",
            helper.make_tensor_type_proto(TensorProto.FLOAT, [1])))

    graph = helper.make_graph(
        onnx_nodes,
        name=model_name,
        inputs=graph_inputs,
        outputs=graph_outputs,
        value_info=value_infos,
    )

    model = helper.make_model(graph)
    model.ir_version = 8
    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = "ai.aten"
    opset.version = 1
    model.doc_string = f"Computation graph for {model_name}"

    return model


def _fused_op_display_name(fused_op: str, module_class: str) -> str:
    """Create a clean display name for a fused op.

    Examples::

        "self_attn.q_a_proj (Linear)"           → "q_a_proj"
        "input_layernorm (DeepseekV3RMSNorm)"   → "RMSNorm"
        "self_attn (DeepseekV3Attention)"        → "Attention"
        "mlp (DeepseekV3MLP)"                    → "MLP"
        "model.layers.0"                         → "residual"
        "self_attn"                              → "self_attn"
    """
    # If module_class is present, extract a readable short form
    if module_class:
        # DeepseekV3RMSNorm → RMSNorm, DeepseekV3Attention → Attention
        short_class = module_class
        for prefix in ("DeepseekV3", "DeepseekV2", "Deepseek",
                        "Qwen2", "Qwen", "Llama", "Mistral", "Mixtral"):
            if short_class.startswith(prefix):
                short_class = short_class[len(prefix):]
                break
        if short_class:
            return short_class

    # Extract the last meaningful segment
    # "self_attn.q_a_proj (Linear)" → take part before " ("
    base = fused_op.split(" (")[0] if " (" in fused_op else fused_op
    # "self_attn.q_a_proj" → "q_a_proj"
    if "." in base:
        return base.rsplit(".", 1)[-1]
    return base


def export_onnx_from_records(
    records: List[Dict[str, Any]],
    output_path: Path,
    model_name: str,
    is_fused: bool = False,
) -> Path:
    """Export traced records directly to ONNX."""
    onnx_model = _build_onnx_from_records(records, model_name, is_fused)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, str(output_path))
    logger.info("Exported ONNX to %s (%d nodes)",
                output_path, len(onnx_model.graph.node))
    return output_path


# ── Export all ────────────────────────────────────────────────────────────────

def export_all(
    raw_graph: nx.DiGraph,
    fused_graph: Optional[nx.DiGraph],
    raw_records: List[Dict[str, Any]],
    fused_records: List[Dict[str, Any]],
    output_dir: Path,
    model_name: str,
    phase: str = "forward",
) -> Dict[str, Path]:
    """Export all graph artifacts to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    # Raw op graph
    raw_json_path = output_dir / f"{model_name}_{phase}_raw_graph.json"
    results["raw_graph_json"] = export_graph_json(raw_graph, raw_json_path)

    raw_onnx_path = output_dir / f"{model_name}_{phase}_raw_graph.onnx"
    results["raw_graph_onnx"] = export_onnx_from_records(
        raw_records, raw_onnx_path, f"{model_name}_raw_{phase}")

    # Fused op graph
    if fused_graph is not None and fused_records:
        fused_json_path = output_dir / f"{model_name}_{phase}_fused_graph.json"
        results["fused_graph_json"] = export_graph_json(fused_graph, fused_json_path)

        fused_onnx_path = output_dir / f"{model_name}_{phase}_fused_graph.onnx"
        results["fused_graph_onnx"] = export_onnx_from_records(
            fused_records, fused_onnx_path, f"{model_name}_fused_{phase}",
            is_fused=True)

    return results
