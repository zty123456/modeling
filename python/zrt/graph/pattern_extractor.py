"""Extract aten op sequences from torch._inductor SFDP attention patterns.

Uses ``make_fx`` to trace each pattern function with the pre-defined
example inputs from ``_get_sfdp_patterns()``, producing a complete
aten-level FX graph that includes all implicit ops (``_to_copy``,
``clone``, ``expand``, etc.).

Usage::

    python -m python.zrt.graph.pattern_extractor [--output-dir DIR] [--device cpu]
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.fx.experimental.proxy_tensor import make_fx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_sfdp_patterns(
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Extract aten op sequences from all SFDP attention fusion patterns.

    Returns a list of dicts, each containing:
      - name: pattern name (e.g. ``_sfdp_pattern_7_half_training``)
      - pattern_fn: source function name
      - aten_ops: ordered list of aten op strings
      - aten_ops_dedup: deduplicated (preserving order) op list
      - num_ops: total op count
      - input_shapes: list of input tensor shapes
      - input_dtypes: list of input tensor dtypes
      - scalar_workaround: any scalar kwargs applied
      - trace_mode: ``training`` or ``inference``
    """
    from torch._inductor.fx_passes.fuse_attention import _get_sfdp_patterns

    input_device = torch.device(device)
    results: List[Dict[str, Any]] = []
    seen_op_seqs: Dict[str, str] = {}  # op_seq_key -> first_name (dedup)

    for name, kwargs in _get_sfdp_patterns(input_device):
        search_fn = kwargs["search_fn"]
        example_inputs = kwargs["example_inputs"]
        scalar_workaround = kwargs.get("scalar_workaround", {})

        # Apply scalar workarounds (e.g. dropout_p=0.113377)
        fn = search_fn
        if scalar_workaround:
            fn = functools.partial(fn, **scalar_workaround)

        try:
            gm = make_fx(fn)(*example_inputs)
        except RuntimeError:
            # Patterns like 7-10 contain hardcoded .to(torch.float16) which
            # fails on CPU with float32 inputs.  Retry with half inputs.
            try:
                half_inputs = [
                    inp.half() if isinstance(inp, torch.Tensor) and inp.is_floating_point()
                    else inp
                    for inp in example_inputs
                ]
                gm = make_fx(fn)(*half_inputs)
                example_inputs = half_inputs  # for metadata below
            except Exception as e2:
                logger.warning("Failed to trace %s: %s", name, e2)
                results.append({
                    "name": name,
                    "pattern_fn": getattr(search_fn, "__name__", str(search_fn)),
                    "error": str(e2),
                })
                continue
        except Exception as e:
            logger.warning("Failed to trace %s: %s", name, e)
            results.append({
                "name": name,
                "pattern_fn": getattr(search_fn, "__name__", str(search_fn)),
                "error": str(e),
            })
            continue

        # Extract aten ops from FX graph
        aten_ops = []
        for node in gm.graph.nodes:
            if node.op == "call_function":
                target = str(node.target)
                aten_ops.append(target)

        # Deduplicate while preserving order
        aten_ops_dedup = list(OrderedDict.fromkeys(aten_ops))

        # Check for duplicate op sequences
        op_seq_key = " -> ".join(aten_ops)
        is_dup = op_seq_key in seen_op_seqs
        dup_of = seen_op_seqs.get(op_seq_key, None)
        if not is_dup:
            seen_op_seqs[op_seq_key] = name

        # Input metadata
        input_shapes = []
        input_dtypes = []
        for inp in example_inputs:
            if isinstance(inp, torch.Tensor):
                input_shapes.append(list(inp.shape))
                input_dtypes.append(str(inp.dtype))
            else:
                input_shapes.append(None)
                input_dtypes.append(type(inp).__name__)

        trace_mode = "training" if "training" in name else "inference"

        results.append({
            "name": name,
            "pattern_fn": getattr(search_fn, "__name__", str(search_fn)),
            "trace_mode": trace_mode,
            "aten_ops": aten_ops,
            "aten_ops_dedup": aten_ops_dedup,
            "aten_op_sequence": " -> ".join(aten_ops),
            "num_ops": len(aten_ops),
            "input_shapes": input_shapes,
            "input_dtypes": input_dtypes,
            "scalar_workaround": scalar_workaround if scalar_workaround else None,
            "is_duplicate_seq": is_dup,
            "duplicate_of": dup_of,
        })

    logger.info("Extracted %d patterns (%d unique op sequences)",
                len(results),
                len(seen_op_seqs))
    return results


def export_patterns(
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Path]:
    """Export extracted patterns to JSON and summary text."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # Full JSON
    json_path = output_dir / "sfdp_patterns_full.json"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    paths["full_json"] = json_path
    logger.info("Exported full patterns to %s", json_path)

    # Unique patterns only (skip duplicates)
    unique = [r for r in results if not r.get("is_duplicate_seq", False)]
    unique_json_path = output_dir / "sfdp_patterns_unique.json"
    unique_json_path.write_text(
        json.dumps(unique, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    paths["unique_json"] = unique_json_path
    logger.info("Exported %d unique patterns to %s", len(unique), unique_json_path)

    # Human-readable summary
    summary_path = output_dir / "sfdp_patterns_summary.txt"
    lines = []
    lines.append(f"SFDP Attention Fusion Patterns — {len(results)} total, {len(unique)} unique op sequences\n")
    lines.append("=" * 100 + "\n")
    for r in results:
        if r.get("error"):
            lines.append(f"[ERROR] {r['name']}: {r['error']}\n")
            continue
        dup_mark = "  [DUP]" if r.get("is_duplicate_seq") else ""
        lines.append(f"\n{'─' * 80}")
        lines.append(f"Pattern: {r['name']}{dup_mark}")
        if r.get("duplicate_of"):
            lines.append(f"  (same op seq as: {r['duplicate_of']})")
        lines.append(f"  Function:  {r['pattern_fn']}")
        lines.append(f"  Mode:      {r['trace_mode']}")
        lines.append(f"  Ops ({r['num_ops']}):")
        for i, op in enumerate(r["aten_ops"]):
            lines.append(f"    {i:3d}. {op}")
        lines.append(f"  Inputs:")
        for i, (shape, dtype) in enumerate(zip(r["input_shapes"], r["input_dtypes"])):
            lines.append(f"    [{i}] shape={shape}  dtype={dtype}")
        if r.get("scalar_workaround"):
            lines.append(f"  Scalar workaround: {r['scalar_workaround']}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    paths["summary"] = summary_path
    logger.info("Exported summary to %s", summary_path)

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Extract aten op sequences from SFDP attention patterns.")
    parser.add_argument("--output-dir", "-o", default="output/patterns",
                        help="Output directory (default: output/patterns)")
    parser.add_argument("--device", default="cpu",
                        help="Device for tracing (default: cpu)")
    args = parser.parse_args()

    results = extract_sfdp_patterns(device=args.device)
    paths = export_patterns(results, Path(args.output_dir))

    print(f"\nDone. {len(results)} patterns extracted.")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
