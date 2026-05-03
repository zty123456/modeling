"""Platform-aware operator fusion engine.

Three-phase pipeline
--------------------
Pass 1  (leaf)    – group consecutive records with the same module_path + layer.
Pass 2  (parent)  – merge leaf groups that share a common parent module,
                    subject to platform-specific op-count / child-count limits.
Pass 3  (labels)  – semantic relabelling:
                      a. module_class → SEMANTIC_LABELS lookup
                      b. sub-pattern matching (platform-specific fused ops)
                      c. cross-boundary Add+Norm → AddRMSNorm detection

Transparent-op handling
-----------------------
PATTERN_SKIP ops (shape-only, always-transparent, init) are skipped as
*wildcards* during sub-pattern matching.  They are NOT removed from records;
they remain in ``_children`` for downstream writers.  This means fusion
results are invariant to whether the display layer later chooses to filter
those ops out.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from python.zrt.graph.tracker import ModuleTracker
from python.zrt.graph.fusion_rules import (
    get_semantic_label,
    get_subpatterns,
    get_platform_settings,
    PATTERN_SKIP,
    CONTAINER_SEMANTICS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FusionSpec:
    """A fusion pattern auto-discovered from dispatch tracing."""
    module_class: str
    aten_op_sequence: List[str]
    num_sub_ops: int
    fusion_level: str
    example_module_path: str
    occurrences: int = 1
    fused_input_shapes: str = ""
    fused_input_dtypes: str = ""
    fused_input_sources: str = ""
    fused_output_shapes: str = ""
    fused_output_dtypes: str = ""
    fused_output_sources: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_shape_list(s: str) -> List[str]:
    """Split '[1, 128], [64]' into ['[1, 128]', '[64]']."""
    if not s:
        return []
    result, depth, current = [], 0, []
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


def _strip_layer_prefix(module_path: str) -> str:
    parts = module_path.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                int(parts[i + 1])
                return ".".join(parts[i + 2:]) or module_path
            except ValueError:
                pass
    return module_path


def _parent_path(module_path: str) -> str:
    if "." in module_path:
        return module_path.rsplit(".", 1)[0]
    return ""


def _compute_fused_io(ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    produced_ids: set = set()
    for op in ops:
        for tid in op.get("_output_ids", []):
            produced_ids.add(tid)

    consumed_ids: set = set()
    for op in ops:
        for tid in op.get("_input_ids", []):
            consumed_ids.add(tid)

    external_input_ids  = consumed_ids - produced_ids
    external_output_ids = produced_ids - consumed_ids

    seen_in: set = set()
    fused_input_ids = []
    for op in ops:
        for tid in op.get("_input_ids", []):
            if tid in external_input_ids and tid not in seen_in:
                seen_in.add(tid)
                fused_input_ids.append(tid)

    seen_out: set = set()
    fused_output_ids = []
    for op in reversed(ops):
        for tid in op.get("_output_ids", []):
            if tid in external_output_ids and tid not in seen_out:
                seen_out.add(tid)
                fused_output_ids.insert(0, tid)

    input_shapes, input_dtypes, input_sources = [], [], []
    for tid in fused_input_ids:
        for op in ops:
            if tid in op.get("_input_ids", []):
                idx    = op["_input_ids"].index(tid)
                shapes = _split_shape_list(op["input_shapes"])
                dtypes = op["input_dtypes"].split(", ")
                input_shapes.append(shapes[idx]  if idx < len(shapes)  else "?")
                input_dtypes.append(dtypes[idx]  if idx < len(dtypes)  else "?")
                input_sources.append(f"{op['aten_op']} (in[{idx}])")
                break

    output_shapes, output_dtypes, output_sources = [], [], []
    for tid in fused_output_ids:
        for op in reversed(ops):
            if tid in op.get("_output_ids", []):
                idx    = op["_output_ids"].index(tid)
                shapes = _split_shape_list(op["output_shapes"])
                dtypes = op["output_dtypes"].split(", ")
                output_shapes.append(shapes[idx]  if idx < len(shapes)  else "?")
                output_dtypes.append(dtypes[idx]  if idx < len(dtypes)  else "?")
                output_sources.append(f"{op['aten_op']} (out[{idx}])")
                break

    return {
        "fused_input_shapes":  ", ".join(input_shapes),
        "fused_input_dtypes":  ", ".join(input_dtypes),
        "fused_input_sources": " | ".join(input_sources),
        "fused_output_shapes":  ", ".join(output_shapes),
        "fused_output_dtypes":  ", ".join(output_dtypes),
        "fused_output_sources": " | ".join(output_sources),
        # Tensor-ID lists for ONNX edge building (external ports only)
        "_input_ids":  fused_input_ids,
        "_output_ids": fused_output_ids,
    }


def _make_fused_entry(
    ops: List[Dict[str, Any]],
    tracker: ModuleTracker,
    fusion_level: str = "leaf",
) -> Dict[str, Any]:
    """Build one fused-op record from a list of raw op records.

    The ``fused_op`` display label is initially set from the module class name
    (with optional semantic translation).  Pass-3 sub-pattern matching may
    later refine the label to a more specific platform op name.
    """
    first, last = ops[0], ops[-1]
    path         = first["module_path"]
    module_class = tracker.path_to_class.get(path, first.get("module_class", ""))

    aten_ops = list(dict.fromkeys(r["aten_op"] for r in ops))

    short_path    = _strip_layer_prefix(path) if path else ""

    # Path-based override: lm_head is always "lm_head" regardless of class
    # name.  Some models use a weight-tied nn.Embedding for the output
    # projection, whose class would otherwise match the "embedding" semantic.
    _path_lower = path.lower()
    if _path_lower == "lm_head" or _path_lower.endswith(".lm_head"):
        semantic = "lm_head"
    else:
        semantic = get_semantic_label(module_class) if module_class else None

    # effective_ops: exclude shape/transparent/init ops for display purposes.
    effective_ops = [r["aten_op"] for r in ops if r["aten_op"] not in PATTERN_SKIP]

    # Build an aten-op–based fallback label from the first effective op.
    # For container modules (attn/mlp/moe) this becomes the actual display label
    # until Pass-3a subpattern matching replaces it with a specific kernel name.
    ref_op = effective_ops[0] if effective_ops else first["aten_op"]
    fn_parts   = ref_op.split(".")
    aten_label = fn_parts[1] if len(fn_parts) >= 2 else ref_op

    if semantic and semantic not in CONTAINER_SEMANTICS:
        # Terminal semantics (rms_norm, rope, embedding, lm_head, …) are
        # unambiguous even for single-op groups — always use them.
        label = semantic
    else:
        # Container semantics (attn, mlp, moe_*) or no semantic at all:
        # show the actual first-effective-op name.  Pass-3a will upgrade this
        # to a specific kernel label (flash_attn, sdpa, gated_mlp, …) if the
        # op sequence matches a platform subpattern.
        label = aten_label

    io = _compute_fused_io(ops)

    return {
        "fused_op":      label,
        "_semantic":     semantic,       # may be None; used by pass-3 checks
        "module_path":   path,
        "module_class":  module_class,
        "fusion_level":  fusion_level,
        "aten_ops":      " \u2192 ".join(aten_ops),
        "num_sub_ops":   len(ops),
        "layer":         first["layer"],
        "input_shapes":  first["input_shapes"],
        "input_dtypes":  first["input_dtypes"],
        "output_shapes": last["output_shapes"],
        "output_dtypes": last["output_dtypes"],
        "component":     first.get("component", ""),
        **io,
        "_children":     ops,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pass-3: semantic relabelling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _group_aten_ops(group: Dict[str, Any]) -> List[str]:
    """Return the aten_op strings for all children in *group*."""
    return [r["aten_op"] for r in group.get("_children", [])]


def _apply_subpatterns(
    groups: List[Dict[str, Any]],
    platform: str,
) -> List[Dict[str, Any]]:
    """Relabel groups whose op sequence matches a platform-specific pattern.

    Pattern matching skips PATTERN_SKIP ops (shape / transparent / init), so
    the result is invariant to display-layer filter changes.
    """
    patterns = get_subpatterns(platform)
    if not patterns:
        return groups

    result = []
    for g in groups:
        mc = g.get("module_class", "")
        matched = False
        for sp in patterns:                  # already sorted by priority desc
            if sp.matches_class(mc):
                op_names = _group_aten_ops(g)
                if sp.matches_ops(op_names):
                    g = dict(g)
                    g["fused_op"] = sp.name
                    g["_semantic"] = sp.name
                    matched = True
                    break
        result.append(g)
    return result


def _is_add_norm_pair(
    g_prev: Dict[str, Any],
    g_norm: Dict[str, Any],
) -> bool:
    """Return True when *g_prev* ends with a residual-add and *g_norm* is a norm.

    Conditions:
      1. g_norm has a norm semantic label (rms_norm or layer_norm).
      2. g_prev's effective ops contain an ``add`` op.
      3. g_prev is the direct parent of g_norm in the module hierarchy.
    """
    norm_label = g_norm.get("_semantic", "")
    if norm_label not in ("rms_norm", "layer_norm"):
        return False
    # Check that g_prev contains a residual add
    prev_ops = _group_aten_ops(g_prev)
    effective = [op for op in prev_ops if op not in PATTERN_SKIP]
    if not any("add" in op for op in effective):
        return False
    # g_prev must be the module-hierarchy parent of g_norm
    norm_parent = _parent_path(g_norm.get("module_path", ""))
    if not norm_parent or norm_parent != g_prev.get("module_path", ""):
        return False
    return True


def _merge_add_norm(
    g_add: Dict[str, Any],
    g_norm: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge a residual-add group and a norm group into an Add+Norm fused op."""
    all_ops  = g_add["_children"] + g_norm["_children"]
    norm_sem = g_norm.get("_semantic", "rms_norm")
    label    = "add_rms_norm" if norm_sem == "rms_norm" else "add_layer_norm"
    io       = _compute_fused_io(all_ops)
    aten_ops = list(dict.fromkeys(r["aten_op"] for r in all_ops))
    return {
        "fused_op":      label,
        "_semantic":     label,
        "module_path":   g_add["module_path"],
        "module_class":  g_add.get("module_class", ""),
        "fusion_level":  "cross_boundary",
        "aten_ops":      " \u2192 ".join(aten_ops),
        "num_sub_ops":   len(all_ops),
        "layer":         g_add["layer"],
        "input_shapes":  g_add["input_shapes"],
        "input_dtypes":  g_add["input_dtypes"],
        "output_shapes": g_norm["output_shapes"],
        "output_dtypes": g_norm["output_dtypes"],
        "component":     g_add.get("component", ""),
        **io,
        "_children":     all_ops,
    }


def _detect_add_norm(
    groups: List[Dict[str, Any]],
    enabled: bool,
) -> List[Dict[str, Any]]:
    """Scan for adjacent (add-group, norm-group) pairs and merge them.

    Only runs when *enabled* is True (set by platform settings).
    """
    if not enabled:
        return groups

    result: List[Dict[str, Any]] = []
    i = 0
    while i < len(groups):
        if i + 1 < len(groups) and _is_add_norm_pair(groups[i], groups[i + 1]):
            result.append(_merge_add_norm(groups[i], groups[i + 1]))
            i += 2
        else:
            result.append(groups[i])
            i += 1
    return result


def _make_individual_entry(op: Dict[str, Any]) -> Dict[str, Any]:
    """Build a singleton fused-op record from one raw dispatch record."""
    fn_parts   = op["aten_op"].split(".")
    label      = fn_parts[1] if len(fn_parts) >= 2 else op["aten_op"]
    io         = _compute_fused_io([op])
    return {
        "fused_op":      label,
        "_semantic":     None,
        "module_path":   op["module_path"],
        "module_class":  op.get("module_class", ""),
        "fusion_level":  "unfused",
        "aten_ops":      op["aten_op"],
        "num_sub_ops":   1,
        "layer":         op["layer"],
        "input_shapes":  op["input_shapes"],
        "input_dtypes":  op["input_dtypes"],
        "output_shapes": op["output_shapes"],
        "output_dtypes": op["output_dtypes"],
        "component":     op.get("component", ""),
        **io,
        "_children":     [op],
    }


def _expand_unfused_containers(
    groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Pass 4: expand multi-op container groups that were not matched by any
    hardware subpattern back into one row per constituent op.

    After Pass-3a, a group's ``_semantic`` is updated to the matched pattern
    name (e.g. ``"npu_fusion_attention"``), which is NOT in CONTAINER_SEMANTICS.
    Groups that *still* carry a CONTAINER_SEMANTICS ``_semantic`` were not
    recognised as any specific kernel, so the individual ops inside them are
    not truly fused and must be visible in the output graph.
    """
    result: List[Dict[str, Any]] = []
    for g in groups:
        if (g.get("_semantic") in CONTAINER_SEMANTICS
                and g.get("num_sub_ops", 1) > 1):
            for child in g.get("_children", []):
                result.append(_make_individual_entry(child))
        else:
            result.append(g)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FusionEngine
# ─────────────────────────────────────────────────────────────────────────────

class FusionEngine:
    """Three-pass automatic fusion engine.

    Parameters
    ----------
    tracker:
        ``ModuleTracker`` whose ``path_to_class`` / ``path_to_children`` maps
        were populated during the forward pass.
    platform:
        One of ``"cuda"``, ``"ascend_npu"``, ``"cpu"``, ``"generic"``.
        Controls parent-merge thresholds, sub-pattern library, and whether
        Add+Norm cross-boundary fusion is enabled.
        Default ``"generic"`` reproduces the original two-pass behaviour.
    debug:
        If True, log detailed fusion decisions (pass-level + per-group) at
        DEBUG level.  Useful for ``--fusion-debug`` CLI flag.
    """

    def __init__(self, tracker: ModuleTracker, platform: str = "generic",
                 debug: bool = False):
        self._tracker  = tracker
        self._platform = platform
        self._debug    = debug
        cfg = get_platform_settings(platform)
        self._max_parent_ops  = cfg["max_parent_ops"]
        self._max_children    = cfg["max_children"]
        self._add_norm_fusion = cfg["add_norm_fusion"]

    # ── Public interface ──────────────────────────────────────────────────────

    def fuse(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the full three-pass pipeline; strip ``_children`` from output."""
        groups = self._run_all_passes(records)
        for i, g in enumerate(groups):
            g["node_id"] = i
            g.pop("_children", None)
            g.pop("_semantic", None)
        return groups

    def fuse_keep_children(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Same as ``fuse`` but preserve ``_children`` for graph building."""
        groups = self._run_all_passes(records)
        for i, g in enumerate(groups):
            g["node_id"] = i
            g.pop("_semantic", None)
        return groups

    def extract_specs(
        self, fused: List[Dict[str, Any]]
    ) -> List[FusionSpec]:
        specs_by_key: Dict[Tuple[str, str], FusionSpec] = {}
        for g in fused:
            if g["num_sub_ops"] <= 1:
                continue
            key = (g["module_class"], g["fusion_level"])
            if key in specs_by_key:
                specs_by_key[key].occurrences += 1
            else:
                specs_by_key[key] = FusionSpec(
                    module_class=g["module_class"],
                    aten_op_sequence=g["aten_ops"].split(" \u2192 "),
                    num_sub_ops=g["num_sub_ops"],
                    fusion_level=g["fusion_level"],
                    example_module_path=g["module_path"],
                    occurrences=1,
                    fused_input_shapes=g.get("fused_input_shapes",  ""),
                    fused_input_dtypes=g.get("fused_input_dtypes",  ""),
                    fused_input_sources=g.get("fused_input_sources", ""),
                    fused_output_shapes=g.get("fused_output_shapes",  ""),
                    fused_output_dtypes=g.get("fused_output_dtypes",  ""),
                    fused_output_sources=g.get("fused_output_sources", ""),
                )
        return sorted(specs_by_key.values(), key=lambda s: -s.occurrences)

    # ── Internal passes ───────────────────────────────────────────────────────

    def _run_all_passes(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run the full 5-pass pipeline with error boundaries.

        Each pass is wrapped in try/except — a failure in one pass leaves the
        groups from the previous pass intact and logs a warning so the pipeline
        never silently degrades.
        """
        if self._debug:
            logger.info("[fusion-debug] Platform: %s | Input records: %d",
                        self._platform, len(records))

        # Pass 1: Leaf grouping
        try:
            groups = self._pass1_leaf(records)
            if self._debug:
                logger.info("[fusion-debug] Pass 1 (leaf): %d → %d groups",
                            len(records), len(groups))
        except Exception as e:
            logger.warning(
                "Fusion Pass 1 (leaf) failed: %s.  Returning raw records "
                "without fusion grouping.", e, exc_info=True)
            # Fallback: each record becomes its own group
            groups = [_make_fused_entry(
                [r], self._tracker, "leaf_fallback") for r in records]
            return groups

        # Pass 2: Parent merge
        try:
            n_before = len(groups)
            groups = self._pass2_parent(groups)
            if self._debug:
                logger.info("[fusion-debug] Pass 2 (parent): %d → %d groups",
                            n_before, len(groups))
        except Exception as e:
            logger.warning(
                "Fusion Pass 2 (parent) failed: %s.  Keeping leaf groups.", e,
                exc_info=True)

        # Pass 3a: Platform sub-pattern relabelling
        try:
            groups = _apply_subpatterns(groups, self._platform)
            if self._debug:
                sub_matched = sum(
                    1 for g in groups
                    if g.get("fused_op", "") not in (g.get("_semantic", ""),))
                logger.info(
                    "[fusion-debug] Pass 3a (sub-patterns): %d groups, "
                    "%d sub-pattern matches", len(groups), sub_matched)
        except Exception as e:
            logger.warning(
                "Fusion Pass 3a (sub-patterns) failed: %s.  Keeping "
                "pre-pattern labels.", e, exc_info=True)

        # Pass 3b: Cross-boundary Add+Norm → AddRMSNorm
        try:
            groups = _detect_add_norm(groups, self._add_norm_fusion)
        except Exception as e:
            logger.warning(
                "Fusion Pass 3b (add_norm) failed: %s.  Continuing.", e,
                exc_info=True)

        # Pass 4: expand unfused container groups back to individual op rows.
        try:
            n_before = len(groups)
            groups = _expand_unfused_containers(groups)
            if self._debug:
                logger.info("[fusion-debug] Pass 4 (expand): %d → %d groups",
                            n_before, len(groups))
        except Exception as e:
            logger.warning(
                "Fusion Pass 4 (expand) failed: %s.  Keeping merged groups.",
                e, exc_info=True)

        if self._debug:
            logger.info("[fusion-debug] Final: %d fused operators", len(groups))

        return groups

    def _pass1_leaf(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Group consecutive records with the same module_path + layer."""
        if not records:
            return []
        groups       = []
        current_group = [records[0]]
        for rec in records[1:]:
            same_path  = rec["module_path"] == current_group[0]["module_path"]
            same_layer = rec["layer"]       == current_group[0]["layer"]
            has_path   = rec["module_path"] != ""
            if same_path and same_layer and has_path:
                current_group.append(rec)
            else:
                groups.append(
                    _make_fused_entry(current_group, self._tracker, "leaf"))
                current_group = [rec]
        groups.append(_make_fused_entry(current_group, self._tracker, "leaf"))
        return groups

    def _pass2_parent(
        self, groups: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge leaf groups that share a fusible common parent module."""
        if not groups:
            return []

        parent_child_count: Dict[str, set] = defaultdict(set)
        parent_total_ops:   Dict[str, int]  = defaultdict(int)
        for g in groups:
            p = _parent_path(g["module_path"])
            if p:
                parent_child_count[p].add(g["module_path"])
                parent_total_ops[p] += g["num_sub_ops"]

        def _is_fusible_parent(parent: str) -> bool:
            if parent not in self._tracker.path_to_class:
                return False
            children = self._tracker.path_to_children.get(parent, [])
            if not children:
                return False
            if len(parent_child_count.get(parent, set())) > self._max_children:
                return False
            if parent_total_ops.get(parent, 0) > self._max_parent_ops:
                return False
            # Rule 1: root-level modules (scope has no parent) are structural
            # wrappers (the model itself), never computational kernels.
            if not _parent_path(parent):
                return False
            # Rule 2: if any child scope is itself a container with
            # sub-children, the parent is a structural wrapper
            # (e.g. DeepseekV3DecoderLayer wraps self_attn + mlp + norms),
            # not a fusible kernel.
            if any(
                child in self._tracker.path_to_children
                for child in children
            ):
                return False
            return True

        result: List[Dict[str, Any]] = []
        i = 0
        while i < len(groups):
            parent = _parent_path(groups[i]["module_path"])

            if parent and _is_fusible_parent(parent):
                j         = i + 1
                total_ops = groups[i]["num_sub_ops"]
                while j < len(groups):
                    g        = groups[j]
                    g_parent = _parent_path(g["module_path"])
                    if ((g_parent == parent or g["module_path"] == parent)
                            and g["layer"] == groups[i]["layer"]):
                        total_ops += g["num_sub_ops"]
                        if total_ops > self._max_parent_ops:
                            break
                        j += 1
                    else:
                        break

                if j > i + 1:
                    merged_ops   = []
                    for g in groups[i:j]:
                        merged_ops.extend(g["_children"])
                    parent_class = self._tracker.path_to_class[parent]
                    short        = _strip_layer_prefix(parent)
                    aten_ops     = list(dict.fromkeys(
                        r["aten_op"] for r in merged_ops))
                    io           = _compute_fused_io(merged_ops)
                    semantic     = get_semantic_label(parent_class)
                    # Container semantics (attn/mlp/moe) stay hidden until a
                    # subpattern match in Pass-3a upgrades to a kernel name.
                    if semantic and semantic not in CONTAINER_SEMANTICS:
                        label = semantic
                    else:
                        label = f"{short} ({parent_class})" if parent_class else short
                    result.append({
                        "fused_op":      label,
                        "_semantic":     semantic,
                        "module_path":   parent,
                        "module_class":  parent_class,
                        "fusion_level":  "parent",
                        "aten_ops":      " \u2192 ".join(aten_ops),
                        "num_sub_ops":   len(merged_ops),
                        "layer":         groups[i]["layer"],
                        "input_shapes":  merged_ops[0]["input_shapes"],
                        "input_dtypes":  merged_ops[0]["input_dtypes"],
                        "output_shapes": merged_ops[-1]["output_shapes"],
                        "output_dtypes": merged_ops[-1]["output_dtypes"],
                        "component":     merged_ops[0].get("component", groups[i].get("component", "")),
                        **io,
                        "_children":     merged_ops,
                    })
                    i = j
                    continue

            result.append(groups[i])
            i += 1

        return result
