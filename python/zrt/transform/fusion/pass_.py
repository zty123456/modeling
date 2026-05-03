"""FusionPass: apply module-scope fusion rules directly to OpGraph IR.

Three-pass algorithm (mirrors FusionEngine but operates on OpGraph nodes):

  Pass 1 (leaf)   — group consecutive same-scope+layer compute nodes.
  Pass 2 (parent) — merge consecutive leaf groups that share a fusible parent.
  Pass 3 (label)  — assign semantic label via module_class → SEMANTIC_LABELS
                    and platform-specific sub-pattern matching.

The scope→class mapping is rebuilt from OpNode.scope / OpNode.module_class,
so no ModuleTracker is required at transform time.

Communication nodes always break groups (they are left as-is).
Nodes with an empty scope are always single-node groups.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import TYPE_CHECKING

from python.zrt.transform.base import GraphPass
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import TensorMeta, DType

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


# ── scope helpers ─────────────────────────────────────────────────────────────

def _parent(scope: str) -> str:
    return scope.rsplit(".", 1)[0] if "." in scope else ""


def _build_scope_maps(
    graph: "OpGraph",
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Rebuild path_to_class and path_to_children from graph nodes."""
    path_to_class: dict[str, str] = {}
    path_to_children: dict[str, set[str]] = defaultdict(set)
    for node in graph.nodes.values():
        if node.scope and node.module_class:
            path_to_class[node.scope] = node.module_class
        if node.scope and "." in node.scope:
            path_to_children[_parent(node.scope)].add(node.scope)
    return path_to_class, dict(path_to_children)


# ── I/O extraction ────────────────────────────────────────────────────────────

def _external_io(
    graph: "OpGraph",
    group_ids: set[str],
) -> tuple[list[TensorMeta], list[TensorMeta]]:
    """Return (ext_inputs, ext_outputs) TensorMeta for a node group.

    ext_inputs  — tensors flowing INTO the group from outside.
    ext_outputs — tensors flowing OUT OF the group to outside.
    De-duplicated by (tensor_id, slot) when tensor_id is available.
    """
    seen_in:  set = set()
    seen_out: set = set()
    inputs:  list[TensorMeta] = []
    outputs: list[TensorMeta] = []

    for e in graph.edges:
        if e.src not in group_ids and e.dst in group_ids:
            key = (e.tensor_id, e.dst_idx)
            if key not in seen_in:
                seen_in.add(key)
                if e.tensor is not None:
                    inputs.append(e.tensor)

        if e.src in group_ids and e.dst not in group_ids:
            key = (e.tensor_id, e.src_idx)
            if key not in seen_out:
                seen_out.add(key)
                if e.tensor is not None:
                    outputs.append(e.tensor)

    return inputs, outputs


# ── Pass 1 ────────────────────────────────────────────────────────────────────

def _pass1_leaf(topo: list[OpNode]) -> list[list[OpNode]]:
    """Group consecutive compute+memory nodes with identical scope+layer+phase.

    Phase (fwd/bwd in stitched train graphs) is part of the group key so that
    fusion never spans the forward/backward boundary — those nodes share scope
    but represent distinct stages and must remain separate for FLOPs / phase
    accounting downstream.
    """
    groups: list[list[OpNode]] = []
    current: list[OpNode] = []

    def _phase(node: OpNode) -> str:
        return node.annotations.get("phase", "")

    for node in topo:
        # comm nodes and scopeless nodes are always standalone group-breakers
        if node.category == "communication" or not node.scope:
            if current:
                groups.append(current)
                current = []
            groups.append([node])
            continue

        if current:
            first = current[0]
            if (
                node.scope == first.scope
                and node.layer == first.layer
                and _phase(node) == _phase(first)
            ):
                current.append(node)
                continue
            groups.append(current)
            current = []

        current = [node]

    if current:
        groups.append(current)
    return groups


# ── Pass 2 ────────────────────────────────────────────────────────────────────

def _pass2_parent(
    leaf_groups: list[list[OpNode]],
    path_to_class: dict[str, str],
    path_to_children: dict[str, set[str]],
    max_parent_ops: int,
    max_children: int,
) -> list[list[OpNode]]:
    """Merge consecutive leaf groups that share a fusible common parent scope."""
    if not leaf_groups:
        return []

    # Build per-parent stats across all leaf groups
    parent_child_scopes: dict[str, set[str]] = defaultdict(set)
    parent_total_ops:    dict[str, int]       = defaultdict(int)
    for g in leaf_groups:
        scope = g[0].scope
        if not scope:
            continue
        p = _parent(scope)
        if p:
            parent_child_scopes[p].add(scope)
            parent_total_ops[p] += len(g)

    def _is_fusible(p: str) -> bool:
        if p not in path_to_class:
            return False
        if p not in path_to_children:
            return False
        if len(parent_child_scopes.get(p, set())) > max_children:
            return False
        if parent_total_ops.get(p, 0) > max_parent_ops:
            return False
        # Rule 1: root-level modules (scope has no parent) are structural
        # wrappers, never computational kernels.
        if not _parent(p):
            return False
        # Rule 2: if any child scope has sub-children, the parent is a
        # structural wrapper, not a fusible kernel.
        children = path_to_children.get(p, set())
        if any(child in path_to_children for child in children):
            return False
        return True

    def _phase(g: list[OpNode]) -> str:
        return g[0].annotations.get("phase", "") if g else ""

    result: list[list[OpNode]] = []
    i = 0
    while i < len(leaf_groups):
        g     = leaf_groups[i]
        scope = g[0].scope if g else ""
        p     = _parent(scope) if scope else ""
        layer = g[0].layer if g else ""
        phase = _phase(g)

        # Comm/scopeless groups are never merged upward
        if not scope or g[0].category == "communication":
            result.append(g)
            i += 1
            continue

        if p and _is_fusible(p):
            j         = i + 1
            total_ops = len(g)
            while j < len(leaf_groups):
                ng     = leaf_groups[j]
                nscope = ng[0].scope if ng else ""
                np_    = _parent(nscope) if nscope else ""
                if (
                    (np_ == p or nscope == p)
                    and ng[0].layer == layer
                    and _phase(ng) == phase
                ):
                    total_ops += len(ng)
                    if total_ops > max_parent_ops:
                        break
                    j += 1
                else:
                    break

            if j > i + 1:
                merged: list[OpNode] = []
                for g2 in leaf_groups[i:j]:
                    merged.extend(g2)
                result.append(merged)
                i = j
                continue

        result.append(g)
        i += 1

    return result


# ── Pass 3 ────────────────────────────────────────────────────────────────────

def _semantic_label(
    group: list[OpNode],
    path_to_class: dict[str, str],
    platform: str,
) -> str:
    """Determine the fused op_type label for a group of nodes.

    Priority: sub-patterns (most specific) > semantic label > module_class.
    Sub-patterns win because they match both class AND op sequence; semantic
    labels only match the class name and can be overly broad (e.g. "mlp" for
    any MLP class, while "gated_mlp" requires the silu+mul pattern).
    """
    from python.zrt.graph.fusion_rules import (
        get_semantic_label, get_subpatterns,
    )

    scope    = group[0].scope
    mc       = path_to_class.get(scope, group[0].module_class)
    op_types = [n.op_type for n in group]

    # 1. Sub-pattern: specific class + op-sequence match (highest priority)
    for sp in get_subpatterns(platform):
        if sp.matches_class(mc) and sp.matches_ops(op_types):
            return sp.name

    # 2. Semantic label from module class name
    label = get_semantic_label(mc) if mc else None
    if label:
        return label

    # 3. Fallback: module class or first op_type
    return mc if mc else group[0].op_type


# ── Fused node constructor ────────────────────────────────────────────────────

def _fused_node(
    fused_id:       str,
    group:          list[OpNode],
    label:          str,
    inputs:         list[TensorMeta],
    outputs:        list[TensorMeta],
    path_to_class:  dict[str, str],
    level:          str,
) -> OpNode:
    first      = group[0]
    mc         = path_to_class.get(first.scope, first.module_class)
    fused_from = list(dict.fromkeys(n.op_type for n in group))

    # Propagate invariant annotations from source group.
    # Fusion must not cross stage or phase boundaries — mixed values indicate
    # a bug in pass ordering.  Log loudly so it surfaces in CI.
    import logging as _logging
    _fused_logger = _logging.getLogger(__name__)
    propagated: dict[str, object] = {}
    for key in ("stage_id", "phase"):
        vals = {n.annotations.get(key) for n in group if key in n.annotations}
        if len(vals) == 1:
            propagated[key] = vals.pop()
        elif len(vals) > 1:
            _fused_logger.error(
                "_fused_node: group %r has mixed %r values %r — "
                "fusion is crossing a %s boundary; annotation will be dropped.",
                fused_id, key, vals, key,
            )

    node = OpNode(
        id           = fused_id,
        op_type      = label,
        inputs       = inputs,
        outputs      = outputs,
        scope        = first.scope,
        category     = first.category,
        module_class = mc,
        layer        = first.layer,
        component    = first.component,
        fused_from   = fused_from,
        num_sub_ops  = len(group),
        fusion_level = level,
    )
    node.annotations.update(propagated)
    return node


# ── FusionPass ────────────────────────────────────────────────────────────────

class FusionPass(GraphPass):
    """Apply module-scope fusion rules to the OpGraph.

    Converts groups of consecutive same-scope aten ops into single fused nodes
    with semantic labels (e.g. ``flash_attn``, ``gated_mlp``, ``rms_norm``).
    Communication nodes are never fused and always act as group-breakers.

    **Incremental mode**: when all non-comm nodes already carry ``fusion_level``
    (from Stage-1 graph capture), Pass 1/2 (grouping + parent merge) are skipped.
    Only Pass 3 (semantic relabel) and Pass 4 (expand unfused containers) run,
    making this O(n) instead of O(n²) on pre-fused graphs.
    """

    name = "fusion"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        from python.zrt.graph.fusion_rules import (
            get_platform_settings, CONTAINER_SEMANTICS,
        )

        # Infer platform from hw_spec vendor if available
        platform = _infer_platform(ctx)
        cfg      = get_platform_settings(platform)

        g = graph.clone()
        path_to_class, path_to_children = _build_scope_maps(g)

        # ── No-op detection: skip full regrouping if graph already fused ──────
        topo = g.topo_sort()
        compute_nodes = [n for n in topo if n.category != "communication"]
        all_prefused = all(
            getattr(n, "fusion_level", "") for n in compute_nodes
        ) if compute_nodes else True

        if all_prefused:
            # Graph already fused by Stage-1 capture → only Pass 4 needed
            fusions = self._pass4_expand_containers(
                topo, platform, path_to_class, CONTAINER_SEMANTICS, g)
        else:
            # ── Pass 1 + 2: incremental grouping/merging ──────────────────────
            leaf_groups = _pass1_leaf(topo)

            final_groups = _pass2_parent(
                leaf_groups,
                path_to_class,
                path_to_children,
                max_parent_ops = cfg["max_parent_ops"],
                max_children   = cfg["max_children"],
            )

            # ── Pass 3 + 4: semantic label + expand unfused containers ────────
            fusions, expanded_topo = self._pass3_and_4(
                final_groups, topo, platform, path_to_class, CONTAINER_SEMANTICS, g)

            topo = expanded_topo  # use expanded topology for apply step

        # ── Apply replacements ────────────────────────────────────────────────
        # Groups are non-overlapping; order does not affect correctness.
        for group_ids, new_node in fusions:
            g.replace_subgraph(group_ids, new_node)

        return g

    # ── Pass 3 + 4 ──────────────────────────────────────────────────────────

    def _pass3_and_4(
        self,
        final_groups: list,
        topo: list,
        platform: str,
        path_to_class: dict,
        container_semantics: set,
        g: "OpGraph",
    ) -> tuple[list[tuple[set[str], "OpNode"]], list]:
        """Pass 3 (semantic label) + Pass 4 (expand unfused containers).

        Returns (fusions, expanded_topo) where expanded_topo replaces *topo*
        after container expansion.
        """
        fusions: list[tuple[set[str], "OpNode"]] = []
        fuse_idx = 0
        expanded_topo: list = []

        for group in final_groups:
            if group[0].category == "communication":
                expanded_topo.extend(group)
                continue

            group_ids = {n.id for n in group}
            label     = _semantic_label(group, path_to_class, platform)

            # ── Pass 4: expand unfused container groups ───────────────────
            if (
                len(group) > 1
                and self._is_unfused_container(label, group, container_semantics)
            ):
                # This container was not matched by any hardware subpattern;
                # expand back to individual nodes.
                for node in group:
                    expanded_topo.append(node)
                continue

            if len(group) == 1:
                # Single-node: relabel op_type if semantic label differs.
                node = group[0]
                if label != node.op_type and node.module_class:
                    original_op       = node.op_type
                    node.op_type      = label
                    node.fused_from   = [original_op]
                    node.num_sub_ops  = 1
                    node.fusion_level = "leaf"
                expanded_topo.append(node)
                continue

            level     = "parent" if len(group) > 3 else "leaf"
            inputs, outputs = _external_io(g, group_ids)
            fused_id  = f"fused_{fuse_idx}_{group[0].id}"
            fuse_idx += 1
            new_node  = _fused_node(
                fused_id, group, label, inputs, outputs, path_to_class, level)
            fusions.append((group_ids, new_node))
            expanded_topo.append(new_node)

        return fusions, expanded_topo

    # ── Pass 4 (incremental mode) ───────────────────────────────────────────

    def _pass4_expand_containers(
        self,
        topo: list,
        platform: str,
        path_to_class: dict,
        container_semantics: set,
        g: "OpGraph",
    ) -> list[tuple[set[str], "OpNode"]]:
        """Pass 4 only: scan for multi-op container groups that weren't
        matched by any hardware subpattern and split them back to individual ops.

        This is the safety net for pre-fused graphs — Stage-1 fusion may leave
        container groups (attn/mlp/moe_block) that have no matching kernel
        pattern, making them unfusible.  We detect and expand them here.
        """
        from python.zrt.graph.fusion_rules import get_subpatterns

        fusions: list[tuple[set[str], "OpNode"]] = []  # empty: no new fusions
        patterns = get_subpatterns(platform)

        for node in topo:
            if node.category == "communication":
                continue
            if getattr(node, "num_sub_ops", 1) <= 1:
                continue

            mc    = node.module_class or ""
            label = node.op_type

            # Check if this node is an unfused container
            if label not in container_semantics:
                continue

            # It's a container — check if any hardware subpattern matches
            matched = False
            for sp in patterns:
                if sp.matches_class(mc):
                    # Can't check op sequence on already-fused node easily;
                    # assume if class matches AND there's a pattern for it, it's fusible
                    matched = True
                    break

            if matched:
                continue  # container IS fusible — keep it

            # Unfused container: split into individual child nodes
            # We don't have _children like FusionEngine; instead we just
            # remove the fusion metadata so downstream treats these as individual ops.
            # The original fused_from list gives us the constituent op types.
            node.fused_from   = []
            node.num_sub_ops  = 0
            node.fusion_level = ""

        return fusions  # empty — no node replacements, just metadata clearing

    @staticmethod
    def _is_unfused_container(
        label: str,
        group: list,
        container_semantics: set,
    ) -> bool:
        """Check if *group* is a multi-op container with no specific kernel match."""
        if not group or label not in container_semantics:
            return False
        if len(group) <= 1:
            return False
        # _semantic_label already tried subpatterns; if label is still
        # in CONTAINER_SEMANTICS, no hardware pattern matched → unfused.
        return True


# ── helpers ───────────────────────────────────────────────────────────────────

def _infer_platform(ctx: "TransformContext") -> str:
    """Best-effort platform from hw_spec vendor string."""
    if ctx.hw_spec is None:
        return "generic"
    vendor = getattr(ctx.hw_spec, "vendor", "").lower()
    if "nvidia" in vendor or "cuda" in vendor:
        return "cuda"
    if "ascend" in vendor or "npu" in vendor:
        return "ascend_npu"
    return "generic"
