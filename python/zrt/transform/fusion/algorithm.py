"""Fusion algorithm: bucket → (rule-fuse | structural-collapse) → Add+Norm.

Single-pass implementation: ``fuse(graph, ctx)`` is the only public entry.

For each (scope, module_class) bucket produced by :func:`bucket_into_groups`:

* If a fusion rule matches the bucket's actual aten op_type sequence,
  build a rich fused node via :func:`build_fused_node` (semantic
  ``op_type``, ``rule.annotations``, and ``sem_*`` fields derived from
  ``rule.shape_derivation`` / ``flops`` / ``memory_bytes`` / ``dtype``).

* Otherwise, fall back to a plain structural collapse via
  :func:`_build_collapsed_node` — ``op_type = module_class``, no
  semantic fields, ``fused_from`` records the absorbed aten ops.

Singleton groups are skipped unless they match a ``class_only`` rule.
Communication and scopeless nodes are never collapsed.

Both code paths propagate split-stage annotations (``stage_id``,
``phase``, ``tp_split``, ``ep_*``, ``cp_split``, ``recompute``) from
the absorbed children when all children agree.

See ``docs/fusion_v2_rich_rules_zh.md`` for the rule contract.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from .provenance import FusedIOPort, resolve_io
from .registry import lookup_rule
from .rule import ModuleFusionRule
from .semantics import annotate_fused_node

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.ir.node import OpNode

logger = logging.getLogger(__name__)


# Annotations that propagate child→parent when all children agree.
_SCALAR_PROPAGATE_KEYS = (
    "stage_id", "phase", "ep_experts_local", "ep_a2a_inserted", "recompute",
)
_DICT_PROPAGATE_KEYS = ("tp_split", "ep_needs_a2a", "cp_split")


# ── Intermediate group representation ────────────────────────────────────────

@dataclass
class FusionGroup:
    """A contiguous run of nodes sharing the same leaf module scope."""
    scope: str
    module_class: str
    module_class_obj: Optional[type]
    child_ops: list["OpNode"]
    leaf_attr: str
    call_id: int = 0
    is_full_forward: bool = False  # True iff bucket contains every op of the call


# ── Pass 1: bucket by leaf module ────────────────────────────────────────────

def bucket_nodes_by_leaf_module(graph: "OpGraph") -> list[FusionGroup]:
    """Group nodes by ``call_id`` (forward-call instance).

    Each module forward invocation has a unique ``call_id`` recorded by
    :class:`ModuleTracker` and stored on every dispatched op.  Grouping
    by ``call_id`` keeps fragments of the same parent scope (interleaved
    with child-module calls) **separate**, which is the only correct
    granularity for ``class_only`` rules — those rules assume the bucket
    represents an entire forward call.

    Communication ops and scopeless ops always become singleton groups.
    Within a single ``call_id``, ops are kept in topological order.
    Legacy graphs without ``call_id`` (e.g. captured before this field
    existed) fall back to ``(scope, module_class, layer)`` keying so old
    fixtures and torch.compile-mode captures still produce sensible
    groups.
    """
    groups: list[FusionGroup] = []
    current: Optional[FusionGroup] = None
    last_key: Any = None
    seen_call_ids_in_topo: set[int] = set()

    def _flush() -> None:
        nonlocal current
        current = None

    for node in graph.topo_sort():
        # Comm / scopeless nodes break the current group
        if node.category == "communication" or not node.scope:
            _flush()
            groups.append(FusionGroup(
                scope=node.scope,
                module_class=node.module_class,
                module_class_obj=None,
                child_ops=[node],
                leaf_attr=node.name,
                call_id=getattr(node, "call_id", 0) or 0,
                is_full_forward=False,
            ))
            last_key = None
            continue

        node_call_id = getattr(node, "call_id", 0) or 0
        if node_call_id > 0:
            key = ("call", node_call_id)
        else:
            # Legacy fallback: bucket by scope/class/layer.
            key = ("legacy", node.scope, node.module_class, node.layer)

        if current is not None and key == last_key:
            current.child_ops.append(node)
            continue

        current = FusionGroup(
            scope=node.scope,
            module_class=node.module_class,
            module_class_obj=_extract_class_obj(node),
            child_ops=[node],
            leaf_attr=node.scope.rsplit(".", 1)[-1] if node.scope else "",
            call_id=node_call_id,
            is_full_forward=False,
        )
        groups.append(current)
        last_key = key

    # Mark each group as "full-forward" iff every op tagged with that call_id
    # is contained in this single group (i.e. no child call interleaved).
    if any(g.call_id > 0 for g in groups):
        call_id_total: dict[int, int] = {}
        for n in graph.nodes.values():
            cid = getattr(n, "call_id", 0) or 0
            if cid:
                call_id_total[cid] = call_id_total.get(cid, 0) + 1
        for g in groups:
            if g.call_id > 0 and len(g.child_ops) == call_id_total.get(g.call_id, -1):
                g.is_full_forward = True

    return groups


# ── Pass 2: parent merge ─────────────────────────────────────────────────────

def _parent(scope: str) -> str:
    return scope.rsplit(".", 1)[0] if "." in scope else ""


def bucket_into_groups(
    graph: "OpGraph",
    *,
    max_parent_ops: int = 60,
) -> list[FusionGroup]:
    """Bucket the graph into FusionGroups (leaf bucket + parent merge).

    Wraps :func:`bucket_nodes_by_leaf_module` followed by
    :func:`_merge_parent_groups` so the two-step pattern lives in one place.
    """
    leaf_groups = bucket_nodes_by_leaf_module(graph)
    return _merge_parent_groups(leaf_groups, graph, max_parent_ops=max_parent_ops)


def _merge_parent_groups(
    leaf_groups: list[FusionGroup],
    graph: "OpGraph",
    *,
    max_parent_ops: int = 60,
) -> list[FusionGroup]:
    """Merge consecutive leaf groups that share a common parent scope."""
    if not leaf_groups:
        return []

    merged: list[FusionGroup] = []
    i = 0
    while i < len(leaf_groups):
        group = leaf_groups[i]

        if len(group.child_ops) <= 1 or not group.scope:
            merged.append(group)
            i += 1
            continue

        parent_scope = _parent(group.scope)
        if not parent_scope:
            merged.append(group)
            i += 1
            continue

        total_ops = len(group.child_ops)
        j = i + 1
        while j < len(leaf_groups):
            next_group = leaf_groups[j]
            if not next_group.scope:
                break
            if _parent(next_group.scope) != parent_scope:
                break
            if next_group.module_class != group.module_class:
                break
            if total_ops + len(next_group.child_ops) > max_parent_ops:
                break
            total_ops += len(next_group.child_ops)
            group = _merge_two_groups(group, next_group)
            j += 1

        merged.append(group)
        i = j

    return merged


def _merge_two_groups(a: FusionGroup, b: FusionGroup) -> FusionGroup:
    return FusionGroup(
        scope=a.scope,
        module_class=a.module_class,
        module_class_obj=a.module_class_obj,
        child_ops=a.child_ops + b.child_ops,
        leaf_attr=a.leaf_attr,
    )


def _extract_class_obj(node: "OpNode") -> Optional[type]:
    obj = node.annotations.get("module_class_obj")
    if isinstance(obj, type):
        return obj
    return None


# ── Annotation propagation helper ───────────────────────────────────────────

def _propagated_annotations(group: FusionGroup) -> dict:
    """Return annotations that all children of *group* agree on.

    Scalar (hashable) keys use set-based dedup; dict-valued keys
    compare element-wise (since dicts are unhashable).
    """
    propagated: dict = {}
    for key in _SCALAR_PROPAGATE_KEYS:
        vals = {op.annotations.get(key) for op in group.child_ops
                if key in op.annotations}
        if len(vals) == 1:
            propagated[key] = vals.pop()
    for key in _DICT_PROPAGATE_KEYS:
        seen = [op.annotations[key] for op in group.child_ops
                if key in op.annotations]
        if seen and all(d == seen[0] for d in seen):
            propagated[key] = seen[0]
    return propagated


# ── Fused node construction ─────────────────────────────────────────────────

def build_fused_node(
    group: FusionGroup,
    rule: ModuleFusionRule,
    graph: "OpGraph",
    fuse_idx: int,
) -> "OpNode":
    """Construct a fused OpNode from a FusionGroup and its matched rule."""
    from python.zrt.ir.node import OpNode

    first = group.child_ops[0]
    tc = rule.target_class
    op_type = rule.op_type or (
        tc.__name__ if isinstance(tc, type) else str(tc)
    )

    # Collect source op_types (deduplicated, ordered)
    fused_from = list(dict.fromkeys(op.op_type for op in group.child_ops))

    # External IO from graph edges
    group_ids = {op.id for op in group.child_ops}
    ext_inputs, ext_outputs = _external_io(graph, group_ids)

    # Build provenance from rule's IO specs
    provenance_parts: list[FusedIOPort] = []
    if rule.inputs or rule.outputs:
        for spec in rule.inputs:
            port = resolve_io(group.child_ops, spec)
            if port is not None:
                provenance_parts.append(port)
        for spec in rule.outputs:
            port = resolve_io(group.child_ops, spec)
            if port is not None:
                provenance_parts.append(port)

    propagated = _propagated_annotations(group)
    level = "parent" if len(group.child_ops) > 3 else "leaf"

    node = OpNode(
        id=f"fused_{fuse_idx}_{first.id}",
        op_type=op_type,
        inputs=ext_inputs,
        outputs=ext_outputs,
        scope=group.scope,
        category=first.category,
        module_class=group.module_class,
        layer=first.layer,
        component=first.component,
        fused_from=fused_from,
        num_sub_ops=len(group.child_ops),
        fusion_level=level,
        name=group.leaf_attr,
        provenance=tuple(provenance_parts),
    )
    node.annotations.update(propagated)
    node.annotations.update(rule.annotations)
    # Derive semantic fields (batch_size / seq_len / dtype / flops / memory)
    # from rule.io_roles + shape_derivation + flops/memory formulas and
    # write them into node.annotations.  Failures are logged, not raised.
    annotate_fused_node(node, group.child_ops, rule)
    return node


def _build_collapsed_node(
    group: FusionGroup,
    graph: "OpGraph",
    fuse_idx: int,
) -> "OpNode":
    """Plain structural collapse: ``op_type = module_class``, no rule fields.

    Used when a multi-op (scope, module_class) bucket has no matching
    fusion rule.  Records absorbed children in ``fused_from`` and
    propagates split-stage annotations that all children agree on.
    """
    from python.zrt.ir.node import OpNode

    first = group.child_ops[0]
    group_ids = {op.id for op in group.child_ops}
    ext_inputs, ext_outputs = _external_io(graph, group_ids)
    fused_from = list(dict.fromkeys(op.op_type for op in group.child_ops))
    propagated = _propagated_annotations(group)
    level = "parent" if len(group.child_ops) > 3 else "leaf"

    node = OpNode(
        id=f"collapsed_{fuse_idx}_{first.id}",
        op_type=group.module_class,
        inputs=ext_inputs,
        outputs=ext_outputs,
        scope=group.scope,
        category=first.category,
        module_class=group.module_class,
        layer=first.layer,
        component=first.component,
        fused_from=fused_from,
        num_sub_ops=len(group.child_ops),
        fusion_level=level,
        name=group.leaf_attr,
    )
    node.annotations.update(propagated)
    return node


def _external_io(
    graph: "OpGraph",
    group_ids: set[str],
) -> tuple[list, list]:
    """Return (ext_inputs, ext_outputs) for a node group."""
    seen_in: set = set()
    seen_out: set = set()
    inputs: list = []
    outputs: list = []

    for e in graph.edges:
        if e.src not in group_ids and e.dst in group_ids:
            key = (e.tensor_id, e.dst_idx)
            if key not in seen_in and e.tensor is not None:
                seen_in.add(key)
                inputs.append(e.tensor)
        if e.src in group_ids and e.dst not in group_ids:
            key = (e.tensor_id, e.src_idx)
            if key not in seen_out and e.tensor is not None:
                seen_out.add(key)
                outputs.append(e.tensor)

    return inputs, outputs


# ── Public entry: bucket → (rule | collapse) → Add+Norm ─────────────────────

def fuse(graph: "OpGraph", ctx=None) -> "OpGraph":
    """Bucket the graph and rewrite each bucket in one pass.

    For every (scope, module_class) bucket:

    * Singleton bucket with a matching ``class_only`` rule → fused node.
    * Multi-op bucket with a matching rule → fused node.
    * Multi-op bucket with no matching rule → plain collapsed node
      (``op_type = module_class``).
    * Otherwise the bucket is left untouched.

    Add+Norm composition runs at the end on the rewritten graph.
    """
    g = graph.clone()
    groups = bucket_into_groups(g)
    fuse_idx = 0

    for group in groups:
        # ── Singleton bucket ────────────────────────────────────────────────
        if len(group.child_ops) <= 1:
            node = group.child_ops[0]
            if node.category == "communication" or not node.module_class:
                continue
            rule = lookup_rule((node.op_type,), module_class=node.module_class)
            if rule is None:
                continue
            if rule.pattern is None or rule.pattern.kind != "class_only":
                continue
            # class_only rules require the bucket to span the full forward
            # call.  A singleton bucket only does so when no child fired.
            if not group.is_full_forward:
                continue
            fused = build_fused_node(group, rule, g, fuse_idx)
            fuse_idx += 1
            g.replace_subgraph({node.id}, fused)
            continue

        # ── Multi-op bucket ─────────────────────────────────────────────────
        if not group.scope or not group.module_class:
            continue

        actual_seq = tuple(op.op_type for op in group.child_ops)
        rule = lookup_rule(actual_seq, module_class=group.module_class)
        # ``class_only`` rules only verify group size; they must not fire on
        # fragments of a forward call (e.g. the inline RMSNorm sub-block of
        # Attention.forward) — the resulting node would be tagged with the
        # parent's semantic op_type and produce wrong shape/FLOPs.
        if (rule is not None
                and rule.pattern is not None
                and rule.pattern.kind == "class_only"
                and not group.is_full_forward):
            rule = None

        group_ids = {op.id for op in group.child_ops}

        if rule is not None:
            replacement = build_fused_node(group, rule, g, fuse_idx)
        else:
            replacement = _build_collapsed_node(group, g, fuse_idx)
        fuse_idx += 1
        g.replace_subgraph(group_ids, replacement)

    g = _compose_add_norm(g)
    return g


# ── Add+Norm compositor ─────────────────────────────────────────────────────

def _compose_add_norm(graph: "OpGraph") -> "OpGraph":
    """Merge adjacent Add + RMSNorm/LayerNorm nodes into AddNorm.

    Operates on already-fused nodes (not raw aten ops).  Looks for
    ``op_type`` ending in "add" followed by one ending in "norm".
    """
    from python.zrt.ir.node import OpNode

    topo = graph.topo_sort()
    if len(topo) < 2:
        return graph

    norm_types = {"rms_norm", "layer_norm", "RMSNorm", "LayerNorm"}
    add_types = {"add", "residual_add"}

    pairs: list[tuple[int, int]] = []
    for i in range(len(topo) - 1):
        a, b = topo[i], topo[i + 1]
        a_label = a.op_type.lower()
        b_label = b.op_type.lower()
        if (a_label in add_types or a_label.endswith("add")) and \
           (b_label in norm_types or b_label.endswith("norm")):
            has_edge = any(
                e.src == a.id and e.dst == b.id for e in graph.edges
            )
            if has_edge:
                pairs.append((i, i + 1))

    if not pairs:
        return graph

    used: set[str] = set()
    for add_i, norm_i in reversed(pairs):
        add_node = topo[add_i]
        norm_node = topo[norm_i]
        if add_node.id in used or norm_node.id in used:
            continue

        merged = OpNode(
            id=f"composed_{add_node.id}_{norm_node.id}",
            op_type="AddNorm",
            inputs=add_node.inputs,
            outputs=norm_node.outputs,
            scope=norm_node.scope,
            category="compute",
            module_class=norm_node.module_class,
            layer=norm_node.layer,
            component=norm_node.component,
            fused_from=add_node.fused_from + norm_node.fused_from,
            num_sub_ops=add_node.num_sub_ops + norm_node.num_sub_ops,
            fusion_level="parent",
            name=norm_node.name,
        )
        merged.annotations.update(add_node.annotations)
        merged.annotations.update(norm_node.annotations)
        graph.replace_subgraph({add_node.id, norm_node.id}, merged)
        used.add(add_node.id)
        used.add(norm_node.id)

    return graph
