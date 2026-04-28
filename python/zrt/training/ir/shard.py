"""Sharding pass — apply TP/CP/EP sharding to IR and insert collectives."""

from __future__ import annotations

from zrt.training.ir.training_graph import Collective, Graph
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.strategy import CPKind, Strategy


class ShardPlan:
    """Holds sharding parameters derived from Strategy."""

    def __init__(self, strategy: Strategy):
        self.tp = strategy.tp
        self.cp = strategy.cp
        self.ep = strategy.ep
        self.dp = strategy.dp
        self.pp = strategy.pp
        self.cp_kind = strategy.cp_kind
        self.sp = strategy.tp > 1  # Megatron SP on when TP>1

    def shard_col_parallel(self, n: int) -> int:
        """Shard column dimension by TP."""
        return n // self.tp

    def shard_row_parallel(self, n: int) -> int:
        """Shard row dimension by TP (output dimension stays full via RS)."""
        return n // self.tp

    def shard_seq_parallel(self, s: int) -> int:
        """Shard sequence dimension by CP (Ulysses)."""
        return s // self.cp if self.cp > 1 else s

    def shard_expert_parallel(self, n: int) -> int:
        """Shard experts by EP (each rank gets n/experts)."""
        return n // self.ep if self.ep > 1 else n

    @property
    def has_cp(self) -> bool:
        """Whether context parallelism is enabled."""
        return self.cp > 1

    @property
    def has_ep(self) -> bool:
        """Whether expert parallelism is enabled."""
        return self.ep > 1

    @property
    def has_tp(self) -> bool:
        """Whether tensor parallelism is enabled."""
        return self.tp > 1


def insert_collectives(graph: Graph, shard: ShardPlan, model: ModelSpec) -> None:
    """Insert TP/CP/EP collectives into the graph IN-PLACE.

    Megatron-TP pattern per dense layer:
      - AG before QKV (if input was RS'd by previous layer)
      - RS after O_proj
      - AG before FFN up/gate
      - RS after FFN down

    Ulysses-CP pattern:
      - A2A before attention (splits sequence across ranks)
      - A2A after attention (gathers sequence)

    EP pattern (MoE only):
      - All-to-All before routed expert FFN
      - All-to-All after routed expert FFN
    """
    collectives: list[Collective] = []

    # Insert TP collectives
    if shard.has_tp:
        _insert_tp_collectives(graph, shard, model, collectives)

    # Insert CP collectives (Ulysses sequence parallel)
    if shard.has_cp:
        _insert_cp_collectives(graph, shard, model, collectives)

    # Insert EP collectives (expert parallel for MoE)
    if shard.has_ep:
        _insert_ep_collectives(graph, shard, model, collectives)

    graph.collectives.extend(collectives)


def _insert_tp_collectives(
    graph: Graph, shard: ShardPlan, model: ModelSpec,
    collectives: list[Collective],
) -> None:
    """Insert TP collectives (AG/RS pairs) into the graph."""
    seq = model.seq_len
    h = model.hidden
    h_attn = model.num_heads * model.head_dim
    h_kv = model.num_kv_heads * model.head_dim
    ffn = model.ffn
    act_bytes = model.act_dtype.bytes

    for layer_id, (start, end) in graph.layer_index.items():
        # Compute payload sizes (before sharding, divided by TP for per-rank)
        ag_attn_bytes = seq * h * act_bytes  # AG before QKV
        rs_attn_bytes = seq * h * act_bytes  # RS after O_proj
        ag_ffn_bytes = seq * h * act_bytes   # AG before FFN up
        rs_ffn_bytes = seq * h * act_bytes   # RS after FFN down

        for i in range(start, end):
            op = graph.ops[i]

            # AG before QKV projection (gathers seq-sharded input for col-parallel)
            if op.kind == "matmul" and "qkv" in op.name:
                collectives.append(Collective(
                    name=f"ag_{op.name}",
                    kind="AG", group="TP",
                    bytes_=ag_attn_bytes,
                    inserted_after=op.name,
                ))

            # RS after O projection
            if op.kind == "matmul" and "o_proj" in op.name:
                collectives.append(Collective(
                    name=f"rs_{op.name}",
                    kind="RS", group="TP",
                    bytes_=rs_attn_bytes,
                    inserted_after=op.name,
                ))

            # AG before FFN up projection
            if op.kind == "matmul" and "up_proj" in op.name:
                collectives.append(Collective(
                    name=f"ag_{op.name}",
                    kind="AG", group="TP",
                    bytes_=ag_ffn_bytes,
                    inserted_after=op.name,
                ))

            # RS after FFN down projection
            if op.kind == "matmul" and "down_proj" in op.name:
                collectives.append(Collective(
                    name=f"rs_{op.name}",
                    kind="RS", group="TP",
                    bytes_=rs_ffn_bytes,
                    inserted_after=op.name,
                ))

        # Adjust tensor shapes for TP sharding
        _apply_tp_sharding(graph, start, end, shard, h, h_attn, h_kv, ffn, seq, act_bytes)


def _insert_cp_collectives(
    graph: Graph, shard: ShardPlan, model: ModelSpec,
    collectives: list[Collective],
) -> None:
    """Insert CP collectives (A2A pairs for Ulysses sequence parallel).

    Ulysses-CP pattern:
      - A2A before attention core (splits sequence across ranks)
      - A2A after attention core (gathers sequence)

    Ring-CP is handled via send/recv pairs but modeled as A2A for simplicity.
    """
    if shard.cp_kind != CPKind.ULYSSES:
        # Ring-CP not yet modeled in Stack A IR
        return

    h = model.hidden
    act_bytes = model.act_dtype.bytes

    for layer_id, (start, end) in graph.layer_index.items():
        # A2A payload size: full sequence × hidden
        a2a_bytes = model.seq_len * h * act_bytes

        for i in range(start, end):
            op = graph.ops[i]

            # A2A before attention core
            if op.kind == "attn_core":
                collectives.append(Collective(
                    name=f"a2a_before_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_before=op.name,
                ))

            # A2A after attention core (gathers attention output)
            if op.kind == "attn_core":
                collectives.append(Collective(
                    name=f"a2a_after_{op.name}",
                    kind="A2A", group="CP",
                    bytes_=a2a_bytes,
                    inserted_after=op.name,
                ))

        # Adjust tensor shapes for CP sharding (sequence dimension)
        _apply_cp_sharding(graph, start, end, shard, model.seq_len)


def _insert_ep_collectives(
    graph: Graph, shard: ShardPlan, model: ModelSpec,
    collectives: list[Collective],
) -> None:
    """Insert EP collectives (A2A pairs for expert parallel).

    EP pattern (MoE only):
      - A2A before routed expert FFN (routes tokens to expert ranks)
      - A2A after routed expert FFN (gathers expert outputs)
    """
    h = model.hidden
    moe_ffn = model.moe_ffn if model.moe_ffn > 0 else model.ffn
    act_bytes = model.act_dtype.bytes

    for layer_id, (start, end) in graph.layer_index.items():
        # Check if this is an MoE layer
        if layer_id >= len(model.layers):
            continue
        if model.layers[layer_id].value != "moe":
            continue

        # A2A payload size: tokens × hidden
        a2a_bytes = model.seq_len * h * act_bytes

        for i in range(start, end):
            op = graph.ops[i]

            # A2A before routed expert FFN
            if op.kind == "matmul" and "routed_expert" in op.name:
                collectives.append(Collective(
                    name=f"a2a_before_{op.name}",
                    kind="A2A", group="EP",
                    bytes_=a2a_bytes,
                    inserted_before=op.name,
                ))

            # A2A after routed expert FFN
            if op.kind == "matmul" and "routed_expert" in op.name:
                collectives.append(Collective(
                    name=f"a2a_after_{op.name}",
                    kind="A2A", group="EP",
                    bytes_=a2a_bytes,
                    inserted_after=op.name,
                ))

        # Adjust tensor shapes for EP sharding (experts sharded across ranks)
        _apply_ep_sharding(graph, start, end, shard, model.num_experts, model.top_k)


def _apply_tp_sharding(
    graph: Graph, start: int, end: int, shard: ShardPlan,
    h: int, h_attn: int, h_kv: int, ffn: int, seq: int, act_bytes: int,
) -> None:
    """Adjust tensor shape_local for TP sharding on ops in [start, end)."""
    if shard.tp <= 1:
        return

    for i in range(start, end):
        op = graph.ops[i]

        if op.kind == "matmul":
            m, n, k = op.meta["m"], op.meta["n"], op.meta["k"]

            if "qkv" in op.name:
                # Col-parallel: shard n dimension (output) by TP
                n_local = n // shard.tp
                op.meta["n_local"] = n_local
                for t in op.outputs:
                    t.shape_local = (t.shape_logical[0], n_local)
            elif "o_proj" in op.name:
                # Row-parallel: shard k dimension (input) by TP
                k_local = k // shard.tp
                op.meta["k_local"] = k_local
                for t in op.inputs:
                    if t.shape_logical[-1] == k:
                        t.shape_local = (t.shape_logical[0], k_local)
            elif "up_proj" in op.name or "gate_proj" in op.name:
                # Col-parallel: shard n by TP
                n_local = n // shard.tp
                op.meta["n_local"] = n_local
                for t in op.outputs:
                    t.shape_local = (t.shape_logical[0], n_local)
            elif "down_proj" in op.name:
                # Row-parallel: shard k by TP
                k_local = k // shard.tp
                op.meta["k_local"] = k_local
                for t in op.inputs:
                    if t.shape_logical[-1] == k:
                        t.shape_local = (t.shape_logical[0], k_local)
        elif op.kind == "attn_core":
            if "heads" in op.meta:
                op.meta["heads"] = max(1, op.meta["heads"] // shard.tp)
            if "h_kv" in op.meta:
                op.meta["h_kv"] = max(1, op.meta["h_kv"] // shard.tp)
            for t in op.inputs + op.outputs:
                if t.shape_logical and t.shape_logical[-1] in (h_attn, h_kv):
                    t.shape_local = (t.shape_logical[0], max(1, t.shape_logical[-1] // shard.tp))
        elif op.kind in ("mhc_pre", "mhc_post", "mhc_head", "hc_expand"):
            # Hyper-Connections are token-local: TP does not shard hc or h.
            # The mixes-Linear operates on (hc·h) features that semantically
            # belong to *one* residual stream replicated across hc copies; the
            # learned hc_*_fn matrix is small (hc·h × mix_hc) and replicated on
            # every TP rank, so introducing TP here would only add comm without
            # reducing compute.  CP (sequence-parallel) handling — when added —
            # would scale the seq dim of meta["s"] independently.
            pass
        elif op.kind in ("ln", "rope", "swiglu", "add"):
            if "bytes_fwd" in op.meta:
                op.meta["bytes_fwd"] = int(op.meta["bytes_fwd"]) // shard.tp
            for t in op.inputs + op.outputs:
                if t.shape_logical and t.shape_logical[-1] in (h, h_attn, h_kv, ffn):
                    t.shape_local = (t.shape_logical[0], max(1, t.shape_logical[-1] // shard.tp))


def _apply_cp_sharding(
    graph: Graph, start: int, end: int, shard: ShardPlan, seq: int,
) -> None:
    """Adjust tensor shape_local for CP sharding (Ulysses sequence parallel).

    For Ulysses CP, the sequence dimension is split across ranks.
    Each rank processes seq/cp tokens.
    """
    if not shard.has_cp:
        return

    seq_local = seq // shard.cp

    for i in range(start, end):
        op = graph.ops[i]

        # Adjust sequence dimension for all ops
        for t in op.inputs + op.outputs:
            if t.shape_logical and len(t.shape_logical) > 0:
                # First dimension is typically sequence/batch
                if t.shape_logical[0] == seq:
                    t.shape_local = (seq_local,) + t.shape_logical[1:]

        # Scale compute metadata for FLOPs model
        # The FLOPs model reads meta['m'] for matmul and meta['s'] for attn_core
        if op.kind == "matmul":
            if "m" in op.meta:
                op.meta["m"] = op.meta["m"] // shard.cp
        elif op.kind == "attn_core":
            if "s" in op.meta:
                op.meta["s"] = op.meta["s"] // shard.cp
        elif op.kind in ("ln", "rope", "swiglu", "add"):
            if "bytes_fwd" in op.meta:
                op.meta["bytes_fwd"] = int(op.meta["bytes_fwd"]) // shard.cp


def _apply_ep_sharding(
    graph: Graph, start: int, end: int, shard: ShardPlan,
    num_experts: int, top_k: int,
) -> None:
    """Adjust tensor shape_local for EP sharding (expert parallel).

    For EP, each rank handles num_experts/ep experts.
    The routed expert FFN only processes a subset of experts.
    """
    if not shard.has_ep:
        return

    experts_per_rank = num_experts // shard.ep

    for i in range(start, end):
        op = graph.ops[i]

        # Only routed expert FFN is affected by EP sharding
        if op.kind == "matmul" and "routed_expert" in op.name:
            # Update num_experts in metadata
            if "num_experts" in op.meta:
                op.meta["num_experts_local"] = experts_per_rank
            # FLOPs scale by local expert count
            # (top_k experts per token, but only experts_per_rank available locally)
            if "num_experts" in op.meta:
                original_experts = op.meta["num_experts"]
                # Effective FLOPs scale ratio: (experts_per_rank / original_experts)
                # But we keep top_k the same since each token still goes to top_k experts
                pass  # FLOPs scaling handled in flops.py

        # Router output: num_experts -> experts_per_rank
        if op.kind == "matmul" and "router" in op.name:
            for t in op.outputs:
                if len(t.shape_logical) > 1:
                    # Router output: (seq, num_experts) -> (seq, experts_per_rank)
                    if t.shape_logical[1] == num_experts:
                        t.shape_local = (t.shape_logical[0], experts_per_rank)
