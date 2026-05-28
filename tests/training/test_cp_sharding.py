"""Regression tests for Ulysses / Ring / Hybrid CP sharding metadata."""

import pytest

from zrt.training.ir.builders import build_graph
from zrt.training.spec import LayerKind, ModelSpec
from zrt.training.spec import CPKind, Strategy


def _model() -> ModelSpec:
    return ModelSpec(
        hidden=4096,
        ffn=16384,
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
        vocab=32000,
        seq_len=8192,
        layers=[LayerKind.DENSE] * 2,
    )


def _strategy(cp_kind: CPKind) -> Strategy:
    return Strategy(tp=2, cp=4, pp=1, ep=1, dp=1, micro_batch=1, cp_kind=cp_kind)


def _attn_ops(cp_kind: CPKind):
    model = _model()
    strategy = _strategy(cp_kind)
    graph = build_graph(model, strategy)
    return [
        op for op in graph.ops
        if op.kind in ("attn_core", "sparse_attn")
    ], model, strategy


@pytest.mark.parametrize("cp_kind", [CPKind.ULYSSES, CPKind.RING, CPKind.HYBRID])
def test_global_embed_and_final_norm_are_cp_sharded(cp_kind):
    """Global token ops outside layer_index still operate on CP-local sequence."""
    model = _model()
    strategy = _strategy(cp_kind)
    graph = build_graph(model, strategy)
    expected_s = model.seq_len // strategy.cp

    embed = next(op for op in graph.ops if op.kind == "embed")
    final_ln = next(op for op in graph.ops if op.name == "final_ln")

    assert embed.inputs[0].shape_local[0] == expected_s
    assert embed.outputs[0].shape_local[0] == expected_s
    assert embed.meta["m"] == expected_s
    assert final_ln.inputs[0].shape_local[0] == expected_s
    assert final_ln.outputs[0].shape_local[0] == expected_s
    assert final_ln.meta["bytes_fwd"] == (
        model.seq_len * model.hidden * model.act_dtype.bytes * 2 // strategy.cp
    )


def test_global_embed_and_final_norm_are_unchanged_without_cp():
    """CP=1 should leave global token-op sequence dimensions unchanged."""
    model = _model()
    strategy = Strategy(tp=2, cp=1, pp=1, ep=1, dp=1, micro_batch=1)
    graph = build_graph(model, strategy)

    embed = next(op for op in graph.ops if op.kind == "embed")
    final_ln = next(op for op in graph.ops if op.name == "final_ln")

    assert embed.inputs[0].shape_local[0] == model.seq_len
    assert embed.outputs[0].shape_local[0] == model.seq_len
    assert embed.meta["m"] == model.seq_len
    assert final_ln.inputs[0].shape_local[0] == model.seq_len
    assert final_ln.outputs[0].shape_local[0] == model.seq_len


class TestUlyssesCPSharding:
    """Ulysses A2A swaps sequence sharding for head sharding during attention."""

    def test_ulysses_attn_s_preserved(self):
        """Ulysses-CP: s must NOT be divided — full seq is on this rank."""
        attn_ops, model, strategy = _attn_ops(CPKind.ULYSSES)
        assert attn_ops
        for op in attn_ops:
            assert op.meta.get("s") == model.seq_len, (
                f"{op.name}: expected s={model.seq_len}, got {op.meta.get('s')}"
            )

    def test_ulysses_attn_heads_divided(self):
        """Ulysses-CP: heads = heads_tp // cp."""
        attn_ops, model, strategy = _attn_ops(CPKind.ULYSSES)
        expected_heads = max(1, (model.num_heads // strategy.tp) // strategy.cp)
        for op in attn_ops:
            assert op.meta.get("heads") == expected_heads, (
                f"{op.name}: expected heads={expected_heads}, got {op.meta.get('heads')}"
            )

    def test_ulysses_heads_not_gathered(self):
        """Debug marker records that heads are scattered, not gathered."""
        attn_ops, model, strategy = _attn_ops(CPKind.ULYSSES)
        for op in attn_ops:
            assert op.meta.get("heads_gathered_by_cp") is False, (
                f"{op.name}: heads_gathered_by_cp should be False"
            )

    def test_first_attn_core_metadata(self):
        """Spot-check the first attn_core op for expected metadata values."""
        attn_ops, model, strategy = _attn_ops(CPKind.ULYSSES)
        op = next(op for op in attn_ops if op.kind == "attn_core")
        assert op.meta["s"] == 8192
        assert op.meta["heads"] == 4
        assert op.meta["heads_tp"] == 16


class TestRingCPSharding:
    """Verify that Ring-CP divides s and leaves heads unchanged."""

    def test_ring_attn_s_divided(self):
        """Ring-CP: s must be divided by cp."""
        attn_ops, model, strategy = _attn_ops(CPKind.RING)
        expected_s = model.seq_len // strategy.cp
        for op in attn_ops:
            assert op.meta.get("s") == expected_s, (
                f"{op.name}: expected s={expected_s}, got {op.meta.get('s')}"
            )

    def test_ring_attn_heads_unchanged(self):
        """Ring-CP: heads must equal heads_tp (unchanged)."""
        attn_ops, model, strategy = _attn_ops(CPKind.RING)
        expected_heads = model.num_heads // strategy.tp
        for op in attn_ops:
            assert op.meta.get("heads") == expected_heads, (
                f"{op.name}: expected heads={expected_heads}, got {op.meta.get('heads')}"
            )


class TestHybridCPSharding:
    """Hybrid/USP combines Ulysses head sharding with Ring sequence tiling."""

    def test_hybrid_attn_keeps_ring_sequence_tiling(self):
        attn_ops, model, strategy = _attn_ops(CPKind.HYBRID)
        expected_s = model.seq_len // strategy.cp
        for op in attn_ops:
            assert op.meta.get("s") == expected_s, (
                f"{op.name}: expected ring-tiled s={expected_s}, got {op.meta.get('s')}"
            )
            assert op.meta.get("cp_tiles") == strategy.cp

    def test_hybrid_attn_keeps_ulysses_head_sharding(self):
        attn_ops, model, strategy = _attn_ops(CPKind.HYBRID)
        expected_heads = max(1, (model.num_heads // strategy.tp) // strategy.cp)
        for op in attn_ops:
            assert op.meta.get("heads") == expected_heads, (
                f"{op.name}: expected Ulysses-sharded heads={expected_heads}, "
                f"got {op.meta.get('heads')}"
            )
            assert op.meta.get("heads_gathered_by_cp") is False
