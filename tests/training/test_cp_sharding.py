"""Regression tests for Ulysses / Ring CP sharding metadata correctness."""

import pytest

from zrt.training.io.config_loader import load_specs
from zrt.training.ir.builders import build_graph
from zrt.training.spec import CPKind, Strategy


_A5_CONFIG = "python/zrt/training/configs/deepseek_v3_2_3d_A5.yaml"


class TestUlyssesCPSharding:
    """Verify that Ulysses-CP scatters heads and gathers seq for attention."""

    @pytest.fixture
    def a5_graph(self):
        model, system, strategy = load_specs(_A5_CONFIG)
        graph = build_graph(model, strategy)
        return graph, model, strategy

    def test_ulysses_attn_s_preserved(self, a5_graph):
        """Ulysses-CP: s must NOT be divided — full seq is on this rank."""
        graph, model, strategy = a5_graph
        for op in graph.ops:
            if op.kind in ("attn_core", "sparse_attn"):
                assert op.meta.get("s") == model.seq_len, (
                    f"{op.name}: expected s={model.seq_len}, got {op.meta.get('s')}"
                )

    def test_ulysses_attn_heads_divided(self, a5_graph):
        """Ulysses-CP: heads = heads_tp // cp."""
        graph, model, strategy = a5_graph
        tp = strategy.tp
        cp = strategy.cp
        expected_heads = max(1, (model.num_heads // tp) // cp)
        for op in graph.ops:
            if op.kind in ("attn_core", "sparse_attn"):
                assert op.meta.get("heads") == expected_heads, (
                    f"{op.name}: expected heads={expected_heads}, got {op.meta.get('heads')}"
                )

    def test_ulysses_heads_not_gathered(self, a5_graph):
        """Ulysses-CP: heads_gathered_by_cp must be False (heads are scattered)."""
        graph, model, strategy = a5_graph
        for op in graph.ops:
            if op.kind in ("attn_core", "sparse_attn"):
                assert op.meta.get("heads_gathered_by_cp") is False, (
                    f"{op.name}: heads_gathered_by_cp should be False"
                )

    def test_first_attn_core_metadata(self, a5_graph):
        """Spot-check the first attn_core op for expected metadata values."""
        graph, model, strategy = a5_graph
        for op in graph.ops:
            if op.kind == "attn_core":
                assert op.meta["s"] == 65536
                assert op.meta["heads"] == 1
                assert op.meta["heads_tp"] == 64
                break


class TestRingCPSharding:
    """Verify that Ring-CP divides s and leaves heads unchanged."""

    def test_ring_attn_s_divided(self):
        """Ring-CP: s must be divided by cp."""
        model, system, strategy = load_specs(_A5_CONFIG)
        ring_strategy = Strategy(
            tp=strategy.tp, cp=strategy.cp, pp=strategy.pp,
            ep=strategy.ep, dp=strategy.dp,
            micro_batch=strategy.micro_batch,
            cp_kind=CPKind.RING,
        )
        graph = build_graph(model, ring_strategy)
        expected_s = model.seq_len // ring_strategy.cp
        for op in graph.ops:
            if op.kind in ("attn_core", "sparse_attn"):
                assert op.meta.get("s") == expected_s, (
                    f"{op.name}: expected s={expected_s}, got {op.meta.get('s')}"
                )

    def test_ring_attn_heads_unchanged(self):
        """Ring-CP: heads must equal heads_tp (unchanged)."""
        model, system, strategy = load_specs(_A5_CONFIG)
        ring_strategy = Strategy(
            tp=strategy.tp, cp=strategy.cp, pp=strategy.pp,
            ep=strategy.ep, dp=strategy.dp,
            micro_batch=strategy.micro_batch,
            cp_kind=CPKind.RING,
        )
        graph = build_graph(model, ring_strategy)
        tp = ring_strategy.tp
        expected_heads = model.num_heads // tp
        for op in graph.ops:
            if op.kind in ("attn_core", "sparse_attn"):
                assert op.meta.get("heads") == expected_heads, (
                    f"{op.name}: expected heads={expected_heads}, got {op.meta.get('heads')}"
                )
