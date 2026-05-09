"""DeepSeek-V4 fusion rules.

V4's FP8-quantised Linear forward produces ~40+ distinct topological
orderings due to non-deterministic branch capture, so strict tuple
matching is impractical.  Only the unfused-call fast path is registered
as a known sequence here; other patterns should be provided via YAML.
"""
from __future__ import annotations

from ..registry import register_rule
from ..rule import ModuleFusionRule


def import_deepseek_v4_rules() -> None:
    """Register fusion rules for DeepSeek-V4.

    Most V4-specific classes (Attention, Compressor, Indexer, Gate,
    Expert, MoE, Block) produce non-deterministic op orderings that
    are better handled by YAML rules discovered via ``discover.py``.
    """
    pass
