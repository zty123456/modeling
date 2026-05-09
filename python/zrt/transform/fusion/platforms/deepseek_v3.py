"""DeepSeek-V3 fusion rules.

Registers string-keyed rules for DeepSeek-V3 custom classes.
Rules without verified ``op_sequences`` are omitted — the algorithm
only fuses groups that match a registered op sequence.
"""
from __future__ import annotations

from ..registry import register_rule
from ..rule import ModuleFusionRule


def import_deepseek_v3_rules() -> None:
    """Register fusion rules for DeepSeek-V3.

    NOTE: DeepSeek-V3's Linear/ColumnParallel/RowParallel decompose to
    a single aten op (after SKIP_OPS filtering), so no fusion rule is
    needed.  Complex modules (MoE, Block, etc.) require model-specific
    op sequences which should be provided via YAML rules.
    """
    # Placeholder — model-specific YAML rules handle the actual sequences.
    # As op sequences are verified for V3-specific classes, add them here
    # or in a deepseek_v3.yaml file.
    pass
