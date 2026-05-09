"""Llama fusion rules.

Llama's custom ``LlamaRMSNorm`` produces the same op sequence as the
built-in RMSNorm rule (custom variant with ``aten.add.Tensor``), so it
is already covered by the built-in rules registered in ``builtins.py``.

Additional model-specific sequences should be provided via YAML.
"""
from __future__ import annotations


def import_llama_rules() -> None:
    """Register Llama-specific rules.

    LlamaRMSNorm is already matched by the built-in ``"RMSNorm"`` rule
    whose op_sequence uses ``aten.add.Tensor``.  No additional rules
    needed unless Llama introduces new multi-op modules.
    """
    pass
