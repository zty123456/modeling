"""Platform-specific fusion rule sets.

Three-layer loading:
    1. Built-in common rules (Python code, verified decompositions)
    2. YAML custom rules (model-specific)
    3. Legacy platform Python rules (model-specific, for models with
       custom classes that have distinct op sequences)
"""
from __future__ import annotations

from ..registry import clear_rules

# Layer 1
from ..builtins import import_builtin_rules


def load_platform_rules(model_id: str = "") -> None:
    """Load all rule layers.

    Layer 1: Built-in decomposition rules (always loaded).
    Layer 2: YAML rules for this model (if available).
    Layer 3: Legacy platform Python rules (model-specific).
    """
    clear_rules()

    # Layer 1: Built-in common rules
    import_builtin_rules()

    # Layer 2: YAML rules (load after builtins so model-specific
    # rules with higher priority can override)
    if model_id:
        from ..yaml_loader import load_model_yaml_rules
        load_model_yaml_rules(model_id)

    # Layer 3: Legacy platform Python rules
    mid = model_id.lower()

    if "deepseek" in mid and ("v4" in mid or "pro" in mid):
        from .deepseek_v4 import import_deepseek_v4_rules
        import_deepseek_v4_rules()
    elif "deepseek" in mid and ("v3" in mid):
        from .deepseek_v3 import import_deepseek_v3_rules
        import_deepseek_v3_rules()
    elif "llama" in mid or "meta-llama" in mid:
        from .llama import import_llama_rules
        import_llama_rules()
    elif "mixtral" in mid:
        from .mixtral import import_mixtral_rules
        import_mixtral_rules()
