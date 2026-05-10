"""Deprecated backward-compat shim for ``python.zrt.transform.fusion.platforms``.

Step-4: the legacy ``registry/platforms/`` Python rules and the
``import_builtin_rules`` Python registration path were removed in
favour of a YAML-only registry.  ``load_platform_rules(model_id)`` is
preserved for one release cycle as a forwarder to the new
:func:`python.zrt.transform.fusion.loading.initialize_rules`.
"""
from __future__ import annotations

import warnings

from .loading import initialize_rules


def load_platform_rules(model_id: str = "") -> None:
    """Deprecated; use :func:`initialize_rules` instead.

    Kept so existing callers (CLI / tests) keep working while the
    transition completes.
    """
    warnings.warn(
        "load_platform_rules is deprecated; use "
        "python.zrt.transform.fusion.loading.initialize_rules instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    initialize_rules(model_id)


__all__ = ["load_platform_rules"]
