"""Backward-compatibility shim.

FusionEngine has been replaced by FusionPass.fuse_records().
This module re-exports FusionSpec and provides a thin FusionEngine wrapper
so existing callers (graph/main.py, report/excel_writer.py) continue to work.
"""
from __future__ import annotations

from typing import Any, Dict, List

from python.zrt.transform.fusion._dict_bridge import FusionSpec


class FusionEngine:
    """Thin adapter: delegates to FusionPass.fuse_records().

    All fusion logic now lives in ``python.zrt.transform.fusion.core``.
    This class exists only for backward compatibility with existing callers.
    """

    def __init__(self, tracker: Any, platform: str = "generic",
                 debug: bool = False, max_leaf_ops: int = 0):
        self._tracker = tracker
        self._platform = platform
        self._debug = debug
        self._max_leaf_ops = max_leaf_ops

    def fuse(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from python.zrt.transform.fusion.pass_ import FusionPass
        return FusionPass.fuse_records(
            records, self._tracker,
            platform=self._platform,
            max_leaf_ops=self._max_leaf_ops,
            keep_children=False,
            debug=self._debug,
        )

    def fuse_keep_children(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        from python.zrt.transform.fusion.pass_ import FusionPass
        return FusionPass.fuse_records(
            records, self._tracker,
            platform=self._platform,
            max_leaf_ops=self._max_leaf_ops,
            keep_children=True,
            debug=self._debug,
        )

    def extract_specs(
        self, fused: List[Dict[str, Any]]
    ) -> List[FusionSpec]:
        from python.zrt.transform.fusion._dict_bridge import extract_fusion_specs
        return extract_fusion_specs(fused)
