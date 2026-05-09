"""Load fusion rules from YAML files.

Supports both new (rich) and legacy schemas — see
``docs/fusion_v2_rich_rules_zh.md`` §3 for the contract.

A single YAML entry with multiple ``observed_op_seqs`` alternatives is
expanded into one ``ModuleFusionRule`` per alternative (each rule keeps
the same target_class / io_roles / shape_derivation / etc.).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from .registry import register_rule
from .rule import ModuleFusionRule

logger = logging.getLogger(__name__)


# Default directory for built-in YAML rule files.
_BUILTIN_YAML_DIR = Path(__file__).parent / "rules"
_USER_YAML_DIRS: list[Path] = []


def add_yaml_search_dir(path: Path | str) -> None:
    """Add a directory to the YAML rule search path."""
    _USER_YAML_DIRS.append(Path(path))


def load_yaml_rules(path: Path) -> list[ModuleFusionRule]:
    """Load rules from a single YAML file.

    Returns an empty list when the file does not exist or is empty.
    Malformed entries are skipped with a warning.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("YAML rules file not found: %s", path)
        return []

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return []

    if not isinstance(data, list):
        logger.error("YAML rules file must contain a list at top level: %s", path)
        return []

    rules: list[ModuleFusionRule] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            logger.warning("Skipping non-dict entry %d in %s", i, path)
            continue
        if not entry.get("enabled", True):
            logger.info(
                "Skipping disabled rule: %s",
                entry.get("target_class", f"entry {i}"),
            )
            continue
        try:
            rules.extend(_expand_entry(entry))
        except Exception as e:
            logger.warning(
                "Failed to parse rule entry %d (%s) in %s: %s",
                i, entry.get("target_class", "?"), path, e,
            )

    return rules


def _expand_entry(entry: dict) -> list[ModuleFusionRule]:
    """Expand one YAML entry into one or more rules.

    Multi-sequence ``observed_op_seqs`` is split into N rules.  Otherwise
    one rule is produced.
    """
    raw_seqs = entry.get("observed_op_seqs") or []
    has_match = entry.get("match") is not None

    # New schema or single-sequence legacy → just one rule.
    if has_match or len(raw_seqs) <= 1:
        return [ModuleFusionRule.from_yaml_dict(entry)]

    # Multi-sequence legacy → one rule per sequence.
    rules: list[ModuleFusionRule] = []
    for seq in raw_seqs:
        sub: dict[str, Any] = dict(entry)
        sub["observed_op_seqs"] = [seq]
        rules.append(ModuleFusionRule.from_yaml_dict(sub))
    return rules


def load_yaml_rules_from_dir(directory: Path) -> list[ModuleFusionRule]:
    """Load all .yaml/.yml files from a directory."""
    rules: list[ModuleFusionRule] = []
    for path in sorted(directory.glob("*.yaml")):
        rules.extend(load_yaml_rules(path))
    for path in sorted(directory.glob("*.yml")):
        rules.extend(load_yaml_rules(path))
    return rules


def load_model_yaml_rules(model_id: str) -> None:
    """Load model-specific YAML rules and register them.

    Searches ``rules/`` and any user-registered directories for files
    matching the model_id (exact stem match first, then prefix match).
    """
    model_key = _model_id_to_key(model_id)
    search_dirs = [_BUILTIN_YAML_DIR] + _USER_YAML_DIRS

    seen_files: set[Path] = set()

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Exact match: e.g. deepseek_v4.yaml
        exact = search_dir / f"{model_key}.yaml"
        if exact.exists() and exact not in seen_files:
            seen_files.add(exact)
            for rule in load_yaml_rules(exact):
                register_rule(rule)
            logger.info("Loaded model-specific rules from %s", exact)
            continue

        # Prefix match: e.g. deepseek_v4.yaml when model_key is "deepseek_v4_lite"
        for path in sorted(search_dir.glob("*.yaml")):
            if path in seen_files:
                continue
            stem = path.stem.replace("_", "").replace("-", "")
            prefix = model_key.split("_")[0].replace("-", "")
            if prefix and stem.startswith(prefix):
                seen_files.add(path)
                for rule in load_yaml_rules(path):
                    register_rule(rule)
                logger.info("Loaded model-specific rules from %s", path)


def _model_id_to_key(model_id: str) -> str:
    """Convert ``"deepseek-ai/DeepSeek-V4"`` → ``"deepseek_v4"``."""
    mid = model_id.lower()
    if "/" in mid:
        mid = mid.rsplit("/", 1)[-1]
    key = mid.replace("-", "_")
    noise = {"ai", "hf", "hub", "models", "instruct", "chat", "base"}
    parts = [p for p in key.split("_") if p and p not in noise]
    return "_".join(parts) if parts else key
