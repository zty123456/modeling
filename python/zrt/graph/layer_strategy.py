"""Layer structure inference and typical layer selection strategy.

This module provides automatic layer type inference from model config,
minimal typical layer selection for efficient capture, and scaling
logic to estimate full model costs from captured typical layers.

Supports:
- Dense models (Llama, Qwen, Mistral)
- Pure MoE models (Mixtral)
- Hybrid MoE models (DeepSeek-V3/V3.2)
- Compressed attention models (DeepSeek-V4-pro/flash)

Usage::

    from python.zrt.graph.layer_strategy import infer_layer_profile, scale_layer_costs
    
    # Infer layer structure from config
    profile = infer_layer_profile(config)
    
    # Typical layers to capture
    typical_indices = profile.typical_indices  # e.g., [0, 3] for DSV3
    
    # After capture, scale costs to full model
    layer_costs = {0: 1.5, 3: 2.3}  # captured per-layer costs
    scaled = scale_layer_costs(layer_costs, profile)
    # scaled = {"dense": 4.5, "moe": 133.4, "total": 137.9}
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """Layer attention/compression type for strategy selection."""
    DENSE = "dense"
    MOE = "moe"
    HCA_HASH = "hca_hash"   # HCA with hash routing (layer_id < num_hash_layers)
    HCA_TOPK = "hca_topk"   # HCA with indexer topk (layer_id >= num_hash_layers)
    CSA_HASH = "csa_hash"   # CSA with hash routing (layer_id < num_hash_layers)
    CSA_TOPK = "csa_topk"   # CSA with indexer topk (layer_id >= num_hash_layers)
    SWA_HASH = "swa_hash"   # SWA with hash routing (layer_id < num_hash_layers)
    SWA_TOPK = "swa_topk"   # SWA with indexer topk (layer_id >= num_hash_layers)
    HCA = "hca"             # Legacy alias
    CSA = "csa"             # Legacy alias
    SWA = "swa"             # Legacy alias


@dataclass
class LayerProfile:
    """Full layer composition inferred from model config."""
    layer_types: List[LayerType]
    typical_indices: List[int]
    
    num_dense: int = 0
    num_moe: int = 0
    num_hca_hash: int = 0   # Hash-routing HCA layers
    num_hca_topk: int = 0   # Topk-indexer HCA layers
    num_hca: int = 0        # Legacy: sum of hash + topk
    num_csa_hash: int = 0   # Hash-routing CSA layers
    num_csa_topk: int = 0   # Topk-indexer CSA layers
    num_csa: int = 0        # Legacy: sum of hash + topk
    num_swa_hash: int = 0   # Hash-routing SWA layers
    num_swa_topk: int = 0   # Topk-indexer SWA layers
    num_swa: int = 0        # Legacy: sum of hash + topk
    
    @property
    def total_layers(self) -> int:
        return len(self.layer_types)
    
    


def infer_layer_profile(config: Any) -> LayerProfile:
    """
    Infer complete layer structure from model config.
    
    Uses 4-tier priority logic:
    1. V4 style (compress_ratios array)
    2. DSV3/V3.2 style (first_k_dense_replace + moe_layer_freq)
    3. Pure MoE (num_local_experts / n_routed_experts)
    4. Default dense
    
    Args:
        config: PretrainedConfig from model
    
    Returns:
        LayerProfile with full layer_types and typical_indices
    """
    compress_ratios = getattr(config, "compress_ratios", None)
    if compress_ratios is not None and len(compress_ratios) > 0:
        return _infer_v4_profile(config, compress_ratios)
    
    if getattr(config, "first_k_dense_replace", None) is not None:
        return _infer_dsv3_profile(config)
    
    if getattr(config, "num_local_experts", None) or getattr(config, "n_routed_experts", None):
        return _infer_pure_moe_profile(config)
    
    return _infer_dense_profile(config)


def _get_full_num_hidden_layers(config: Any) -> int:
    """Get full model layer count, preferring saved value over current truncated value."""
    return getattr(config, "_full_num_hidden_layers", None) or getattr(config, "num_hidden_layers", 0)


def _infer_v4_profile(config: Any, compress_ratios: List[int]) -> LayerProfile:
    """
    Infer layer profile for DeepSeek-V4 style models.
    
    compress_ratios array mapping:
    - 128 → HCA (High-Compression Attention)
    - 4   → CSA (Compressed Sparse Attention)
    - 0   → SWA (Sliding-Window)
    
    num_hash_layers applies to ALL layers (including SWA):
    - layer_id < num_hash_layers → hash routing
    - layer_id >= num_hash_layers → indexer topk
    
    This produces 6 layer types:
    - HCA_HASH / HCA_TOPK: HCA with hash or topk routing
    - CSA_HASH / CSA_TOPK: CSA with hash or topk routing
    - SWA_HASH / SWA_TOPK: SWA with hash or topk routing
    
    Typical layer selection (up to 7 layers):
    - First of each type (HCA_HASH, HCA_TOPK, CSA_HASH, CSA_TOPK, SWA_HASH, SWA_TOPK)
    
    Note: MTP (Multi-Token Prediction) layer is NOT included.
    """
    num_hidden = _get_full_num_hidden_layers(config)
    num_hash_layers = getattr(config, "num_hash_layers", 0) or 0
    
    # Truncate compress_ratios to num_hidden_layers if longer (ignore MTP)
    if len(compress_ratios) > num_hidden:
        logger.info(
            "Ignoring last %d elements of compress_ratios (MTP layers)",
            len(compress_ratios) - num_hidden,
        )
        compress_ratios = compress_ratios[:num_hidden]
    
    if len(compress_ratios) != num_hidden:
        logger.warning(
            "compress_ratios length (%d) != num_hidden_layers (%d)",
            len(compress_ratios), num_hidden
        )
    
    # Build layer_types: hash/topk based on layer_id, not on compression status
    layer_types: List[LayerType] = []
    for i, ratio in enumerate(compress_ratios):
        use_hash = i < num_hash_layers
        if ratio == 128:
            layer_types.append(LayerType.HCA_HASH if use_hash else LayerType.HCA_TOPK)
        elif ratio == 4:
            layer_types.append(LayerType.CSA_HASH if use_hash else LayerType.CSA_TOPK)
        elif ratio == 0:
            layer_types.append(LayerType.SWA_HASH if use_hash else LayerType.SWA_TOPK)
        else:
            logger.warning(
                "Unknown compress_ratio %d at layer %d, treating as SWA",
                ratio, i
            )
            layer_types.append(LayerType.SWA_HASH if use_hash else LayerType.SWA_TOPK)
    
    counts = Counter(layer_types)
    
    logger.info(
        "DSV4 profile: hca_hash=%d, hca_topk=%d, csa_hash=%d, csa_topk=%d, "
        "swa_hash=%d, swa_topk=%d, typical_indices=%s",
        counts[LayerType.HCA_HASH], counts[LayerType.HCA_TOPK],
        counts[LayerType.CSA_HASH], counts[LayerType.CSA_TOPK],
        counts[LayerType.SWA_HASH], counts[LayerType.SWA_TOPK],
        _select_first_of_each_type(layer_types),
    )
    
    return LayerProfile(
        layer_types=layer_types,
        typical_indices=_select_first_of_each_type(layer_types),
        num_hca_hash=counts[LayerType.HCA_HASH],
        num_hca_topk=counts[LayerType.HCA_TOPK],
        num_hca=counts[LayerType.HCA_HASH] + counts[LayerType.HCA_TOPK],
        num_csa_hash=counts[LayerType.CSA_HASH],
        num_csa_topk=counts[LayerType.CSA_TOPK],
        num_csa=counts[LayerType.CSA_HASH] + counts[LayerType.CSA_TOPK],
        num_swa_hash=counts[LayerType.SWA_HASH],
        num_swa_topk=counts[LayerType.SWA_TOPK],
        num_swa=counts[LayerType.SWA_HASH] + counts[LayerType.SWA_TOPK],
    )


def _infer_dsv3_profile(config: Any) -> LayerProfile:
    """
    Infer layer profile for DeepSeek-V3/V3.2 style models.
    
    - layers [0, first_k_dense_replace) → DENSE
    - layers >= first_k_dense_replace and (i - first_k) % moe_layer_freq == 0 → MOE
    - others → DENSE
    
    Note: moe_layer_freq counting starts from first_k_dense_replace, not from 0.
          When freq=2: Layer first_k, first_k+2, first_k+4, ... are MoE.
    """
    total = _get_full_num_hidden_layers(config)
    first_k = getattr(config, "first_k_dense_replace", 0) or 0
    freq = getattr(config, "moe_layer_freq", 1) or 1
    
    layer_types: List[LayerType] = []
    for i in range(total):
        if i < first_k:
            layer_types.append(LayerType.DENSE)
        elif (i - first_k) % freq == 0:
            layer_types.append(LayerType.MOE)
        else:
            layer_types.append(LayerType.DENSE)
    
    counts = Counter(layer_types)
    return LayerProfile(
        layer_types=layer_types,
        typical_indices=_select_first_of_each_type(layer_types),
        num_dense=counts[LayerType.DENSE],
        num_moe=counts[LayerType.MOE],
    )


def _infer_pure_moe_profile(config: Any) -> LayerProfile:
    """
    Infer layer profile for pure MoE models (e.g., Mixtral).
    
    All layers are MOE (no dense prefix layers).
    """
    total = _get_full_num_hidden_layers(config)
    layer_types = [LayerType.MOE] * total
    return LayerProfile(
        layer_types=layer_types,
        typical_indices=[0],
        num_moe=total,
    )


def _infer_dense_profile(config: Any) -> LayerProfile:
    """
    Infer layer profile for standard dense models (Llama, Qwen, Mistral).
    
    All layers are DENSE.
    """
    total = _get_full_num_hidden_layers(config)
    layer_types = [LayerType.DENSE] * total
    return LayerProfile(
        layer_types=layer_types,
        typical_indices=[0],
        num_dense=total,
    )


def _select_first_of_each_type(layer_types: List[LayerType]) -> List[int]:
    """
    Select first occurrence of each layer type.
    
    Examples:
        [DENSE, DENSE, MOE, MOE] → [0, 2]
        [HCA, HCA, CSA, HCA] → [0, 2]
        [SWA, SWA, CSA, HCA] → [0, 2, 3]
    """
    seen: set[LayerType] = set()
    indices: List[int] = []
    for i, layer_type in enumerate(layer_types):
        if layer_type not in seen:
            seen.add(layer_type)
            indices.append(i)
    return indices


def scale_layer_costs(
    layer_costs: Dict[int, float],
    profile: LayerProfile,
) -> Dict[str, float]:
    """
    Scale captured layer costs to full model using linear multiplication.
    
    Args:
        layer_costs: {layer_idx: captured_cost} from typical layer capture
        profile: LayerProfile with layer type counts
    
    Returns:
        {"dense": ..., "moe": ..., "hca_hash": ..., "hca_topk": ..., 
         "csa_hash": ..., "csa_topk": ..., "swa_hash": ..., "swa_topk": ..., "total": ...}
    """
    type_to_cost: Dict[LayerType, float] = {}
    for idx in profile.typical_indices:
        if idx < len(profile.layer_types):
            layer_type = profile.layer_types[idx]
            type_to_cost[layer_type] = layer_costs.get(idx, 0.0)
    
    scaled: Dict[str, float] = {}
    for attr in ["dense", "moe", "hca_hash", "hca_topk", "csa_hash", "csa_topk", "swa_hash", "swa_topk"]:
        count = getattr(profile, f"num_{attr}", 0)
        if count > 0:
            # Map attr name to LayerType (e.g., "hca_hash" -> LayerType.HCA_HASH)
            layer_type_name = attr.upper()
            layer_type = getattr(LayerType, layer_type_name, None)
            if layer_type:
                cost_per_layer = type_to_cost.get(layer_type, 0.0)
                scaled[attr] = cost_per_layer * count
    
    scaled["total"] = sum(scaled.values())
    return scaled