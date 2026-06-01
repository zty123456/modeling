"""Pipeline Parallel stage partition strategies.

This module provides unified partition logic for PP stage assignment,
supporting:
- Uniform distribution (fallback when no cost data)
- Greedy bin-packing (default when typical_costs available)
- VPP interleaved (when vpp_chunks > 1 and pp_schedule supports VPP)
- Explicit assignment (user-provided pp_layer_assignment)

Single data source: typical_costs
- Typical layer granularity: {LayerType: cost} → shared cost per layer type
- Full layer granularity: {layer_idx: cost} → independent cost per layer

Usage::

    from python.zrt.graph.partition import partition_layers_by_strategy
    from python.zrt.graph.layer_strategy import LayerProfile, infer_layer_profile
    
    profile = infer_layer_profile(config)
    
    # Typical layer granularity (LayerType keys)
    typical_costs = {LayerType.HCA_HASH: 1250.0, LayerType.HCA_TOPK: 980.0}
    assignment = partition_layers_by_strategy(profile, typical_costs, pp=4)
    
    # Full layer granularity (PipelineParallelPass path)
    typical_costs = {0: 1250.0, 1: 1238.0, 2: 1262.0}
    assignment = partition_layers_by_strategy(profile, typical_costs, pp=4)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from python.zrt.graph.layer_strategy import LayerProfile, LayerType

logger = logging.getLogger(__name__)


def partition_layers_by_strategy(
    layer_profile: LayerProfile,
    typical_costs: Optional[Dict[Any, float]],
    pp: int,
    pp_schedule: str = "1f1b",
    vpp_chunks: int = 1,
    pp_layer_assignment: Optional[List[int]] = None,
) -> List[int]:
    """根据划分策略生成 layer_assignment.
    
    单一数据来源: typical_costs
      - 典型层粒度: {LayerType: cost_per_layer}
        → 同类型层共享相同成本
        → 来自典型层捕获 (7个典型层)
      
      - 全量层粒度: {layer_idx: cost_per_layer}
        → 每层独立成本
        → 来自全量捕获或 DAGScheduler annotation
    
    策略推断优先级:
      1. 用户显式指定 (pp_layer_assignment) → 最高优先级
      2. VPP interleaved (vpp_chunks > 1 + interleaved/dualpipev) → interleaved
      3. typical_costs 可用 → 贪婪 bin-packing
      4. typical_costs 不可用 → 均匀分布 (fallback)
    
    Args:
        layer_profile: 完整层结构 (total_layers, layer_types)
        typical_costs: 成本数据，支持两种格式
            - {LayerType: cost}: 典型层粒度
            - {layer_idx: cost}: 全量层粒度
            - None 或 {}: 触发 fallback 到均匀分布
        pp: Pipeline 并行度
        pp_schedule: PP schedule 类型 (1f1b/interleaved/dualpipe/dualpipev/zb)
        vpp_chunks: Virtual pipeline chunks (VPP)
        pp_layer_assignment: 用户显式层到 stage 映射
    
    Returns:
        layer_assignment: [stage_id for each layer], 镜度 = total_layers
    """
    total = layer_profile.total_layers
    
    # 1. 用户显式指定 (最高优先级)
    if pp_layer_assignment and len(pp_layer_assignment) == total:
        return [max(0, min(s, pp - 1)) for s in pp_layer_assignment]
    
    # 2. VPP interleaved
    is_vpp = vpp_chunks > 1 and pp_schedule in ("interleaved", "i1f1b", "dualpipev")
    if is_vpp:
        return _interleaved_partition(layer_profile, pp, vpp_chunks)
    
    # 3. typical_costs 可用 → 贪婪 bin-packing
    # 注意: 空字典 {} 被判断为"不可用"
    if typical_costs:
        return _greedy_partition(layer_profile, typical_costs, pp)
    
    # 4. fallback → 均匀分布
    return _uniform_partition(layer_profile, pp)


def _uniform_partition(profile: LayerProfile, pp: int) -> List[int]:
    """连续块分布（fallback）.
    
    触发条件: typical_costs 不可用
    
    Example: 61 layers, pp=4
      stage 0: layer 0-15   (16 layers)
      stage 1: layer 16-30  (15 layers)
      stage 2: layer 31-45  (15 layers)
      stage 3: layer 46-60  (15 layers)
    
    Returns: [0,0,0,...,0, 1,1,1,...,1, 2,2,2,...,2, 3,3,3,...,3]
    """
    total = profile.total_layers
    stage_size = max(1, total // pp)
    return [min(i // stage_size, pp - 1) for i in range(total)]


def _greedy_partition(
    profile: LayerProfile,
    typical_costs: Dict[Any, float],
    pp: int,
) -> List[int]:
    """按负载均衡（默认策略）.
    
    触发条件: typical_costs 可用
    
    自动识别数据格式:
      - {LayerType: cost}: 按 layer_type 查找
      - {layer_idx: cost}: 按 layer_idx 查找
    
    Returns: 可能不连续（按负载均衡）
    """
    layer_assignment: List[int] = []
    stage_load = [0.0] * pp
    
    # 自动识别数据格式
    is_by_layer_type = any(isinstance(k, LayerType) for k in typical_costs.keys())
    
    for layer_idx, layer_type in enumerate(profile.layer_types):
        if is_by_layer_type:
            load = typical_costs.get(layer_type, 0.0)
        else:
            load = typical_costs.get(layer_idx, 0.0)
        
        min_stage = int(min(range(pp), key=lambda i: stage_load[i]))
        layer_assignment.append(min_stage)
        stage_load[min_stage] += load
    
    return layer_assignment


def _interleaved_partition(
    profile: LayerProfile,
    pp: int,
    vpp_chunks: int,
) -> List[int]:
    """Virtual pipeline parallel 的 round-robin 分布.
    
    触发条件: vpp_chunks > 1 + pp_schedule in ("interleaved", "dualpipev")
    
    Example: 61 layers, pp=2, vpp_chunks=2
      total_chunks = 2 * 2 = 4
      Device 0: chunk 0 + chunk 2 → [L0, L1, L4, L5, L8, L9, ...]
      Device 1: chunk 1 + chunk 3 → [L2, L3, L6, L7, L10, L11, ...]
    
    Returns: 交错分布 [0, 0, 1, 1, 0, 0, 1, 1, ...]
    """
    total = profile.total_layers
    total_chunks = pp * vpp_chunks
    layers_per_chunk = max(1, total // total_chunks)
    
    layer_assignment: List[int] = []
    for idx in range(total):
        chunk_id = min(idx // layers_per_chunk, total_chunks - 1)
        stage = chunk_id % pp
        layer_assignment.append(stage)
    
    return layer_assignment
