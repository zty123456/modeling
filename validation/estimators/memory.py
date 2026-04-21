"""
内存预算估算
"""
from typing import Optional
from python.zrt.memory.budget import MemoryBudget
from validation.models import ValidationScenario


def estimate_memory_budget(scenario: ValidationScenario) -> Optional[MemoryBudget]:
    """根据验证场景估算内存预算"""
    try:
        model = scenario.model
        hw = scenario.hardware

        # 权重: (params) / (TP) 如果有分片
        tp = hw.tensor_parallel_size
        params_per_card = model.num_params_b * 1e9 / tp

        # 考虑量化 (字节数/参数)
        quantization_factor = {
            "FP16": 2.0,
            "FP8": 1.0,
            "NVFP4": 0.5,
            "W8A8": 1.0,
        }.get(model.quantization, 2.0)

        weights_bytes = params_per_card * quantization_factor
        weights_mb = weights_bytes / 1e6

        # KV Cache: (batch_size * seq_len * num_layers * hidden_size * 2) / TP * quantization
        batch = scenario.inference.batch_size
        seq_len = scenario.inference.input_seq_len + scenario.inference.output_seq_len
        kv_cache_bytes = (batch * seq_len * model.num_hidden_layers * model.hidden_size * 2
                          * quantization_factor) / tp
        kv_cache_mb = kv_cache_bytes / 1e6

        total_mb = weights_mb + kv_cache_mb
        capacity_mb = hw.total_memory_gb * 1024

        return MemoryBudget(
            weights_mb=weights_mb,
            kv_cache_mb=kv_cache_mb,
            activation_peak_mb=50.0,  # 估算激活
            comm_buffer_mb=10.0,
            framework_overhead_mb=100.0,
            total_mb=total_mb + 160.0,
            capacity_mb=capacity_mb,
            is_feasible=(total_mb + 160.0) < capacity_mb,
        )
    except Exception as e:
        return None
