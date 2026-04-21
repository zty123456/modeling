"""
计算时间估算
"""
from validation.models import ValidationScenario


def estimate_compute_time(scenario: ValidationScenario, throughput_multiplier: float = 1.0) -> float:
    """估算单个 token 的计算时间 (ms)"""
    # 简化模型: 基于吞吐量反推计算时间
    # 计算时间 = 1 / throughput (受硬件能力限制)

    hw = scenario.hardware
    model = scenario.model
    tp = hw.tensor_parallel_size

    # 估算峰值 FLOPs（简化）
    # 每个 token 需要 ~6 * num_params * 2 (FLOPs / token)
    # 但是要除以 TP （并行分片）
    flops_per_token = 6 * model.num_params_b * 1e9 * 2 / tp

    # 硬件 FLOPS（估算）
    hw_flops_map = {
        "A100 40GB": 312e12,  # FP16 FLOPS
        "A100 80GB": 312e12,
        "Ascend 910B": 512e12,  # 估算
        "Ascend 910C": 640e12,  # 估算
        "B200 Blackwell": 1440e12,  # 估算
    }

    hw_flops = hw_flops_map.get(scenario.hardware.device_name, 312e12)

    # 计算时间: flops_per_token / hw_flops (seconds)
    compute_time_s = flops_per_token / (hw_flops * throughput_multiplier)
    compute_time_ms = compute_time_s * 1000

    return max(0.001, compute_time_ms)
