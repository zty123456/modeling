"""
通信延迟估算
"""
from python.zrt.transform.analysis.comm_latency import _estimate_comm_latency
from validation.models import ValidationScenario


def estimate_comm_latency(scenario: ValidationScenario) -> float:
    """估算通信延迟 (microseconds)"""
    hw = scenario.hardware
    model = scenario.model

    # 梯度通信大小: 权重大小（不分片）
    model_bytes = model.num_params_b * 1e9 * 2.0  # FP16

    # 使用 AllReduce （标准的梯度同步）
    collective = "all_reduce"
    group_size = hw.num_devices

    bandwidth_bps = hw.interconnect_bandwidth_gbs * 1e9 / 8.0
    link_latency_us = 0.1

    latency_us = _estimate_comm_latency(
        collective, group_size, model_bytes, bandwidth_bps, link_latency_us
    )

    return latency_us
