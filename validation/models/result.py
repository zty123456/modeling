"""
验证结果数据模型
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PredictionResult:
    """预测结果"""
    scenario_id: str
    model_name: str
    hardware: str

    # 内存预测
    predicted_weights_mb: Optional[float] = None
    predicted_kv_cache_mb: Optional[float] = None
    predicted_total_memory_mb: Optional[float] = None
    memory_feasible: Optional[bool] = None

    # 通信延迟预测 (microseconds)
    predicted_comm_latency_us: Optional[float] = None

    # 计算时间估算 (milliseconds)
    predicted_compute_time_ms: Optional[float] = None

    # 推理延迟预测（综合）
    predicted_latency_ms: Optional[float] = None
    predicted_throughput_tok_s: Optional[float] = None

    # 实测值
    measured_latency_ms: Optional[float] = None
    measured_throughput_tok_s: Optional[float] = None

    # 误差分析
    throughput_error_pct: Optional[float] = None
    is_accurate: bool = field(default=False)

    def error_message(self) -> str:
        if not self.is_accurate:
            return f"Throughput error: {self.throughput_error_pct:.1f}%"
        return "PASS"
