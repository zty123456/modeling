"""
验证场景的数据模型定义
"""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class HardwareConfig:
    """硬件配置"""
    device_name: str  # "A100", "B200", "Ascend 910C"
    num_devices: int
    interconnect: str  # "NVLink", "HCCS", "RoCE"
    interconnect_bandwidth_gbs: float
    hbm_bandwidth_gbs: float  # 单卡 HBM 带宽
    total_memory_gb: int
    tensor_parallel_size: int


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    num_params_b: int
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    quantization: str  # "FP16", "FP8", "NVFP4", "W8A8"
    is_moe: bool = False
    num_experts: Optional[int] = None


@dataclass
class InferenceConfig:
    """推理配置"""
    batch_size: int
    input_seq_len: int
    output_seq_len: int
    phase: str = "prefill"  # "prefill" | "decode" | "both"


@dataclass
class ValidationScenario:
    """完整验证场景"""
    scenario_id: str
    hardware: HardwareConfig
    model: ModelConfig
    inference: InferenceConfig
    measured_latency_ms: Optional[float] = None
    measured_throughput_tok_s: Optional[float] = None
    framework: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self):
        return asdict(self)
