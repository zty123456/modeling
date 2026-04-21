"""
验证场景定义（已迁移到 validation 模块）

此文件为 backward-compatible stub，实际代码已移至 validation/scenarios.py
"""

# 转发所有导出到新模块
from validation.models import (
    HardwareConfig,
    ModelConfig,
    InferenceConfig,
    ValidationScenario,
)
from validation.scenarios import (
    SCENARIO_1_A100_LLAMA2_70B_TP4,
    SCENARIO_2_A100_QWEN_72B_TP4,
    SCENARIO_3_ASCEND910B_16CARD_DEEPSEEK,
    SCENARIO_4_B200_8CARD_LLAMA4,
    VALIDATION_SCENARIOS,
    export_scenarios_to_json,
)

__all__ = [
    "HardwareConfig",
    "ModelConfig",
    "InferenceConfig",
    "ValidationScenario",
    "SCENARIO_1_A100_LLAMA2_70B_TP4",
    "SCENARIO_2_A100_QWEN_72B_TP4",
    "SCENARIO_3_ASCEND910B_16CARD_DEEPSEEK",
    "SCENARIO_4_B200_8CARD_LLAMA4",
    "VALIDATION_SCENARIOS",
    "export_scenarios_to_json",
]


if __name__ == "__main__":
    export_scenarios_to_json()

    # Print scenario overview
    print("\n" + "=" * 80)
    print("Validation Scenarios Overview")
    print("=" * 80)
    for scenario in VALIDATION_SCENARIOS:
        print(f"\n[{scenario.scenario_id}]")
        print(f"  Hardware: {scenario.hardware.num_devices}x {scenario.hardware.device_name} ({scenario.hardware.interconnect})")
        print(f"  Model: {scenario.model.name} ({scenario.model.num_params_b}B)")
        print(f"  Config: TP={scenario.hardware.tensor_parallel_size}, batch={scenario.inference.batch_size}")
        if scenario.measured_throughput_tok_s:
            print(f"  Measured: {scenario.measured_throughput_tok_s:.1f} tok/s")
        if scenario.measured_latency_ms:
            print(f"  Measured: {scenario.measured_latency_ms:.1f} ms")
        print(f"  Source: {scenario.source}")
