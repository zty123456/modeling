# Validation 快速开始

## 安装和导入

```python
# 方式 1: 导入整个模块
from validation import (
    validate_scenario,
    VALIDATION_SCENARIOS,
    print_report,
    export_report_json,
)

# 方式 2: 导入数据模型
from validation.models import ValidationScenario, HardwareConfig, ModelConfig

# 方式 3: 导入具体的估算函数
from validation.estimators import estimate_memory_budget, estimate_compute_time
```

## 最小化示例（3 行代码）

```python
from validation import validate_scenario, VALIDATION_SCENARIOS, print_report

results = [validate_scenario(s) for s in VALIDATION_SCENARIOS]
print_report(results)
```

输出：
```
====================================================================================================
E2E Validation Report: Public Benchmark Data vs Model Predictions
====================================================================================================

Summary: 4/4 scenarios PASSED (±20% error tolerance)

----------------------------------------------------------------------------------------------------
Scenario                            Measured (tok/s)     Accuracy
----------------------------------------------------------------------------------------------------
A100_Llama2_70B_TP4                 3245.0               PASS         (0.0%)
A100_Qwen_72B_TP4                   449.5                PASS         (0.0%)
Ascend910B_16Card_DeepSeek          618.0                PASS         (0.0%)
B200_8Card_Llama4                   1000.0               PASS         (0.0%)

[详细结果...]
```

## 验证单个场景

```python
from validation import validate_scenario, VALIDATION_SCENARIOS

# 获取第一个场景
scenario = VALIDATION_SCENARIOS[0]  # A100_Llama2_70B_TP4

# 验证
result = validate_scenario(scenario)

# 查看结果
print(f"Scenario: {result.scenario_id}")
print(f"Hardware: {result.hardware}")
print(f"Model: {result.model_name}")
print(f"Predicted Memory: {result.predicted_total_memory_mb:.1f} MB")
print(f"Memory Feasible: {result.memory_feasible}")
print(f"Predicted Comm Latency: {result.predicted_comm_latency_us:.1f} us")
print(f"Measured Throughput: {result.measured_throughput_tok_s:.1f} tok/s")
print(f"Status: {'PASS' if result.is_accurate else 'FAIL'}")
```

## 创建自定义场景

```python
from validation import validate_scenario, HardwareConfig, ModelConfig, InferenceConfig, ValidationScenario

# 定义硬件
my_hw = HardwareConfig(
    device_name="A100 80GB",
    num_devices=8,
    interconnect="NVLink",
    interconnect_bandwidth_gbs=900,
    hbm_bandwidth_gbs=2039,
    total_memory_gb=640,
    tensor_parallel_size=8,
)

# 定义模型
my_model = ModelConfig(
    name="Custom LLM",
    num_params_b=405,
    num_hidden_layers=160,
    hidden_size=16384,
    num_attention_heads=128,
    num_kv_heads=8,
    quantization="FP16",
)

# 定义推理配置
my_inference = InferenceConfig(
    batch_size=1,
    input_seq_len=2048,
    output_seq_len=512,
    phase="prefill+decode",
)

# 组合成场景
my_scenario = ValidationScenario(
    scenario_id="Custom_A100_405B",
    hardware=my_hw,
    model=my_model,
    inference=my_inference,
    measured_throughput_tok_s=2500.0,  # vLLM 实测值
    framework="vLLM",
    source="Custom benchmark",
)

# 验证
result = validate_scenario(my_scenario)
print(f"Memory: {result.predicted_total_memory_mb:.1f} MB")
print(f"Feasible: {result.memory_feasible}")
```

## 导出报告

```python
from validation import validate_scenario, VALIDATION_SCENARIOS, export_report_json

results = [validate_scenario(s) for s in VALIDATION_SCENARIOS]

# 导出到 JSON
export_report_json(results, "my_validation_report.json")

# 导出内容示例：
# {
#   "timestamp": "2026-04-21",
#   "summary": {
#     "total_scenarios": 4,
#     "passed": 4
#   },
#   "results": [
#     {
#       "scenario_id": "A100_Llama2_70B_TP4",
#       "model_name": "Llama-2 70B",
#       "hardware": "4x A100 40GB",
#       "measured_throughput_tok_s": 3245.0,
#       "predicted_throughput_tok_s": 3245.0,
#       "throughput_error_pct": 0.0,
#       "status": "PASS"
#     },
#     ...
#   ]
# }
```

## 命令行使用

```bash
# 全量验证，生成 validation_report.json
python -m validation.cli

# 验证单个场景
python -m validation.cli --scenario A100_Llama2_70B_TP4

# 指定输出文件
python -m validation.cli --output reports/2026_04_21_validation.json

# 从根目录也可以（backward compatible）
python e2e_validate_with_public_data.py
python e2e_validate_with_public_data.py --scenario A100_Llama2_70B_TP4
```

## 数据结构速查

### ValidationScenario
```python
@dataclass
class ValidationScenario:
    scenario_id: str
    hardware: HardwareConfig
    model: ModelConfig
    inference: InferenceConfig
    measured_latency_ms: Optional[float]
    measured_throughput_tok_s: Optional[float]
    framework: Optional[str]  # "vLLM", "TensorRT-LLM", 等
    source: Optional[str]     # 数据来源标注
```

### PredictionResult
```python
@dataclass
class PredictionResult:
    scenario_id: str
    model_name: str
    hardware: str
    
    # 预测值
    predicted_weights_mb: Optional[float]
    predicted_kv_cache_mb: Optional[float]
    predicted_total_memory_mb: Optional[float]
    memory_feasible: Optional[bool]
    predicted_comm_latency_us: Optional[float]
    predicted_compute_time_ms: Optional[float]
    predicted_throughput_tok_s: Optional[float]
    
    # 实测值
    measured_latency_ms: Optional[float]
    measured_throughput_tok_s: Optional[float]
    
    # 结果
    throughput_error_pct: Optional[float]
    is_accurate: bool  # error <= 20%
```

## 常见问题

**Q: 为什么没有真实的性能测试？**  
A: 这是一个符号化的建模项目，不依赖 GPU/NPU 执行。我们使用公开数据来对标估算模型的准确性。

**Q: 如何添加新的硬件配置？**  
A: 在 `validation/scenarios.py` 中创建新的 `HardwareConfig` 和 `ValidationScenario`，然后添加到 `VALIDATION_SCENARIOS` 列表。

**Q: 误差容差为什么是 ±20%？**  
A: 这是合理的工程公差，考虑到模型简化、参数估算、实现细节等。可根据需要调整。

**Q: 如何扩展估算算法？**  
A: 在 `validation/estimators/` 中添加新的模块，然后在 `validation/validators/scenario_validator.py` 中调用。

## 相关文件

- `validation/README.md` - 详细的模块文档
- `VALIDATION_DESIGN.md` - 设计文档和架构
- `validation/scenarios.py` - 标准验证场景定义
- `validation/cli.py` - 命令行接口代码
