# Validation 模块 - 性能精度对标

端到端验证我们的性能预测模型（CommLatencyPass + MemoryModel）在多卡推理场景下的准确度，对标开源框架公开数据。

## 目录结构

```
validation/
├── models/              # 数据模型（场景、硬件、模型配置、结果）
├── estimators/          # 估算模块（内存、计算时间、通信延迟）
├── validators/          # 验证逻辑（场景验证）
├── reporters/           # 报告生成（打印、导出 JSON）
├── scenarios.py         # 标准验证场景集合
├── cli.py              # 命令行接口
├── __init__.py         # 公开 API
└── baselines/          # 参考数据库（待补充）
```

## 使用方法

### 1. Python API

```python
from validation import validate_scenario, print_report, VALIDATION_SCENARIOS

# 验证所有场景
results = [validate_scenario(s) for s in VALIDATION_SCENARIOS]

# 打印报告
print_report(results)

# 导出 JSON 报告
from validation import export_report_json
export_report_json(results, "my_report.json")
```

### 2. 命令行

```bash
# 验证所有场景并生成报告
python -m validation.cli

# 验证单个场景
python -m validation.cli --scenario A100_Llama2_70B_TP4

# 指定输出文件
python -m validation.cli --output my_report.json
```

## 数据模型

### ValidationScenario
完整的验证场景，包括：
- `hardware`: HardwareConfig（GPU/NPU 配置）
- `model`: ModelConfig（模型参数）
- `inference`: InferenceConfig（推理配置）
- `measured_*`: 实测吞吐/延迟

### PredictionResult
验证结果，包括：
- 预测的内存、通信延迟、计算时间
- 实测值与预测值的误差
- `is_accurate`: 是否通过验证（误差 ≤20%）

## 估算模块

| 模块 | 功能 |
|-----|------|
| `memory.py` | 估算权重、KV Cache、总内存占用 |
| `compute_time.py` | 基于硬件 FLOPS 和参数量估算计算时间 |
| `comm_latency.py` | 估算 AllReduce 通信延迟 |

## 标准验证场景

1. **A100_Llama2_70B_TP4** - Llama2 70B，4 张 A100 40GB，TP=4
2. **A100_Qwen_72B_TP4** - Qwen 72B，4 张 A100 40GB，TP=4
3. **Ascend910B_16Card_DeepSeek** - DeepSeek-R1，16 张 910B，TP=8
4. **B200_8Card_Llama4** - Llama4，8 张 B200，TP=8

## 扩展方向

- [ ] 添加更多公开基准数据（MLPerf、SPEC）
- [ ] 支持量化效果对标（FP8 vs FP16）
- [ ] MoE 模型的专家分布验证
- [ ] 集成到 CI/CD 流水线
