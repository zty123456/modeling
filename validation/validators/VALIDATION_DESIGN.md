# Validation 模块设计文档

## 概述

`validation/` 模块是一个轻量级的验证框架，用于：
1. **性能精度对标** - 对标公开数据集验证预测模型准确度
2. **多硬件配置支持** - 支持 A100、Ascend 910B、B200 等多种硬件
3. **端到端验证** - 从输入场景到报告生成的完整流程

## 关键设计原则

### 1. 不依赖真实执行
- ✅ 基于 FakeTensorMode 的符号化执行，无需 GPU/NPU
- ✅ 使用公开数据作为 baseline，不下载真实权重
- ✅ 估算模块基于参数化模型（FLOPs、带宽、延迟公式）

### 2. 模块化架构
```
validation/
├── models/        数据结构层（场景、配置、结果）
├── estimators/    估算层（内存、计算、通信）
├── validators/    验证层（场景对标）
├── reporters/     输出层（报告生成）
└── scenarios.py   场景库（标准数据）
```

### 3. 关注点分离
| 模块 | 职责 | 示例 |
|-----|------|------|
| `models/` | 数据结构定义 | ValidationScenario, HardwareConfig |
| `estimators/` | 计算和估算 | estimate_memory_budget(), estimate_compute_time() |
| `validators/` | 验证逻辑 | validate_scenario() - 调用所有 estimator |
| `reporters/` | 输出格式 | print_report(), export_report_json() |
| `scenarios.py` | 标准数据库 | VALIDATION_SCENARIOS = [...] |

## 文件清单

### 数据模型层 (`models/`)

**scenario.py** - 验证场景配置
```python
HardwareConfig    # GPU/NPU + 互连配置
ModelConfig       # 模型参数（参数量、层数、head 数等）
InferenceConfig   # 推理配置（batch、seq_len、phase）
ValidationScenario # 场景 = hardware + model + inference + measured_data
```

**result.py** - 验证结果
```python
PredictionResult  # 预测值 + 实测值 + 误差分析 + 通过/失败
```

### 估算层 (`estimators/`)

**memory.py** - `estimate_memory_budget(scenario) -> MemoryBudget`
```
估算内容:
  - weights_mb: (num_params / TP) * quantization_bytes
  - kv_cache_mb: (batch * seq_len * num_layers * hidden_size * 2) / TP * quantization
  - total_mb: weights + cache + overhead
  - is_feasible: total <= gpu_memory
```

**compute_time.py** - `estimate_compute_time(scenario) -> float`
```
估算内容:
  - flops_per_token: 6 * num_params * 2 / TP
  - hw_flops: 根据硬件设备的理论峰值 FLOPS
  - compute_time_ms = flops_per_token / hw_flops * 1000
```

**comm_latency.py** - `estimate_comm_latency(scenario) -> float`
```
估算内容:
  - 模型广播和同步的 AllReduce 延迟
  - 使用 ring-allreduce 算法估算
  - 返回单位: microseconds
```

### 验证层 (`validators/`)

**scenario_validator.py** - `validate_scenario(scenario) -> PredictionResult`
```
流程:
  1. 调用所有 estimator（内存、计算、通信）
  2. 若有实测数据，计算误差
  3. 判断是否通过（±20% 容差）
  4. 返回 PredictionResult 对象
```

### 输出层 (`reporters/`)

**report.py**
```python
print_report(results)           # 打印格式化的表格报告
export_report_json(results)     # 导出 JSON 格式报告
```

### 场景库 (`scenarios.py`)

包含 4 个标准验证场景：
1. `A100_Llama2_70B_TP4` - vLLM 基准
2. `A100_Qwen_72B_TP4` - DatabaseMart 数据
3. `Ascend910B_16Card_DeepSeek` - 华为 Ascend
4. `B200_8Card_Llama4` - NVIDIA Blackwell

## 数据流

```
ValidationScenario
    ↓
validate_scenario()  ← 调度器
    ├→ estimate_memory_budget()
    ├→ estimate_compute_time()
    └→ estimate_comm_latency()
    ↓
PredictionResult  ← 包含预测值 + 实测值 + 误差
    ↓
print_report() / export_report_json()
```

## 使用示例

### 1. 验证单个场景
```python
from validation import validate_scenario, VALIDATION_SCENARIOS

scenario = VALIDATION_SCENARIOS[0]  # A100_Llama2_70B_TP4
result = validate_scenario(scenario)

print(f"内存: {result.predicted_total_memory_mb:.1f} MB")
print(f"吞吐: {result.measured_throughput_tok_s:.1f} tok/s")
print(f"精度: {result.is_accurate}")
```

### 2. 批量验证和报告
```python
from validation import validate_scenario, print_report, export_report_json, VALIDATION_SCENARIOS

results = [validate_scenario(s) for s in VALIDATION_SCENARIOS]
print_report(results)
export_report_json(results, "my_report.json")
```

### 3. 命令行使用
```bash
# 全量验证
python -m validation.cli

# 单个场景
python -m validation.cli --scenario A100_Llama2_70B_TP4

# 自定义输出
python -m validation.cli --output reports/validation_2026_04_21.json
```

## 扩展点

### 添加新的硬件配置
```python
# validation/scenarios.py
NEW_SCENARIO = ValidationScenario(
    scenario_id="H100_Llama_405B",
    hardware=HardwareConfig(
        device_name="H100 80GB",
        num_devices=8,
        interconnect="NVLink 4",
        interconnect_bandwidth_gbs=1800,
        hbm_bandwidth_gbs=3900,
        total_memory_gb=640,
        tensor_parallel_size=8,
    ),
    model=...,
    inference=...,
)
```

### 添加新的估算算法
```python
# validation/estimators/custom_metric.py
def estimate_custom_metric(scenario) -> float:
    ...

# validation/validators/scenario_validator.py
# 在 validate_scenario() 中添加调用
result.custom_metric = estimate_custom_metric(scenario)
```

### 添加新的报告格式
```python
# validation/reporters/report_html.py
def export_report_html(results) -> str:
    ...
```

## 集成点

### 与 python.zrt.graph 的连接
- `run_trace()` 生成的图可用于验证操作符捕获的完整性
- FakeTensorMode 保证内存占用估算的一致性

### 与 python.zrt.memory 的连接
- `MemoryBudget` 数据结构复用
- 内存模型的验证

### 与 python.zrt.transform.analysis 的连接
- `_estimate_comm_latency()` 通信延迟估算
- CommLatencyPass 模型验证

## Backward Compatibility

根目录的两个文件作为转发器：
- `e2e_validate_with_public_data.py` → `validation.cli`
- `validation_scenarios.py` → `validation.models` + `validation.scenarios`

旧的 import 方式仍然可用：

```python
# 旧方式（仍支持）
from validation.validators.validation_scenarios import VALIDATION_SCENARIOS

# 新方式（推荐）
from validation import VALIDATION_SCENARIOS
```

## 测试策略

### 单元测试（`tests/test_validation.py`）
- [ ] 场景数据加载
- [ ] 各估算模块的输出范围
- [ ] 结果数据类的序列化

### 集成测试
- [ ] 完整的验证流程
- [ ] 报告生成
- [ ] CLI 接口

### 回归测试
- [ ] 对标公开数据的误差收敛
- [ ] 新硬件配置的添加

## 性能指标收集

当前跟踪的指标：
| 指标 | 单位 | 来源 |
|-----|-----|------|
| 权重内存 | MB | weights_params / TP * quantization |
| KV Cache | MB | batch * seq_len * layers * hidden / TP |
| 计算时间 | ms | FLOPs / HW_FLOPS |
| 通信延迟 | us | AllReduce 公式 |
| 吞吐 | tok/s | 实测 baseline |
| 误差 | % | \|(predicted - measured) / measured\| |

## 未来方向

- [ ] 支持动态 batch size
- [ ] 融合算子的效果验证
- [ ] MoE 专家分布对性能的影响
- [ ] 量化等级（FP8 vs INT8）的对标
- [ ] 与 MLPerf/SPEC 的集成
- [ ] 持续基准库（baseline tracking）
