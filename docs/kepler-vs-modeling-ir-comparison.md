# Kepler vs Modeling 计算图 IR 对比分析

> 日期: 2026-05-23
> 背景: 评估将 Kepler 项目的拖拽构造计算图机制引入 Modeling 项目的可行性

---

## 1 项目定位差异

| 维度 | Kepler | Modeling |
|------|---------|----------|
| 目标 | **LLM 推理**资源建模（延迟/显存/吞吐预测） | **LLM 训练**资源建模（step time/MFU/memory estimate） |
| 用户画像 | 内部 R&D 团队，Web 界面操作 | 工程师 + CLI/YAML 配置驱动 |
| 图来源 | 用户**手动拖拽构造** | 自动 tracing（Stack B）或从 ModelSpec 合成（Stack A） |
| 交付方式 | FastAPI + React Web 单容器 | Python CLI + Excel/HTML 输出 |
| 依赖框架 | 无推理引擎依赖，纯公式 | Torch tracing 依赖（Stack B）；纯公式（Stack A） |

---

## 2 IR 数据模型核心对比

### 2.1 节点模型

| 维度 | Kepler `OperatorBase` | Modeling `OpNode` / `Op` |
|------|----------------------|--------------------------|
| 核心字段 | op_name + is_vector + compute_flops_str + cfg dict | id + op_type + inputs/outputs(TensorMeta[]) + attrs + scope + annotations |
| 类别机制 | **继承区分**: OpCubeBase/OpVectorBase/OpMixBase/OpCommBase | **属性标注**: category = compute/communication/memory |
| FLOPs表达 | **符号表达式**字符串 `"2*B*S*IN_DIM*OUT_DIM"`，运行时 eval(cfg) 求值 | **数值**: meta dict 存具体维度参数，由 FlopsPass 计算 |
| Tensor模型 | `TensorBase`: shape list[int] + dtype + name + 符号解析(context dict) | `TensorMeta`(OpGraph) / `Tensor`(TrainingGraph): shape tuple + dtype + mem_bytes / shape_logical + shape_local |
| 参数(权重) | 单独 `params: list[TensorBase]` 字段，与 inputs/outputs 三分 | 没有 params 字段; 权重信息在 meta dict（m, n, k）或通过 scope/annotations 推断 |
| 粒度 | **算子级**: MatMul, FlashAttention, RMSNorm, AllReduce | **两层**: Stack A → 模块级(attn_core, swiglu, router); Stack B → aten级 → fusion后模块级 |

**关键差异**: Kepler 用**继承体系**区分算子类别(Cube/Vector/Mix/Comm)，每种类型有不同的 cost 计算路径和 static_cost。Modeling 用**单一 category 字段**，所有算子共用一套 Roofline 公式(compute_bound vs memory_bound)，差异通过 `annotations` 侧信道实现。

**Kepler 算子继承树**:

```
OperatorBase
  ├── OpCubeBase     → MatMul, ColumnParallelLinear, RowParallelLinear, GroupMatMul, MoEGateTopK
  ├── OpVectorBase   → Embedding, RMSNorm, SwiGlu
  ├── OpMixBase      → FlashAttention
  ├── OpCommBase     → AllReduce, AllGather, MoEDispatch, MoECombine
  └── OperatorBase   → StartOperator, EndOperator
```

### 2.2 边模型

| 维度 | Kepler | Modeling |
|------|---------|----------|
| 边定义 | `{from: int, to: int}` — 简整数引用(op_id) | `Edge(src: str, src_idx: int, dst: str, dst_idx: int, tensor: TensorMeta|None)` |
| 边语义 | **拓扑依赖**，无 tensor 元数据 | **数据流**: 携带 TensorMeta + slot index + tensor_id |
| 隐式边 | 未指定 edges 时自动顺序链 | 无隐式边，全部显式 |
| 控制边 | 不区分 | 支持 `tensor=None` 控制边 |
| 断边 | `disconnected: Set<string>` 跟踪用户断开的隐式边 | 无此机制 |

**关键差异**: Kepler 的边是**拓扑骨架**，只关心"谁在谁之后"。Modeling 的边是**完整数据流**，tensor 元数据沿边传播，slot index 实现多输入多输出的精确连接。

### 2.3 层模型

| 维度 | Kepler | Modeling |
|------|---------|----------|
| 层定义 | `LayerConfig`: id + name + repeat + opIndices(显式索引引用) | `layer_index: dict[int, tuple[int,int]]` (start/end range) |
| 层与节点 | **双向引用**: LayerConfig.opIndices → OpNodeData.index; OpNodeData.layer_idx → LayerConfig | **单向**: Op.layer_id 标注所属层; Graph.layer_index 提供索引查询 |
| 重复机制 | `repeat: number` 字段，计算时乘 repeat | 通过 `num_layers` 和 `LayerSpec` 循环构建 |
| 全局算子 | layer_idx=-1(pre-layer) / -2(post-layer) | layer_id<0 标记全局算子 |

### 2.4 通信模型

| 维度 | Kepler | Modeling |
|------|---------|----------|
| 通信算子 | OpCommBase 子类: AllReduce, AllGather, MoEDispatch, MoECombine | `Collective` 独立实体(kind=AG/RS/AR/A2A/P2P + group=TP/CP/EP/DP/PP) |
| 通信时间 | 单层带宽模型: comm_bytes / comm_bw (3档: bwsio/intra/inter) | **N-tier hierarchical α-β 模型**: CommDomain + collective_time_multi_tier |
| 重叠 | 无 overlap 机制 | 完整 overlap 模型(TP/EP/DP/PP/Muon 5维度) |

**关键差异**: 这是最显著的技术差距。Kepler 的通信建模非常简化(flat 3档带宽)，而 Modeling 有完整的分层 α-β 模型 + 5维度 overlap。

### 2.5 硬件模型

| 维度 | Kepler | Modeling |
|------|---------|----------|
| 芯片规格 | AIChipConfig: cube/vect/sfu 三类算力 + 三档通信带宽 + efficiency ratios | ComputeSpec: roofline_peak_tflops + InterconnectSpec(N-tier TopologyTier list) |
| 带宽 | 固定 3档(bwsio/intra/inter) + ratio 衰减 | `effective_bw_bps(group_size)` + kb_efficiency + scale derate + N-tier α-β |
| 内存 | 总量 + noise + L2 cache 模拟 | 详细 memory_breakdown(): 权重/KV/激活/优化器/通信buffer 分项 |
| 芯片注册 | subclass_registry(NVIDIA/Ascend 子类) | 纯数据驱动(YAML spec dict) |

---

## 3 图构造机制对比

| 维度 | Kepler | Modeling |
|------|---------|----------|
| 构造方式 | 前端拖拽 + JSON 导出/导入 | YAML → 自动合成 / Torch tracing → 自动 |
| 用户交互 | React Flow 画布: 拖算子、连线、配层、配参数 | 无前端; CLI + YAML 配置 |
| 算子发现 | `/api/library/operators` → JSON 文件列表 | `builders.py` 稠密函数 + `infer_layer_types()` 自动推断 |
| 自定义算子 | CustomOperatorDialog → localStorage 持久化 | 无自定义算子机制(算子集是硬编码的) |
| 验证 | DFS 环检测 | topo_sort + cycle RuntimeError |
| 布局 | 自动拓扑排序布局(120ms debounce) | 无布局(纯数据结构) |

### Kepler 拖拽机制关键组件

| 组件 | 文件 | 功能 |
|------|------|------|
| OperatorPanel | `frontend/src/components/ModelEditor/OperatorPanel.tsx` | 左侧算子面板, 8类分组, HTML5 drag API |
| ModelCanvas | `frontend/src/components/ModelEditor/ModelCanvas.tsx` | React Flow 画布, onDrop创建节点, onConnect连线 |
| LayerNode | `frontend/src/components/ModelEditor/LayerNode.tsx` | 自定义节点渲染(算子名+参数+category色标) |
| OperatorDetail | `frontend/src/components/ModelEditor/OperatorDetail.tsx` | 右侧详情面板(编辑inputs/params/outputs/compute_flops/module) |
| model store | `frontend/src/stores/model.ts` (Zustand) | 状态管理: nodes, edges, layers, disconnected, export/import |
| ModelConfig | `frontend/src/components/ModelEditor/ModelConfig.tsx` | 层配置(增删层、分配算子、repeat倍率) |
| CustomOperatorDialog | `frontend/src/components/ModelEditor/CustomOperatorDialog.tsx` | 自定义算子创建对话框 |
| operators.ts | `frontend/src/constants/operators.ts` | 8类算子分类 + MODULE_GROUPS |

### Kepler 算子分类

| 类别 | 颜色 | 算子 |
|------|------|------|
| Flow | #9e9e9e | START, END |
| Embedding | #3b6fb6 | Embedding |
| Normalization | #8e6bb8 | RMSNorm |
| Attention | #c4504a | FlashAttention |
| Activation | #e67e22 | SwiGlu |
| Linear | #4d8c57 | ColumnParallelLinear, RowParallelLinear, MatMul, GroupMatMul |
| MoE | #2e86c1 | MoEGateTopK, MoEDispatch, MoECombine |
| Communication | #b866cc | AllGather, AllReduce |
| Custom | #d4842a | 用户自定义(localStorage持久化) |

---

## 4 Executor/仿真引擎对比

| 维度 | Kepler | Modeling |
|------|---------|----------|
| 执行引擎 | `GraphExecutor`: NetworkX DiGraph + topo_sort 顺序执行 | `estimate()` / `estimate_training_from_graphs()`: 分析函数 |
| 每步操作 | node.op(input_tensors, output_tensors) → calc_cost_us(chip) → ExecuteResult | 每个算子 FlopsPass → RooflinePass → CommLatencyPass → 逐步 annotation |
| 结果 | ExecuteResult(compute/mem/comm/bound/total per op) | TrainingReport(step_time, MFU, memory_breakdown, per-op timing) |
| 训练 vs 推理 | 只做推理(prefill + decode) | 训练全流程(fwd + bwd + optimizer + comm + overlap) |

### Kepler Executor 流程

```
GraphExecutor(model: dict, context: dict, chip: AIChipConfig):
  .graph = nx.DiGraph()

  run() -> list[ExecuteResult]:
    1. _build_graph() — 创建Node, parse_tensors(context), fill_operator(context)
    2. _execute_topological() — topo_sort顺序执行, 每节点:
       - node.op(input_tensors, output_tensors)  # 赋值tensor, 计算bw_bytes
       - node.op.calc_cost_us(chip)              # 计算时间开销
       - 收集ExecuteResult
```

### Modeling TransformPipeline 流程

```
TransformPipeline (4阶段):
  split → fuse → optim → analyze

  split:  DP/TP/EP/CP/PP并行切分 + recompute/offload + CommInserter
  fuse:   模块级fusion(scope + module_class分桶 → rich-rule匹配 + Add-Norm组合)
  optim:  Quantization / EPLB / SharedExpert / MTP / ZeRO / Optimizer
  analyze: FlopsPass → RooflinePass → CommLatencyPass → StreamAssignPass → TrainingFlops/Memory/Pipeline
```

---

## 5 序列化对比

| 维度 | Kepler | Modeling |
|------|---------|----------|
| 格式 | JSON(operators[] + layers[] + edges[]) | JSON(OpGraph serde: nodes{} + edges[] + metadata) |
| 算子定义 | 内联在graph JSON + 独立operator JSON文件 | 完全内联(OpNode所有字段) |
| 导入/导出 | 前端 exportModel/importFromJSON 双向 | save_json/load_json 双向 |
| 自定义算子 | localStorage + 文件JSON | 无 |

### Kepler 导出 JSON 结构

```json
{
  "name": "custom-model",
  "num_ops": 8,
  "num_layers": 2,
  "num_edges": 7,
  "operators": [
    {
      "op_id": 0,
      "op_name": "Embedding",
      "layer_idx": -1,
      "op_module": "embed",
      "inputs": [{"name": "x", "shape": "[B, S, hidden_dim]", "dtype": "fp16"}],
      "params": [{"name": "weight", "shape": "[vocab_size, hidden_dim]", "dtype": "fp16"}],
      "outputs": [{"name": "y", "shape": "[B, S, hidden_dim]", "dtype": "fp16"}],
      "compute_flops": "0"
    }
  ],
  "layers": [
    {
      "name": "Layer_0",
      "layer_idx": 0,
      "repeat": 1,
      "op_idx": [1, 2]
    }
  ],
  "edges": [
    {"from": 0, "to": 1}
  ]
}
```

### Modeling OpGraph JSON 结构

```json
{
  "name": "spec_h7168_l60_training",
  "phase": "training",
  "metadata": {"source": "model_spec", "hidden": 7168, "layers": 60},
  "nodes": {
    "op_0": {
      "id": "op_0",
      "op_type": "matmul",
      "inputs": [{"id": "t0", "shape": [1, 7168], "dtype": "bf16", "mem_bytes": 14336}],
      "outputs": [{"id": "t1", "shape": [1, 18432], "dtype": "bf16", "mem_bytes": 36864}],
      "attrs": {"m": 1, "n": 18432, "k": 7168},
      "scope": "model.layers.0.self_attn.qkv_proj",
      "category": "compute",
      "annotations": {"layer_id": 0, "layer_kind": "DENSE"},
      "op_short": "mm"
    }
  },
  "edges": [
    {"src": "op_0", "src_idx": 0, "dst": "op_1", "dst_idx": 0, "tensor": {"id": "t1", ...}, "tensor_id": 1}
  ]
}
```

---

## 6 异同总结

### 共同点

- 目标相同: 都是 LLM 部署资源预测工具，用公式替代实测
- 都有计算图 IR: 都用 DAG 作为核心数据结构
- 都有算子分类: Kepler(继承) / Modeling(category字段)，粒度接近
- 都有 FLOPs 计算: Kepler eval(符号表达式) / Modeling FlopsPass(数值公式)
- 都有 Roofline 模型: Kepler max(compute, bw) / Modeling RooflinePass

### 关键差异

| 差异点 | Kepler | Modeling | 影响 |
|--------|---------|----------|------|
| 场景 | 推理 only | 训练(fwd+bwd+optimizer) | Modeling 需要 fwd/bwd phase 标注 |
| 图来源 | **手动拖拽** | **自动合成/tracing** | Kepler 的拖拽 UI 是核心价值 |
| 边模型 | 简拓扑依赖 `{from,to}` | 完整数据流 Edge(src,src_idx,dst,dst_idx,tensor) | 边模型需要升级 |
| 通信建模 | flat 3档带宽 | N-tier α-β + 5维 overlap | Kepler 通信建模远弱于 Modeling |
| 层次结构 | 无 scope 树 | GraphHierarchy scope 树 | scope 树对 HTML export 很重要 |
| Transform Pipeline | 无(直接执行) | 4阶段 TransformPipeline | 不可替换 |
| 自定义算子 | 支持(前端 UI + JSON) | 不支持(硬编码) | Kepler 更灵活 |
| 前端 | React Flow + Zustand | 无前端(CLI only) | Kepler 有完整可视化交互 |

---

## 7 引入拖拽机制可行性评估

### 可直接移植的部分

1. **前端框架**: React Flow + Zustand 的画布 + 节点 + 连线 UI
2. **算子面板**: 按类别分组的 OperatorPanel 模式
3. **自定义算子对话框**: CustomOperatorDialog 的交互模式
4. **JSON 导入/导出**: 序列化/反序列化逻辑(格式对齐后)

### 需要适配的部分

1. **边模型升级**: Kepler 的 `{from, to}` 需升级为 Modeling 的 `Edge(src, src_idx, dst, dst_idx, tensor)`
   - 前端连线时需要选择端口(slot)，而不是简单节点间连线
   - 每条边需要携带 TensorMeta 元数据
2. **算子定义对齐**: Kepler 的 JSON operator 定义(符号表达式 compute_flops) → Modeling 的 OpNode(数值 meta dict + FlopsPass 注入)
   - 方案A: 前端使用 Modeling 的 meta dict 格式，放弃 Kepler 的符号表达式
   - 方案B: 前端保留 Kepler 符号表达式，后端增加 eval 桥接层
   - **推荐方案A**: 保持 Modeling 的数值 meta dict 一致性
3. **通信算子**: Kepler 只有 AllReduce/AllGather 两个通信算子。Modeling 有完整的 Collective 体系(AG/RS/AR/A2A/P2P per TP/CP/EP/DP/PP)
   - 拖拽 UI 需要展示通信算子但**不允许用户随意添加**
   - 通信算子应由 CommInserterPass 自动插入，拖拽 UI 只做可视化展示
4. **层机制**: Kepler 的 `LayerConfig.repeat` 映射到 Modeling 的 `num_layers` × 层模板模式
5. **训练阶段标注**: Modeling 需要在节点上标注 fwd/bwd/optimizer phase
   - 拖拽 UI 需要区分正反向算子(可考虑自动生成 bwd 对应算子)

### 不建议移植的部分

1. **Kepler 的 Executor**: NetworkX + topo_sort 顺序执行，不如 Modeling 的 TransformPipeline + CommDomain + overlap，不应替换
2. **Kepler 的通信建模**: flat 3档带宽远不如 Modeling 的 N-tier α-β 模型，不应回退
3. **Kepler 的硬件模型**: AIChipConfig 不兼容 Modeling 的 ComputeSpec + InterconnectSpec，需要另做前端编辑器

### 建议的集成架构

```
前端 (React Flow + Zustand) ← 新引入, 模式来自 Kepler
  │
  ├─ OperatorPanel: 从 Modeling 的 builders.py 算子集生成
  │                  (不用 Kepler 的 JSON 文件算子库, 用 Modeling 硬编码算子)
  ├─ ModelCanvas: 拖拽 → 创建 OpNode → 连 Edge
  │               (边携带 slot index + TensorMeta, 不是 Kepler 的简单 {from,to})
  ├─ LayerConfig: 映射到 Modeling 的 layer_index
  │               (repeat 字段 → num_layers 参数)
  ├─ CustomOperatorDialog: 创建自定义 OpNode(存 Modeling JSON 格式)
  │
  │  export → OpGraph JSON (serde.py 格式)
  │  import ← OpGraph JSON
  │
后端 (保持 Modeling 现有架构)
  │
  ├─ OpGraph.from_model_spec() ← 自动合成(保留, 与手动拖拽并存)
  ├─ OpGraph.from_user_json() ← 新增: 从前端 JSON 构造
  ├─ TransformPipeline.run() ← 保持
  ├─ estimate() ← 保持
  └
```

**核心思路**: 前端是 Kepler 的拖拽 UI，后端是 Modeling 的 estimate 引擎。前端生成 Modeling 的 OpGraph JSON 格式，后端消费。不需要替换 Modeling 的 IR 或 TransformPipeline，只需要增加一个"从前端 JSON 构造 OpGraph"的入口。