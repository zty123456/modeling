# PP 流水线并行建模

> Pipeline Parallel stitcher — 从 per-stage DAGScheduler Timeline 构建 stage × microbatch 网格并调度，替代公式化的 PipelineComposer。

## 目录

1. [核心概念：Stage / Layer / Microbatch](#1-核心概念stage--layer--microbatch)
2. [Global Batch 与 Microbatch 的关系](#2-global-batch-与-microbatch-的关系)
3. [前向图与反向图的拼接](#3-前向图与反向图的拼接)
4. [PPStitcher：网格调度引擎](#4-ppstitcher网格调度引擎)
5. [三种边：依赖约束](#5-三种边依赖约束)
6. [三种 Schedule 的 Edge ③ 差异](#6-三种-schedule-的-edge--差异)
7. [Trace 模式 vs Formula 模式](#7-trace-模式-vs-formula-模式)
8. [端到端数据流](#8-端到端数据流)
9. [CLI 使用](#9-cli-使用)

---

## 1. 核心概念：Stage / Layer / Microbatch

### 层次关系

```
Layer  ←───── 模型的最小结构单元 (transformer block)
  │
  │  多个 layer 被分配到同一个 stage（贪心 bin-packing）
  ▼
Stage  ←───── 一张 GPU 上负责的层范围 (pp=4 → 4 个 stage)
  │
  │  一个 stage 内执行一个 microbatch 的完整 FWD + BWD
  ▼
Microbatch  ←── 流水线注入的最小数据单位
```

### Layer → Stage 的分派 (PipelineParallelPass)

使用贪心 bin-packing 策略：对每个 layer，找当前计算负载最小的 stage 放入，使各 stage 负载尽量均匀。

**实例**（61 层模型，pp=4）：

```
Stage 0:  layers 0..14   (15 层)  ← 最接近输入 embedding
Stage 1:  layers 15..29  (15 层)
Stage 2:  layers 30..44  (15 层)
Stage 3:  layers 45..60  (16 层)  ← 最接近 loss
```

**数据流动**：
- **前向**：激活从 Stage 0 → Stage 1 → Stage 2 → Stage 3 逐级传递
- **反向**：梯度从 Stage 3 → Stage 2 → Stage 1 → Stage 0 依次回流

### PPStitcher 视角的 pp×M 网格

```
         m0        m1        m2        m3           ← M 个 microbatch
s0:   [F] [B]   [F] [B]   [F] [B]   [F] [B]
s1:   [F] [B]   [F] [B]   [F] [B]   [F] [B]
s2:   [F] [B]   [F] [B]   [F] [B]   [F] [B]
s3:   [F] [B]   [F] [B]   [F] [B]   [F] [B]
 ↑
 pp 个 stage (= pp 张 GPU，每行 = 一张卡)

每个单元格 = 一个 GridTask:
  ├─ FWD 单元格的 latency = stage_fwd[s] + (接收 P2P 延迟)
  ├─ BWD 单元格的 latency = stage_bwd[s] + (接收 P2P 延迟)
  └─ stream_id = s  (同 stage task 在同 device 串行)
```

> PPStitcher 拿的是 `stage_fwd[s]` / `stage_bwd[s]`（DAGScheduler 已算好的整 stage 总延迟），不区分 layer。Layer 级别的每算子细节保留在 `stage_timelines[s].scheduled_ops` 中供 Chrome Trace 详细视图使用。

---

## 2. Global Batch 与 Microbatch 的关系

### 关键公式

```
M = global_batch / (micro_batch × dp)

其中:
  global_batch  = 一个 training step 处理的样本总数
  micro_batch   = 一次 GPU 前向/反向处理的样本数
  dp            = Data Parallel 并行度
  M             = 每个 step 的 microbatch 数量（即流水线注入次数）
```

**示例**：`global_batch=32, micro_batch=8, dp=1` → `M = 32 / (8 × 1) = 4`

### 一个 Training Step 的执行过程

```
Time ──────────────────────────────────────────────────────────→

Stage 0:  F₀  F₁  F₂  F₃ | B₀  F₄  B₁  F₅  B₂  F₆  B₃ | B₄  B₅  B₆  B₇
Stage 1:      F₀  F₁  F₂ | F₃  B₀  F₄  B₁  F₅  B₂  F₆ | B₃  B₄  B₅  B₆  B₇
Stage 2:          F₀  F₁ | F₂  F₃  B₀  F₄  B₁  F₅  B₂ | F₆  B₃  B₄  B₅  B₆  B₇
Stage 3:              F₀ | F₁  F₂  F₃  B₀  F₄  B₁  F₅ | B₂  F₆  B₃  B₄  B₅  B₆  B₇
         ├── warmup ──┤├──────── steady ────────────────┤├── cooldown ──────┤
```

**三个阶段**：

| 阶段 | 含义 | 时间 |
|------|------|------|
| **Warmup**（注入） | Stage 0→1→2→3 依次收到 microbatch 0 并做 FWD，激活逐级传递 | `(pp-1) × t_fwd` |
| **Steady**（稳态） | 每个 stage 交替执行 FWD 和 BWD（1F1B），Stage 3 完成 FWD₀ 后立即做 BWD₀ | `M × (t_fwd + t_bwd)` |
| **Cooldown**（排空） | 最后一个 microbatch 的 FWD 完成后，剩余的 BWD 依次回流 | `(pp-1) × t_bwd` |

### 每个 Microbatch 包含什么

```
┌─ Microbatch m ────────────────────────────────────────────────┐
│  数据: micro_batch 个样本                                      │
│                                                               │
│  ┌─ FWD ──────────────────────────────────────────────────┐  │
│  │  Stage s 覆盖的所有 layer:                               │  │
│  │    ├─ Attention: QKV投影 → SDPA → Output投影            │  │
│  │    ├─ MoE FFN: Gate → TopK → Expert FFN → Combine      │  │
│  │    ├─ RMSNorm / LayerNorm                               │  │
│  │    └─ Residual Add                                      │  │
│  │  产出: 输出激活 (→ 下一stage) + 中间张量 (→ BWD)         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─ BWD ──────────────────────────────────────────────────┐  │
│  │  Stage s 覆盖的所有 layer (倒序):                        │  │
│  │    ├─ 算子梯度计算 (dInput, dWeight)                     │  │
│  │    ├─ Recompute: 如需重算 FWD 中间结果                   │  │
│  │    └─ Comm: TP RS收集梯度 / DP AR规约梯度               │  │
│  │  产出: dWeight (→ optimizer) + dX (→ 前一stage)         │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### Step 结束：Optimizer Step

```
所有 microbatch 的 BWD 完成后:
  ├─ DP All-Reduce/Reduce-Scatter: 跨 DP ranks 规约梯度
  ├─ Optimizer: Adam/AdamW/Muon 更新权重
  └─ ZeRO-1: All-Gather 分发更新的 optimizer state

Muon 优化器的 NS 旋转可在下个 step 的 FWD 窗口内隐藏（Moonshot optimization）。
```

---

## 3. 前向图与反向图的拼接

### 3.1 `stitch_fwd_bwd()` — 图级别拼接

位于 `python/zrt/ir/adapter.py`。

```
train_forward  OpGraph (1181 节点, phase="fwd")
train_backward OpGraph (1240 节点, phase="bwd")
         │
         ▼  stitch_fwd_bwd()
         │
         ├─ fwd 节点: 保留原 ID + annotation phase="fwd"
         ├─ bwd 节点: ID 加前缀 "bwd_" + annotation phase="bwd"
         ├─ fwd→fwd 边: 复制
         ├─ bwd→bwd 边: 复制 (ID 已加前缀)
         └─ fwd→bwd 边: 通过 tensor-ID 精确匹配重建
              fwd 输出 tensor → TensorTracker 记录 ID
              bwd 输入 tensor → 查找对应 ID
              → 加入依赖边: fwd_node → bwd_xxx
```

两个 phase 共享同一个 `TensorTracker`，保证了 tensor-ID 在跨 phase 时的全局唯一性，使 `stitch_fwd_bwd` 能精确匹配。

### 3.2 `PipelineParallelPass` — Stage 分配 + P2P 插入

位于 `python/zrt/transform/parallel/pipeline_parallel.py`。

在拼接后的统一 OpGraph 上：
1. 按 `node.layer` 聚合每层节点，做贪心 bin-packing 分派到各 stage
2. 检测跨 stage 的边，插入 `comm.send_recv` 节点（`op_type="comm.send_recv"`）
3. 给所有节点写入 `annotations["stage_id"]`

### 3.3 `TrainingPipelinePass` — 提取 per-stage 延迟

位于 `python/zrt/transform/analysis/training.py`。

```
for s in range(pp):
    sub = g.subgraph(stage_s_node_ids)       # 取出 stage s 的子图
    tl = DAGScheduler.schedule(sub)          # 子图做拓扑排序 + list scheduling
    stage_fwd[s] = tl.phase_latency("fwd")   # 提取 fwd 总延迟
    stage_bwd[s] = tl.phase_latency("bwd")   # 提取 bwd 总延迟
    stage_timelines[s] = tl                  # 保留每算子时序
```

此时前向反向已经在一个统一图里，DAGScheduler 按 `annotations["phase"]` 拆分 fwd/bwd，自动处理 compute/comm 重叠。

---

## 4. PPStitcher：网格调度引擎

### 整体算法

```
stitch()
  │
  ├─ 1. _build_grid()           创建 stage×mb 网格，每个单元格一个 GridTask
  │
  ├─ 2. _add_activation_dependency()    边①: F→B (同卡同mb)
  │
  ├─ 3. _add_cross_stage_p2p()          边②: 跨卡 P2P (激活前传 + 梯度回传)
  │
  ├─ 4. _add_device_serial()            边③: 同卡串行化 (schedule-dependent)
  │
  ├─ 5. _list_schedule()         Kahn 拓扑排序 + 贪心 list scheduling
  │
  └─ 6. _build_result()          汇总 warmup/steady/cooldown/bubble
```

### List Scheduler

采用 Kahn's algorithm 进行拓扑排序，贪心选取 earliest-possible-start-time 的 task：

```python
while ready:
    task = min(ready, key=earliest_start_time)
    task.start = max(pred_done, device_free)
    task.end   = task.start + task.latency
    device_free[stream_id] = task.end
    # release dependents
```

包含循环检测：`len(scheduled) < len(tasks)` → `ValueError`。

### Bubble 计算

```python
per_stage = max(stage_fwd[s] + stage_bwd[s] for s in range(pp))
ideal_time = M × per_stage
bubble = max(0, step_time - ideal_time)
```

Bubble = 流水线实际时间 − 理想串行时间（无流水线开销）。Bubble 来源：
- Warmup 阶段：下游 stage 等待第一个 microbatch 的激活到达
- Cooldown 阶段：上游 stage 等待最后一个 microbatch 的梯度回传

---

## 5. 三种边：依赖约束

### 边①：激活依赖 (F→B)

```
同 stage 同 microbatch: fwd 必须完成后 bwd 才能开始
s0 m0 fwd ──→ s0 m0 bwd
s1 m2 fwd ──→ s1 m2 bwd
```

### 边②：跨 Stage P2P

```
前向链 (激活前传):
  s0 m0 fwd ──→ s1 m0 fwd ──→ s2 m0 fwd ──→ s3 m0 fwd

反向链 (梯度回传):
  s3 m0 bwd ──→ s2 m0 bwd ──→ s1 m0 bwd ──→ s0 m0 bwd
```

P2P 传输延迟作为 **delayed_deps 间隙** 建模：接收方 task 的 `latency_us` **不包含** P2P 部分，而是通过依赖边上的 start-time gap 表达——接收方 task 的 `start_us` 必须在发送方 `end_us + p2p_latency_us` 之后。

```
s1 m0 fwd.start ≥ s0 m0 fwd.end + p2p_fwd_us[0→1]   ← 等待 s0 的激活
s2 m0 fwd.start ≥ s1 m0 fwd.end + p2p_fwd_us[1→2]
```

P2P 延迟来自 `_extract_p2p_latency_per_edge()`，按 `(src_stage, dst_stage, phase)` 从图中 `comm.send_recv` 节点的实际 `latency_us`（`CommLatencyPass：msg_bytes / BW + link_latency`）提取瓶颈值。对于 VPP，虚拟 stage 映射到物理 device 后查找对应边界的 P2P 延迟。

### 边③：设备串行化 (1F1B)

让每张卡上的 task 按 1F1B 协议串行化。关键设计：分**两条独立链**，不互相连接。

```
Stage 0 (warmup 深度 w = pp - s = 4):

  Chain A (warmup):  F₀ → F₁ → F₂ → F₃
  Chain B (alternating + cooldown):
                     B₀ → F₄ → B₁ → F₅ → B₂ → B₃ → B₄ → B₅

Stage 3 (warmup 深度 w = 1):

  Chain A (warmup):  F₀
  Chain B (alternating + cooldown):
                     B₀ → F₁ → B₁ → F₂ → B₂ → F₃ → B₃ → F₄ → B₄ → F₅ → B₅
```

**为什么 Chain A 和 Chain B 不连接**：B₀ 何时开始由依赖（边① F₀→B₀ + 边② 反向链）+ list scheduler 的 device-free 共同决定。如果连接，会强制所有 warmup F 完成才开始 B，破坏流水线重叠。

---

## 6. 三种 Schedule 的 Edge ③ 差异

| Schedule | Edge ③ 策略 | 说明 |
|----------|------------|------|
| **1F1B** | `_add_device_serial_1f1b` | 两链：warmup F 链 + alternating F/B 链 |
| **DualPipe** | `_add_device_serial_dualpipe` | 双流反并行：Stream A `F₀→F₁→B₀→F₂→...`，跳过模式 |
| **ZeroBubble** | `_add_device_serial_zb` | bwd 拆分为 bwd_dx + bwd_dw，B_dw 延迟到气泡中执行 |
| **VPP (Interleaved)** | 无 Edge ③ | P2P 约束 + device-free tracking 自然产生 interleaving |

---

## 7. Trace 模式 vs Formula 模式

| 维度 | Trace (`--pp-mode trace`) | Formula (`--pp-mode formula`) |
|------|--------------------------|-------------------------------|
| 调度方式 | PPStitcher: grid × list scheduling | PipelineComposer: `max(t_stage)` 瓶颈公式 |
| 异构阶段 | **保留**，快卡可提前 | **忽略**，全部用瓶颈卡 |
| P2P 延迟 | 从 `comm.send_recv` 节点提取实测值 | 公式计算 `pp_p2p` |
| Bubble | 精确计算，反映实际 grid 气泡 | `(pp-1) × (t_fwd + t_bwd)` 瓶颈公式 |
| Chrome Trace | ✅ 输出 `pp_stitched.json` / `pp_combined.json` | ❌ 不输出 |
| 通信重叠 | Dual-path: trace sweep-line + formula micro-pipelining | Formula only |

**Trace 模式的 step_time 通常更小**：因为它保留了异构阶段特征，快卡不需要等瓶颈卡。

---

## 8. 端到端数据流

```
1. FakeTensor Trace (pipeline.py)
   train_forward  → 1181 aten 算子
   train_backward → 1240 aten 算子
                         │
2. stitch_fwd_bwd (adapter.py)
   统一 OpGraph: 2421 节点, phase="train"
   通过 tensor-ID 精确匹配建立 fwd→bwd 依赖
                         │
3. Transform Pipeline
   ├─ PipelineParallelPass → 分配 stage_id, 插入 comm.send_recv
   ├─ CommInserterPass     → TP/EP/CP 通信插入
   ├─ FusionPass           → aten → module 粒度融合
   ├─ RooflinePass         → 每算子延迟标注
   ├─ TrainingFlopsPass    → FLOPs 计算
   └─ TrainingMemoryPass   → Memory 计算
                         │
4. TrainingPipelinePass
   ├─ DAGScheduler 对每个 stage 子图做拓扑 + list scheduling
   ├─ 输出: stage_fwd[s], stage_bwd[s]  → PPStitcher
   └─ 输出: stage_timelines[s]          → ChromeTraceExporter
                         │
5. PPStitcher.stitch()
   ├─ _build_grid: pp×M 网格
   ├─ 边①②③ 依赖
   ├─ _list_schedule: Kahn + 贪心
   └─ _build_result: metrics
                         │
6. ChromeTraceExporter
   ├─ export_stitched()  → pp_stitched.json
   ├─ export_combined()  → pp_combined.json
   └─ export_per_stage() → pp_per_stage.json
                         │
7. TrainingReport
   ├─ step_time_ms, bubble_fraction, MFU/HFU
   ├─ 各 strategy 的 exposed/hidden comm
   └─ Memory breakdown
```

---

## 9. CLI 使用

```powershell
# Trace 模式（默认）— PPStitcher + Chrome Trace
.venv\Scripts\python.exe -m zrt.cli `
  --model-id hf_models/deepseek_v4 --train --hw nvidia_h100_sxm `
  --hidden 7168 --layers 4 --seq-len 128 `
  --global-batch 32 --micro-batch 8 `
  --pp 4 --tp 1 --pp-schedule 1f1b --pp-mode trace `
  --recompute-policy full --optimizer adam --zero-stage 1

# Formula 模式 — PipelineComposer
.venv\Scripts\python.exe -m zrt.cli `
  --model-id hf_models/deepseek_v4 --train --hw nvidia_h100_sxm `
  --hidden 7168 --layers 4 --seq-len 128 `
  --global-batch 32 --micro-batch 8 `
  --pp 4 --tp 1 --pp-schedule 1f1b --pp-mode formula `
  --recompute-policy full --optimizer adam --zero-stage 1

# DualPipe schedule
.venv\Scripts\python.exe -m zrt.cli `
  --model-id hf_models/deepseek_v4 --train --hw nvidia_h100_sxm `
  --hidden 7168 --layers 4 --seq-len 128 `
  --global-batch 32 --micro-batch 8 `
  --pp 4 --tp 1 --pp-schedule dualpipe --pp-mode trace `
  --recompute-policy full --optimizer adam --zero-stage 1
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pp` | 1 | 流水线并行度 |
| `--pp-schedule` | `1f1b` | 调度策略: `1f1b`, `interleaved`, `dualpipe`, `dualpipev`, `zb` |
| `--pp-mode` | `trace` | PP 建模模式: `trace` = PPStitcher, `formula` = PipelineComposer |
| `--vpp-chunks` | 1 | 虚拟流水线 chunks (VPP/interleaved) |
| `--global-batch` | 32 | 全局 batch size |
| `--micro-batch` | 1 | 每个 GPU 的 micro batch size |

---

## 相关文件

| 文件 | 功能 |
|------|------|
| `python/zrt/executor/pp_stitcher.py` | PPStitcher 核心：网格构建 + 边构建 + list scheduling + 结果汇总 |
| `python/zrt/executor/chrome_trace.py` | Chrome Trace 导出：stitched / combined / per-stage 三种视图 |
| `python/zrt/executor/scheduler.py` | DAGScheduler：单 stage 算子级调度，输出 Timeline |
| `python/zrt/transform/analysis/training.py` | TrainingPipelinePass：统筹 per-stage 调度 + PPStitcher/PipelineComposer 分派 |
| `python/zrt/transform/parallel/pipeline_parallel.py` | PipelineParallelPass：layer → stage 分配 + P2P 节点插入 |
| `python/zrt/ir/adapter.py` | stitch_fwd_bwd：前向反向图拼接 |
| `python/zrt/training/compose/schedules.py` | PipelineComposer：公式化 PP 调度（formula 模式） |
| `python/zrt/training/spec/strategy.py` | Strategy：包括 `num_microbatches()` 等 |
