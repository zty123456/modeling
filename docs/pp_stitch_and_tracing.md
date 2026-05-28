# PP 流水线拼接与 Chrome Trace 可视化

## 概述

本项目实现了**拓扑驱动的流水线并行（Pipeline Parallelism, PP）调度拼接**，以及将拼接结果导出为 **Chrome Trace JSON** 格式进行可视化分析。

### 核心思想

训练中的 PP 调度策略（如 1F1B、DualPipe、ZeroBubble）在 **stage × microbatch 网格** 上表现为三类依赖边的组合。通过显式构建网格任务并施加三类约束边，再用 list scheduler 贪心调度，即可自然产生正确的流水线重叠效果 — 无需硬编码公式。

### 涉及文件

| 文件 | 作用 |
|---|---|
| `python/zrt/executor/pp_stitcher.py` | PP 流水线拼接核心：网格构建、三类边、list scheduling |
| `python/zrt/executor/chrome_trace.py` | Chrome Trace JSON 导出：三种视图模式 |
| `demo_pp_stitcher.py` | PP 拼接演示：多种调度策略、Gantt 图 |
| `demo_trace_export.py` | Trace 导出演示：三种 JSON 输出 |

---

## 一、Stage × Microbatch 网格调度

### 1.1 网格定义

```
             m=0      m=1      m=2      ...      m=M-1
        ┌────────┬────────┬────────┬─────┬────────┐
  s=0   │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ ... │ FM-1 BM-1│  ← GPU 0
        ├────────┼────────┼────────┼─────┼────────┤
  s=1   │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ ... │ FM-1 BM-1│  ← GPU 1
        ├────────┼────────┼────────┼─────┼────────┤
  s=2   │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ ... │ FM-1 BM-1│  ← GPU 2
        ├────────┼────────┼────────┼─────┼────────┤
  s=3   │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ ... │ FM-1 BM-1│  ← GPU 3
        └────────┴────────┴────────┴─────┴────────┘
```

- **Stage（s）**：流水线的一个物理 GPU/Device，持有模型的连续若干层
- **Microbatch（m）**：一个训练数据子切片，是调度的最小原子单元
- **GridTask**：网格中的一个单元格，表示某个 stage 上某个 mb 的 **一次 FWD** 或 **一次 BWD**

### 1.2 三大类约束边（Edge Types）

三条边共同决定整个流水线的执行时序：

```
Edge ①  F→B activation dependency
        G[s][m].fwd → G[s][m].bwd
        （同一 stage+mb 内，前向必须先完成才能开始反向）

Edge ②  cross-stage P2P
        G[s][m].fwd   → G[s+1][m].fwd    （前向激活传递）
        G[s+1][m].bwd → G[s][m].bwd      （反向梯度回传）

Edge ③  device-serial protocol
        取决于调度策略：
        - 1F1B:   warmup chain(F₀→F₁→...→Fw-1) + alternating chain(B₀→Fw→B₁→Fw+1→...)
        - DualPipe: 双流反并行链（每物理设备一个 stage）
        - DualPipeV: 双流反并行链 + 虚拟 stage 分割（每物理设备 v 个虚拟 stage）
        - VPP (interleaved): P2P + 设备空闲约束自然排序（每物理设备 v 个虚拟 stage）
        - ZeroBubble: bwd 拆分为 bwd_dx 和 bwd_dw 两个子阶段
```

**关键设计**：Edge ③ 必须拆分为两条**独立**的链（warmup 链和 alternating 链），二者间无连接。若将 warmup 末端直接连到 alternating 首位（如 `Fw-1 → B₀`），会导致 B₀ 被错误地推迟到所有 warmup 前向完成之后，完全消除流水线重叠。

### 1.3 各调度策略详解

#### 1.3.0 ZeroBubble 调度

ZeroBubble 将 bwd 拆分为 bwd_dx（激活梯度）和 bwd_dw（权重梯度）两个子阶段。
bwd_dw 可以推迟到任意空闲时间执行，用权重梯度计算填满流水线气泡。

**网格布局**（每个 stage 每 mb 三个任务）：
```
G[s][m] = { F[m], B_dx[m], B_dw[m] }
```

**三条 ZB 专有约束**（在通用 Edge ① ② ③ 之外）：

| 约束 | 来源 | 说明 |
|------|------|------|
| `F[m] → B_dx[m]` | Edge ① 扩展 | 前向完成后才能计算激活梯度 |
| `B_dx[m] → B_dw[m]` | Edge ③ 扩展 | 激活梯度完成后才能计算权重梯度（梯度链） |
| `B_dx[s+1][m] → B_dx[s][m]` | Edge ② 扩展 | 跨 stage 激活梯度反向传递 |
| `B_dw[s+1][m] → B_dw[s][m]` | Edge ② 扩展 | 跨 stage 权重梯度反向传递 |

**B_dw 自由填空闲**（ZB 核心特性）：
```
bwd_dw 的依赖链：F → B_dx → B_dw  （B_dw 在 B_dx 之后）
bwd_dw 不受 F[m+1] 约束           （不阻塞下一个前向）
bwd_dw 可被 list scheduler 自由推迟 → 自然填入流水线气泡
```

示例（pp=4, M=6，仅 Stage 0 时间线）：
```
GPU 0: |F₀ B_dx₀ F₁ B_dw₀ B_dx₁ F₂ B_dw₁ B_dx₂ F₃ B_dw₂ ... B_dw₅|
        └── 核心循环：F[m+1] 在 B_dx[m] 后立即启动 ──┘
         └── B_dw[m] 在 B_dx[m] 后任意时刻插入 ──┘
```

#### 1.3.1 1F1B 调度

以 pp=4, M=6 为例，各 stage 的 warmup 前向次数 `w = pp - s`：

```
Stage 0 (w=4):  Chain A:  F₀ → F₁ → F₂ → F₃
                Chain B:            B₀ → F₄ → B₁ → F₅ → B₂ → B₃ → B₄ → B₅

Stage 1 (w=3):  Chain A:  F₀ → F₁ → F₂
                Chain B:         B₀ → F₃ → B₁ → F₄ → B₂ → F₅ → B₃ → B₄ → B₅

Stage 2 (w=2):  Chain A:  F₀ → F₁
                Chain B:      B₀ → F₂ → B₁ → F₃ → B₂ → F₄ → B₃ → F₅ → B₄ → B₅

Stage 3 (w=1):  Chain A:  F₀
                Chain B:  B₀ → F₁ → B₁ → F₂ → B₂ → F₃ → B₃ → F₄ → B₄ → F₅ → B₅
```

加上 Edge ②（跨 stage 前/反向 P2P），list scheduler 自然生成：

```
GPU 0: |F₀ F₁ F₂ F₃ ---- B₀ F₄ B₁ F₅ B₂ B₃ B₄ B₅|
GPU 1: |  F₀ F₁ F₂ -- B₀ F₃ B₁ F₄ B₂ F₅ B₃ B₄ B₅  |
GPU 2: |    F₀ F₁ B₀ F₂ B₁ F₃ B₂ F₄ B₃ F₅ B₄ B₅    |
GPU 3: |      F₀ B₀ F₁ B₁ F₂ B₂ F₃ B₃ F₄ B₄ F₅ B₅  |
        └warmup┘└──── alternating ─────┘└ cooldown ┘
```

#### 1.3.2 DualPipe 调度

DualPipe 采用双流反并行设计：两个 microbatch 流在每物理设备上交叉执行，利用 F&B 重叠（前向与反向可以并行）减少空泡。

核心公式：
- warmup 空泡 = (PP/2-1) × (F&B - 2W)
- cooldown 空泡 = (PP/2-1) × (B - W)
- 总空泡 = (PP/2-1)(F&B + B - 3W)
- 其中 F&B = max(F, B)，W = bwd_dw 时间

Edge ③ 实现：`bwd[m] → fwd[m+2]` 跨步依赖，模拟双流中反向完成后跳过一个微批次再启动下一个前向。

```
Device 0:  F₀ → B₀ → F₂ → B₂ → F₄ → B₄ → ...
           F₁ → B₁ → F₃ → B₃ → F₅ → B₅ → ...  (反并行流)
```

物理设备数 = PP+1（额外一台用于承载两条流的首尾衔接），激活值显存 = PP（每设备缓存全流水线激活）。

#### 1.3.3 DualPipeV 调度

DualPipeV 是 DualPipe 的虚拟 stage 分割版本，结合了 DualPipe 的双流反并行与 VPP 的虚拟 stage 映射：

- **网格维度**：`eff_pp = pp × vpp_chunks` 个虚拟 stage，映射到 `pp+1` 个物理设备
- **虚拟 stage → 物理设备映射**：`device_id = virtual_stage % (pp+1)`（类似 VPP 的循环分配）
- **每虚拟 stage 延迟**：`latency = per_stage_latency / vpp_chunks`
- **Edge ③**：同 DualPipe 的 `bwd[m] → fwd[m+2]` 跨步依赖，但施加在虚拟 stage 维度
- **P2P**：虚拟 stage 间的跨 stage P2P 连接

核心公式（与 DualPipe 相同结构，除以 V）：
- warmup 空泡 = (PP/2-1)/V × (F&B - 2W)
- cooldown 空泡 = (PP/2-1)/V × (B - W)
- 总空泡 = (PP/2-1)(F&B+B-3W)/V
- 参数显存 = 2x（每设备承载两个虚拟 stage 的权重）
- 激活值显存 = PP/2（比 DualPipe 的 PP 更低，因为虚拟 stage 分割后每设备只需缓存一半激活）
- 物理设备数 = PP+1

网格示例（pp=4, vpp_chunks=2, M=6）：
```
               m=0      m=1      m=2      m=3      m=4      m=5
          ┌────────┬────────┬────────┬────────┬────────┬────────┐
  vs=0    │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ F₃ B₃  │ F₄ B₄  │ F₅ B₅  │  ← Device 0
          ├────────┼────────┼────────┼────────┼────────┼────────┤
  vs=1    │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ F₃ B₃  │ F₄ B₄  │ F₅ B₅  │  ← Device 1
          ├────────┼────────┼────────┼────────┼────────┼────────┤
  vs=2    │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ F₃ B₃  │ F₄ B₄  │ F₅ B₅  │  ← Device 2
          ├────────┼────────┼────────┼────────┼────────┼────────┤
  vs=3    │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ F₃ B₃  │ F₄ B₄  │ F₅ B₅  │  ← Device 3
          ├────────┼────────┼────────┼────────┼────────┼────────┤
  vs=4    │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ F₃ B₃  │ F₄ B₄  │ F₅ B₅  │  ← Device 4
          ├────────┼────────┼────────┼────────┼────────┼────────┤
  vs=5    │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ F₃ B₃  │ F₄ B₄  │ F₅ B₅  │  ← Device 5
          ├────────┼────────┼────────┼────────┼────────┼────────┤
  vs=6    │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ F₃ B₃  │ F₄ B₄  │ F₅ B₅  │  ← Device 0
          ├────────┼────────┼────────┼────────┼────────┼────────┤
  vs=7    │ F₀ B₀  │ F₁ B₁  │ F₂ B₂  │ F₃ B₃  │ F₄ B₄  │ F₅ B₅  │  ← Device 1
          └────────┴────────┴────────┴────────┴────────┴────────┘
```

每个虚拟 stage 的 fwd/bwd 延迟为物理 stage 的 1/V。同一物理设备上的多个虚拟 stage
共享设备串行约束，list scheduler 的 device_free 机制确保它们不并行。

当 `vpp_chunks <= 1` 时，DualPipeV 退化为 DualPipe（与 `DualPipeVComposer` 行为一致）。

### 1.4 List Scheduling 算法

```
输入: 带三类依赖边的 GridTask 集合
算法: Kahn 拓扑排序 + 贪心最早开始

1. 构建 in_degree: 每个 task 的未就绪前驱数
2. ready_queue = [所有 in_degree=0 的 task]
3. while ready_queue:
     a. 选 start_time 最小的 task（start = max(所有前驱完成时间, 设备就绪时间)）
     b. 将 task 放入调度序列，更新 finish_time 和 device_free_time
     c. 释放所有依赖此 task 的后继（in_degree -= 1；若为 0 则入 ready_queue）
```

### 1.5 关键指标

```
step_time  = 从第一个任务开始到最后一个任务结束的总时间
warmup     = 从开始到 stage(pp-1) 开始第一个 BWD 的时间
steady     = step_time - warmup - cooldown
cooldown   = 从 stage 0 开始最后一个 BWD 到结束的时间
bubble     = step_time - M × per_stage_bottleneck

公式验证（1F1B，同构 stage）:
  step_time  = (M + pp - 1) × per_stage
  bubble     = (pp - 1) × per_stage
```

---

## 二、Chrome Trace 导出（三种视图）

### 2.1 `stitched.json` — 流水线网格视图

**pid = stage_id（每个 stage 一行），tid = 0**

每个事件 = 一个 GridTask（某个 stage 上某个 mb 的一次 FWD/BWD 完整块）。

对于带虚拟 stage 的调度（DualPipeV, VPP），pid 覆盖 `0 .. eff_pp-1`（所有虚拟 stage），而不仅是物理 stage。metadata 的 `process_name` 事件覆盖所有 pid，确保 Chrome Trace 中每个虚拟 stage 都有一行。物理 stage 数通过 `result.pp` 保存，显示范围通过遍历 `result.tasks` 中的实际 `stage_id` 确定。

```
pid=0 (GPU 0): [F₀][F₁][F₂][F₃]        [B₀][F₄][B₁][F₅][B₂][B₃][B₄][B₅]
pid=1 (GPU 1):    [F₀][F₁][F₂]      [B₀][F₃][B₁][F₄][B₂][F₅][B₃][B₄][B₅]
pid=2 (GPU 2):       [F₀][F₁]   [B₀][F₂][B₁][F₃][B₂][F₄][B₃][F₅][B₄][B₅]
pid=3 (GPU 3):          [F₀][B₀][F₁][B₁][F₂][B₂][F₃][B₃][F₄][B₄][F₅][B₅]
```

附带即时事件（ph="i"）标注 warmup/cooldown 分界线。

**适用场景**：全局宏观视角——看 PP 气泡分布、warmup/steady/cooldown 三段占比、跨 stage 的 FWD/BWD 级联。

### 2.2 `per_stage.json` — 单卡算子细节视图

**pid = stage_id，tid = stream_id（compute=0, TP comm=1, EP comm=2, ...）**

把 stitched 的每个大矩形"展开"成 DAGScheduler 记录的**具体算子序列**。每个 microbatch 的 fwd/bwd 算子按其对应的网格块 `fwd_base`/`bwd_base` 做时间平移。

```
pid=0 (GPU 0):
  tid=0 (compute): [matmul][attn][matmul][ffn]...[matmul_bwd][attn_bwd]...
  tid=1 (TP comm):    [all_reduce]                   [all_reduce]     ...
  tid=2 (EP comm):        [a2a_fwd]                  [a2a_bwd]       ...
```

所有 microbatch 共享同一组物理 stream 行（不按 mb 拆行），mb 号保存在 `args.mb` 中。

**适用场景**：微观视角——看 TP/EP 通信与计算的 overlap、通信气泡、算子粒度、单卡瓶颈分析。

### 2.3 `combined.json` — 两层叠加视图

**pid 0..pp-1 = 网格（同 stitched），pid pp..2*pp-1 = 细节（同 per_stage）**

网格在上方 pids，细节在下方 pids，同一 stage 垂直对齐。通过折叠/展开 pid 分组可在宏观和微观之间切换。

**适用场景**：在一个 trace 文件中同时呈现两种视角，适合做完整的端到端分析报告。

### 2.4 Chrome Trace 事件格式

```json
{
  "traceEvents": [
    {
      "ph": "X",
      "name": "▼ FWD [c] s0 m0",
      "cat": "compute",
      "pid": 0,
      "tid": 0,
      "ts": 0.0,
      "dur": 198.0,
      "args": {"phase": "fwd", "mb": 0, "stage": 0, "dep_count": 0}
    }
  ]
}
```

- `ph="X"`：完整事件（有 duration）
- `ph="i"`：即时事件（标注 warmup/cooldown 分界）
- `pid`：进程 = Stage/GPU
- `tid`：线程 = 流（compute/comm stream）
- `ts`/`dur`：起始时间和持续时长（微秒）

### 2.5 网格视图命名规则

stitched 视图中每个 GridTask 的事件名格式为 `<前缀> <microbatch_id>`：

| 前缀 | `task.phase` | 含义 |
|------|-------------|------|
| `F` | `fwd` | 前向计算（Forward） |
| `B` | `bwd` | 反向计算（Backward，未拆分） |
| `B_dx` | `bwd_dx` | 反向激活梯度计算（Backward dX，对激活值的梯度） |
| `B_dw` | `bwd_dw` | 反向权重梯度计算（Backward dW，对权重的梯度） |

**`B_dx` vs `B_dw` 区别**：
- `B_dx`（激活梯度）：计算 loss 对中间激活张量的梯度，用于向更早的 stage 传播。必须串行执行，无法推迟
- `B_dw`（权重梯度）：计算 loss 对该 stage 权重的梯度，仅在本 stage 使用。可推迟填空闲（ZeroBubble 核心特性）

非 ZB 调度（1F1B、DualPipe 等）全部使用 `B` 前缀。ZB 调度将 `B` 拆分为 `B_dx` + `B_dw`，颜色上 `B_dx` 略深于 `B_dw` 以方便视觉区分。

---

## 三、使用示例

### 3.1 PP 流水线拼接

```python
from python.zrt.executor.pp_stitcher import PPStitcher

stitcher = PPStitcher(
    stage_fwd_us={0: 100, 1: 80, 2: 120, 3: 90},   # 每 stage 前向耗时
    stage_bwd_us={0: 200, 1: 160, 2: 240, 3: 180},  # 每 stage 反向耗时
    pp=4, M=8,                                        # stage 数、microbatch 数
    p2p_latency_us=3,                                 # 单次跨 stage 传输
    schedule="1f1b",                                  # 调度策略
)
result = stitcher.stitch()
print(result.summary())
```

### 3.2 Chrome Trace 导出

```python
from python.zrt.executor.chrome_trace import ChromeTraceExporter

exporter = ChromeTraceExporter(time_unit="us")

# 模式 1: 纯网格
exporter.export_stitched(result, "stitched.json")

# 模式 2: 算子细节
exporter.export_per_stage(timelines, "per_stage.json", M=result.M, pp_stitched=result)

# 模式 3: 组合
exporter.export_combined(result, timelines, "combined.json")
```

### 3.3 运行 Demo

```bash
# PP 调度拼接演示
python demo_pp_stitcher.py

# Trace 导出演示
python demo_trace_export.py
# 输出: demo_trace/stitched.json, per_stage.json, combined.json
# 在 Chrome 中打开 chrome://tracing，加载任一 JSON 即可查看
```

---

## 四、设计决策与注意事项

1. **TP/EP/CP 特性已内嵌**：DAGScheduler 产出的 per-stage Timeline 已经包含了 TP all_reduce、EP all_to_all 等通信算子的耗时。PPStitcher 只负责跨 stage 的流水线编排，不重复计算 intra-stage 并行。

2. **List scheduler 非确定性**：当多个 task 具有相同的 start_time 时，出现顺序由 Python dict 遍历顺序决定。如果需要确定性的输出顺序，应在 grid 构建时对 task 做全排序。

3. **Bubble 公式**：`step_time - M × per_stage_bottleneck` 在异构 stage（各 stage 的 fwd+bwd 不同）时可能低估实际气泡，因为快 stage 的等待时间不仅取决于瓶颈 stage。此时应参考 stitched 视图中的直观空白段。

4. **去重**：`_chain_on_device` 中添加边时必须检查 `prev not in tasks[tid].dependencies`，避免 activation dep 和 chain dep 产生重复边 → in_degree 错误 → 任务永久挂起。

---

## 五、Per-Stage 延迟失衡与重分配机制

### 5.1 问题背景

训练图捕获中，backward ops 由 autograd 生成，缺乏模块上下文（`layer=''`）。PipelineParallelPass 将所有 `layer=''` 的节点分配到 `stage_id=0`，包括：

- 552+ 个 autograd backward ops（无 module 跟踪）
- 21 个 embedding/head forward ops（不属于任何 transformer 层）

这导致 stage 0 的 DAGScheduler 延迟远大于其他 stage（4.3x 失衡），PP 流水线出现严重瓶颈。

**示例数据（DeepSeek-V4, pp=4, tp=8）**：
```
Stage 0: fwd=353.3ms, bwd=353.5ms  (268 fwd_nodes, 823 bwd_nodes)
Stage 1: fwd=82.7ms,  bwd=87.9ms   (238 fwd_nodes, 194 bwd_nodes)
Stage 2: fwd=66.6ms,  bwd=84.2ms   (291 fwd_nodes, 194 bwd_nodes)
Stage 3: fwd=78.0ms,  bwd=78.2ms   (239 fwd_nodes, 195 bwd_nodes)
```

### 5.2 原始重分配逻辑（TrainingPipelinePass）

原始代码有两步修复：

- **Step 1**：若某 stage 的 bwd > 85% of total bwd → 按 fwd 比例重分配 bwd
- **Step 2**：若部分 stage 无 fwd ops → 均匀分布（homogeneous fallback）

**问题**：85% 阈值过于保守。实际失衡比 70% 未触发，但 4.3x 失衡已严重影响流水线效率。且无 fwd 重分配机制。

### 5.3 新设计：基于失衡比的重分配

**触发条件**：任一 stage 的 fwd 或 bwd > 2 × avg（即该 stage 占 >50% 总量），视为失衡。

**重分配算法**：

1. **识别"干净"stage**：fwd 和 bwd 均 ≤ 2×avg 的 stage
2. **从干净 stage 计算参考比例**：`ref_ratio[s] = clean_fwd[s] / total_clean_fwd`
3. **将参考比例扩展到所有 stage**（含失衡 stage）：
   - 对失衡 stage，用最近的干净 stage 比例或平均比例估算
4. **重分配**：
   - `stage_fwd[s] = total_fwd × ref_ratio[s]`
   - `stage_bwd[s] = total_bwd × ref_ratio[s]`
5. **若无干净 stage**（极端情况）：均匀分布 `total/pp`

**预期效果（DeepSeek-V4）**：
```
重分配前: Stage 0 fwd=353ms → 重分配后: ~120ms (均衡)
重分配前: Stage 0 bwd=353ms → 重分配后: ~126ms (均衡)
```

### 5.4 涉及文件

| 文件 | 修改 |
|---|---|
| `python/zrt/transform/analysis/training.py` | TrainingPipelinePass.run()：替换 85% 阈值 + 新增 fwd 重分配 |
| `tests/training/test_pipeline_parallel.py` | 新增失衡重分配的回归测试 |