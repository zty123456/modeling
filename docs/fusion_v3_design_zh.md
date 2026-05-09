# 算子融合 v3 设计 —— 基于 torch.export + DAG 子图匹配

> **状态（2026-05-09 更新）**：本文档为**未来路径**保留。本轮重构选择走 **v2 增量重构**
> （新富规则格式 + 三档匹配 + 语义推导 + AST/runtime discover），落地见
> `docs/fusion_v2_rich_rules_zh.md`。本 v3 设计仍是日后切换 torch.export 抓图通路的
> 参考蓝图，v2 富格式的 `MatchPattern` / `IORole` / `ShapeDerivation` 数据结构届时可
> 直接复用。

_2026-05-09。本文档规划用 PyTorch 官方 `torch.export` + `SubgraphMatcher` / Inductor `register_replacement` 替换当前 v2 的「`(scope, module_class)` 分桶 + aten op-tuple 严格相等」融合方案。_

---

## 1. 背景与动机

### 1.1 v2 现状

v2 在 `python/zrt/transform/fusion/algorithm.py` 实现，核心步骤：

1. `bucket_nodes_by_leaf_module` 把 dispatch 抓出的 OpNode 按 `(scope, module_class)` 分桶
2. `_merge_parent_groups` 把同 parent 下的连续叶桶合并，最大 60 op
3. 对每个桶 `lookup_rule_by_op_sequence`，用 **整 tuple 严格相等**去 yaml 规则的 `op_sequences` 里找

规则的产出方式：跑一遍 `run_trace_phases` 得到 `*_raw_graph.json`，人工或半自动地把每个 `module_class` 下观察到的 op 序列写进 `python/zrt/transform/fusion/rules/<model>.yaml`。

### 1.2 v2 的根本缺陷

| 缺陷 | 表象 |
|---|---|
| **顺序敏感** | 独立分支的 dispatch 顺序在不同进程/版本下不稳定，导致同一 nn.Module 抓出多个 tuple |
| **分支爆炸** | DSV4 yaml 里一个 Attention 类要枚举两条 14/30 算子序列；FP8 Linear 在文档里被记为「~40+ 种排列」 |
| **版本漂移** | `aten.add.Tensor` ↔ `aten.add_.Scalar`、`aten.matmul.default` ↔ `aten.linear.default` 等 PyTorch 版本变迁直接打穿匹配 |
| **匹配键失真** | `(scope, module_class)` 是项目自己从 forward hook 拼出来的，HF 里包含 `.cpu().numpy()` 的模块需要 `patches.py` 兜底；`scope` 在并行子模块下定义不清 |
| **规则编写门槛高** | 一条规则 = 一组 14~30 个 aten 全限定名的 list，肉眼难审，手写易错 |

### 1.3 外部主流做法（已调研）

- **`torch.export` + `nn_module_stack`**：官方 PT2 路径。导出器会在每个 FX node 的 `meta["nn_module_stack"]` 里记录该 node 对应的 nn.Module qualname 与类型链，原生覆盖 v2 想做的 scope 信息。
- **`torch.fx.SubgraphMatcher`**（`replace_pattern_with_filters`）：DAG 子图匹配，PT2 量化、PT2 transformations 教程的官方匹配器。`ignore_literals=True` 解决字面量漂移；`match_filters` 加形状/dtype 谓词。
- **`torch._inductor.pattern_matcher.register_replacement`**：vLLM/SGLang 生产融合通路使用的 API。把 `search_fn` / `replacement_fn` 写成普通 Python，框架自动 trace 成 `PatternExpr` DAG；支持多输出（`MultiOutputPattern`）；可序列化预编译。
- 其他（TVM Relax DFPattern、MLIR PDLL、TASO/PET）生态不匹配或学术级，不交付。

详细调研见会话记录与 `docs/fusion_v3_research_notes.md`（待补）。

---

## 2. 设计目标

### 2.1 目标

1. **规则可读**：一条规则 = 一个 ≤ 30 行的 `nn.Module` 参考实现，肉眼即可审核
2. **顺序无关**：DAG 拓扑等价的子图都能匹配，不再因 dispatch 顺序失败
3. **模块边界对齐官方语义**：用 `nn_module_stack` 替代自造的 `module_path/scope`
4. **CPU 可跑**：FakeTensorMode 下 `torch.export(strict=False)` 能跑通 Llama3/Qwen2/DSV3/DSV4，无 GPU/NPU 依赖
5. **下游零修改**：融合产物仍是 `OpGraph`，`transform/`、`executor/`、`simulator/`、`report/` 不动
6. **v2 可回滚**：CLI/API 提供 `--fusion=v2|v3` 开关，过渡期共存

### 2.2 非目标

- 不替换 `OpGraph` IR
- 不重写 `transform/parallel`、`executor`、`simulator`
- 不在 v3 阶段引入子图自动发现（TASO/PET 风格）—— 规则仍由人工编写，但门槛大幅降低
- 不追求「比 v2 更快」—— v3 单次 export 时间预计 1.5~2× v2 的 dispatch 抓图，可接受
- 不引入 GPU/NPU 才能跑的依赖（如 Inductor 后端编译）

---

## 3. 整体架构

### 3.1 旧 vs 新

```
v2（保留作 fallback）
  HF model
    │ FakeTensorMode + ModuleTracker(forward hooks)
    ▼
  TorchDispatchMode → records[]
    │ records_to_opgraph
    ▼
  OpGraph (raw)
    │ FusionPass(yaml rules)
    │   bucket_nodes_by_leaf_module
    │   _merge_parent_groups
    │   lookup_rule_by_op_sequence (tuple ==)
    ▼
  OpGraph (fused)

v3（新主路径）
  HF model
    │ FakeTensorMode
    ▼
  torch.export.export(model, fake_args, strict=False)
    │
    ▼
  ExportedProgram
    │ run_decompositions(core_aten_decompositions())
    ▼
  FX GraphModule  (每个 node 带 meta["nn_module_stack"], meta["val"])
    │
    │ FusionPassV3
    │   ├─ Tier-1: SubgraphMatcher.replace_pattern_with_filters(rule.pattern_module)
    │   │            for rule in registry sorted by priority
    │   └─ Tier-2: register_replacement(search_fn, replacement_fn, ...)
    │              （多输出 / 复杂控制流的少数规则）
    ▼
  FX GraphModule (fused, 含 fused_call_function 节点)
    │ fx_to_opgraph
    ▼
  OpGraph (fused) ──→ 下游 transform / executor / simulator 不变
```

### 3.2 关键解耦点：FX → OpGraph 桥接

下游所有 pass 与 simulator 都建立在 `OpGraph` 之上。v3 在 FX 层做完融合后，通过 `python/zrt/ir/fx_to_opgraph.py` 把融合后的 GraphModule 转换为 OpGraph：

| FX node 字段 | OpGraph 字段 | 备注 |
|---|---|---|
| `node.name` | `OpNode.id` | 加 `op_` 前缀对齐 v2 |
| `node.target.__qualname__` | `OpNode.op_type` | 融合节点的 target 是替换函数，op_type 取自规则 metadata |
| `node.meta["val"]` | `OpNode.inputs / outputs (TensorMeta)` | FakeTensor → shape + dtype |
| `node.meta["nn_module_stack"]` | `OpNode.scope`, `module_class` | 取最深一层（leaf module）即对齐 v2 语义 |
| `node.meta["nn_module_stack"]`（解析） | `OpNode.layer` | 用 regex `layers\.(\d+)` 抽层号 |
| 规则元数据 | `OpNode.fused_from`, `provenance` | 规则注册时声明 |

转换一次性完成，下游 `OpGraph.metadata` 上加 `fusion_version: "v3"` 标识。

---

## 4. 核心模块设计

### 4.1 `python/zrt/graph/fx_capture_export.py`

```python
def capture_via_export(
    model: nn.Module,
    fake_args: tuple,
    *,
    fake_kwargs: dict | None = None,
    fake_mode: FakeTensorMode | None = None,
    decompose: bool = True,
    dynamic_shapes: dict | None = None,
    strict: bool = False,
) -> torch.fx.GraphModule:
    """Capture *model* via torch.export, optionally apply core_aten decomposition.

    Returns the underlying GraphModule of the ExportedProgram, with
    ``meta['nn_module_stack']`` and ``meta['val']`` preserved on every
    call_function node.
    """
```

要点：
- `strict=False` 默认，避开 HF 自定义代码的导出限制
- `fake_mode` 透传，复用调用方的 FakeTensorMode 上下文（与现有 `model_loader.py` 一致）
- `dynamic_shapes` 留口子，DSV4 Indexer 等控制流敏感子图可声明动态维避免特化
- 失败时打印 `_explain_export_failure(model)`：给出哪个 module 的哪个 forward 在 export 阶段抛错的诊断

### 4.2 `python/zrt/transform/fusion/v3/rule.py`

```python
@dataclass(frozen=True)
class RuleV3:
    pattern_module: nn.Module          # 一段引用实现（PoC：RMSNormPattern）
    op_type: str                       # 融合后节点的 op_type（如 "rms_norm"）
    priority: int = 10
    match_filter: Callable[[InternalMatch], bool] | None = None
    ignore_literals: bool = True
    io_roles: tuple[IORole, ...] = ()  # (role, source_kind, source_index)
    annotations: dict = field(default_factory=dict)
    backend: Literal["fx_matcher", "inductor"] = "fx_matcher"
    # backend == "inductor" 时使用：
    search_fn:      Callable | None = None
    replacement_fn: Callable | None = None
    extra_check:    Callable | None = None
```

设计要点：
- **统一规则容器**：Tier-1 / Tier-2 复用同一 `RuleV3`，靠 `backend` 区分，避免下游分裂
- `pattern_module` 模式（Tier-1）：把规则写成参考 nn.Module，由 SubgraphMatcher 内部做 `torch.export(pattern_module)` 后做 DAG 匹配
- `search_fn / replacement_fn` 模式（Tier-2）：直接给 Inductor pattern_matcher 用
- `io_roles` 替代 v2 的 `IOSpec`：在融合节点上标注哪个输入是 activation / weight / bias 等，供下游 simulator 和 report 消费

### 4.3 `python/zrt/transform/fusion/v3/registry.py`

```python
_RULES: list[RuleV3] = []                              # 全局有序

def register(rule: RuleV3) -> None: ...
def all_rules(*, sorted_by_priority: bool = True) -> list[RuleV3]: ...
def clear() -> None: ...                                # for tests
```

差异点：
- 不再按 module class 名分桶（v2 的 `_STRING_RULES` 模式）。规则之间靠 `priority` 排序，`replace_pattern` 自身能避免重叠匹配（`remove_overlapping_matches=True` 是默认）。
- 规则**全局唯一**：移除 v2 platforms/ 的「按 model_id 字符串前缀加载」启发式。模型差异通过规则文件的 import 显式声明（见 §5.4）。

### 4.4 `python/zrt/transform/fusion/v3/matcher.py`

```python
def fuse_graphmodule(
    gm: torch.fx.GraphModule,
    rules: list[RuleV3],
) -> torch.fx.GraphModule:
    """Apply rules in priority order. Returns a new GraphModule."""
    for rule in rules:
        if rule.backend == "fx_matcher":
            _apply_fx_matcher(gm, rule)
        else:
            _apply_inductor(gm, rule)
        _repair_module_stack(gm, rule)
    return gm
```

`_repair_module_stack` 是关键修复：`replace_pattern_with_filters` 在替换后会丢失 fused 节点的 `nn_module_stack`，需要把被替换子图节点的 LCA（最近公共祖先模块）作为 fused 节点的 `module_stack` 写回去。否则下游 `fx_to_opgraph` 拿不到 scope/module_class。

### 4.5 `python/zrt/ir/fx_to_opgraph.py`

```python
def fx_to_opgraph(
    gm: torch.fx.GraphModule,
    *,
    name: str,
    phase: str,
    metadata: dict | None = None,
) -> OpGraph: ...
```

实现细节：
- 跳过 `placeholder` / `output` / `get_attr` 节点（对应 v2 中过滤掉的 tensor-source 节点）
- 通过 `meta["val"]` 提取 shape/dtype；FakeTensor 已携带，无需额外计算
- 边由 FX node 的 `args` / `users` 关系生成
- 融合节点（target 是规则注册的替换函数）的 `op_type` 取规则元数据，`fused_from` 取被替换前的 aten 名集合（Tier-1 在 `_apply_fx_matcher` 阶段记录到 node.meta；Tier-2 由 `register_replacement` 的回调记录）

### 4.6 `python/zrt/transform/fusion/v3/api.py`

```python
class FusionPassV3(GraphPass):
    name = "fusion_v3"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        # 注意：v3 不接收上游 raw OpGraph，而是从 ctx 拿 GraphModule。
        # 见 §6.1：调用约定
        gm = ctx.fx_graphmodule
        rules = registry.all_rules()
        fused_gm = fuse_graphmodule(gm, rules)
        return fx_to_opgraph(
            fused_gm, name=graph.name, phase=graph.phase,
            metadata={**graph.metadata, "fusion_version": "v3"},
        )
```

---

## 5. 规则编写规范

### 5.1 Tier-1：参考实现 nn.Module（首选）

适用：单输出、无副作用、子图大小 ≤ 30 op 的规则。

```python
# python/zrt/transform/fusion/v3/rules/rms_norm.py
import torch
from torch import nn

class RMSNormPattern(nn.Module):
    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        var = (x.float() ** 2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(var + 1e-6)).to(x.dtype) * weight

RULE = RuleV3(
    pattern_module=RMSNormPattern(),
    op_type="rms_norm",
    priority=10,
    ignore_literals=True,        # 让 1e-6 / dim=-1 这类字面量被通配
    io_roles=(
        IORole("activation", source_kind="input", source_index=0),
        IORole("weight",     source_kind="input", source_index=1),
        IORole("output",     source_kind="output", source_index=0),
    ),
)
```

写规则的人**只需要**：
1. 写一段 PyTorch 代码，描述被融合算子的语义
2. 标注 op_type、priority、IO 角色
3. 不需要列 aten 序列、不需要管 dispatch 顺序、不需要管字面量

### 5.2 Tier-2：search_fn / replacement_fn（多输出、控制流）

适用：MLA Attention（输出 attn_out + KV cache 更新）、Compressor（含 `new_full + copy_` 并行分支）、Indexer（`arange + where` 控制依赖）、FP8 Linear（多种 quant 排列）。

```python
# python/zrt/transform/fusion/v3/rules/v4/linear_fp8.py
def linear_fp8_search(x, w, scale_max=448.0):
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = amax / scale_max
    x_q = (x / scale).to(torch.float8_e4m3fn)
    return torch.mm(x_q.to(x.dtype), w)

def linear_fp8_replace(x, w, scale_max=448.0):
    return _fused_linear_fp8(x, w, scale_max)   # 占位 callable，op_type 走规则元数据

RULE = RuleV3(
    op_type="quant_linear",
    priority=8,
    backend="inductor",
    search_fn=linear_fp8_search,
    replacement_fn=linear_fp8_replace,
    extra_check=lambda match: match.kwargs.get("scale_max") in (448.0, 240.0),
)
```

vLLM `vllm/compilation/passes/fusion/rms_quant_fusion.py` 是参考实现。

### 5.3 IO 角色标注 (`IORole`)

```python
@dataclass(frozen=True)
class IORole:
    role: Literal["activation", "weight", "bias", "output", "kv_cache", ...]
    source_kind: Literal["input", "output"]
    source_index: int                  # pattern_module forward 签名中的位置
```

下游 simulator 通过 fused 节点的 `provenance` 查 IO 角色（与 v2 一致），不变。

### 5.4 模型相关规则的组织

```
python/zrt/transform/fusion/v3/rules/
├── __init__.py          # load_default()：注册 dense + RMSNorm + SDPA + ...
├── rms_norm.py
├── layer_norm.py
├── linear.py
├── attention_sdpa.py
├── swiglu.py
├── moe/
│   ├── __init__.py      # load_moe()
│   ├── gate_topk.py
│   ├── expert_swiglu.py
│   └── reduce.py
└── v4/
    ├── __init__.py      # load_v4()
    ├── hc_pre.py
    ├── hc_post.py
    ├── hc_head.py
    ├── attention.py     # MLA
    ├── compressor.py
    ├── indexer.py
    ├── moe_reduce.py
    └── linear_fp8.py
```

加载策略（替代 v2 的 `platforms/__init__.py:load_platform_rules`）：
- 默认加载 `rules/load_default()` —— 覆盖 dense + 标准 attention
- 用户在 `TransformContext.fusion_rule_packs` 列表中显式声明追加 pack：`["moe", "v4"]`
- 不再用 `model_id` 字符串子串启发式判断（v2 痛点之一）

### 5.5 规则编写流程

```
1. 在 hf_models/<model>/inference/model.py 里找到要融合的 nn.Module
2. 复制其 forward 主干，删去与融合无关的辅助代码（assertion / logging / device dispatch）
3. 整理成 ≤ 30 行的 nn.Module，命名 <Concept>Pattern
4. 在 rules/ 下新建文件，注册 RuleV3
5. 写最小测试：构造一个含该 pattern 的小 nn.Module → 走 v3 pipeline → 断言匹配次数
6. （可选）跑 diff_against_v2.py 看新旧路径产物差异
```

预期：一条规则的编写 + 测试 ≤ 0.5 工作日（v2 yaml 模式平均 1~2 工作日）。

---

## 6. 关键设计决策

### 6.1 调用约定：FX GraphModule 怎么进 FusionPassV3

v2 的 `FusionPass.run(graph: OpGraph, ctx)` 接收 OpGraph。v3 需要 FX GraphModule。两条路：

| 方案 | 描述 | 决策 |
|---|---|---|
| A. `ctx.fx_graphmodule` 上下文字段 | 上游 capture 阶段把 GraphModule 塞进 ctx，v3 pass 从 ctx 取 | **采纳** |
| B. OpGraph.metadata 里塞 GraphModule | 避免改 ctx 类 | 拒绝：OpGraph metadata 应是可序列化的 |

`graph/main.py` 的 capture 改造：
- `--fusion=v2`：跑现有 `_trace_phase`（dispatch），输出 OpGraph，FusionPass v2 处理
- `--fusion=v3`：跑新 `_export_phase`（torch.export），输出 GraphModule，塞进 ctx，FusionPassV3 处理后输出 OpGraph

### 6.2 nn_module_stack 在替换后的修复

`replace_pattern_with_filters` 不会自动给 fused 节点设置 `meta["nn_module_stack"]`，下游 `fx_to_opgraph` 没有 scope 信息。修复方案：

```python
def _repair_module_stack(gm, rule):
    for node in gm.graph.nodes:
        if _is_fused_by_rule(node, rule) and "nn_module_stack" not in node.meta:
            replaced = node.meta.get("_v3_replaced_nodes", [])
            stacks = [n.meta.get("nn_module_stack", {}) for n in replaced]
            node.meta["nn_module_stack"] = _lca_stack(stacks)
```

`_lca_stack`：求 OrderedDict 的最长公共前缀。

### 6.3 字面量的处理

- Tier-1 默认 `ignore_literals=True`：1e-6、`dim=-1` 等通配
- 要锁定字面量时，在 `match_filter` 里写显式条件
- Tier-2 通过 `extra_check` 谓词检查 match 的具体 args

### 6.4 多 rule 优先级

`replace_pattern` 一次只跑一条规则。v3 按 `priority` 降序循环跑所有规则，每跑完一条就 `gm.recompile()`。复杂度 O(rules × graph_size)，rules 数量预期 ≤ 30，graph 节点 ≤ 5000，可接受。

### 6.5 失败时的诊断

v2 路径下未匹配 group 只能从日志看 module_class 名。v3 提供：

```bash
python -m python.zrt.transform.fusion.v3.debug \
    --model hf_models/llama3_8b --layer 0 \
    --rule-file rules/rms_norm.py
```

功能：
- 打印每条规则的匹配次数
- 打印未被任何规则覆盖的 FX 节点子图（按 `nn_module_stack` 分组）
- 输出 graphviz dot 标注 matched / unmatched

---

## 7. 与 v2 对照

| 维度 | v2 | v3 |
|---|---|---|
| 抓图 | `TorchDispatchMode` + `ModuleTracker` | `torch.export` + `run_decompositions` |
| 模块边界 | 自维护 `module_path/scope` + forward hooks | `node.meta["nn_module_stack"]`（官方） |
| 规则形态 | yaml 列出 aten 全名序列 | nn.Module 参考实现 |
| 匹配 | 整 tuple 严格相等 | DAG SubgraphMatcher |
| 字面量 | 内嵌于序列，敏感 | `ignore_literals` 通配 |
| 顺序敏感 | 是（独立分支也敏感） | 否（DAG 等价即可） |
| 多输出 | 不支持，靠多条规则枚举 | Tier-2 原生支持 |
| 模型选择 | `platforms/__init__.py` 字符串前缀启发式 | 显式 `fusion_rule_packs` 声明 |
| 单条规则编写工时 | 1~2 天 | ≤ 0.5 天 |
| CPU/无卡环境 | 支持（FakeTensor） | 支持（FakeTensor + export） |
| 下游 IR | OpGraph | OpGraph（无变化） |

---

## 8. 风险与对冲

| 风险 | 概率 | 对冲 |
|---|---|---|
| HF 自定义模型 `torch.export(strict=False)` 失败 | 中 | 扩展 `python/zrt/graph/patches.py`，新增 `apply_export_patches()`：把 `.cpu().numpy()` / `inference_mode` / 自定义 CUDA kernel 调用 monkey-patch 成 trace 友好等价物。Phase 0 先在 Llama3 跑通；DSV3 / V4 在 Phase 2/3 各做一次 patch 扩展 |
| `core_aten_decompositions` 把规则关心的 op 拆得太碎 | 中 | 用 `decomposition_table.pop(target)` 移除会破坏规则的分解；或干脆不分解（`run_decompositions` 可选） |
| `nn_module_stack` 在 `replace_pattern` 后丢失 | 高（已知） | `_repair_module_stack`（§6.2） |
| Tier-2（Inductor）API 在 PyTorch 升级后变动 | 低 | 通过 `inductor_bridge.py` 单点封装；锁定测试用 PT 版本 |
| FX → OpGraph 桥接信息丢失 | 中 | 在 `meta["_v3_*"]` 命名空间下携带 fused_from / provenance；测试覆盖序列化往返 |
| v2 yaml 规则迁移工作量大 | 中 | 不要求一次迁完。v2 在 `--fusion=v2` 下保留可用，新模型直接走 v3，旧模型按需补规则 |
| Indexer / Compressor 在 export 时控制流被特化 | 中（DSV4） | 用 `dynamic_shapes` 声明动态维；必要时让 Indexer 走 Tier-2 + `extra_check` |

---

## 9. 落地阶段

| 阶段 | 工时 | 主要交付 | 验收 |
|---|---|---|---|
| Phase 0 — 基础设施 + RMSNorm PoC | 1.5–2d | `fx_capture_export.py` / `fx_to_opgraph.py` / `RuleV3` / `FusionPassV3` / `RMSNormPattern` / Llama3 测试 | Llama3 8B v3 路径下 RMSNorm 替换次数 == `2L+1`，端到端 < 30s |
| Phase 1 — Dense 规则集合 | 1.5d | layer_norm / linear / attention_sdpa / swiglu / rotary / dropout 规则 | Llama3、Qwen2、Mistral 三个 dense 模型 v2/v3 op_type 直方图一致 |
| Phase 2 — MoE | 2d | moe gate / expert / reduce 规则；MLA Attention 规则；DSV3 跑通 | DSV3 4 层 v3 覆盖率 ≥ v2，未匹配数 ≤ v2 一半 |
| Phase 3 — DeepSeek-V4 | 2–3d | v4/* 8 类规则；Tier-2 通路与 inductor_bridge | DSV4 4 层 train_forward 匹配率 ≥ 95%；Excel 报告 8 类自定义模块均融合为单节点 |
| Phase 4 — 收尾 | 1d | CLI 默认切 v3；v2 yaml 移到 `_legacy_yaml/`；CLAUDE.md 更新 | `pytest tests/ -v` 在 `--fusion=v3` 默认下通过 |

总计 8–10 工作日。每阶段可独立验收/回滚。

---

## 10. 附录

### 10.1 公开 API 草案

```python
# python/zrt/transform/fusion/v3/__init__.py
from .rule import RuleV3, IORole
from .registry import register, all_rules, clear
from .api import FusionPassV3
from .matcher import fuse_graphmodule

__all__ = [
    "RuleV3", "IORole",
    "register", "all_rules", "clear",
    "FusionPassV3", "fuse_graphmodule",
]
```

### 10.2 测试目录结构

```
tests/transform/fusion_v3/
├── conftest.py                       # 提供 (model_id, expected_fused_classes) fixture
├── test_capture_export.py            # torch.export 抓图单测
├── test_fx_to_opgraph.py             # 桥接单测
├── test_rms_norm_match.py            # PoC 规则匹配单测
├── test_dense_parity.py              # Llama3/Qwen2/Mistral v2/v3 一致性
├── test_moe_parity.py                # Mixtral / DSV3
├── test_v4_coverage.py               # DSV4 8 类覆盖率
└── test_repair_module_stack.py       # nn_module_stack 修复
```

### 10.3 文档清单

| 文档 | 状态 |
|---|---|
| `docs/fusion_v3_design_zh.md` | 本文 |
| `docs/fusion_v3_authoring_zh.md` | Phase 1 期间产出，含 4 个 cookbook 示例 |
| `docs/fusion_v2_to_v3_migration_zh.md` | Phase 4 产出 |
| `docs/fusion_v3_research_notes.md` | 外部调研笔记（torch.fx / Inductor / vLLM / TVM 等链接） |

### 10.4 关键参考链接

- [torch/fx/subgraph_rewriter.py](https://github.com/pytorch/pytorch/blob/main/torch/fx/subgraph_rewriter.py)
- [Pattern Matching with Literal Arguments PR #143147](https://github.com/pytorch/pytorch/pull/143147)
- [Writing Graph Transformations on ATen IR](https://docs.pytorch.org/docs/stable/torch.compiler_transformations.html)
- [How is pattern matching in inductor/fx implemented?](https://dev-discuss.pytorch.org/t/how-is-pattern-matching-in-inductor-fx-implemented/1720)
- [Learn by doing: TorchInductor Pattern Matcher](https://karthick.ai/blog/2026/Learn-By-Doing-Torchinductor-Pattern-Matcher/)
- [vLLM Fusion Passes design doc](https://docs.vllm.ai/en/stable/design/fusions/)
- [torch.export docs (2.9)](https://docs.pytorch.org/docs/stable/export.html)
- [Export sub-graphs at the Aten IR level](https://dev-discuss.pytorch.org/t/export-sub-graphs-at-the-aten-ir-level/1936)
