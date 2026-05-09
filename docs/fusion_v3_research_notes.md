# 算子融合 v3 — 外部调研笔记

_2026-05-09。本文档收录 v3 设计阶段的外部调研产出，与 `docs/fusion_v3_design_zh.md` 配套使用。_

---

## 1. 调研动机与方法

### 1.1 v2 缺陷的一句话回顾

`docs/fusion_v3_design_zh.md` §1.2 已系统列举 v2 五大缺陷（顺序敏感 / 分支爆炸 / 版本漂移 / 匹配键失真 / 编写门槛高）。本文不重复，仅指出共同根因：**「按 dispatch 顺序抓出的 aten 全限定名 tuple 整段相等」是一种比 DAG 更强的等价关系**——任何破坏顺序的扰动都会让规则脱靶。调研的目标即是寻找一种**仅依赖 DAG 拓扑等价**的成熟方案。

### 1.2 调研路径

调研集中在 2026-05-08 进行，方法上沿三条线索展开：

1. **官方文档与源码**：`pytorch/pytorch` 的 `torch/fx/`、`torch/_inductor/`、`torch/export/` 三个目录，以及 PyTorch dev-discuss 上的设计帖。
2. **PR 与 Issue**：通过关键词检索 PyTorch GitHub PR / Issue（如 PR #143147 引入 `ignore_literals`），定位上游近期对 SubgraphMatcher / Inductor pattern_matcher 的改动。
3. **业界推理框架**：vLLM、SGLang、TensorRT-LLM、HuggingFace Optimum 的融合通路，重点看它们「规则用什么形态写 / 用什么 API 匹配 / 多输出如何处理」。

调研结果汇总成本笔记。详细的「3 优先推荐 + 6 拒绝备选」结论已写进设计文档 §1.3。

### 1.3 排除项

- **纯学术工作**（TASO / PET / Tensat / EquivSet）：研究级，无 turn-key 工程实现，调度器 + 验证器都需要二次开发。
- **生态不匹配项**（TVM Relax DFPattern、MLIR PDLL、IREE）：API 思想可借鉴，但要求项目切换运行时（Apache TVM / MLIR），不在 v3 阶段考虑。
- **黑盒商业框架**（TensorRT、cuDNN graph API）：不开源、不可改造，无法承载用户自定义算子的融合规则。

---

## 2. PyTorch 官方融合机制

### 2.1 `torch.export`

#### 2.1.1 export() vs jit.trace 的语义差异

`torch.export.export(model, args, kwargs, *, strict=True, dynamic_shapes=None)` 与早期 `torch.jit.trace` 的本质差异在于：

| 维度 | `torch.jit.trace` | `torch.export.export` |
|---|---|---|
| 输出形态 | `ScriptModule`（TorchScript IR） | `ExportedProgram`（FX GraphModule + signature + state_dict） |
| 动态控制流 | 无法捕获，trace 只看到一条路径 | `strict=False` 下通过 fake tensor + symbolic shape 部分支持；data-dependent 控制流仍受限 |
| Guards | 无 | 输入约束被记录为 guards，运行时校验 |
| 模块层次 | 全部 inline | 通过 `node.meta["nn_module_stack"]` 完整保留 |
| 副作用 | 静默丢弃 | `non-strict` 下作 best-effort 保留；`strict` 下报错 |

`strict=True`（默认）走 TorchDynamo 加严格捕获，对 HF 自定义代码（DeepSeek、Qwen3-MoE）经常失败；`strict=False` 走 non-strict 路径，使用 Python 解释 + FakeTensor 推理，兼容性显著更好但仍非万能。**v3 默认 `strict=False`**（见 `python/zrt/graph/fx_capture_export.py:103-109`）。

`dynamic_shapes` 是规避控制流特化（specialization）的官方手段。例如 DSV4 Indexer 内的 `arange(seq_len)` 若 `seq_len` 被特化成具体常量，规则在不同 `seq_len` 下匹配会失败；声明 `Dim("seq", min=1, max=4096)` 让该维保持符号化即可。

#### 2.1.2 ExportedProgram 与 GraphModule 的关系

```
ExportedProgram
├── graph_module        : torch.fx.GraphModule    ← v3 实际消费的对象
├── graph_signature     : ExportGraphSignature    （input/output 角色）
├── state_dict          : dict[str, Tensor]       （weight/buffer）
├── range_constraints   : dict[Symbol, ValueRange]
└── module_call_graph   : list[ModuleCallEntry]   （用于 unflatten）
```

`ep.module()` 返回的就是 `graph_module`。v3 capture 路径调用 `ep.module()` 后直接传给 SubgraphMatcher（`fx_capture_export.py:112`），无需操作其它字段。

#### 2.1.3 `run_decompositions(core_aten_decompositions())` 的作用与代价

`core_aten_decompositions()`（PT 2.5+ 改名为 `default_decompositions()`）返回一张 ~180 条 op → 实现函数的映射表。`ep.run_decompositions(table)` 会用这些函数把 prim_aten / aten 高层 op 拆成核心 aten 集合，得到的图：

- **优点**：op 集合稳定；版本漂移大幅缓解（高层 API 变了不影响 core aten）。
- **代价**：高层 op 被打散。例如 `nn.Linear(no bias)` 拆成 `aten.permute([1,0]) → aten.mm`；`F.silu` 拆成 `_to_copy → sigmoid → mul → _to_copy`（bf16 输入时）。规则必须按拆分后的形态写。

v3 通过 **「pattern 与 target 走同一条 decomp 路径」** 解决——`matcher.py::_trace_pattern` 显式调用 `capture_via_export(rule.pattern_module, args, decompose=True)`，确保两侧形状对齐。

#### 2.1.4 `nn_module_stack` 与 `val` FakeTensor metadata 的来源

`torch.export` 在每个 call_function node 的 `node.meta` 上写入：

| key | 类型 | 含义 |
|---|---|---|
| `nn_module_stack` | `OrderedDict[str, (qualname, type)]` | 从 root 到该 op 所属 leaf module 的完整模块路径链 |
| `val` | `FakeTensor` 或其他 example value | 该 node 输出的 shape / dtype / stride（FakeTensor 形态） |
| `stack_trace` | `str` | 源代码栈（用户代码行号） |
| `original_aten` | `OpOverload` | 分解前的高层 aten op（仅在 `run_decompositions` 后存在） |

`nn_module_stack` 是 v3 取代 v2 自维护 `(scope, module_class)` 的关键——它由 export 框架原生维护，覆盖率更全（包含分解后的内部 op）、更稳（不依赖 forward hook 的执行顺序）。

#### 2.1.5 已知 export 兼容性陷阱

调研中识别出三类对 HF 模型常见的失败模式：

1. **data-dependent control flow**：例如 `if mask.any(): ...`，`strict=True` 下直接报错；`strict=False` 下走 Python 解释，但只捕获被 FakeTensor 触发的那一支。**对策**：`v3` 在已知热点（DSV4 Indexer、MoE gate）走 Tier-2 + `dynamic_shapes`，不强求 export 完整覆盖。
2. **`.cpu().numpy()` / `.tolist()`**：FakeTensor 上无对应实现，立即抛 `NotImplementedError`。**对策**：项目 `python/zrt/graph/patches.py` 已经针对 MoE 和 DSV3.2 Indexer 用 monkey-patch 兜底（详见 CLAUDE.md「patches.py 清单」段落）；v3 复用同一组 patch。
3. **`torch.inference_mode()` 上下文**：导致部分 autograd 元信息丢失，`run_decompositions` 后图结构不完整。**对策**：调用方需在 `eval()` 模式下不进入 `inference_mode`。

**参考帖**：[Export sub-graphs at the Aten IR level](https://dev-discuss.pytorch.org/t/export-sub-graphs-at-the-aten-ir-level/1936) 讨论了这些陷阱与缓解方案。

### 2.2 `torch.fx.SubgraphMatcher`

#### 2.2.1 DAG 子图匹配算法

源码：`torch/fx/passes/utils/matcher_utils.py`（核心类 `SubgraphMatcher`）和 `torch/fx/subgraph_rewriter.py`（封装 `replace_pattern_with_filters`）。

算法本质是一个回溯式同构搜索：从 pattern 的 output node 出发，自顶向下、按 `args` 边逐步尝试与 target node 配对；任意一步失败即回溯。相比早期 `subgraph_rewriter` 仅支持线性序列，`SubgraphMatcher` 是**真正的 DAG 匹配**，能识别同一 `dropout` 同时被多个下游消费这种 fork-join 结构。

#### 2.2.2 关键参数

| 参数 | 作用 | v3 取值 |
|---|---|---|
| `match_output` | pattern 的 output 节点是否必须与 target output 对齐 | `False`（pattern 通常是子图） |
| `match_placeholder` | pattern 的 placeholder 是否必须对齐 target placeholder | `False`（外部输入可以是任何 producer） |
| `ignore_literals` | 标量字面量是否通配 | `True`（默认） |
| `remove_overlapping_matches` | 重叠匹配只保留先到的 | `True` |

`ignore_literals=True` 解决了 v2 的「字面量漂移」问题（`eps=1e-6` vs `eps=1e-5`、`dim=-1` vs `dim=2` 等）。该参数由 [PR #143147](https://github.com/pytorch/pytorch/pull/143147) 引入，明确为 PT2 量化团队设计——量化 pass 中大量出现 `quantize_per_tensor(x, scale, zp)` 类形如「op + 标量参数」的图，固定字面量会让规则无法跨 calibration 复用。

#### 2.2.3 InternalMatch 数据结构

匹配结果以 `InternalMatch` namedtuple 返回：

```python
@dataclass
class InternalMatch:
    anchors: list[Node]                       # pattern output 在 target 中的对应节点
    nodes_map: dict[Node, Node]               # pattern node → target node
    placeholder_nodes: list[Node]             # pattern placeholder 在 target 中的对应（外部 producer）
    returning_nodes: list[Node]               # pattern output 节点
```

v3 主要消费 `nodes_map`：遍历 pattern 中的 call_function 节点，取 target 端对应节点打 fused 标签。`placeholder_nodes` 在某些场景（见 §2.2.5 的 KeyError）需要被故意置空。

#### 2.2.4 `replace_pattern` / `replace_pattern_with_filters` 接口

```python
def replace_pattern_with_filters(
    gm: GraphModule,
    pattern: Union[Callable, GraphModule],
    replacement: Union[Callable, GraphModule],
    match_filters: list[Callable[[InternalMatch, Graph, Graph], bool]] | None = None,
    ignore_literals: bool = False,
) -> list[ReplacedPatterns]
```

`pattern` 与 `replacement` 既可以是 `Callable`（内部会 `make_fx` trace 成 GraphModule），也可以直接传 GraphModule。`match_filters` 是谓词列表，对每个候选 match 全部通过才接受。

**v3 没有调用此 API**——见 §2.2.5 的决策记录。

#### 2.2.5 Phase 0 实施过程中发现的真实坑

以下 4 个坑均在 Phase 0 实施过程中（2026-05-08 ~ 2026-05-09）调试出来，是本笔记最有价值的内容。它们决定了 `python/zrt/transform/fusion/v3/matcher.py` 当前的 "tag, do not replace" 策略。

##### 坑 1：pattern 中 `nn.Parameter` 与目标 FakeTensor 不通过严格 `type()` 检查

**现象**：把 RMSNorm 的 weight 写成 `self.weight = nn.Parameter(torch.randn(d))`，trace 后变成 `get_attr` 节点持有 `Parameter`。匹配 HF Llama 时永远 0 命中。

**根因**：`SubgraphMatcher._match_attributes` 对两个 `get_attr` 节点的属性做 `type(a) is type(b)` 严格类型比较。pattern 端是真实 `Parameter`，target 端在 FakeTensorMode + export 后是 `FakeTensor`，类型不一致直接判否。

**对策**：把 weight 提升为 forward 参数。这样 pattern 端变成 `placeholder`，配合 `match_placeholder=False` 后该 placeholder 被当作通配符，能匹配 target 中任意 producer（包括 `get_attr Parameter`）。`python/zrt/transform/fusion/v3/rules/rms_norm.py:8-13` 的 docstring 完整记录了这个坑。

```python
# 错误写法
class RMSNormPattern(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))  # 永远不匹配
    def forward(self, x):
        ...
        return self.weight * h

# 正确写法
class RMSNormPattern(nn.Module):
    def forward(self, x, weight):  # weight 改作 forward 参数
        ...
        return weight * h
```

##### 坑 2：`ignore_literals=True` + placeholder 缺失节点导致 KeyError

**现象**：开启 `ignore_literals=True` 后，`SubgraphMatcher.match()` 在某些规则上抛 `KeyError: <某个 placeholder 节点>`，整轮匹配结果为空。

**根因**：当 pattern 的某个 call_function 的 arg 在 target 中匹配到一个**字面量（非 Node）**时，`ignore_literals=True` 让匹配静默接受这一对，但**不会向 `nodes_map` 写入对应的 placeholder**。回溯到底端时 `placeholder_nodes = [nodes_map[pn] for pn in self.pattern_placeholder_nodes]` 这行查表 KeyError，把整个搜索连带已经成功的分支一并撕掉。

**对策**：v3 不消费 `placeholder_nodes`（自己只看 `nodes_map`），所以可以**直接把 matcher 的 `pattern_placeholder_nodes` 置空**绕过该 list comprehension。代码：

```python
# python/zrt/transform/fusion/v3/matcher.py:99-108
matcher = SubgraphMatcher(
    pattern=pattern_gm.graph,
    match_output=False,
    match_placeholder=False,
    remove_overlapping_matches=True,
    ignore_literals=rule.ignore_literals,
)
matcher.pattern_placeholder_nodes = []  # workaround
```

注释中明确标注「Workaround for upstream SubgraphMatcher」，避免后人误删。

##### 坑 3：replace 后 `nn_module_stack` metadata 丢失

**现象**：调用 `replace_pattern_with_filters` 替换子图后，新插入的 fused 节点 `node.meta["nn_module_stack"]` 为空。下游 `fx_to_opgraph` 拿不到 scope 信息，融合节点的 `module_class` 字段为 `None`。

**根因**：`subgraph_rewriter` 对插入的新节点不会自动继承被替换子图的 metadata。Pytorch 上游对此长期未修复（[Export sub-graphs at the Aten IR level](https://dev-discuss.pytorch.org/t/export-sub-graphs-at-the-aten-ir-level/1936) 讨论提到过）。

**对策**：v3 的 `matcher.py` 改为「**tag, do not replace**」策略——根本不调 `replace_pattern_with_filters`，只用 `SubgraphMatcher.match()` 定位匹配子图，在被匹配节点的 `node.meta` 上打三个 tag：

```python
node.meta["_v3_fused_op_type"]  = rule.op_type
node.meta["_v3_fused_match_id"] = "v3_<op_type>_<n>"
node.meta["_v3_fused_priority"] = rule.priority
```

`fx_to_opgraph` 桥接时按 `_v3_fused_match_id` 把同一 match 的所有节点折叠成单个 `OpNode`，`nn_module_stack` 直接从被打 tag 的某个内部节点取（它本来就携带正确的 stack）。设计文档 §6.2 描述的 `_repair_module_stack` 因此变成「不需要修复」。

代码：`python/zrt/transform/fusion/v3/matcher.py:1-19` 的模块 docstring 解释了 tag-only 策略的动机。

##### 坑 4：export 注入的 `_assert_tensor_metadata` / `_guards_fn` 节点干扰匹配

**现象**：直接对 `ep.module()` 跑 SubgraphMatcher，明明 pattern 形状对得上，匹配结果仍 0 命中。打印 FX 图发现散布大量 `aten._assert_tensor_metadata.default` 调用和一个 `call_module` 形式的 `_guards_fn` 节点。

**根因**：`torch.export` 出于运行时保护目的，在 graph 头部插入 `_guards_fn`（symbolic-shape 校验），并在每个产生 FakeTensor 的位置插入 `_assert_tensor_metadata`（dtype/device 校验）。这些节点没有用户（`node.users` 为空），但**位于 FX graph 中**，让 pattern 与 target 的 DAG 拓扑对不上。

**对策**：在 `capture_via_export` 出口前 strip 掉这些副作用节点：

```python
# python/zrt/graph/fx_capture_export.py:37-60
def _strip_export_artifacts(gm: torch.fx.GraphModule) -> None:
    to_erase = []
    for n in gm.graph.nodes:
        if n.op == "call_function" and "_assert_tensor_metadata" in str(n.target):
            to_erase.append(n)
        elif n.op == "call_module" and isinstance(n.target, str) and n.target.startswith("_guards"):
            to_erase.append(n)
    for n in to_erase:
        if not n.users:
            gm.graph.erase_node(n)
```

只删无 user 的节点，不影响图语义。

##### 坑 5：分解后 `_to_copy` 出现/消失取决于输入 dtype

**现象**：用 fp32 example inputs 写 RMSNormPattern，对 bf16 真实模型 0 命中。

**根因**：HF RMSNorm 写法是 `h = x.to(torch.float32); ...; return weight * h.to(input_dtype)`。当输入已经是 fp32 时，两个 `.to()` 都是 no-op，`_to_copy` aten 节点根本不出现；输入是 bf16 时则出现两个 `_to_copy`。pattern 与 target 在 cast 个数上不匹配。

**对策**：所有 example inputs 一律用 bf16（与 HF 模型默认 dtype 一致）：

```python
# python/zrt/transform/fusion/v3/rules/rms_norm.py:74-81
def _example_inputs():
    return (
        torch.randn(1, 4, 4, dtype=torch.bfloat16),
        torch.randn(4, dtype=torch.bfloat16),
    )
```

##### 坑 6：结构相似的子图（RotaryEmbedding 内的 pow→mean→mul）需要 match_filter 排除

**现象**：RMSNorm pattern 在 Llama3 上匹配数高于预期，多余的命中分布在 `LlamaRotaryEmbedding` scope 内。

**根因**：RotaryEmbedding 计算 `inv_freq` 时也有 `pow→mean→add→rsqrt→mul→mul` 形态的 DAG，结构上等价于 RMSNorm，DAG 匹配会同时命中。

**对策**：每条规则强制配 `match_filter`，按 `nn_module_stack` 末位的 leaf module class 名筛选。`python/zrt/transform/fusion/v3/rules/rms_norm.py:34-48` 的 `_all_internal_nodes_in_rmsnorm` 实现了这一点；其它规则（`linear.py`、`swiglu.py`、`attention_sdpa.py`、`rotary.py`）沿用同一模式。

> 这条不是 PyTorch 的坑，而是「DAG 同构匹配本身比模块级匹配更宽松」的固有副作用。**结论：v3 规则必须 always 配 match_filter，没有例外**。

##### 坑 7：`_is_contained` 拒绝多输出模块的内部节点泄漏

**现象**：MoEGate pattern 在 harness 上 0 命中，而 self-match 通过。pattern 和 target 的 DAG 结构完全一致（`linear→sigmoid→add→topk→getitem→gather→sum→div→mul`），但 SubgraphMatcher 返回 0 matches。

**根因**：`SubgraphMatcher._is_contained` 检查每个被匹配的 target 节点的所有 user 是否也在 match 内部。MoEGate 返回 `(topk_idx, topk_weight)` 二元组，其中 `getitem(topk, 1)`（topk_idx）被 match 内部的 `gather` 消费的同时，也作为 gate 的返回值被 match 外部的 `output` 节点消费。`_is_contained` 发现 `getitem_1` 有外部 user（output），且 `getitem_1` 对应的 pattern 节点不是 returning node，于是拒绝整个 match。

**对策**：v3 的 tag-only 策略不需要 containment — 我们不替换节点，只是标注。因此安全地 bypass `_is_contained`：

```python
# python/zrt/transform/fusion/v3/matcher.py
matcher._is_contained = lambda nodes_map: True
```

这个 workaround 安全因为：(1) 我们只标注不修改 FX 图结构；(2) `remove_overlapping_matches=True` 仍然防止同一节点被多条规则重复 claim；(3) 被标注节点的 external users 不受影响。

代码：`python/zrt/transform/fusion/v3/matcher.py:99-112`。

> **结论**：`_is_contained` 是为 replacement 设计的安全检查，tag-only 策略不需要它。HF 模型中多输出模块很常见（MoEGate、Attention 的 KV cache 输出等），bypass 此检查对 tag-only 场景是必要的。

### 2.3 `torch._inductor.pattern_matcher`

源码：`torch/_inductor/pattern_matcher.py`，是 Inductor 自身做高层融合（如 RMSNorm + quant、attention SDPA decomp）的引擎。

#### 2.3.1 `register_replacement` / `search_fn` / `replacement_fn` 三件套

```python
from torch._inductor.pattern_matcher import register_replacement, fwd_only

def search_fn(x, w):
    return some_decomp(x, w)

def replacement_fn(x, w):
    return fused_kernel(x, w)

register_replacement(
    search_fn=search_fn,
    replacement_fn=replacement_fn,
    example_inputs=[torch.randn(2, 8), torch.randn(8, 8)],
    trace_fn=fwd_only,
    pass_dict=my_patterns,           # PatternMatcherPass
    extra_check=lambda match: ...,   # 可选谓词
)
```

`search_fn` / `replacement_fn` 都是普通 Python；框架内部用 `make_fx` 把它们 trace 成 FX，再翻成 `PatternExpr` 树。匹配阶段直接遍历 `PatternExpr`，比从 source code 重新解析快得多。

#### 2.3.2 `PatternExpr` / `MultiOutputPattern`

PatternExpr 是 Inductor 自定义的 IR：

| 类 | 含义 |
|---|---|
| `CallFunction(target, *args)` | 一个 call_function 节点 |
| `CallMethod(method, self, *args)` | 一个 call_method 节点 |
| `KeywordArg(name)` | 命名通配符（外部输入） |
| `Ignored` | 任意值通配符 |
| `MultiOutputPattern([p1, p2])` | 顶层是多个输出 |
| `ListOf(pattern)` | 同一 pattern 重复多次（如 MoE 多 experts） |

`MultiOutputPattern` 是 Tier-2 相对 Tier-1 SubgraphMatcher 的核心优势——SubgraphMatcher 只支持单输出，MLA Attention 同时输出 attn_out 和 KV cache 更新，必须用 Tier-2。

#### 2.3.3 `extra_check` 谓词的用途

`extra_check(match: Match) -> bool` 在初步结构匹配通过后被调用，可读 `match.kwargs`（命名通配符的实际绑定值）做 shape / dtype / 数值范围检查。例如 FP8 Linear 规则要求 `scale_max in (448.0, 240.0)`，否则可能命中其它 quant 路径。

#### 2.3.4 序列化预编译（pickle 缓存）

`gen_register_replacement(unique_name, search_fn, replacement_fn, ...)` 在导入时 trace 一次并把 `PatternExpr` pickle 到磁盘（路径在 `torch/_inductor/fx_passes/serialized_patterns/`）。后续 import 直接反序列化，省掉每次启动 trace 的成本。

v3 Phase 3 才会引入 Tier-2，目前 `RuleV3` 已经预留 `backend="inductor"` 字段（`python/zrt/transform/fusion/v3/rule.py:72`）。

---

## 3. 业界实践

### 3.1 vLLM 融合通路

源码：`vllm/compilation/passes/`（GitHub: [vllm-project/vllm](https://github.com/vllm-project/vllm)）。

#### 3.1.1 目录结构

```
vllm/compilation/passes/
├── fusion/
│   ├── rms_quant_fusion.py       # RMSNorm + fp8 quant
│   ├── act_quant_fusion.py       # SiLU + fp8 quant
│   ├── attn_fusion.py            # Attention rewrite
│   └── ...
└── inductor_pass.py              # 接入 torch.compile pipeline
```

#### 3.1.2 典型规则：`rms_quant_fusion`

vLLM 的 RMSNorm + FP8 quant 融合，正是 §2.3 描述的 `register_replacement` API 的标准用法。`search_fn` 写一段「先 RMSNorm 再做 per-tensor quant」的纯 PyTorch 实现，`replacement_fn` 调用 vLLM 自带的 fused CUDA kernel。

设计要点：

- 规则写在 Python 里，**不是 yaml**——这与 v2 的 yaml 规则形成鲜明对比，也是 v3 转向 nn.Module 模式的依据之一。
- 多输出场景（`fused_add_rms_norm` 同时输出 normed_x 和 residual）通过 `MultiOutputPattern` 描述。

#### 3.1.3 与 torch.compile 的耦合方式

vLLM 把这些 pattern_matcher 规则注入 `torch._inductor.config.post_grad_custom_post_pass`，在 Inductor 后端 lowering 之前跑。流程是：

```
forward() → torch.compile(backend="inductor")
           → AOTAutograd → joint_graph
           → 走 vLLM 的 PatternMatcherPass
           → Inductor 继续 lowering
```

参考：[vLLM Fusion Passes design doc](https://docs.vllm.ai/en/stable/design/fusions/)、[vLLM torch.compile blog](https://blog.vllm.ai/2025/08/20/torch-compile.html)。

> v3 不复用 torch.compile 通路（项目无 GPU 依赖、需 CPU/FakeTensor 跑通），但**借鉴其规则写法**——这正是 §10.4 引用 vLLM 的原因。

### 3.2 SGLang

SGLang 的算子融合 RFC 见 [Issue #10118](https://github.com/sgl-project/sglang/issues/10118)。结论与 vLLM 一致：**统一到 Inductor pattern_matcher 路径**，不再维护私有的 yaml/DSL。这也间接验证了 v3 选择 PyTorch 原生 API 的方向。

### 3.3 HF transformers attention_interface

#### 3.3.1 三种实现的切换机制

`transformers >= 4.36` 的 Llama / Qwen2 / Mistral 在 `LlamaAttention.__init__` 中根据 `config._attn_implementation` 切换 forward 中调用的 attention 函数：

| 取值 | 使用的函数 | 备注 |
|---|---|---|
| `"eager"` | `eager_attention_forward(...)` | Python 实现，含显式 `repeat_kv` + `q @ k.T * scale + mask + softmax + dropout + @ v` |
| `"sdpa"` | `torch.nn.functional.scaled_dot_product_attention` | 单一 op，无中间张量 |
| `"flash_attention_2"` | `flash_attn_func` | 依赖 flash-attn 包 |

切换不通过子类化，而是通过函数指针赋值（`ALL_ATTENTION_FUNCTIONS`）。

#### 3.3.2 对 v3 SDPA pattern 的影响

v3 的 `attention_sdpa.py` 规则**只支持 `eager` 路径**，原因：

1. `sdpa` 路径在 FX 上是单个 `torch.ops.aten.scaled_dot_product_attention.default` 节点，不需要融合。
2. `flash_attention_2` 走外部 CUDA kernel，TorchDispatch 抓不到内部 op。

`python/zrt/graph/model_loader.py` 已强制 `config._attn_implementation = "eager"`，保证 v3 规则能命中。pattern 必须**精确镜像** `eager_attention_forward`，包括无操作的 `F.dropout(p=0, training=False)`（会引入一个 `clone` aten 节点）和 `repeat_kv` 的 `unsqueeze → expand → reshape` 链路。`python/zrt/transform/fusion/v3/rules/attention_sdpa.py:69-93` 完整复刻了这段。

---

## 4. 拒绝的备选方案

### 4.1 TVM Relax DFPattern — 为什么不选

TVM Relax 的 `DFPattern` 提供了非常优雅的 builder DSL：

```python
matmul = is_op("relax.matmul")(wildcard(), wildcard())
add    = is_op("relax.add")(matmul, wildcard())
relu   = is_op("relax.nn.relu")(add)
```

驱动接口：`relax.transform.FuseOpsByPattern(patterns)`。

**拒绝原因**：

1. **运行时切换成本**：要求项目引入 Apache TVM (~80MB compiled deps)，重写 capture pipeline 为 Relax IR。
2. **算子库不对齐**：TVM 的 op set 是自维护的（`relax.matmul`、`relax.nn.gelu`），与 PyTorch aten 不能直接互换；规则要重新编译一遍。
3. **项目下游已建立在 OpGraph IR 上**（`transform/`、`executor/`、`simulator/`、`report/` 全部消费 OpGraph）；切换到 Relax 等于推倒重来。

**保留借鉴**：DFPattern 的「`is_op | wildcard | >>`」builder 风格在未来可考虑作为 v3 规则的语法糖，但当前优先级低。

参考：[TVM Relax FuseOpsByPattern](https://tvm.apache.org/docs/reference/api/python/relax/transform.html)、[TVM dataflow pattern tests](https://github.com/apache/tvm/blob/main/tests/python/relay/test_dataflow_pattern.py)。

### 4.2 MLIR PDLL — 为什么不选

PDLL（Pattern Descriptor Language Lite）是 MLIR 的声明式 DSL，被 IREE / linalg 用于 tile+fuse：

```pdll
Pattern => replace op<linalg.matmul>(...) with op<my.fused>(...);
```

**拒绝原因**：

1. **基础设施太重**：要求 MLIR 工具链（`mlir-opt`、`pdll-tool`），而 v3 目标是「Python 项目内自包含」。
2. **文本 DSL 调试代价高**：编译错误信息是 MLIR 风格，对纯 Python 项目维护者不友好。
3. **覆盖面不匹配**：PDLL 是为 linalg 这种规整 tensor algebra 设计的；HF 模型里的 RMSNorm / RoPE / GQA 含大量 `view`/`reshape`/`unsqueeze`，在 linalg 域内表达不自然。

参考：[MLIR PDLL docs](https://mlir.llvm.org/docs/PDLL/)、[IREE/MLIR/Linalg tutorial](https://iree.dev/community/blog/2024-01-29-iree-mlir-linalg-tutorial/)。

### 4.3 TASO / PET / Tensat — 为什么不选

| 论文 | 一句话描述 |
|---|---|
| TASO (SOSP'19) | 从算子代数等价规则自动生成等价子图改写候选 + Z3 验证 |
| PET (OSDI'21) | 扩展 TASO 支持「部分等价」改写 + 误差校正 |
| Tensat (MLSys'21) | 用 e-graph 表示等价类，比 TASO 搜索更高效 |

**拒绝原因**：

1. **研究级实现**：开源代码（[jiazhihao/TASO](https://github.com/jiazhihao/TASO)）依赖特定 CUDA 版本，无 PT2 后端。
2. **目标错位**：这些工作解决「**自动发现新的等价 fused op**」；v3 当前需求是「**用人写好的规则在已知子图上做匹配**」。融合规则数量预期 ≤ 30，人工写完全可控。
3. **缺乏测量基础**：自动生成的改写候选需要在真实硬件上 benchmark 选优；项目侧重 simulator，无法承担 calibration 闭环。

未来若要引入「自动从 dispatch trace 发现新 op tuple」功能，可借鉴 e-graph 思想。当前不交付。

参考：[TASO 论文](https://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf)、[PET 论文](https://www.usenix.org/system/files/osdi21-wang-haojie.pdf)。

### 4.4 自研 op-tuple 完善版（v2 演进）— 为什么不选

理论上可以在 v2 基础上做以下改良：

- 把 tuple 替换成「拓扑排序后再哈希」的 normalized signature（解决顺序敏感）
- 支持 wildcard op type（解决版本漂移）
- 加 shape/dtype 谓词（解决匹配键失真）

**拒绝原因**：做完上述改良后，等价于在重新发明 `SubgraphMatcher`——既然 PyTorch 官方已有成熟实现，没有理由维护一个项目专属的等价物。`SubgraphMatcher` 还自带 PT 团队的长期维护投入（PR #143147 这种 case 项目自己跟进成本太高）。

---

## 5. 关键决策与对应证据

下表把 v3 设计文档中的核心决策对应到外部参考或本项目的实证发现：

| 决策 | 来源 / 证据 |
|---|---|
| 用 `nn.Module` 当 pattern 而不是 FX graph | [torch.compiler_transformations](https://docs.pytorch.org/docs/stable/torch.compiler_transformations.html) 教程 + PT2 量化代码使用相同模式 |
| 用 `torch.export(strict=False)` 而不是 `torch.jit.trace` | HF 自定义代码（DeepSeek、Qwen3-MoE）需要 non-strict；jit.trace 已退出维护 |
| `run_decompositions(core_aten_decompositions())` 默认开启 | Inductor 主路径如此；vLLM `register_replacement` 同样依赖核心 aten 集合 |
| `match_placeholder=False` + `match_output=False` | `SubgraphMatcher` 默认子图匹配模式；与 PT2 量化 demo 一致 |
| 默认 `ignore_literals=True` | [PR #143147](https://github.com/pytorch/pytorch/pull/143147) 引入此参数的原始动机 |
| pattern 的 weight 走 forward 参数而不是 `nn.Parameter` | 本项目 Phase 0 实证（坑 1，§2.2.5） |
| 每条规则强制 `match_filter` 校验 leaf module class | 本项目 Phase 0 实证（坑 6，§2.2.5） |
| example inputs 用 bf16 | 本项目 Phase 0 实证（坑 5，§2.2.5） |
| 改用「tag, do not replace」策略 | 本项目 Phase 0 实证（坑 3，§2.2.5）+ [Export sub-graphs at the Aten IR level](https://dev-discuss.pytorch.org/t/export-sub-graphs-at-the-aten-ir-level/1936) |
| 在 capture 出口 strip `_assert_tensor_metadata` / `_guards_fn` | 本项目 Phase 0 实证（坑 4，§2.2.5） |
| 把 matcher 的 `pattern_placeholder_nodes` 置空绕过 KeyError | 本项目 Phase 0 实证（坑 2，§2.2.5） |
| 强制 HF model `_attn_implementation="eager"` | HF transformers `attention_interface` 规范（§3.3）；非 eager 路径无中间 op 可融 |
| Tier-2 规则用 `register_replacement`（多输出 / 控制流） | vLLM `vllm/compilation/passes/fusion/rms_quant_fusion.py` 同样选择 |
| DSV4 用 Tier-2 不用 Tier-1 | DSV4 modeling 含 MLA Attention 多输出 + Indexer 控制流（设计文档 §5.2 列举的四类）|
| 不再按 `model_id` 字符串前缀启发式加载规则 | v2 痛点之一；调研中 vLLM/SGLang 都改为显式声明 |

---

## 6. 参考链接（按章节归类）

> 全部链接均出自 `docs/fusion_v3_design_zh.md` §10.4 或调研 Agent 的 source list。链接最后一次访问于 2026-05-08。

### 6.1 PyTorch 官方源码

- [torch/fx/subgraph_rewriter.py](https://github.com/pytorch/pytorch/blob/main/torch/fx/subgraph_rewriter.py) — `replace_pattern` / `replace_pattern_with_filters` 入口。
- `torch/fx/passes/utils/matcher_utils.py` — `SubgraphMatcher` 实现（同仓库；调研中没有单独 URL，按文件路径检索）。
- `torch/_inductor/pattern_matcher.py` — `register_replacement` / `gen_pattern` / `PatternExpr` 类层级。
- `torch/_inductor/fx_passes/serialized_patterns/` — Inductor 预编译 pattern 序列化目录，参考其 pickle 结构。

### 6.2 PR / Issue

- [Pattern Matching with Literal Arguments (PR #143147)](https://github.com/pytorch/pytorch/pull/143147) — 引入 `ignore_literals=True`。

### 6.3 PyTorch 官方文档

- [torch.export docs (2.9)](https://docs.pytorch.org/docs/stable/export.html) — ExportedProgram / strict / dynamic_shapes / decompositions。
- [Writing Graph Transformations on ATen IR](https://docs.pytorch.org/docs/stable/torch.compiler_transformations.html) — SubgraphMatcher / replace_pattern 的官方教程。

### 6.4 PyTorch dev-discuss

- [How is pattern matching in inductor/fx implemented?](https://dev-discuss.pytorch.org/t/how-is-pattern-matching-in-inductor-fx-implemented/1720) — Inductor pattern_matcher 设计讨论。
- [Export sub-graphs at the Aten IR level](https://dev-discuss.pytorch.org/t/export-sub-graphs-at-the-aten-ir-level/1936) — replace 后 metadata 丢失的官方讨论帖。

### 6.5 第三方教程

- [Learn by doing: TorchInductor Pattern Matcher](https://karthick.ai/blog/2026/Learn-By-Doing-Torchinductor-Pattern-Matcher/) — `register_replacement` 实战 walkthrough。

### 6.6 业界推理框架

- [vLLM Fusion Passes design doc](https://docs.vllm.ai/en/stable/design/fusions/) — vLLM 融合 pass 总览。
- [vLLM torch.compile blog](https://blog.vllm.ai/2025/08/20/torch-compile.html) — vLLM 与 torch.compile 的耦合方式。
- [SGLang unified fusion RFC #10118](https://github.com/sgl-project/sglang/issues/10118) — SGLang 收敛到 Inductor pattern_matcher 的 RFC。
- [TensorRT-LLM pattern-matcher gap (Issue #8374)](https://github.com/NVIDIA/TensorRT-LLM/issues/8374) — TRT-LLM 自研 pattern matcher 的覆盖率问题，反向佐证「不要重新发明轮子」。
- [HuggingFace BetterTransformer overview](https://huggingface.co/docs/optimum/bettertransformer/overview) — HF 侧没有图级匹配，只做 module-class swap，作为 v3 决策的对照。

### 6.7 拒绝的备选方案文档

- [TVM Relax FuseOpsByPattern](https://tvm.apache.org/docs/reference/api/python/relax/transform.html) — 拒绝原因见 §4.1。
- [TVM dataflow pattern tests](https://github.com/apache/tvm/blob/main/tests/python/relay/test_dataflow_pattern.py) — DFPattern 用例。
- [MLIR PDLL docs](https://mlir.llvm.org/docs/PDLL/) — 拒绝原因见 §4.2。
- [IREE/MLIR/Linalg tutorial](https://iree.dev/community/blog/2024-01-29-iree-mlir-linalg-tutorial/) — IREE pattern 应用案例。
- [TASO 论文 (SOSP'19)](https://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf) — 拒绝原因见 §4.3。
- [TASO 仓库](https://github.com/jiazhihao/TASO) — 实现状态参考。
- [PET 论文 (OSDI'21)](https://www.usenix.org/system/files/osdi21-wang-haojie.pdf) — 部分等价改写 + 校正。

---

_文档结束。新发现的坑请追加到 §2.2.5；新引入的拒绝备选请追加到 §4。_
