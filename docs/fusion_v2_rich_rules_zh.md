# 算子融合规则富格式（rich rules） —— v2 增量重构契约

_2026-05-09。本文档是 `python/zrt/transform/fusion/` 增量重构的**规则契约**。所有融合规则的注册路径（Python 注册 / YAML 加载 / discover 自动产出）必须严格遵守此契约。下游 matcher、semantics 推导器、simulator 都按此契约消费规则。_

---

## 1. 设计目标（重申）

| 目标 | 含义 |
|---|---|
| **匹配灵活** | 取代旧 v2 的「全 tuple 严格相等」。三档匹配：仅按类、有序正则（含 skip 通配）、DAG 直方图 |
| **语义可推** | 融合节点能从 IO 反推 batch_size / seq_len / hidden_dim / head_dim / dtype，并据此计算 FLOPs / 访存量 |
| **配置友好** | 同一规则可通过 Python 注册 *或* YAML 声明；YAML 优先用于模型相关规则（discover 工具自动产出） |
| **向后兼容** | 老 YAML 字段（`observed_op_seqs`、`inputs`、`outputs`）继续可用，自动翻译到新字段 |

---

## 2. 数据结构（Python 端契约）

文件：`python/zrt/transform/fusion/rule.py`

### 2.1 `MatchPattern`：三档匹配

```python
MatchKind = Literal["class_only", "ordered_regex", "dag_signature"]

@dataclass(frozen=True)
class MatchPattern:
    kind: MatchKind = "ordered_regex"
    op_regexes:  tuple[str, ...] = ()                 # ordered_regex 用
    op_multiset: tuple[tuple[str, int], ...] = ()     # dag_signature 用
    skip_ops:    frozenset[str] = DEFAULT_SKIP_OPS    # 这些 op 可在序列中被跳过
    min_ops:     int = 1                              # 子图最小算子数
    max_ops:     int = 1024                           # 子图最大算子数
```

| 档位 | 何时用 | 匹配语义 |
|---|---|---|
| **class_only** | leaf module 一一对应一个融合算子（如 `RMSNorm`、`Compressor`、`Indexer`），无需关心 aten 序 | 仅校验 group 的 `module_class` 与 `target_class` 字符串/正则一致 |
| **ordered_regex** | 子图算子有稳定的拓扑顺序，但要容忍 view/permute/dtype-cast 等中间噪声 | 按 `op_regexes` 顺序消费 group 的 op_type；`skip_ops` 中的算子可被跳过；每个 regex 用 `re.fullmatch` |
| **dag_signature** | 分支多、顺序不稳定（如 FP8 Linear 40+ 排列） | 按 `op_multiset = ((regex, min_count), ...)` 直方图比对：每条 regex 必须在 group 中至少出现 `min_count` 次；顺序无关 |

`DEFAULT_SKIP_OPS`（默认跳过集，所有规则共享）：
```
aten.view.default          aten.reshape.default       aten.permute.default
aten.transpose.int         aten.expand.default        aten.squeeze.dim
aten.unsqueeze.default     aten.flatten.using_ints    aten.contiguous.memory_format
aten._to_copy.default      aten.to.dtype              aten.to.dtype_layout
aten.clone.default         aten.detach.default        aten.alias.default
```

### 2.2 `IORole`：IO 角色 + 符号 shape

```python
@dataclass(frozen=True)
class IORole:
    role: str                                          # "activation" / "weight" / "output" / ...
    source_kind: Literal["input", "output"] = "input"
    source_op_index: int = -1                          # group.child_ops 索引，-1 = 末尾
    source_arg_index: int = -1                         # 张量槽位
    shape_role: Optional[str] = None                   # 符号轴，如 "[B,S,H]"
    expected_dtype: Optional[str] = None               # dtype 校验
```

`role` 推荐取值（不强制）：
```
activation     输入激活
weight         主权重（mm/linear 的 W）
bias           偏置
scale          量化缩放
output         输出激活
kv_cache       KV 缓存
router_weight  MoE 路由器权重
expert_indices MoE topk 索引
freqs_cis      RoPE 频率
attn_sink      attention sink
```

`shape_role` 是符号字符串，仅供 `shape_derivation` 与人工审阅参考。常见取值：
```
[B,S,H]         batch × seq × hidden
[B,S,Hin]       hidden_in
[B,S,Hout]      hidden_out
[Hout,Hin]      矩阵权重
[H]             向量权重（norm scale）
[B,S,N,D]       batch × seq × heads × head_dim
[E,H]           experts × hidden
[V,H]           vocab × hidden
```

`IOSpec` 是 `IORole` 的别名（保留老 import 路径）。

### 2.3 `ShapeDerivation`：符号轴的求值表达式

```python
@dataclass(frozen=True)
class ShapeDerivation:
    batch_size:       Optional[str] = None
    seq_len:          Optional[str] = None
    hidden_in:        Optional[str] = None
    hidden_out:       Optional[str] = None
    num_heads:        Optional[str] = None
    head_dim:         Optional[str] = None
    intermediate_dim: Optional[str] = None
    vocab_size:       Optional[str] = None
    num_experts:      Optional[str] = None
    topk:             Optional[str] = None
    extra: tuple[tuple[str, str], ...] = ()           # ((name, expr), ...)
```

每个值是字符串表达式，由 `_safe_eval.py` 在 `{role: TensorView}` 命名空间中安全求值。`TensorView` 接口：

```python
class TensorView:
    shape: tuple[int, ...]
    dtype: str         # "bf16" / "fp32" / ...
    bytes: int         # 总字节数
    numel: int         # 元素个数
    itemsize: float    # 单元素字节数
```

合法表达式示例：
```python
"activation.shape[0]"
"activation.shape[-1]"
"weight.shape[0] * weight.shape[1]"
"max(activation.shape[1], 1)"
```

### 2.4 `ModuleFusionRule`：完整规则

```python
@dataclass(frozen=True)
class ModuleFusionRule:
    target_class: type | str | tuple[str, ...]      # 单类、类名或类名 regex 列表

    op_type: Optional[str] = None                   # 融合后 op_type，None → target_class 名

    # ── 匹配 ──
    pattern: Optional[MatchPattern] = None          # 新：结构化匹配
    op_sequences: tuple[tuple[str, ...], ...] = ()  # 老：自动转 ordered_regex

    # ── IO 角色 ──
    io_roles: tuple[IORole, ...] = ()
    inputs:  tuple[IORole, ...] = ()                # 老别名，自动并入 io_roles
    outputs: tuple[IORole, ...] = ()                # 老别名，自动并入 io_roles

    # ── 语义推导 ──
    shape_derivation: Optional[ShapeDerivation] = None

    flops_formula:  Optional[str] = None            # 字符串公式，AST 安全求值
    memory_formula: Optional[str] = None
    flops_callable:  Optional[Callable[[dict], float]] = None   # Python 注册路径专用
    memory_callable: Optional[Callable[[dict], float]] = None

    flops_kind: Literal["sum_children", "from_io", "formula", "custom"] = "sum_children"
    custom_resolver: Optional[Callable] = None

    # ── 元信息 ──
    annotations: dict = field(default_factory=dict)
    priority: int = 10
```

**字段优先级与回退：**
1. `pattern is not None` → 直接用
2. `pattern is None and op_sequences` → 自动构造 `MatchPattern(kind="ordered_regex", op_regexes=tuple(re.escape(o) for o in seq))`；多条 sequence 在 YAML 加载阶段拆成多条 rule
3. `pattern is None and not op_sequences` → 默认 `MatchPattern(kind="class_only")`

**FLOPs 计算优先级：**
1. `flops_callable` → 调用
2. `flops_formula` → AST 求值
3. `flops_kind="from_io"` → `2 * input_bytes / itemsize_main`（粗估）
4. `flops_kind="sum_children"` → 子算子 FLOPs 求和（默认，需上游算好）
5. `flops_kind="custom"` → 由 `custom_resolver` 处理

`memory_*` 同理但只有 `formula/callable/from_io`，无 `sum_children`。

---

## 3. YAML 端契约

文件落盘位置：`python/zrt/transform/fusion/rules/<model>.yaml`

### 3.1 新格式（推荐）

```yaml
# 规则文件 schema_version: 1
# 顶层是规则数组；每条规则一个 dict。

- target_class: RMSNorm
  op_type: rms_norm
  priority: 20
  enabled: true                          # 可选，默认 true
  match:
    kind: ordered_regex                  # class_only | ordered_regex | dag_signature
    op_regexes:
      - "aten\\.pow\\.Tensor_Scalar"
      - "aten\\.mean\\.dim"
      - "aten\\.(add|add_)\\.(Tensor|Scalar)"
      - "aten\\.rsqrt\\.default"
      - "aten\\.mul\\.Tensor"
      - "aten\\.mul\\.Tensor"
    skip_ops_extra: ["aten.detach.default"]   # 可选：在 DEFAULT_SKIP_OPS 之外再加
    min_ops: 5
  io_roles:
    - {role: activation, source_kind: input,  source_op_index: 0,  source_arg_index: 0,  shape_role: "[B,S,H]"}
    - {role: weight,     source_kind: input,  source_op_index: -1, source_arg_index: 0,  shape_role: "[H]"}
    - {role: output,     source_kind: output, source_op_index: -1, source_arg_index: -1, shape_role: "[B,S,H]"}
  shape_derivation:
    batch_size: "activation.shape[0]"
    seq_len:    "activation.shape[1]"
    hidden_in:  "activation.shape[-1]"
    hidden_out: "output.shape[-1]"
  flops_formula:  "4 * batch_size * seq_len * hidden_in"
  memory_formula: "activation.bytes + weight.bytes + output.bytes"
  annotations:
    layer_norm_kind: rms

# class_only 示例（Compressor 子图复杂度高，靠 nn.Module 边界一锤定音）
- target_class: Compressor
  op_type: kv_compressor
  priority: 30
  match:
    kind: class_only
  io_roles:
    - {role: activation, source_kind: input,  source_op_index: 0,  source_arg_index: 0, shape_role: "[B,S,H]"}
    - {role: output,     source_kind: output, source_op_index: -1, source_arg_index: -1, shape_role: "[B,S/R,D]"}
  shape_derivation:
    batch_size:  "activation.shape[0]"
    seq_len:     "activation.shape[1]"
    hidden_in:   "activation.shape[-1]"
    head_dim:    "output.shape[-1]"
    extra:
      - [compress_ratio, "max(1, activation.shape[1] // output.shape[1])"]
  flops_formula: "2 * batch_size * seq_len * hidden_in * head_dim + 8 * batch_size * (seq_len // compress_ratio) * head_dim"
  memory_formula: "activation.bytes + output.bytes * 2"

# dag_signature 示例（FP8 Linear 顺序不稳定）
- target_class: Linear
  op_type: fp8_linear
  priority: 25
  match:
    kind: dag_signature
    op_multiset:
      - ["aten\\.mm\\.default|aten\\.matmul\\..*", 1]
      - ["fp8_gemm|fp4_gemm|act_quant", 1]
  io_roles:
    - {role: activation, source_kind: input,  source_op_index: 0,  source_arg_index: 0, shape_role: "[B,S,Hin]"}
    - {role: weight,     source_kind: input,  source_op_index: -1, source_arg_index: 1, shape_role: "[Hout,Hin]"}
    - {role: output,     source_kind: output, source_op_index: -1, source_arg_index: -1, shape_role: "[B,S,Hout]"}
  shape_derivation:
    batch_size: "activation.shape[0]"
    seq_len:    "activation.shape[1]"
    hidden_in:  "activation.shape[-1]"
    hidden_out: "output.shape[-1]"
  flops_formula:  "2 * batch_size * seq_len * hidden_in * hidden_out"
  memory_formula: "activation.bytes + weight.bytes + output.bytes"
  annotations:
    quant: fp8
```

### 3.2 老格式（继续支持，自动翻译）

```yaml
- target_class: ColumnParallelLinear
  op_type: ColumnParallelLinear
  observed_op_seqs:                         # → MatchPattern(kind="ordered_regex", op_regexes=[re.escape(o), ...])
    - [aten.mm.default, aten.add.Tensor]
    - [aten.linear.default]                 # 多条 → 在 YAML 加载阶段拆成 2 条 rule
  inputs:                                   # → io_roles 的输入侧
    - {role: activation, source_op_index: 0, source_arg_index: 0}
  outputs:
    - {role: output, source_op_index: -1, source_arg_index: -1, source_kind: output}
  priority: 20
  annotations:
    parallel: column
```

老格式不包含 `shape_derivation` / `flops_formula`，下游 `flops_kind` 默认 `sum_children`，行为与现网一致，不会破坏。

---

## 4. 安全求值器 `_safe_eval.py`

文件：`python/zrt/transform/fusion/_safe_eval.py`

```python
def safe_eval(expr: str, namespace: dict) -> float | int | tuple:
    """AST 求值。允许：
       - 数字字面量（int/float）
       - 名字（来自 namespace）
       - 一级属性访问（x.shape / x.dtype / x.bytes / x.numel / x.itemsize）
       - 下标（x.shape[0]、x.shape[-1]）
       - 二元/一元算术：+ - * / // % **
       - 内置函数白名单：min, max, abs, log, log2, sqrt, ceil, floor
       
       禁止：import / lambda / 多级属性 / 推导式 / 函数定义 / call 非白名单
       
       所有不合法的 AST 节点直接 raise ValueError。
    """
```

`namespace` 由 semantics 推导阶段构造：
```python
namespace = {
    role_name: TensorView(shape=meta.shape, dtype=meta.dtype.value,
                          bytes=meta.mem_bytes, numel=prod(meta.shape),
                          itemsize=meta.dtype.itemsize)
    for role_name, meta in resolved_io.items()
}
# 加上派生轴：
namespace.update(derived_shapes)   # batch_size, seq_len, hidden_in, ...
```

---

## 5. 公开 API（不变 + 新增）

```python
# python/zrt/transform/fusion/__init__.py
from .api import FusionPass
from .rule import (
    ModuleFusionRule,
    IORole, IOSpec,                      # IOSpec 是 IORole 别名
    MatchPattern, MatchKind,
    ShapeDerivation,
    DEFAULT_SKIP_OPS,
)
from .registry import register_rule, clear_rules, all_rules
from .yaml_loader import load_yaml_rules, add_yaml_search_dir
```

---

## 6. 验收（由 Step 2 校验）

- [ ] `pytest tests/transform/fusion/test_match.py` 全过：三档匹配各覆盖至少 3 个用例
- [ ] `pytest tests/transform/fusion/test_semantics.py` 全过：shape_derivation + flops/memory 公式各覆盖至少 3 个用例
- [ ] `pytest tests/fusion/test_discover.py` 全过：AST + runtime + joiner 各覆盖核心路径
- [ ] `tests/transform/fusion/test_dsv4_rules.py` 全过：DSv4 16 类规则在 mock OpGraph 上能被正确选中并产出语义字段
- [ ] 老 `tests/test_fusion_pass.py`（如果还有用例）保持通过或在 docstring 中明确标注废弃

---

## 7. 不在本次范围内

- 不切换抓图通路（继续用 dispatch；torch.export 路径见 `docs/fusion_v3_design_zh.md`）
- 不重写 OpGraph IR
- 不改 simulator / report 模块（只通过 fused node `annotations` 透传新字段）
- 不做 Inductor pattern_matcher 集成

---

## 8. 文件清单（本次重构涉及）

```
新增：
  docs/fusion_v2_rich_rules_zh.md                          ← 本文
  python/zrt/transform/fusion/_safe_eval.py
  python/zrt/transform/fusion/match.py
  python/zrt/transform/fusion/semantics.py
  python/zrt/transform/fusion/rules/deepseek_v4.yaml
  python/zrt/fusion/discover/__init__.py
  python/zrt/fusion/discover/ast_scanner.py
  python/zrt/fusion/discover/runtime_tracer.py
  python/zrt/fusion/discover/joiner.py
  python/zrt/fusion/discover/cli.py
  python/zrt/fusion/discover/templates.py
  .claude/skills/discover-fusion-rules/SKILL.md
  tests/transform/fusion/test_match.py
  tests/transform/fusion/test_semantics.py
  tests/transform/fusion/test_dsv4_rules.py
  tests/fusion/__init__.py
  tests/fusion/test_discover.py
  tests/fusion/test_ast_scanner.py

修改：
  python/zrt/transform/fusion/rule.py            ← 新 schema
  python/zrt/transform/fusion/yaml_loader.py     ← 新解析
  python/zrt/transform/fusion/registry.py        ← 用新 matcher
  python/zrt/transform/fusion/algorithm.py       ← 接 matcher + semantics
  python/zrt/transform/fusion/__init__.py        ← 暴露新符号
  python/zrt/fusion/discover.py                  ← 重定向到新包

兼容性：
  现有 builtins.py、platforms/*.py 不变（继续走老路径）
  现有 *.yaml 文件（如有）不变
```
