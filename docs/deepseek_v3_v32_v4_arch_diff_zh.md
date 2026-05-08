# DeepSeek V3 / V3.2 / V4 架构对照表

> 来源：V3 论文 (arxiv 2412.19437) §3, V3.2 论文 (arxiv 2512.02556) §2, V4 论文 §2-3, V4-Pro `inference/model.py`

## 全局参数

| 维度 | V3 | V3.2 | V4-Flash | V4-Pro |
|---|---|---|---|---|
| 总层数 / hidden | 61 / 7168 | 61 / 7168 | 43 / 4096 | 61 / 7168 |
| head_dim / n_heads / n_kv_heads | 128 / 128 / 128 (MLA) | 同 V3 | 512 / 64 / 1 | 512 / 128 / 1 |
| vocab | 129280 | 129280 | 129280 | 129280 |
| 首段 attn | 3 dense + 58 MoE | 3 dense + 58 MoE | 2 SWA + 41 CSA/HCA 交替 | 61 MoE (前 3 hash, 后 58 score); 含 HCA-only + CSA/HCA 交替 |

## MLA 投影链（V3 / V3.2）

```
q_a_proj (d → q_lora=1536)
  → q_a_layernorm
  → q_b_proj (q_lora → n_heads × head_dim = 128×128)
  → split(qk_nope=128, qk_rope=64)
  → RoPE(roped part)
  → concat → attn_core → o_proj

kv_a_proj_with_mqa (d → kv_lora=512)
  → kv_a_layernorm
  → kv_b_proj (kv_lora → qk_nope + v_head = 128+128)
  → split(k_nope, v)
  → RoPE(k_nope roped part)
  → concat(k_rope, k_nope, v) → KV cache
```

参数：`q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128`

## V3.2 新增：Lightning Indexer (DSA)

V3.2 = V3 + DSA（Dynamic Sparse Attention），在 MLA 之上增加稀疏 KV 选择：

- `index_n_heads=64, index_head_dim=128, index_topk=2048`
- 索引查询：`index_q_proj (d → n_h^I × c^I)`
- 索引键：从 KV cache 中取（V3.2 路径无 compressor）
- 评分：`einsum(q^I, k_cache) → ReLU → 加权求和`
- top-k 选择 → 稀疏 attention（仅 topk 个 KV token）

## V4 注意力架构（CSA / HCA / SWA）

### Compressor（KV 压缩，V4-only）

```
wkv: d → coff × head_dim   (coff = 1+overlap; overlap when m=4)
wgate: d → coff × head_dim
ape: (compress_ratio, coff × head_dim)   # 绝对位置偏置

# prefill:
kv = wkv(x).unflatten(ratio)
score = wgate(x).unflatten(ratio) + ape
kv_compress = sum(kv * softmax(score), dim=ratio)  # → (seq/m, head_dim)

# m=4 (CSA): overlapping windows, coff=2
# m=128 (HCA): non-overlapping, coff=1
```

### Indexer（CSA-only，top-k 稀疏选择）

```
compressor_indexer = Compressor(m=4, rotate=True, head_dim=index_head_dim=128)
  # 自带 compressor，带 Hadamard rotation + FP4 quant
wq_b: q_lora_rank → n_heads^I × head_dim^I  (ColumnParallel)
weights_proj: d → n_heads^I  (ColumnParallel)

q_index = wq_b(qr) → apply_rotary → rotate → fp4_quant
kv_index = compressor_indexer(x)  # → (seq/m, index_head_dim)
score = einsum("bshd,btd->bsht", q_index, kv_index) → ReLU × weights → sum
topk_indices = score.topk(index_topk)
```

- V4-Pro: `index_n_heads=64, index_head_dim=128, index_topk=1024`
- V4-Flash: `index_n_heads=64, index_head_dim=128, index_topk=512`

### Attention 投影链（V4）

```
# Q: low-rank with rsqrt normalization
wq_a: d → q_lora_rank (1536)
  → q_norm (RMSNorm)
  → wq_b: q_lora_rank → n_heads × head_dim (ColumnParallel)
  → rsqrt(q².mean + eps) normalization
  → apply_rotary(roped part)

# KV: single-head MQA
wkv: d → head_dim (512)   # single KV head!
  → kv_norm (RMSNorm)
  → apply_rotary(roped part)
  → FP8 quant (non-roped dims)

# Attention: sparse_attn kernel
#   topk_indices = concat(window_topk, compress_topk)
#   for CSA: compress_topk from Indexer
#   for HCA: compress_topk from get_compress_topk_idxs (no indexer)

# O: grouped low-rank projection
wo_a: (n_heads × head_dim / n_groups) → (n_groups × o_lora_rank)  # ColumnParallel
wo_b: (n_groups × o_lora_rank) → d  # RowParallel (all-reduce)
# implemented as: reshape → einsum("bsgd,grd->bsgr") → wo_b
```

### V4-Pro 层分布（compress_ratios）

```
[128, 128, 4, 128, 4, 128, 4, ..., 4, 128, 0]
  L0  L1  L2  L3  L4  L5  L6    L59 L60 L61(MTP)

L0, L1: compress_ratio=128 → HCA (无 indexer, dense MQA on compressed KV)
L2-L60: alternating CSA(4)/HCA(128)
  - CSA(m=4): compressor + indexer(topk=1024) + sparse_attn + SWA aux
  - HCA(m=128): compressor(m'=128) + dense MQA + SWA aux (无 indexer)
L61: MTP (compress_ratio=0)

总计: 31 HCA + 30 CSA + 1 MTP = 61 + 1 = 62 entries
```

## Hyper-Connections (mHC, V4-only)

```
hc_mult = 4  # 维护 4 份隐藏状态副本
hc_sinkhorn_iters = 20

hc_pre:
  x_flat = flatten(x, hc_dim)  # [b, s, hc×d]
  mixes = linear(x_flat, hc_fn) × rsqrt(x².mean + eps)  # hc_fn: [mix_hc, hc×d]
  pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)
  y = sum(pre × x, dim=hc)  # → [b, s, d]

hc_post:
  y = post × x + sum(comb × residual, dim=hc)  # → [b, s, hc, d]
```

## MoE 差异

| 维度 | V3 | V3.2 | V4-Flash | V4-Pro |
|---|---|---|---|---|
| n_routed_experts | 256 | 256 | 256 | 384 |
| top_k | 8 | 8 | 6 | 6 |
| moe_ffn | 2048 | 2048 | 2048 | 3072 |
| scoring | sigmoid | sigmoid | sqrtsoftplus | sqrtsoftplus |
| expert dtype | bf16 | bf16 | fp4 | fp4 |
| hash routing | 无 | 无 | 前 3 MoE 层 | 前 3 MoE 层 |
| shared_experts | 1 | 1 | 1 | 1 |

## 优化器

- V3 / V3.2: AdamW
- V4: Muon (hybrid Newton-Schulz, 10=8+2 steps), 仅用于非 embed/head/RMSNorm/mHC-bias 参数

## 建模复用关系

```
V3 算子序 = MLA attention + sigmoid-MoE (已建模)

V3.2 算子序 = V3 MLA + Lightning Indexer + sparse_attn_core
  → 新增: indexer 投影 + einsum 评分 + topk + sparse attn

V4 CSA 算子序 = token-level KV compressor + indexer + shared-KV MQA + SWA aux + grouped o_proj
V4 HCA 算子序 = token-level KV compressor(m'=128) + dense MQA + SWA aux + grouped o_proj (无 indexer)
V4 SWA 算子序 = 纯 sliding window attention (n_win=128)

V4 通用 = mHC (Sinkhorn) 替代 residual + hash routing (前 3 层) + sqrt-softplus 评分 + FP4 expert
```
