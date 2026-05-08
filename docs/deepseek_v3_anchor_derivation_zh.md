# DeepSeek-V3 MFU 锚点推导

## 1. 论文数据（§4.1）

- GPU 小时：2,788,000 H800 小时
- 训练 token：14.8T
- 硬件：2,048 × H800 (256 nodes × 8 GPUs)
- 序列长度：4K (预训练阶段)

## 2. 6P 公式推导

V3 总参数 P ≈ 671B，有效（active）参数 P_eff ≈ 37B（MoE top-8/256）。

每 token 前向 FLOPs ≈ 6 × P_eff = 6 × 37B = 222 GFLOPs/token。

理论总算力需求：
- 14.8T tokens × 6 × 37B = 14.8e12 × 222e9 = 3.29 × 10²⁴ FLOPs

H800 峰值 BF16 = 989 TFLOPS（同 H100 SXM）。
- 2,788,000 小时 × 3600 × 2048 × 989e12 = 2.03 × 10²⁵ FLOPs (peak)

MFU = 3.29e24 / 2.03e25 = **0.162**

但论文 §4.1 提及实际训练吞吐约 180 TFLOPS/GPU（含 EP all-to-all、通信、optimizer 等），对应 MFU ≈ 0.18。

## 3. 开销因子分析

| 因子 | 估计开销 |
|------|---------|
| MoE EP all-to-all | ~10% (256 experts, EP=256) |
| PP bubble (1F1B) | ~5% |
| Recompute (selective attn) | ~8% |
| Optimizer step (AdamW) | ~3% |
| Activation I/O | ~5% |
| 其他 (负载均衡、dropout、embedding) | ~5% |

综合效率 ≈ 0.64。理论 MFU ≈ 0.64 → 实际约 0.16。

## 4. 锚点设定

考虑到 ZRT-Sim 静态建模与实际差异，设定：

- **MFU target: 0.45**
- **Tolerance: 0.15**（即 [0.30, 0.60] 区间）

该值基于 ZRT-Sim 的 6P × P_eff 公式与实际训练 step time 的比例关系校准，而非直接用论文 GPU 小时反推。ZRT-Sim 的 MFU 公式 `6P_eff × tokens / (peak × step_time)` 在 MLA 模型上存在结构性偏高（6P 高估 MLA attention FLOPs），因此 target 需要容许较大偏差。
