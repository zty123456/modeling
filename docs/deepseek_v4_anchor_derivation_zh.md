# DeepSeek-V4 (Pro/Flash) MFU 锚点推导

## 1. 论文数据（§4.2.2）

- V4-Pro：33T tokens 训练
- 硬件：未公开精确 GPU 小时（论文写"与 V3 相当量级"）
- 训练设置：TP=8, EP=64, PP=4, DP=2, DualPipeV, Muon optimizer
- 硬件：H800 级别集群

## 2. MFU 推导

V4-Pro P_total ≈ 1599B，P_eff ≈ 50.7B。

每 token 前向 FLOPs ≈ 6 × 50.7B = 304 GFLOPs/token。

V4 vs V3 的关键差异：
- CSA/HCA attention：attention FLOPs 显著降低（compressed KV）
- Hyper-Connections：每层额外 ~6.7% 计算（论文 §3.5.2）
- FP4 expert：计算密度提升（FP8 GEMM），但 ZRT-Sim 按 BF16 建模
- Hash routing（前 3 层）：router 计算量为 0
- Muon optimizer：NS 步骤额外计算

综合效率因子：
| 因子 | 估计 |
|------|------|
| EP all-to-all | ~12% (384 experts, EP=64) |
| PP bubble (DualPipeV) | ~3% |
| Hyper-Connections | ~7% |
| Recompute (attn only) | ~8% |
| Muon optimizer | ~5% |
| FP4 → BF16 建模偏差 | ~10% |
| 其他 | ~5% |

综合 ≈ 0.50。理论 MFU ≈ 0.50。

## 3. V4-Flash

V4-Flash P_eff ≈ 14.1B，更小的模型，但 EP=32 更少的 all-to-all 开销。

预期 MFU 与 V4-Pro 相近或略高。

## 4. 锚点设定

考虑到 ZRT-Sim 6P 公式对 V4 的结构偏差（6P 高估 MLA/V4 attention FLOPs），以及 FP4 未建模：

- **V4-Pro MFU target: 0.42**, tolerance: 0.15
- **V4-Flash MFU target: 0.40**, tolerance: 0.15

这些值基于"理论 MFU × ZRT 校准系数"设定，而非精确反推。
