# DeepSeek-V3.2 MFU 锚点推导

## 1. 论文数据（§3）

V3.2 论文目前未公开完整训练 wall-clock 数据。基于以下信息推导：

- 架构 = V3 + Lightning Indexer（DSA），其余相同
- Indexer 开销：每 MoE 层新增 ~6 ops，FLOPs 增量 ≈ 2 × seq² × 64 × 128 per layer
- 训练设置类似 V3：H800 集群，EP 路由

## 2. MFU 推导

V3.2 P_eff ≈ 37.7B（V3 的 37B + indexer 参数增量 ~0.7B）。

理论 FLOPs/token = 6 × 37.7B = 226.2 GFLOPs。

相比 V3：
- Indexer 增加 FLOPs ≈ 0.7B / 37B ≈ 2%（参数增量）
- 但 DSA 减少 attention FLOPs（top-k=2048 vs full seq），净效果取决于 seq_len
- 在 seq=4K 时，indexer 开销 < 1%

预期 MFU 与 V3 相近或略低（indexer 计算开销 vs attention FLOPs 节省）。

## 3. 开销因子

与 V3 相同，额外增加：
- Indexer top-k 选择：每层额外 O(L²) 计算（但 seq=4K 时量级小）

综合预期 MFU ≈ V3 的 95% → ~0.43。

## 4. 锚点设定

- **MFU target: 0.43**
- **Tolerance: 0.15**（即 [0.28, 0.58]）

注：ZRT-Sim 的 6P 公式对 MLA+DSA 模型仍有结构性偏高，因此 tolerance 保持宽松。
