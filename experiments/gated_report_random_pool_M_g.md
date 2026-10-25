# Gated library report random_pool_M_g

- 输入候选文件: experiments/random_candidates_pool.json
- 合并后 LLM/随机候选: 120
- 合并去重数: 0
- 经典规则直接入库: 0
- 规模/难度: M/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 120
- 行为相似度淘汰 Sim >= 0.9: 0
- 互补性贪心未入选: 0
- 最终规则数: 0 / K=6
- 固定参考状态数: 527

## 最终库

| # | Rule | Source | Selection reason | Type | f_H | max Sim | marginal gain |
|---:|---|---|---|---:|---:|---:|---:|

## 选择集 seeds 600–619

No rule passed the gate.

## 保留集 seeds 700–719

No rule passed the gate.
