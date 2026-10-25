# Gated library report pool_L_g

- 输入候选文件: experiments/llm_candidates_p3b1.json, experiments/llm_candidates_p3b2.json, experiments/llm_candidates_p3b3.json
- 合并后 LLM/随机候选: 76
- 合并去重数: 5
- 经典规则直接入库: 0
- 规模/难度: L/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 74
- 行为相似度淘汰 Sim >= 0.9: 0
- 互补性贪心未入选: 0
- 最终规则数: 2 / K=6
- 固定参考状态数: 1158

## 最终库

| # | Rule | Source | Selection reason | Type | f_H | max Sim | marginal gain |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | Balanced_Tardiness_Workload | llm:p3b1 | best_normalized_objective | 2 | 0.982214 | 0.785838 | 0.017786 |
| 2 | Aggressive_SPT_Checkpoint | llm:p3b2 | greedy_marginal_gain | 1 | 0.992936 | 0.785838 | 0.018941 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| Balanced_Tardiness_Workload | 0.982214 |
| Aggressive_SPT_Checkpoint | 0.992936 |

- 库内最优单规则: Balanced_Tardiness_Workload (0.982214)
- 库内逐实例最优标准化目标: 0.963273
- 互补性差距: 0.018941 (1.93%)
- 库内最优单规则原始平均 F: 1099.405195
- 库内逐实例最优原始平均 F: 1079.752612

- 最优单规则 vs SPT: -2.30% (参照平均 F=1125.274200)
- 最优单规则 vs ATC: -7.60% (参照平均 F=1189.783094)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| Aggressive_SPT_Checkpoint | 0.977619 |
| Balanced_Tardiness_Workload | 1.009340 |

- 库内最优单规则: Aggressive_SPT_Checkpoint (0.977619)
- 库内逐实例最优标准化目标: 0.970420
- 互补性差距: 0.007198 (0.74%)
- 库内最优单规则原始平均 F: 1162.243543
- 库内逐实例最优原始平均 F: 1153.543543

- 最优单规则 vs SPT: -1.95% (参照平均 F=1185.318824)
- 最优单规则 vs ATC: -4.77% (参照平均 F=1220.419840)
