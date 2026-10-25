# Gated library report p3b1_L_g

- 输入候选: 29
- 规模/难度: L/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 28
- 行为相似度淘汰 Sim >= 0.9: 0
- 互补性贪心未入选: 0
- 最终规则数: 1 / K=6
- 固定参考状态数: 1158

## 最终库

| # | Rule | Source | Type | f_H | max Sim | marginal gain |
|---:|---|---|---:|---:|---:|---:|
| 1 | Balanced_Tardiness_Workload | llm | 2 | 0.982214 | 0.000000 | 0.017786 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| Balanced_Tardiness_Workload | 0.982214 |

- 库内最优单规则: Balanced_Tardiness_Workload (0.982214)
- 库内逐实例最优标准化目标: 0.982214
- 互补性差距: 0.000000 (0.00%)
- 库内最优单规则原始平均 F: 1099.405195
- 库内逐实例最优原始平均 F: 1099.405195

- 最优单规则 vs SPT: -2.30% (参照平均 F=1125.274200)
- 最优单规则 vs ATC: -7.60% (参照平均 F=1189.783094)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| Balanced_Tardiness_Workload | 1.009340 |

- 库内最优单规则: Balanced_Tardiness_Workload (1.009340)
- 库内逐实例最优标准化目标: 1.009340
- 互补性差距: 0.000000 (0.00%)
- 库内最优单规则原始平均 F: 1197.903311
- 库内逐实例最优原始平均 F: 1197.903311

- 最优单规则 vs SPT: +1.06% (参照平均 F=1185.318824)
- 最优单规则 vs ATC: -1.84% (参照平均 F=1220.419840)
