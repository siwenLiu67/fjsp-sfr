# Gated library report p3b2_S_g

- 输入候选: 26
- 规模/难度: S/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 24
- 行为相似度淘汰 Sim >= 0.9: 1
- 互补性贪心未入选: 0
- 最终规则数: 1 / K=6
- 固定参考状态数: 282

## 最终库

| # | Rule | Source | Type | f_H | max Sim | marginal gain |
|---:|---|---|---:|---:|---:|---:|
| 1 | Aggressive_SPT_Checkpoint | llm | 1 | 0.970157 | 0.000000 | 0.029843 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| Aggressive_SPT_Checkpoint | 0.970157 |

- 库内最优单规则: Aggressive_SPT_Checkpoint (0.970157)
- 库内逐实例最优标准化目标: 0.970157
- 互补性差距: 0.000000 (0.00%)
- 库内最优单规则原始平均 F: 310.596146
- 库内逐实例最优原始平均 F: 310.596146

- 最优单规则 vs SPT: -3.60% (参照平均 F=322.196146)
- 最优单规则 vs ATC: -5.36% (参照平均 F=328.171146)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| Aggressive_SPT_Checkpoint | 0.984894 |

- 库内最优单规则: Aggressive_SPT_Checkpoint (0.984894)
- 库内逐实例最优标准化目标: 0.984894
- 互补性差距: 0.000000 (0.00%)
- 库内最优单规则原始平均 F: 315.903754
- 库内逐实例最优原始平均 F: 315.903754

- 最优单规则 vs SPT: -1.50% (参照平均 F=320.728754)
- 最优单规则 vs ATC: -11.48% (参照平均 F=356.878754)
