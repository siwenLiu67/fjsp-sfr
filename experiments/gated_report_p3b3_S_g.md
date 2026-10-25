# Gated library report p3b3_S_g

- 输入候选: 26
- 规模/难度: S/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 24
- 行为相似度淘汰 Sim >= 0.9: 0
- 互补性贪心未入选: 0
- 最终规则数: 2 / K=6
- 固定参考状态数: 282

## 最终库

| # | Rule | Source | Type | f_H | max Sim | marginal gain |
|---:|---|---|---:|---:|---:|---:|
| 1 | Slack_Driven_SPT | llm | 1 | 0.987894 | 0.744681 | 0.012106 |
| 2 | Integrated_Checkpoint_Index | llm | 3 | 0.990591 | 0.744681 | 0.051453 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| Slack_Driven_SPT | 0.987894 |
| Integrated_Checkpoint_Index | 0.990591 |

- 库内最优单规则: Slack_Driven_SPT (0.987894)
- 库内逐实例最优标准化目标: 0.936441
- 互补性差距: 0.051453 (5.21%)
- 库内最优单规则原始平均 F: 315.096146
- 库内逐实例最优原始平均 F: 299.471146

- 最优单规则 vs SPT: -2.20% (参照平均 F=322.196146)
- 最优单规则 vs ATC: -3.98% (参照平均 F=328.171146)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| Slack_Driven_SPT | 1.014193 |
| Integrated_Checkpoint_Index | 1.023715 |

- 库内最优单规则: Slack_Driven_SPT (1.014193)
- 库内逐实例最优标准化目标: 0.954749
- 互补性差距: 0.059444 (5.86%)
- 库内最优单规则原始平均 F: 323.903754
- 库内逐实例最优原始平均 F: 307.878754

- 最优单规则 vs SPT: +0.99% (参照平均 F=320.728754)
- 最优单规则 vs ATC: -9.24% (参照平均 F=356.878754)
