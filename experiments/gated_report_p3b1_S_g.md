# Gated library report p3b1_S_g

- 输入候选: 29
- 规模/难度: S/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 27
- 行为相似度淘汰 Sim >= 0.9: 0
- 互补性贪心未入选: 0
- 最终规则数: 2 / K=6
- 固定参考状态数: 282

## 最终库

| # | Rule | Source | Type | f_H | max Sim | marginal gain |
|---:|---|---|---:|---:|---:|---:|
| 1 | Multi_Feature_Index | llm | 3 | 0.941734 | 0.684397 | 0.058266 |
| 2 | Hybrid_Urgency_Slack | llm | 1 | 0.999938 | 0.684397 | 0.014194 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| Multi_Feature_Index | 0.941734 |
| Hybrid_Urgency_Slack | 0.999938 |

- 库内最优单规则: Multi_Feature_Index (0.941734)
- 库内逐实例最优标准化目标: 0.927540
- 互补性差距: 0.014194 (1.51%)
- 库内最优单规则原始平均 F: 301.321146
- 库内逐实例最优原始平均 F: 296.346146

- 最优单规则 vs SPT: -6.48% (参照平均 F=322.196146)
- 最优单规则 vs ATC: -8.18% (参照平均 F=328.171146)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| Multi_Feature_Index | 1.031322 |
| Hybrid_Urgency_Slack | 1.164607 |

- 库内最优单规则: Multi_Feature_Index (1.031322)
- 库内逐实例最优标准化目标: 1.012946
- 互补性差距: 0.018377 (1.78%)
- 库内最优单规则原始平均 F: 328.203754
- 库内逐实例最优原始平均 F: 322.853754

- 最优单规则 vs SPT: +2.33% (参照平均 F=320.728754)
- 最优单规则 vs ATC: -8.03% (参照平均 F=356.878754)
