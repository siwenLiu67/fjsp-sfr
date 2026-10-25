# Gated library report p3b1_S_mix

- 输入候选文件: experiments/llm_candidates_p3b1.json
- 合并后 LLM/随机候选: 29
- 合并去重数: 0
- 经典规则直接入库: 2
- 规模/难度: S/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 27
- 行为相似度淘汰 Sim >= 0.9: 0
- 互补性贪心未入选: -2
- 最终规则数: 4 / K=6
- 固定参考状态数: 282

## 最终库

| # | Rule | Source | Selection reason | Type | f_H | max Sim | marginal gain |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | SPT | classic | mandatory_classic_bypass_quality_gate | classic | 1.000000 | 0.748227 | 0.000000 |
| 2 | ATC_K2 | classic | mandatory_classic_bypass_quality_gate | classic | 1.027784 | 0.698582 | 0.000000 |
| 3 | Multi_Feature_Index | llm:p3b1 | greedy_marginal_gain | 3 | 0.941734 | 0.748227 | 0.037152 |
| 4 | Hybrid_Urgency_Slack | llm:p3b1 | greedy_marginal_gain | 1 | 0.999938 | 0.698582 | 0.008353 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| Multi_Feature_Index | 0.941734 |
| Hybrid_Urgency_Slack | 0.999938 |
| SPT | 1.000000 |
| ATC_K2 | 1.027784 |

- 库内最优单规则: Multi_Feature_Index (0.941734)
- 库内逐实例最优标准化目标: 0.914356
- 互补性差距: 0.027377 (2.91%)
- 库内最优单规则原始平均 F: 301.321146
- 库内逐实例最优原始平均 F: 292.371146

- 最优单规则 vs SPT: -6.48% (参照平均 F=322.196146)
- 最优单规则 vs ATC: -8.18% (参照平均 F=328.171146)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| SPT | 1.000000 |
| Multi_Feature_Index | 1.031322 |
| ATC_K2 | 1.128626 |
| Hybrid_Urgency_Slack | 1.164607 |

- 库内最优单规则: SPT (1.000000)
- 库内逐实例最优标准化目标: 0.964622
- 互补性差距: 0.035378 (3.54%)
- 库内最优单规则原始平均 F: 320.728754
- 库内逐实例最优原始平均 F: 309.253754

- 最优单规则 vs SPT: +0.00% (参照平均 F=320.728754)
- 最优单规则 vs ATC: -10.13% (参照平均 F=356.878754)
