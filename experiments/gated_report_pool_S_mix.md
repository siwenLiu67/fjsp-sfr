# Gated library report pool_S_mix

- 输入候选文件: experiments/llm_candidates_p3b1.json, experiments/llm_candidates_p3b2.json, experiments/llm_candidates_p3b3.json
- 合并后 LLM/随机候选: 76
- 合并去重数: 5
- 经典规则直接入库: 2
- 规模/难度: S/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 70
- 行为相似度淘汰 Sim >= 0.9: 2
- 互补性贪心未入选: -1
- 最终规则数: 5 / K=6
- 固定参考状态数: 282

## 最终库

| # | Rule | Source | Selection reason | Type | f_H | max Sim | marginal gain |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | SPT | classic | mandatory_classic_bypass_quality_gate | classic | 1.000000 | 0.748227 | 0.000000 |
| 2 | ATC_K2 | classic | mandatory_classic_bypass_quality_gate | classic | 1.027784 | 0.702128 | 0.000000 |
| 3 | Multi_Feature_Index | llm:p3b1 | greedy_marginal_gain | 3 | 0.941734 | 0.872340 | 0.037152 |
| 4 | Integrated_Checkpoint_Index | llm:p3b3 | greedy_marginal_gain | 3 | 0.990591 | 0.872340 | 0.011384 |
| 5 | Hybrid_Urgency_Slack | llm:p3b1 | greedy_marginal_gain | 1 | 0.999938 | 0.787234 | 0.004050 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| Multi_Feature_Index | 0.941734 |
| Integrated_Checkpoint_Index | 0.990591 |
| Hybrid_Urgency_Slack | 0.999938 |
| SPT | 1.000000 |
| ATC_K2 | 1.027784 |

- 库内最优单规则: Multi_Feature_Index (0.941734)
- 库内逐实例最优标准化目标: 0.907275
- 互补性差距: 0.034459 (3.66%)
- 库内最优单规则原始平均 F: 301.321146
- 库内逐实例最优原始平均 F: 289.996146

- 最优单规则 vs SPT: -6.48% (参照平均 F=322.196146)
- 最优单规则 vs ATC: -8.18% (参照平均 F=328.171146)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| SPT | 1.000000 |
| Integrated_Checkpoint_Index | 1.023715 |
| Multi_Feature_Index | 1.031322 |
| ATC_K2 | 1.128626 |
| Hybrid_Urgency_Slack | 1.164607 |

- 库内最优单规则: SPT (1.000000)
- 库内逐实例最优标准化目标: 0.951017
- 互补性差距: 0.048983 (4.90%)
- 库内最优单规则原始平均 F: 320.728754
- 库内逐实例最优原始平均 F: 306.203754

- 最优单规则 vs SPT: +0.00% (参照平均 F=320.728754)
- 最优单规则 vs ATC: -10.13% (参照平均 F=356.878754)
