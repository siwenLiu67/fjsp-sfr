# Gated library report p3b3_S_mix

- 输入候选文件: experiments/llm_candidates_p3b3.json
- 合并后 LLM/随机候选: 26
- 合并去重数: 0
- 经典规则直接入库: 2
- 规模/难度: S/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 24
- 行为相似度淘汰 Sim >= 0.9: 0
- 互补性贪心未入选: -2
- 最终规则数: 4 / K=6
- 固定参考状态数: 282

## 最终库

| # | Rule | Source | Selection reason | Type | f_H | max Sim | marginal gain |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | SPT | classic | mandatory_classic_bypass_quality_gate | classic | 1.000000 | 0.943262 | 0.000000 |
| 2 | ATC_K2 | classic | mandatory_classic_bypass_quality_gate | classic | 1.027784 | 0.702128 | 0.000000 |
| 3 | Integrated_Checkpoint_Index | llm:p3b3 | greedy_marginal_gain | 3 | 0.990591 | 0.744681 | 0.022543 |
| 4 | Slack_Driven_SPT | llm:p3b3 | greedy_marginal_gain | 1 | 0.987894 | 0.943262 | 0.010723 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| Slack_Driven_SPT | 0.987894 |
| Integrated_Checkpoint_Index | 0.990591 |
| SPT | 1.000000 |
| ATC_K2 | 1.027784 |

- 库内最优单规则: Slack_Driven_SPT (0.987894)
- 库内逐实例最优标准化目标: 0.926595
- 互补性差距: 0.061299 (6.21%)
- 库内最优单规则原始平均 F: 315.096146
- 库内逐实例最优原始平均 F: 296.446146

- 最优单规则 vs SPT: -2.20% (参照平均 F=322.196146)
- 最优单规则 vs ATC: -3.98% (参照平均 F=328.171146)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| SPT | 1.000000 |
| Slack_Driven_SPT | 1.014193 |
| Integrated_Checkpoint_Index | 1.023715 |
| ATC_K2 | 1.128626 |

- 库内最优单规则: SPT (1.000000)
- 库内逐实例最优标准化目标: 0.948529
- 互补性差距: 0.051471 (5.15%)
- 库内最优单规则原始平均 F: 320.728754
- 库内逐实例最优原始平均 F: 305.853754

- 最优单规则 vs SPT: +0.00% (参照平均 F=320.728754)
- 最优单规则 vs ATC: -10.13% (参照平均 F=356.878754)
