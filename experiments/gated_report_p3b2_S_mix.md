# Gated library report p3b2_S_mix

- 输入候选文件: experiments/llm_candidates_p3b2.json
- 合并后 LLM/随机候选: 26
- 合并去重数: 0
- 经典规则直接入库: 2
- 规模/难度: S/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 24
- 行为相似度淘汰 Sim >= 0.9: 1
- 互补性贪心未入选: -2
- 最终规则数: 3 / K=6
- 固定参考状态数: 282

## 最终库

| # | Rule | Source | Selection reason | Type | f_H | max Sim | marginal gain |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | SPT | classic | mandatory_classic_bypass_quality_gate | classic | 1.000000 | 0.971631 | 0.000000 |
| 2 | ATC_K2 | classic | mandatory_classic_bypass_quality_gate | classic | 1.027784 | 0.627660 | 0.000000 |
| 3 | Aggressive_SPT_Checkpoint | llm:p3b2 | greedy_marginal_gain | 1 | 0.970157 | 0.971631 | 0.012853 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| Aggressive_SPT_Checkpoint | 0.970157 |
| SPT | 1.000000 |
| ATC_K2 | 1.027784 |

- 库内最优单规则: Aggressive_SPT_Checkpoint (0.970157)
- 库内逐实例最优标准化目标: 0.947008
- 互补性差距: 0.023148 (2.39%)
- 库内最优单规则原始平均 F: 310.596146
- 库内逐实例最优原始平均 F: 302.721146

- 最优单规则 vs SPT: -3.60% (参照平均 F=322.196146)
- 最优单规则 vs ATC: -5.36% (参照平均 F=328.171146)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| Aggressive_SPT_Checkpoint | 0.984894 |
| SPT | 1.000000 |
| ATC_K2 | 1.128626 |

- 库内最优单规则: Aggressive_SPT_Checkpoint (0.984894)
- 库内逐实例最优标准化目标: 0.966278
- 互补性差距: 0.018616 (1.89%)
- 库内最优单规则原始平均 F: 315.903754
- 库内逐实例最优原始平均 F: 309.878754

- 最优单规则 vs SPT: -1.50% (参照平均 F=320.728754)
- 最优单规则 vs ATC: -11.48% (参照平均 F=356.878754)
