# Gated library report p3b2_L_mix

- 输入候选文件: experiments/llm_candidates_p3b2.json
- 合并后 LLM/随机候选: 26
- 合并去重数: 0
- 经典规则直接入库: 2
- 规模/难度: L/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 25
- 行为相似度淘汰 Sim >= 0.9: 0
- 互补性贪心未入选: -2
- 最终规则数: 3 / K=6
- 固定参考状态数: 1158

## 最终库

| # | Rule | Source | Selection reason | Type | f_H | max Sim | marginal gain |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | SPT | classic | mandatory_classic_bypass_quality_gate | classic | 1.000000 | 0.895509 | 0.000000 |
| 2 | ATC_K2 | classic | mandatory_classic_bypass_quality_gate | classic | 1.064561 | 0.537133 | 0.000000 |
| 3 | Aggressive_SPT_Checkpoint | llm:p3b2 | greedy_marginal_gain | 1 | 0.992936 | 0.895509 | 0.025666 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| Aggressive_SPT_Checkpoint | 0.992936 |
| SPT | 1.000000 |
| ATC_K2 | 1.064561 |

- 库内最优单规则: Aggressive_SPT_Checkpoint (0.992936)
- 库内逐实例最优标准化目标: 0.957983
- 互补性差距: 0.034953 (3.52%)
- 库内最优单规则原始平均 F: 1113.642983
- 库内逐实例最优原始平均 F: 1076.275570

- 最优单规则 vs SPT: -1.03% (参照平均 F=1125.274200)
- 最优单规则 vs ATC: -6.40% (参照平均 F=1189.783094)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| Aggressive_SPT_Checkpoint | 0.977619 |
| SPT | 1.000000 |
| ATC_K2 | 1.035587 |

- 库内最优单规则: Aggressive_SPT_Checkpoint (0.977619)
- 库内逐实例最优标准化目标: 0.950093
- 互补性差距: 0.027525 (2.82%)
- 库内最优单规则原始平均 F: 1162.243543
- 库内逐实例最优原始平均 F: 1126.720633

- 最优单规则 vs SPT: -1.95% (参照平均 F=1185.318824)
- 最优单规则 vs ATC: -4.77% (参照平均 F=1220.419840)
