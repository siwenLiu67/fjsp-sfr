# Gated library report p3b3_M_mix

- 输入候选文件: experiments/llm_candidates_p3b3.json
- 合并后 LLM/随机候选: 26
- 合并去重数: 0
- 经典规则直接入库: 2
- 规模/难度: M/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 25
- 行为相似度淘汰 Sim >= 0.9: 0
- 互补性贪心未入选: -2
- 最终规则数: 3 / K=6
- 固定参考状态数: 527

## 最终库

| # | Rule | Source | Selection reason | Type | f_H | max Sim | marginal gain |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | SPT | classic | mandatory_classic_bypass_quality_gate | classic | 1.000000 | 0.897533 | 0.000000 |
| 2 | ATC_K2 | classic | mandatory_classic_bypass_quality_gate | classic | 1.043237 | 0.645161 | 0.000000 |
| 3 | Slack_Driven_SPT | llm:p3b3 | greedy_marginal_gain | 1 | 0.984888 | 0.897533 | 0.028933 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| Slack_Driven_SPT | 0.984888 |
| SPT | 1.000000 |
| ATC_K2 | 1.043237 |

- 库内最优单规则: Slack_Driven_SPT (0.984888)
- 库内逐实例最优标准化目标: 0.937576
- 互补性差距: 0.047312 (4.80%)
- 库内最优单规则原始平均 F: 564.990875
- 库内逐实例最优原始平均 F: 540.365875

- 最优单规则 vs SPT: -2.55% (参照平均 F=579.765875)
- 最优单规则 vs ATC: -6.11% (参照平均 F=601.765875)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| SPT | 1.000000 |
| Slack_Driven_SPT | 1.059682 |
| ATC_K2 | 1.060534 |

- 库内最优单规则: SPT (1.000000)
- 库内逐实例最优标准化目标: 0.975782
- 互补性差距: 0.024218 (2.42%)
- 库内最优单规则原始平均 F: 560.752161
- 库内逐实例最优原始平均 F: 544.477161

- 最优单规则 vs SPT: +0.00% (参照平均 F=560.752161)
- 最优单规则 vs ATC: -5.02% (参照平均 F=590.413947)
