# Gated library report p3b1_M_mix

- 输入候选文件: experiments/llm_candidates_p3b1.json
- 合并后 LLM/随机候选: 29
- 合并去重数: 0
- 经典规则直接入库: 2
- 规模/难度: M/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 28
- 行为相似度淘汰 Sim >= 0.9: 0
- 互补性贪心未入选: -2
- 最终规则数: 3 / K=6
- 固定参考状态数: 527

## 最终库

| # | Rule | Source | Selection reason | Type | f_H | max Sim | marginal gain |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | SPT | classic | mandatory_classic_bypass_quality_gate | classic | 1.000000 | 0.700190 | 0.000000 |
| 2 | ATC_K2 | classic | mandatory_classic_bypass_quality_gate | classic | 1.043237 | 0.643264 | 0.000000 |
| 3 | Multi_Feature_Index | llm:p3b1 | greedy_marginal_gain | 3 | 0.979029 | 0.700190 | 0.026654 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| Multi_Feature_Index | 0.979029 |
| SPT | 1.000000 |
| ATC_K2 | 1.043237 |

- 库内最优单规则: Multi_Feature_Index (0.979029)
- 库内逐实例最优标准化目标: 0.939856
- 互补性差距: 0.039173 (4.00%)
- 库内最优单规则原始平均 F: 565.952319
- 库内逐实例最优原始平均 F: 544.390875

- 最优单规则 vs SPT: -2.38% (参照平均 F=579.765875)
- 最优单规则 vs ATC: -5.95% (参照平均 F=601.765875)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| SPT | 1.000000 |
| Multi_Feature_Index | 1.060047 |
| ATC_K2 | 1.060534 |

- 库内最优单规则: SPT (1.000000)
- 库内逐实例最优标准化目标: 0.957566
- 互补性差距: 0.042434 (4.24%)
- 库内最优单规则原始平均 F: 560.752161
- 库内逐实例最优原始平均 F: 534.588947

- 最优单规则 vs SPT: +0.00% (参照平均 F=560.752161)
- 最优单规则 vs ATC: -5.02% (参照平均 F=590.413947)
