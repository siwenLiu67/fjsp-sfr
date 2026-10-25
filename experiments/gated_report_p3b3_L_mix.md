# Gated library report p3b3_L_mix

- 输入候选文件: experiments/llm_candidates_p3b3.json
- 合并后 LLM/随机候选: 26
- 合并去重数: 0
- 经典规则直接入库: 2
- 规模/难度: L/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 26
- 行为相似度淘汰 Sim >= 0.9: 0
- 互补性贪心未入选: -2
- 最终规则数: 2 / K=6
- 固定参考状态数: 1158

## 最终库

| # | Rule | Source | Selection reason | Type | f_H | max Sim | marginal gain |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | SPT | classic | mandatory_classic_bypass_quality_gate | classic | 1.000000 | 0.528497 | 0.000000 |
| 2 | ATC_K2 | classic | mandatory_classic_bypass_quality_gate | classic | 1.064561 | 0.528497 | 0.000000 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| SPT | 1.000000 |
| ATC_K2 | 1.064561 |

- 库内最优单规则: SPT (1.000000)
- 库内逐实例最优标准化目标: 0.983650
- 互补性差距: 0.016350 (1.64%)
- 库内最优单规则原始平均 F: 1125.274200
- 库内逐实例最优原始平均 F: 1104.424200

- 最优单规则 vs SPT: +0.00% (参照平均 F=1125.274200)
- 最优单规则 vs ATC: -5.42% (参照平均 F=1189.783094)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| SPT | 1.000000 |
| ATC_K2 | 1.035587 |

- 库内最优单规则: SPT (1.000000)
- 库内逐实例最优标准化目标: 0.979725
- 互补性差距: 0.020275 (2.03%)
- 库内最优单规则原始平均 F: 1185.318824
- 库内逐实例最优原始平均 F: 1158.845121

- 最优单规则 vs SPT: +0.00% (参照平均 F=1185.318824)
- 最优单规则 vs ATC: -2.88% (参照平均 F=1220.419840)
