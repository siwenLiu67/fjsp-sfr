# Gated library report random_pool_L_nogate_mix

- 输入候选文件: experiments/random_candidates_pool.json
- 合并后 LLM/随机候选: 120
- 合并去重数: 0
- 经典规则直接入库: 2
- 规模/难度: L/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- 质量门槛: 关闭
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 0
- 行为相似度淘汰 Sim >= 0.9: 22
- 互补性贪心未入选: 94
- 最终规则数: 4 / K=6
- 固定参考状态数: 1158

## 最终库

| # | Rule | Source | Selection reason | Type | f_H | max Sim | marginal gain |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | SPT | classic | mandatory_classic_bypass_quality_gate | classic | 1.000000 | 0.528497 | 0.000000 |
| 2 | ATC_K2 | classic | mandatory_classic_bypass_quality_gate | classic | 1.064561 | 0.875648 | 0.000000 |
| 3 | random_t2_016 | random | greedy_marginal_gain | 2 | 1.062987 | 0.875648 | 0.005633 |
| 4 | random_t3_008 | random | greedy_marginal_gain | 3 | 1.133941 | 0.531088 | 0.002547 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| SPT | 1.000000 |
| random_t2_016 | 1.062987 |
| ATC_K2 | 1.064561 |
| random_t3_008 | 1.133941 |

- 库内最优单规则: SPT (1.000000)
- 库内逐实例最优标准化目标: 0.975470
- 互补性差距: 0.024530 (2.45%)
- 库内最优单规则原始平均 F: 1125.274200
- 库内逐实例最优原始平均 F: 1095.549200

- 最优单规则 vs SPT: +0.00% (参照平均 F=1125.274200)
- 最优单规则 vs ATC: -5.42% (参照平均 F=1189.783094)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| SPT | 1.000000 |
| ATC_K2 | 1.035587 |
| random_t2_016 | 1.053675 |
| random_t3_008 | 1.173115 |

- 库内最优单规则: SPT (1.000000)
- 库内逐实例最优标准化目标: 0.968386
- 互补性差距: 0.031614 (3.16%)
- 库内最优单规则原始平均 F: 1185.318824
- 库内逐实例最优原始平均 F: 1144.586055

- 最优单规则 vs SPT: +0.00% (参照平均 F=1185.318824)
- 最优单规则 vs ATC: -2.88% (参照平均 F=1220.419840)
