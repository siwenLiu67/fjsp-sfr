# Gated library report random_pool_L_nogate

- 输入候选文件: experiments/random_candidates_pool.json
- 合并后 LLM/随机候选: 120
- 合并去重数: 0
- 经典规则直接入库: 0
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
| 1 | random_t1_028 | random | best_normalized_objective | 1 | 1.000000 | 0.528497 | 0.000000 |
| 2 | random_t2_016 | random | greedy_marginal_gain | 2 | 1.062987 | 0.763385 | 0.012900 |
| 3 | random_t2_026 | random | greedy_marginal_gain | 2 | 1.123590 | 0.763385 | 0.003125 |
| 4 | random_t3_008 | random | greedy_marginal_gain | 3 | 1.133941 | 0.531088 | 0.005151 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| random_t1_028 | 1.000000 |
| random_t2_016 | 1.062987 |
| random_t2_026 | 1.123590 |
| random_t3_008 | 1.133941 |

- 库内最优单规则: random_t1_028 (1.000000)
- 库内逐实例最优标准化目标: 0.978823
- 互补性差距: 0.021177 (2.12%)
- 库内最优单规则原始平均 F: 1125.274200
- 库内逐实例最优原始平均 F: 1099.131785

- 最优单规则 vs SPT: +0.00% (参照平均 F=1125.274200)
- 最优单规则 vs ATC: -5.42% (参照平均 F=1189.783094)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| random_t1_028 | 1.000000 |
| random_t2_016 | 1.053675 |
| random_t2_026 | 1.066160 |
| random_t3_008 | 1.173115 |

- 库内最优单规则: random_t1_028 (1.000000)
- 库内逐实例最优标准化目标: 0.966289
- 互补性差距: 0.033711 (3.37%)
- 库内最优单规则原始平均 F: 1185.318824
- 库内逐实例最优原始平均 F: 1147.486055

- 最优单规则 vs SPT: +0.00% (参照平均 F=1185.318824)
- 最优单规则 vs ATC: -2.88% (参照平均 F=1220.419840)
