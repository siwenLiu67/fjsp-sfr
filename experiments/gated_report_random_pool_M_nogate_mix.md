# Gated library report random_pool_M_nogate_mix

- 输入候选文件: experiments/random_candidates_pool.json
- 合并后 LLM/随机候选: 120
- 合并去重数: 0
- 经典规则直接入库: 2
- 规模/难度: M/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- 质量门槛: 关闭
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 0
- 行为相似度淘汰 Sim >= 0.9: 32
- 互补性贪心未入选: 82
- 最终规则数: 6 / K=6
- 固定参考状态数: 527

## 最终库

| # | Rule | Source | Selection reason | Type | f_H | max Sim | marginal gain |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | SPT | classic | mandatory_classic_bypass_quality_gate | classic | 1.000000 | 0.641366 | 0.000000 |
| 2 | ATC_K2 | classic | mandatory_classic_bypass_quality_gate | classic | 1.043237 | 0.916509 | 0.000000 |
| 3 | random_t1_027 | random | greedy_marginal_gain | 1 | 1.081917 | 0.808349 | 0.015484 |
| 4 | random_t1_016 | random | greedy_marginal_gain | 1 | 1.112777 | 0.808349 | 0.010895 |
| 5 | random_t2_026 | random | greedy_marginal_gain | 2 | 1.045647 | 0.840607 | 0.010540 |
| 6 | random_t2_016 | random | greedy_marginal_gain | 2 | 1.042813 | 0.916509 | 0.005998 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| SPT | 1.000000 |
| random_t2_016 | 1.042813 |
| ATC_K2 | 1.043237 |
| random_t2_026 | 1.045647 |
| random_t1_027 | 1.081917 |
| random_t1_016 | 1.112777 |

- 库内最优单规则: SPT (1.000000)
- 库内逐实例最优标准化目标: 0.923593
- 互补性差距: 0.076407 (7.64%)
- 库内最优单规则原始平均 F: 579.765875
- 库内逐实例最优原始平均 F: 532.065875

- 最优单规则 vs SPT: +0.00% (参照平均 F=579.765875)
- 最优单规则 vs ATC: -3.66% (参照平均 F=601.765875)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| SPT | 1.000000 |
| random_t2_016 | 1.040177 |
| ATC_K2 | 1.060534 |
| random_t2_026 | 1.172241 |
| random_t1_016 | 1.218910 |
| random_t1_027 | 1.254673 |

- 库内最优单规则: SPT (1.000000)
- 库内逐实例最优标准化目标: 0.966168
- 互补性差距: 0.033832 (3.38%)
- 库内最优单规则原始平均 F: 560.752161
- 库内逐实例最优原始平均 F: 540.213947

- 最优单规则 vs SPT: +0.00% (参照平均 F=560.752161)
- 最优单规则 vs ATC: -5.02% (参照平均 F=590.413947)
