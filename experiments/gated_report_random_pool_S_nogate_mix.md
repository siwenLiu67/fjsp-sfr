# Gated library report random_pool_S_nogate_mix

- 输入候选文件: experiments/random_candidates_pool.json
- 合并后 LLM/随机候选: 120
- 合并去重数: 0
- 经典规则直接入库: 2
- 规模/难度: S/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- 质量门槛: 关闭
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 0
- 行为相似度淘汰 Sim >= 0.9: 58
- 互补性贪心未入选: 56
- 最终规则数: 6 / K=6
- 固定参考状态数: 282

## 最终库

| # | Rule | Source | Selection reason | Type | f_H | max Sim | marginal gain |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | SPT | classic | mandatory_classic_bypass_quality_gate | classic | 1.000000 | 0.627660 | 0.000000 |
| 2 | ATC_K2 | classic | mandatory_classic_bypass_quality_gate | classic | 1.027784 | 0.868794 | 0.000000 |
| 3 | random_t2_026 | random | greedy_marginal_gain | 2 | 1.025862 | 0.868794 | 0.022303 |
| 4 | random_t2_008 | random | greedy_marginal_gain | 2 | 1.205826 | 0.677305 | 0.011259 |
| 5 | random_t1_012 | random | greedy_marginal_gain | 1 | 1.055880 | 0.656028 | 0.005853 |
| 6 | random_t3_002 | random | greedy_marginal_gain | 3 | 1.061470 | 0.677305 | 0.003663 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| SPT | 1.000000 |
| random_t2_026 | 1.025862 |
| ATC_K2 | 1.027784 |
| random_t1_012 | 1.055880 |
| random_t3_002 | 1.061470 |
| random_t2_008 | 1.205826 |

- 库内最优单规则: SPT (1.000000)
- 库内逐实例最优标准化目标: 0.916784
- 互补性差距: 0.083216 (8.32%)
- 库内最优单规则原始平均 F: 322.196146
- 库内逐实例最优原始平均 F: 293.271146

- 最优单规则 vs SPT: +0.00% (参照平均 F=322.196146)
- 最优单规则 vs ATC: -1.82% (参照平均 F=328.171146)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| SPT | 1.000000 |
| random_t2_026 | 1.079142 |
| random_t3_002 | 1.102298 |
| ATC_K2 | 1.128626 |
| random_t1_012 | 1.163271 |
| random_t2_008 | 1.277242 |

- 库内最优单规则: SPT (1.000000)
- 库内逐实例最优标准化目标: 0.962537
- 互补性差距: 0.037463 (3.75%)
- 库内最优单规则原始平均 F: 320.728754
- 库内逐实例最优原始平均 F: 308.128754

- 最优单规则 vs SPT: +0.00% (参照平均 F=320.728754)
- 最优单规则 vs ATC: -10.13% (参照平均 F=356.878754)
