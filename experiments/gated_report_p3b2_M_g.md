# Gated library report p3b2_M_g

- 输入候选: 26
- 规模/难度: M/service_heavy
- 特征集: theory11_plus3
- 质量门槛基线: best_classic
- best_classic 口径: 按选择集平均 F 选定单一固定基线 SPT; f_H 的逐实例分母均来自该规则
- 质量门槛淘汰 f_H >= 1: 24
- 行为相似度淘汰 Sim >= 0.9: 1
- 互补性贪心未入选: 0
- 最终规则数: 1 / K=6
- 固定参考状态数: 527

## 最终库

| # | Rule | Source | Type | f_H | max Sim | marginal gain |
|---:|---|---|---:|---:|---:|---:|
| 1 | Aggressive_SPT_Checkpoint | llm | 1 | 0.989797 | 0.000000 | 0.010203 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| Aggressive_SPT_Checkpoint | 0.989797 |

- 库内最优单规则: Aggressive_SPT_Checkpoint (0.989797)
- 库内逐实例最优标准化目标: 0.989797
- 互补性差距: 0.000000 (0.00%)
- 库内最优单规则原始平均 F: 572.965875
- 库内逐实例最优原始平均 F: 572.965875

- 最优单规则 vs SPT: -1.17% (参照平均 F=579.765875)
- 最优单规则 vs ATC: -4.79% (参照平均 F=601.765875)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| Aggressive_SPT_Checkpoint | 1.017581 |

- 库内最优单规则: Aggressive_SPT_Checkpoint (1.017581)
- 库内逐实例最优标准化目标: 1.017581
- 互补性差距: 0.000000 (0.00%)
- 库内最优单规则原始平均 F: 566.288947
- 库内逐实例最优原始平均 F: 566.288947

- 最优单规则 vs SPT: +0.99% (参照平均 F=560.752161)
- 最优单规则 vs ATC: -4.09% (参照平均 F=590.413947)
