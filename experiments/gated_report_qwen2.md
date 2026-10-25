# Gated library report qwen2

- 输入候选: 39
- 质量门槛淘汰 f_H >= 1: 13
- 行为相似度淘汰 Sim >= 0.9: 5
- 互补性贪心未入选: 15
- 最终规则数: 6 / K=6
- 固定参考状态数: 538

## 最终库

| # | Rule | Type | f_H | max Sim | marginal gain |
|---:|---|---:|---:|---:|---:|
| 1 | value_density_priority | 1 | 0.707706 | 0.680297 | 0.292294 |
| 2 | weighted_service_tardiness | 3 | 0.716584 | 0.802974 | 0.008942 |
| 3 | balanced_service_risk | 3 | 0.716899 | 0.802974 | 0.069307 |
| 4 | composite_risk_score | 3 | 0.817215 | 0.513011 | 0.028389 |
| 5 | log_urgency_weighted | 3 | 0.853318 | 0.592937 | 0.003336 |
| 6 | deficit_urgency_mix | 3 | 0.886541 | 0.680297 | 0.009121 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| value_density_priority | 0.707706 |
| weighted_service_tardiness | 0.716584 |
| balanced_service_risk | 0.716899 |
| composite_risk_score | 0.817215 |
| log_urgency_weighted | 0.853318 |
| deficit_urgency_mix | 0.886541 |

- 库内最优单规则: value_density_priority (0.707706)
- 库内逐实例最优标准化目标: 0.588610
- 互补性差距: 0.119096 (16.83%)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| value_density_priority | 0.649567 |
| balanced_service_risk | 0.700349 |
| weighted_service_tardiness | 0.713047 |
| log_urgency_weighted | 0.792128 |
| composite_risk_score | 0.813035 |
| deficit_urgency_mix | 0.835535 |

- 库内最优单规则: value_density_priority (0.649567)
- 库内逐实例最优标准化目标: 0.603017
- 互补性差距: 0.046550 (7.17%)
