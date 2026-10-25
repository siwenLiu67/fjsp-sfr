# Gated library report qwen1

- 输入候选: 37
- 质量门槛淘汰 f_H >= 1: 6
- 行为相似度淘汰 Sim >= 0.9: 12
- 互补性贪心未入选: 13
- 最终规则数: 6 / K=6
- 固定参考状态数: 538

## 最终库

| # | Rule | Type | f_H | max Sim | marginal gain |
|---:|---|---:|---:|---:|---:|
| 1 | weighted_service_balance | 3 | 0.692311 | 0.700743 | 0.307689 |
| 2 | multi_value_tardiness_blend | 3 | 0.739627 | 0.579926 | 0.020032 |
| 3 | forecast_insufficiency_filter | 4 | 0.758709 | 0.643123 | 0.041570 |
| 4 | urgent_tardiness_focus | 1 | 0.775741 | 0.700743 | 0.011240 |
| 5 | slack_gap_balance | 3 | 0.795629 | 0.544610 | 0.013738 |
| 6 | conditional_forecast_recovery | 2 | 0.934165 | 0.643123 | 0.007734 |

## 选择集 seeds 600–619

| Rule | f_H |
|---|---:|
| weighted_service_balance | 0.692311 |
| multi_value_tardiness_blend | 0.739627 |
| forecast_insufficiency_filter | 0.758709 |
| urgent_tardiness_focus | 0.775741 |
| slack_gap_balance | 0.795629 |
| conditional_forecast_recovery | 0.934165 |

- 库内最优单规则: weighted_service_balance (0.692311)
- 库内逐实例最优标准化目标: 0.597996
- 互补性差距: 0.094315 (13.62%)

## 保留集 seeds 700–719

| Rule | f_H |
|---|---:|
| weighted_service_balance | 0.702003 |
| multi_value_tardiness_blend | 0.707763 |
| urgent_tardiness_focus | 0.728283 |
| slack_gap_balance | 0.734307 |
| forecast_insufficiency_filter | 0.764487 |
| conditional_forecast_recovery | 0.851425 |

- 库内最优单规则: weighted_service_balance (0.702003)
- 库内逐实例最优标准化目标: 0.616453
- 互补性差距: 0.085550 (12.19%)
