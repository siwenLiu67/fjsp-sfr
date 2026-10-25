# 正式实验前核查 2

## 结论

- service_heavy/S: 三个库中最优服务规则为 qwen2_g/weighted_service_tardiness (F=333.53)，相对 SPT 为 +12.08。
- service_heavy/M: 三个库中最优服务规则为 qwen1_g/weighted_service_balance (F=597.53)，相对 SPT 为 +52.67。
- service_heavy/L: 三个库中最优服务规则为 qwen2_g/weighted_service_tardiness (F=1270.43)，相对 SPT 为 +136.98。
- Service-heavy 下服务规则相对 SPT 的百分比差距在 S、M、L 三个规模均缩小。

## Service-heavy 难度校准

| 规模 | Best(SPT,ATC) 平均 F | TT/F | WSF/F | 35%–50% 目标 |
|---|---:|---:|---:|:---:|
| S | 310.04 | 50.34% | 49.66% | 达标 |
| M | 529.45 | 54.19% | 45.81% | 达标 |
| L | 1107.63 | 58.45% | 41.55% | 达标 |
| X | 2285.69 | 58.37% | 41.63% | 达标 |

## 服务成本加重前后：最优服务规则相对 SPT 的差距

| 规模 | Balanced 最优服务规则 | Balanced 相对 SPT | Service-heavy 最优服务规则 | Service-heavy 相对 SPT | 相对差距是否缩小 |
|---|---|---:|---|---:|:---:|
| S | qwen2_g/weighted_service_tardiness | +19.42 (+9.61%) | qwen2_g/weighted_service_tardiness | +12.08 (+3.76%) | 是 |
| M | qwen2_g/weighted_service_tardiness | +47.94 (+13.47%) | qwen1_g/weighted_service_balance | +52.67 (+9.67%) | 是 |
| L | qwen2_g/balanced_service_risk | +150.07 (+20.72%) | qwen2_g/weighted_service_tardiness | +136.98 (+12.08%) | 是 |

## Service-heavy：全部固定规则

| 规模 | 规则库 | 规则 | 平均 F | 平均 TT | 平均 WSF |
|---|---|---|---:|---:|---:|
| S | classic | SPT | 321.45 | 161.34 | 160.11 |
| S | classic | EDD | 391.90 | 218.84 | 173.06 |
| S | classic | ATC_K2 | 337.50 | 181.54 | 155.96 |
| S | classic | MIN_SLACK | 543.81 | 340.30 | 203.51 |
| S | default | due_date_slack | 448.07 | 257.96 | 190.11 |
| S | handwritten | gap_priority | 419.25 | 236.34 | 182.91 |
| S | handwritten | tardiness_avoidance | 341.89 | 182.18 | 159.71 |
| S | handwritten | due_date_first | 448.07 | 257.96 | 190.11 |
| S | handwritten | service_value | 365.42 | 209.96 | 155.46 |
| S | handwritten | gap_tard_combo | 544.71 | 339.80 | 204.91 |
| S | handwritten | capacity_aware | 361.08 | 204.62 | 156.46 |
| S | qwen1_g | weighted_service_balance | 335.45 | 179.84 | 155.61 |
| S | qwen1_g | multi_value_tardiness_blend | 355.57 | 192.66 | 162.91 |
| S | qwen1_g | forecast_insufficiency_filter | 369.51 | 204.70 | 164.81 |
| S | qwen1_g | urgent_tardiness_focus | 341.89 | 182.18 | 159.71 |
| S | qwen1_g | slack_gap_balance | 373.06 | 207.60 | 165.46 |
| S | qwen1_g | conditional_forecast_recovery | 424.35 | 240.24 | 184.11 |
| S | qwen2_g | value_density_priority | 356.40 | 202.94 | 153.46 |
| S | qwen2_g | weighted_service_tardiness | 333.53 | 177.82 | 155.71 |
| S | qwen2_g | balanced_service_risk | 339.85 | 184.24 | 155.61 |
| S | qwen2_g | composite_risk_score | 390.10 | 224.44 | 165.66 |
| S | qwen2_g | log_urgency_weighted | 419.58 | 230.52 | 189.06 |
| S | qwen2_g | deficit_urgency_mix | 385.29 | 219.48 | 165.81 |
| M | classic | SPT | 544.86 | 294.48 | 250.38 |
| M | classic | EDD | 715.12 | 413.88 | 301.24 |
| M | classic | ATC_K2 | 564.20 | 318.20 | 246.00 |
| M | classic | MIN_SLACK | 988.46 | 633.90 | 354.56 |
| M | default | due_date_slack | 829.47 | 493.60 | 335.87 |
| M | handwritten | gap_priority | 783.10 | 466.58 | 316.52 |
| M | handwritten | tardiness_avoidance | 605.32 | 348.16 | 257.16 |
| M | handwritten | due_date_first | 829.47 | 493.60 | 335.87 |
| M | handwritten | service_value | 626.00 | 381.18 | 244.82 |
| M | handwritten | gap_tard_combo | 930.88 | 591.56 | 339.32 |
| M | handwritten | capacity_aware | 637.82 | 384.16 | 253.66 |
| M | qwen1_g | weighted_service_balance | 597.53 | 349.40 | 248.13 |
| M | qwen1_g | multi_value_tardiness_blend | 629.28 | 368.68 | 260.60 |
| M | qwen1_g | forecast_insufficiency_filter | 661.38 | 391.62 | 269.76 |
| M | qwen1_g | urgent_tardiness_focus | 605.32 | 348.16 | 257.16 |
| M | qwen1_g | slack_gap_balance | 677.82 | 408.84 | 268.98 |
| M | qwen1_g | conditional_forecast_recovery | 762.91 | 454.44 | 308.47 |
| M | qwen2_g | value_density_priority | 632.74 | 379.66 | 253.08 |
| M | qwen2_g | weighted_service_tardiness | 602.11 | 351.56 | 250.55 |
| M | qwen2_g | balanced_service_risk | 599.53 | 350.70 | 248.83 |
| M | qwen2_g | composite_risk_score | 669.29 | 402.44 | 266.85 |
| M | qwen2_g | log_urgency_weighted | 729.69 | 428.36 | 301.33 |
| M | qwen2_g | deficit_urgency_mix | 731.54 | 443.60 | 287.94 |
| L | classic | SPT | 1133.45 | 660.22 | 473.23 |
| L | classic | EDD | 1502.73 | 940.32 | 562.41 |
| L | classic | ATC_K2 | 1203.37 | 731.94 | 471.43 |
| L | classic | MIN_SLACK | 2234.63 | 1534.14 | 700.49 |
| L | default | due_date_slack | 1861.56 | 1205.28 | 656.28 |
| L | handwritten | gap_priority | 1710.19 | 1100.76 | 609.43 |
| L | handwritten | tardiness_avoidance | 1314.57 | 805.32 | 509.25 |
| L | handwritten | due_date_first | 1861.56 | 1205.28 | 656.28 |
| L | handwritten | service_value | 1401.06 | 921.16 | 479.90 |
| L | handwritten | gap_tard_combo | 1974.03 | 1329.40 | 644.63 |
| L | handwritten | capacity_aware | 1451.33 | 946.30 | 505.03 |
| L | qwen1_g | weighted_service_balance | 1285.58 | 811.08 | 474.50 |
| L | qwen1_g | multi_value_tardiness_blend | 1398.33 | 867.54 | 530.79 |
| L | qwen1_g | forecast_insufficiency_filter | 1450.95 | 918.76 | 532.19 |
| L | qwen1_g | urgent_tardiness_focus | 1314.57 | 805.32 | 509.25 |
| L | qwen1_g | slack_gap_balance | 1516.76 | 975.74 | 541.02 |
| L | qwen1_g | conditional_forecast_recovery | 1689.47 | 1082.34 | 607.13 |
| L | qwen2_g | value_density_priority | 1342.22 | 867.96 | 474.26 |
| L | qwen2_g | weighted_service_tardiness | 1270.43 | 800.66 | 469.77 |
| L | qwen2_g | balanced_service_risk | 1271.45 | 797.50 | 473.95 |
| L | qwen2_g | composite_risk_score | 1491.45 | 964.12 | 527.33 |
| L | qwen2_g | log_urgency_weighted | 1633.78 | 1034.36 | 599.42 |
| L | qwen2_g | deficit_urgency_mix | 1668.61 | 1089.38 | 579.23 |

## Service-heavy：各库最优单规则配对比较

| 规模 | 规则库 | 最优单规则 | 平均 F | ΔF=规则−SPT (95% CI) | ΔF=规则−ATC (95% CI) |
|---|---|---|---:|---:|---:|
| S | handwritten | tardiness_avoidance | 341.89 | +20.44 [+7.83, +32.97] | +4.39 [-6.83, +15.54] |
| S | qwen1_g | weighted_service_balance | 335.45 | +14.00 [+1.33, +26.82] | -2.05 [-16.28, +12.14] |
| S | qwen2_g | weighted_service_tardiness | 333.53 | +12.08 [+0.09, +24.32] | -3.97 [-17.84, +9.93] |
| M | handwritten | tardiness_avoidance | 605.32 | +60.46 [+36.87, +84.65] | +41.12 [+19.41, +63.46] |
| M | qwen1_g | weighted_service_balance | 597.53 | +52.67 [+29.44, +75.97] | +33.33 [+13.35, +53.76] |
| M | qwen2_g | balanced_service_risk | 599.53 | +54.67 [+31.47, +77.88] | +35.33 [+18.14, +52.68] |
| L | handwritten | tardiness_avoidance | 1314.57 | +181.11 [+146.25, +215.57] | +111.20 [+74.93, +149.21] |
| L | qwen1_g | weighted_service_balance | 1285.58 | +152.13 [+112.42, +191.94] | +82.21 [+48.34, +117.06] |
| L | qwen2_g | weighted_service_tardiness | 1270.43 | +136.98 [+94.51, +178.65] | +67.06 [+30.25, +104.36] |

## Balanced/M：SPT、ATC 与 Qwen 库最优单规则分解

| 规则库 | 规则 | 平均 F | 平均 TT | 平均 WSF | TT/F | WSF/F |
|---|---|---:|---:|---:|---:|---:|
| classic | SPT | 355.92 | 272.86 | 83.06 | 76.66% | 23.34% |
| classic | ATC_K2 | 378.69 | 298.40 | 80.29 | 78.80% | 21.20% |
| qwen1_g | weighted_service_balance | 407.41 | 326.52 | 80.89 | 80.15% | 19.85% |
| qwen2_g | weighted_service_tardiness | 403.86 | 323.10 | 80.76 | 80.00% | 20.00% |
