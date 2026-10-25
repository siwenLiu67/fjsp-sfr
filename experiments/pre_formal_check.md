# 正式实验前固定规则核查

## 结论

- balanced/S: handwritten/tardiness_avoidance=227.57 (vs SPT +25.52; vs ATC +8.42)；qwen1_g/weighted_service_balance=223.21 (vs SPT +21.16; vs ATC +4.06)；qwen2_g/weighted_service_tardiness=221.47 (vs SPT +19.42; vs ATC +2.32)。
- balanced/M: handwritten/tardiness_avoidance=414.58 (vs SPT +58.65; vs ATC +35.89)；qwen1_g/weighted_service_balance=407.41 (vs SPT +51.48; vs ATC +28.72)；qwen2_g/weighted_service_tardiness=403.86 (vs SPT +47.94; vs ATC +25.18)。
- balanced/L: handwritten/tardiness_avoidance=911.80 (vs SPT +187.60; vs ATC +109.03)；qwen1_g/weighted_service_balance=906.79 (vs SPT +182.60; vs ATC +104.02)；qwen2_g/balanced_service_risk=874.27 (vs SPT +150.07; vs ATC +71.50)。
- hard/S: handwritten/tardiness_avoidance=281.51 (vs SPT +31.66; vs ATC +10.82)；qwen1_g/weighted_service_balance=274.57 (vs SPT +24.72; vs ATC +3.88)；qwen2_g/balanced_service_risk=273.21 (vs SPT +23.36; vs ATC +2.52)。
- hard/M: handwritten/tardiness_avoidance=536.30 (vs SPT +61.30; vs ATC +37.33)；qwen1_g/weighted_service_balance=515.42 (vs SPT +40.41; vs ATC +16.44)；qwen2_g/weighted_service_tardiness=521.12 (vs SPT +46.11; vs ATC +22.14)。
- hard/L: handwritten/tardiness_avoidance=1224.27 (vs SPT +163.13; vs ATC +102.88)；qwen1_g/urgent_tardiness_focus=1224.27 (vs SPT +163.13; vs ATC +102.88)；qwen2_g/weighted_service_tardiness=1218.74 (vs SPT +157.59; vs ATC +97.35)。
- M/balanced 与旧 15-job 默认生成器完全一致的实例为 0/50，因此结论为不一致。

## Balanced：全部规则

| 规模 | 规则库 | 规则 | 平均 F | 平均 TT | 平均 WSF | 配对差 ΔF=规则−SPT | 95% CI |
|---|---|---|---:|---:|---:|---:|---:|
| S | classic | SPT | 202.05 | 147.38 | 54.67 | 0.00 | [0.00, 0.00] |
| S | classic | EDD | 266.03 | 206.80 | 59.23 | 63.98 | [48.60, 79.76] |
| S | classic | ATC_K2 | 219.15 | 167.04 | 52.11 | 17.10 | [7.88, 26.82] |
| S | classic | MIN_SLACK | 398.09 | 324.74 | 73.35 | 196.04 | [176.62, 215.94] |
| S | default | due_date_slack | 305.03 | 237.96 | 67.07 | 102.98 | [84.42, 122.34] |
| S | handwritten | gap_priority | 284.45 | 222.22 | 62.23 | 82.40 | [65.60, 99.54] |
| S | handwritten | tardiness_avoidance | 227.57 | 173.06 | 54.51 | 25.52 | [14.22, 37.42] |
| S | handwritten | due_date_first | 305.03 | 237.96 | 67.07 | 102.98 | [84.58, 122.12] |
| S | handwritten | service_value | 233.85 | 184.68 | 49.17 | 31.80 | [20.14, 44.10] |
| S | handwritten | gap_tard_combo | 397.27 | 324.08 | 73.19 | 195.22 | [174.28, 216.60] |
| S | handwritten | capacity_aware | 240.57 | 188.44 | 52.13 | 38.52 | [26.78, 50.54] |
| S | qwen1_g | weighted_service_balance | 223.21 | 174.10 | 49.11 | 21.16 | [10.44, 32.34] |
| S | qwen1_g | multi_value_tardiness_blend | 231.52 | 177.56 | 53.96 | 29.47 | [17.59, 41.20] |
| S | qwen1_g | forecast_insufficiency_filter | 244.27 | 189.88 | 54.39 | 42.22 | [31.24, 52.94] |
| S | qwen1_g | urgent_tardiness_focus | 227.57 | 173.06 | 54.51 | 25.52 | [13.88, 37.58] |
| S | qwen1_g | slack_gap_balance | 253.83 | 198.68 | 55.15 | 51.78 | [37.76, 66.32] |
| S | qwen1_g | conditional_forecast_recovery | 291.17 | 227.90 | 63.27 | 89.12 | [72.64, 105.56] |
| S | qwen2_g | value_density_priority | 224.41 | 175.08 | 49.33 | 22.36 | [11.88, 33.32] |
| S | qwen2_g | weighted_service_tardiness | 221.47 | 172.44 | 49.03 | 19.42 | [8.16, 30.98] |
| S | qwen2_g | balanced_service_risk | 225.53 | 175.30 | 50.23 | 23.48 | [11.92, 35.58] |
| S | qwen2_g | composite_risk_score | 263.19 | 208.42 | 54.77 | 61.14 | [46.14, 77.64] |
| S | qwen2_g | log_urgency_weighted | 275.49 | 212.62 | 62.87 | 73.44 | [59.70, 87.20] |
| S | qwen2_g | deficit_urgency_mix | 254.09 | 197.62 | 56.47 | 52.04 | [37.90, 66.52] |
| M | classic | SPT | 355.92 | 272.86 | 83.06 | 0.00 | [0.00, 0.00] |
| M | classic | EDD | 455.03 | 357.52 | 97.51 | 99.11 | [75.62, 123.03] |
| M | classic | ATC_K2 | 378.69 | 298.40 | 80.29 | 22.76 | [3.89, 41.87] |
| M | classic | MIN_SLACK | 691.68 | 569.02 | 122.66 | 335.75 | [303.07, 369.54] |
| M | default | due_date_slack | 562.34 | 449.80 | 112.54 | 206.42 | [179.25, 234.95] |
| M | handwritten | gap_priority | 536.48 | 429.00 | 107.48 | 180.55 | [150.58, 211.13] |
| M | handwritten | tardiness_avoidance | 414.58 | 327.46 | 87.12 | 58.65 | [40.21, 77.16] |
| M | handwritten | due_date_first | 562.34 | 449.80 | 112.54 | 206.42 | [179.87, 234.63] |
| M | handwritten | service_value | 436.64 | 355.80 | 80.84 | 80.72 | [63.73, 97.50] |
| M | handwritten | gap_tard_combo | 641.54 | 524.40 | 117.14 | 285.61 | [254.09, 318.31] |
| M | handwritten | capacity_aware | 432.38 | 348.28 | 84.10 | 76.46 | [56.89, 95.72] |
| M | qwen1_g | weighted_service_balance | 407.41 | 326.52 | 80.89 | 51.48 | [35.87, 67.37] |
| M | qwen1_g | multi_value_tardiness_blend | 419.71 | 332.38 | 87.33 | 63.78 | [45.30, 83.52] |
| M | qwen1_g | forecast_insufficiency_filter | 455.59 | 365.66 | 89.93 | 99.66 | [77.78, 122.47] |
| M | qwen1_g | urgent_tardiness_focus | 414.58 | 327.46 | 87.12 | 58.65 | [40.42, 77.26] |
| M | qwen1_g | slack_gap_balance | 472.93 | 379.72 | 93.21 | 117.01 | [96.51, 138.21] |
| M | qwen1_g | conditional_forecast_recovery | 529.99 | 423.98 | 106.01 | 174.06 | [148.20, 200.56] |
| M | qwen2_g | value_density_priority | 418.44 | 338.88 | 79.56 | 62.51 | [44.36, 81.49] |
| M | qwen2_g | weighted_service_tardiness | 403.86 | 323.10 | 80.76 | 47.94 | [31.94, 64.09] |
| M | qwen2_g | balanced_service_risk | 408.26 | 326.98 | 81.28 | 52.34 | [33.21, 71.92] |
| M | qwen2_g | composite_risk_score | 457.42 | 368.38 | 89.04 | 101.50 | [81.69, 120.98] |
| M | qwen2_g | log_urgency_weighted | 494.90 | 394.82 | 100.08 | 138.98 | [121.17, 157.89] |
| M | qwen2_g | deficit_urgency_mix | 499.90 | 404.02 | 95.88 | 143.97 | [114.44, 174.12] |
| L | classic | SPT | 724.20 | 576.86 | 147.34 | 0.00 | [0.00, 0.00] |
| L | classic | EDD | 1039.01 | 850.78 | 188.23 | 314.81 | [284.15, 345.72] |
| L | classic | ATC_K2 | 802.77 | 655.04 | 147.73 | 78.57 | [53.40, 104.17] |
| L | classic | MIN_SLACK | 1622.78 | 1379.58 | 243.20 | 898.58 | [831.54, 966.83] |
| L | default | due_date_slack | 1295.52 | 1071.80 | 223.72 | 571.32 | [528.34, 614.42] |
| L | handwritten | gap_priority | 1175.99 | 974.06 | 201.93 | 451.79 | [409.14, 493.89] |
| L | handwritten | tardiness_avoidance | 911.80 | 747.72 | 164.08 | 187.60 | [156.84, 217.58] |
| L | handwritten | due_date_first | 1295.52 | 1071.80 | 223.72 | 571.32 | [528.68, 614.46] |
| L | handwritten | service_value | 948.14 | 798.50 | 149.64 | 223.94 | [189.10, 259.14] |
| L | handwritten | gap_tard_combo | 1437.44 | 1215.46 | 221.98 | 713.24 | [645.49, 779.69] |
| L | handwritten | capacity_aware | 985.82 | 823.46 | 162.36 | 261.62 | [229.30, 295.22] |
| L | qwen1_g | weighted_service_balance | 906.79 | 756.72 | 150.07 | 182.60 | [145.98, 218.79] |
| L | qwen1_g | multi_value_tardiness_blend | 931.93 | 765.48 | 166.45 | 207.73 | [178.85, 236.57] |
| L | qwen1_g | forecast_insufficiency_filter | 1022.59 | 848.40 | 174.19 | 298.39 | [267.55, 329.21] |
| L | qwen1_g | urgent_tardiness_focus | 911.80 | 747.72 | 164.08 | 187.60 | [157.24, 217.58] |
| L | qwen1_g | slack_gap_balance | 1045.88 | 871.30 | 174.58 | 321.68 | [285.24, 358.11] |
| L | qwen1_g | conditional_forecast_recovery | 1186.62 | 980.12 | 206.50 | 462.42 | [420.28, 506.35] |
| L | qwen2_g | value_density_priority | 918.33 | 764.28 | 154.05 | 194.13 | [165.39, 224.62] |
| L | qwen2_g | weighted_service_tardiness | 897.51 | 749.38 | 148.13 | 173.31 | [138.32, 207.64] |
| L | qwen2_g | balanced_service_risk | 874.27 | 723.20 | 151.07 | 150.07 | [121.98, 178.96] |
| L | qwen2_g | composite_risk_score | 1053.81 | 885.22 | 168.59 | 329.61 | [300.49, 359.11] |
| L | qwen2_g | log_urgency_weighted | 1108.54 | 912.50 | 196.04 | 384.34 | [351.88, 416.54] |
| L | qwen2_g | deficit_urgency_mix | 1179.57 | 985.78 | 193.79 | 455.37 | [416.35, 496.44] |

## Balanced：各库最优单规则

| 规模 | 规则库 | 库内最优单规则 | 平均 F | 较 SPT 差值 | 较 SPT | 较 ATC 差值 | 较 ATC |
|---|---|---|---:|---:|:---:|---:|:---:|
| S | handwritten | tardiness_avoidance | 227.57 | 25.52 (+12.63%) | 否 | 8.42 (+3.84%) | 否 |
| S | qwen1_g | weighted_service_balance | 223.21 | 21.16 (+10.47%) | 否 | 4.06 (+1.85%) | 否 |
| S | qwen2_g | weighted_service_tardiness | 221.47 | 19.42 (+9.61%) | 否 | 2.32 (+1.06%) | 否 |
| M | handwritten | tardiness_avoidance | 414.58 | 58.65 (+16.48%) | 否 | 35.89 (+9.48%) | 否 |
| M | qwen1_g | weighted_service_balance | 407.41 | 51.48 (+14.46%) | 否 | 28.72 (+7.58%) | 否 |
| M | qwen2_g | weighted_service_tardiness | 403.86 | 47.94 (+13.47%) | 否 | 25.18 (+6.65%) | 否 |
| L | handwritten | tardiness_avoidance | 911.80 | 187.60 (+25.90%) | 否 | 109.03 (+13.58%) | 否 |
| L | qwen1_g | weighted_service_balance | 906.79 | 182.60 (+25.21%) | 否 | 104.02 (+12.96%) | 否 |
| L | qwen2_g | balanced_service_risk | 874.27 | 150.07 (+20.72%) | 否 | 71.50 (+8.91%) | 否 |

## Hard：全部规则

| 规模 | 规则库 | 规则 | 平均 F | 平均 TT | 平均 WSF | 配对差 ΔF=规则−SPT | 95% CI |
|---|---|---|---:|---:|---:|---:|---:|
| S | classic | SPT | 249.85 | 182.48 | 67.37 | 0.00 | [0.00, 0.00] |
| S | classic | EDD | 308.85 | 237.34 | 71.51 | 59.00 | [45.50, 73.24] |
| S | classic | ATC_K2 | 270.69 | 203.22 | 67.47 | 20.84 | [10.14, 31.86] |
| S | classic | MIN_SLACK | 449.71 | 369.22 | 80.49 | 199.86 | [178.52, 221.36] |
| S | default | due_date_slack | 347.85 | 271.06 | 76.79 | 98.00 | [82.70, 114.60] |
| S | handwritten | gap_priority | 347.75 | 271.12 | 76.63 | 97.90 | [82.12, 114.04] |
| S | handwritten | tardiness_avoidance | 281.51 | 212.64 | 68.87 | 31.66 | [21.36, 42.10] |
| S | handwritten | due_date_first | 347.85 | 271.06 | 76.79 | 98.00 | [82.54, 114.32] |
| S | handwritten | service_value | 296.01 | 230.56 | 65.45 | 46.16 | [34.20, 58.62] |
| S | handwritten | gap_tard_combo | 450.69 | 369.88 | 80.81 | 200.84 | [179.78, 221.80] |
| S | handwritten | capacity_aware | 308.67 | 240.18 | 68.49 | 58.82 | [47.12, 70.68] |
| S | qwen1_g | weighted_service_balance | 274.57 | 208.44 | 66.13 | 24.72 | [12.54, 37.60] |
| S | qwen1_g | multi_value_tardiness_blend | 298.93 | 228.86 | 70.07 | 49.08 | [36.22, 62.36] |
| S | qwen1_g | forecast_insufficiency_filter | 312.65 | 242.42 | 70.23 | 62.80 | [49.72, 75.80] |
| S | qwen1_g | urgent_tardiness_focus | 281.51 | 212.64 | 68.87 | 31.66 | [21.20, 42.08] |
| S | qwen1_g | slack_gap_balance | 327.43 | 255.12 | 72.31 | 77.58 | [60.58, 95.18] |
| S | qwen1_g | conditional_forecast_recovery | 347.09 | 270.56 | 76.53 | 97.24 | [81.54, 113.64] |
| S | qwen2_g | value_density_priority | 292.55 | 226.86 | 65.69 | 42.70 | [30.02, 55.72] |
| S | qwen2_g | weighted_service_tardiness | 275.39 | 208.86 | 66.53 | 25.54 | [12.90, 39.06] |
| S | qwen2_g | balanced_service_risk | 273.21 | 206.96 | 66.25 | 23.36 | [12.14, 35.16] |
| S | qwen2_g | composite_risk_score | 310.87 | 241.52 | 69.35 | 61.02 | [47.08, 75.24] |
| S | qwen2_g | log_urgency_weighted | 336.07 | 258.74 | 77.33 | 86.22 | [73.56, 99.18] |
| S | qwen2_g | deficit_urgency_mix | 305.99 | 236.96 | 69.03 | 56.14 | [43.76, 68.54] |
| M | classic | SPT | 475.01 | 366.82 | 108.19 | 0.00 | [0.00, 0.00] |
| M | classic | EDD | 598.50 | 480.68 | 117.82 | 123.49 | [97.77, 149.00] |
| M | classic | ATC_K2 | 498.98 | 392.78 | 106.20 | 23.97 | [9.97, 38.22] |
| M | classic | MIN_SLACK | 895.92 | 752.12 | 143.80 | 420.91 | [389.49, 453.40] |
| M | default | due_date_slack | 692.63 | 561.14 | 131.49 | 217.62 | [190.50, 244.55] |
| M | handwritten | gap_priority | 696.43 | 564.38 | 132.05 | 221.42 | [194.69, 248.23] |
| M | handwritten | tardiness_avoidance | 536.30 | 426.32 | 109.98 | 61.30 | [41.41, 81.62] |
| M | handwritten | due_date_first | 692.63 | 561.14 | 131.49 | 217.62 | [190.13, 243.95] |
| M | handwritten | service_value | 575.76 | 472.02 | 103.74 | 100.75 | [79.57, 121.79] |
| M | handwritten | gap_tard_combo | 843.24 | 702.60 | 140.64 | 368.23 | [340.52, 395.90] |
| M | handwritten | capacity_aware | 592.14 | 483.30 | 108.84 | 117.13 | [91.83, 142.66] |
| M | qwen1_g | weighted_service_balance | 515.42 | 411.58 | 103.84 | 40.41 | [21.43, 59.61] |
| M | qwen1_g | multi_value_tardiness_blend | 571.50 | 459.80 | 111.70 | 96.49 | [73.42, 119.42] |
| M | qwen1_g | forecast_insufficiency_filter | 590.94 | 477.06 | 113.88 | 115.93 | [92.34, 139.66] |
| M | qwen1_g | urgent_tardiness_focus | 536.30 | 426.32 | 109.98 | 61.30 | [41.62, 81.65] |
| M | qwen1_g | slack_gap_balance | 642.16 | 520.78 | 121.38 | 167.15 | [141.94, 192.78] |
| M | qwen1_g | conditional_forecast_recovery | 697.33 | 566.64 | 130.69 | 222.32 | [192.20, 251.40] |
| M | qwen2_g | value_density_priority | 555.06 | 450.64 | 104.42 | 80.05 | [58.46, 101.50] |
| M | qwen2_g | weighted_service_tardiness | 521.12 | 416.36 | 104.76 | 46.11 | [27.83, 63.78] |
| M | qwen2_g | balanced_service_risk | 527.88 | 422.74 | 105.14 | 52.87 | [34.66, 71.41] |
| M | qwen2_g | composite_risk_score | 625.79 | 510.98 | 114.81 | 150.78 | [124.58, 178.33] |
| M | qwen2_g | log_urgency_weighted | 648.82 | 522.02 | 126.80 | 173.81 | [152.90, 194.29] |
| M | qwen2_g | deficit_urgency_mix | 664.86 | 546.18 | 118.68 | 189.85 | [163.34, 216.81] |
| L | classic | SPT | 1061.15 | 859.02 | 202.13 | 0.00 | [0.00, 0.00] |
| L | classic | EDD | 1388.23 | 1162.14 | 226.09 | 327.08 | [289.87, 364.79] |
| L | classic | ATC_K2 | 1121.39 | 923.70 | 197.69 | 60.24 | [31.20, 90.59] |
| L | classic | MIN_SLACK | 2161.39 | 1875.44 | 285.95 | 1100.24 | [1038.27, 1163.38] |
| L | default | due_date_slack | 1659.71 | 1400.26 | 259.45 | 598.56 | [546.92, 649.90] |
| L | handwritten | gap_priority | 1640.91 | 1387.86 | 253.05 | 579.76 | [536.91, 623.37] |
| L | handwritten | tardiness_avoidance | 1224.27 | 1014.46 | 209.81 | 163.13 | [128.93, 197.60] |
| L | handwritten | due_date_first | 1659.71 | 1400.26 | 259.45 | 598.56 | [546.97, 651.15] |
| L | handwritten | service_value | 1354.15 | 1152.96 | 201.19 | 293.00 | [257.41, 329.40] |
| L | handwritten | gap_tard_combo | 1924.67 | 1654.36 | 270.31 | 863.52 | [802.31, 924.76] |
| L | handwritten | capacity_aware | 1399.72 | 1190.30 | 209.42 | 338.57 | [297.80, 379.46] |
| L | qwen1_g | weighted_service_balance | 1237.47 | 1040.12 | 197.35 | 176.32 | [146.50, 206.03] |
| L | qwen1_g | multi_value_tardiness_blend | 1375.50 | 1153.08 | 222.42 | 314.35 | [276.93, 352.21] |
| L | qwen1_g | forecast_insufficiency_filter | 1399.31 | 1181.08 | 218.23 | 338.16 | [297.50, 379.35] |
| L | qwen1_g | urgent_tardiness_focus | 1224.27 | 1014.46 | 209.81 | 163.13 | [129.50, 196.73] |
| L | qwen1_g | slack_gap_balance | 1479.57 | 1249.40 | 230.17 | 418.42 | [370.12, 466.79] |
| L | qwen1_g | conditional_forecast_recovery | 1616.99 | 1362.10 | 254.89 | 555.84 | [507.54, 603.70] |
| L | qwen2_g | value_density_priority | 1322.67 | 1119.68 | 202.99 | 261.52 | [224.57, 298.51] |
| L | qwen2_g | weighted_service_tardiness | 1218.74 | 1019.22 | 199.52 | 157.59 | [130.35, 184.40] |
| L | qwen2_g | balanced_service_risk | 1227.96 | 1028.62 | 199.34 | 166.82 | [134.94, 200.00] |
| L | qwen2_g | composite_risk_score | 1450.93 | 1233.92 | 217.01 | 389.78 | [346.26, 434.69] |
| L | qwen2_g | log_urgency_weighted | 1570.33 | 1319.24 | 251.09 | 509.18 | [467.97, 551.17] |
| L | qwen2_g | deficit_urgency_mix | 1569.89 | 1334.62 | 235.27 | 508.75 | [463.83, 552.36] |

## Hard：各库最优单规则

| 规模 | 规则库 | 库内最优单规则 | 平均 F | 较 SPT 差值 | 较 SPT | 较 ATC 差值 | 较 ATC |
|---|---|---|---:|---:|:---:|---:|:---:|
| S | handwritten | tardiness_avoidance | 281.51 | 31.66 (+12.67%) | 否 | 10.82 (+4.00%) | 否 |
| S | qwen1_g | weighted_service_balance | 274.57 | 24.72 (+9.89%) | 否 | 3.88 (+1.43%) | 否 |
| S | qwen2_g | balanced_service_risk | 273.21 | 23.36 (+9.35%) | 否 | 2.52 (+0.93%) | 否 |
| M | handwritten | tardiness_avoidance | 536.30 | 61.30 (+12.90%) | 否 | 37.33 (+7.48%) | 否 |
| M | qwen1_g | weighted_service_balance | 515.42 | 40.41 (+8.51%) | 否 | 16.44 (+3.29%) | 否 |
| M | qwen2_g | weighted_service_tardiness | 521.12 | 46.11 (+9.71%) | 否 | 22.14 (+4.44%) | 否 |
| L | handwritten | tardiness_avoidance | 1224.27 | 163.13 (+15.37%) | 否 | 102.88 (+9.17%) | 否 |
| L | qwen1_g | urgent_tardiness_focus | 1224.27 | 163.13 (+15.37%) | 否 | 102.88 (+9.17%) | 否 |
| L | qwen2_g | weighted_service_tardiness | 1218.74 | 157.59 (+14.85%) | 否 | 97.35 (+8.68%) | 否 |

## Hard：缺省排序的目标构成

| 规模 | 缺省排序平均 F | 平均 TT | TT/F | 平均 WSF | WSF/F |
|---|---:|---:|---:|---:|---:|
| S | 347.85 | 271.06 | 77.93% | 76.79 | 22.07% |
| M | 692.63 | 561.14 | 81.02% | 131.49 | 18.98% |
| L | 1659.71 | 1400.26 | 84.37% | 259.45 | 15.63% |

## M/balanced 与旧默认生成器一致性

| 检查项 | 结果 |
|---|---|
| 种子 100–149 中完全一致的实例数 | 0/50 |
| `checkpoint_tightness` | 旧默认 `1.0` → M/balanced `0.9` |
| `service_commitment_scale_range` | 旧默认 `(0.8, 1.05)` → M/balanced `(0.95, 1.15)` |
| `job_due_date_factor_range` | 旧默认 `(0.85, 1.15)` → M/balanced `(0.7, 1.0)` |
| `group_name` | 旧默认 `framework_validation` → M/balanced `formal_M_balanced` |
