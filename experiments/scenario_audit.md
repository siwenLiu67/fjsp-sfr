# 场景生成、预测边界与检查点语义审计

> 核对范围：`generation/scenario_generator.py`、`generation/scenario_config.json`、`generation/instance_generator.py`、`generation/scale_config.py`、`core/forecast.py`、`core/models.py`、`core/online.py`、`env/ca_rule_selection.py`、`env/ca_observation.py`、`dispatch/ca_metrics.py`、`dispatch/theory_features.py`、`rl/graph.py` 与 `core/objective.py`。统计口径为中心场景 seed 25000–25049，共 50 个实例、3500 个作业。本文档只做代码与生成数据核对，未运行任何调度实验。

## 1. 作业服务数量 \(q_j\)

### 定义与生成

- `generation/scenario_config.json:24-27` 将数量范围设为 `[1, 5]`；`generation/scenario_generator.py:89-90,116-123` 对每个作业调用 `rng.randint(1, 5)`。因此 \(q_j\) 是离散均匀分布 \(U\{1,2,3,4,5\}\) 的整数抽样，不需要额外取整，理论范围为 1–5、理论均值为 3、理论方差为 2。
- 加工时间在 `generation/scenario_generator.py:103-114` 先按工序和可选机器生成，\(q_j\) 随后在第 121 行单独抽样；公式中不引用加工时间。两者共用同一个伪随机数流，但在生成模型中没有函数依赖关系。

### 中心场景 50 个验证实例的数值

| 统计量 | 数值 |
|---|---:|
| 作业数 | 3500 |
| \(q_j\) 均值 | 2.9731428571 |
| \(q_j\) 总体方差（分母 \(N\)） | 1.9895644082 |
| \(q_j\) 样本方差（分母 \(N-1\)） | 1.9901330176 |
| \(q_j\) 与“各工序的可选机器加工时间均值之和”的合并 Pearson 相关系数 | 0.0113676763 |

50 个实例分别计算相关系数后，其均值为 0.0166559203（最小 -0.2714070634，最大 0.3099384829）。合并相关几乎为 0，与“数量与加工时间独立抽样”一致。

## 2. 实体最终截止时间、作业分配与总需求

### 实体最终截止时间 \(D_r\)

`generation/scenario_generator.py:136-142` 先取实体 \(r\) 所属作业组 \(J_r\)，然后定义

\[
D_r=\max_{j\in J_r} d_j.
\]

`max(2, final_deadline)` 在当前正整数交货期生成下不改变该值。最后一个检查点时刻就是 \(D_r\) (`generation/scenario_generator.py:145-148`)。

### 实体分配

`generation/scenario_generator.py:96-99` 令实体数 \(R=\max(1,\operatorname{round}(J/8))\)。分配时先放入 `0,...,R-1`，确保每个实体至少有一个作业；余下 \(J-R\) 个实体标号用 `randrange(R)` 独立均匀抽样，再对整个标号列表洗牌。所以它是“保证非空后的均匀随机分配”，不是每个实体作业数严格相等的均分。

### 实体总需求 \(Q_r\)

`generation/scenario_generator.py:143-147` 定义

\[
Q_r=\max\!\left(1,\rho_Q\sum_{j\in J_r}q_j\right).
\]

当前 `scenario_config.json:3` 中 `commitment_ratio` \(\rho_Q=1\)，且 \(q_j\ge1\)，因此精确有 \(Q_r=\sum_{j\in J_r}q_j\)。第一个检查点的累计需求为 \(0.55Q_r\)，最后一个为 \(Q_r\)。

## 3. 作业交货期 \(d_j\)

`generation/scenario_generator.py:103-120` 对工序 \(o\) 先计算其所有可选机器加工时间的算术均值，再对路由内所有工序求和：

\[
\bar P_j=\sum_{o\in O_j}\frac{1}{|M_o|}\sum_{m\in M_o}p_{jom},
\qquad
d_j=\max\{2,\ r_j+1,\ \operatorname{round}(r_j+\mathrm{ddt}\,\bar P_j)\}.
\]

- 会取整：代码使用 Python `round`（半数情形采用 ties-to-even），最终 `due_date` 为整数。
- 中心场景 `ddt=1.0` (`scenario_config.json:30-34`)。对 seed 25000–25049 的 3500 个作业，满足
  \(d_j-r_j<\sum_o\min_{m\in M_o}p_{jom}\) 的作业数为 0，比例为 **0.0000%**。这与中心场景中“可选机器均值不小于最小值”一致；因而这批数据里没有“到达即开工、无等待且每道工序取最快机器仍必然拖期”的作业。

## 4. `forecast_provider` 与第 3.1.6 节的信息边界

### 新场景实际构造结果

`make_scenario_instance` 只转调 `make_instance` (`generation/scenario_generator.py:172-182`)，而 `make_instance` 在构造 `FJSPInstance` 时只传入作业、实体、机器和 metadata (`generation/scenario_generator.py:150-169`)，**没有构造或传入 `forecast_provider`**。`FJSPInstance.forecast_provider` 的默认值是 `None` (`core/models.py:139-149`)。因此当前中心/L9 新场景实例中：

- 预测对象：无；
- 时间窗口：无；
- 噪声来源与大小：无；
- 未释放作业的身份、工序、机器资格和加工时间：均没有用于预测，因为预测根本未启用。

环境发现 provider 为 `None` 时，会构造每个实体均为空 bucket 的 epoch-0 快照 (`env/ca_rule_selection.py:369-374`; `core/forecast.py:113-119`)。后续的预测数量因而为 0，并不是一条有噪声的实体级滚动预测。

### 旧生成器中已实现但新场景未接入的预测机制

`generation/instance_generator.py:235-256,490-530` 先生成不依赖实际作业的 latent demand：对每个实体和固定日历时间桶给出期望服务数量 \(\mu_{rb}\)。`generation/instance_generator.py:321-335` 将这些实体级聚合均值传给 `build_forecast_provider_from_latent_demand`。在 S/M/L/X 默认设置下：

- 预测对象是“实体 \(r\) ×日历时间桶 \(b\)”的聚合服务数量，单位与 \(q_j\) 和检查点累计需求相同，不是作业个数 (`core/forecast.py:18-24,122-140`)。
- `InstanceConfig` 默认 `bucket_size=10`、`forecast_horizon_buckets=6`、`forecast_update_interval=10` (`generation/instance_generator.py:98-103`)；每次快照从当前 epoch 所属的日历桶起向前覆盖 6 个桶，即 60 个时间单位 (`core/forecast.py:172-189`)。
- 对时间桶 \(b\) 和前瞻索引 \(h\)，不确定度是
  \[
  \sigma_{rbh}=s\sqrt{\mu_{rb}}(1+0.15h),
  \]
  其中默认 `error_scale` \(s=1\)；点预测是 \(\hat q_{rbh}=\max(0,\mu_{rb}+z_{rb}\sigma_{rbh})\)，\(z_{rb}\sim N(0,1)\)。同一实体/日历桶在整个 episode 复用同一个 \(z_{rb}\)，随时间滚动时不重新抽方向 (`core/forecast.py:163-187`)。
- provider 的唯一需求输入是 `(entity_id, bucket_start, mean_quantity)` (`core/forecast.py:122-161`)。旧生成器用不同的 RNG 流独立生成实际到达和预测噪声 (`generation/instance_generator.py:176-188,235-256,325-335`)，不向 provider 传入任何未释放作业对象、作业 ID、工序、机器资格或加工时间。

### 与第 3.1.6 节“仅限实体级滚动聚合数量预测及不确定性”逐项对照

| 第 3.1.6 节要求 | 旧 `instance_generator`/`forecast.py` 机制 | 当前 `make_scenario_instance` |
|---|---|---|
| 实体级 | 符合：索引是 entity ID | 未接入预测，无可供使用的实体级预测 |
| 滚动更新 | 符合：按 `update_interval` 选取时间快照 | 不符合功能性描述：始终是 epoch-0 空快照 |
| 只给聚合数量 | 符合：给出每个时间桶的数量均值 | 无预测数量 |
| 给出不确定性 | 符合：每个 bucket 有 `sigma` | 无不确定性（空 bucket） |
| 不使用未释放作业的具体信息 | 符合：只使用 latent 实体级 bucket 均值 | 没有泄漏，但原因是 provider 未启用，不能由此认定已实现第 3.1.6 节的预测功能 |

**结论：** `core/forecast.py` 与旧生成器所实现的接口符合第 3.1.6 节的信息边界；但当前新中心/L9 场景没有将该 provider 挂接到实例，所以对这批正式场景而言，第 3.1.6 节所述“有滚动聚合预测及不确定性”并未实际成立。

## 5. 1–3 个检查点的支持情况

### 环境、图构造与特征计算

**这些运行时模块本身支持每个实体 1–3 个检查点，并不假定恰好 2 个。**

- 数据模型只要求 checkpoints 非空、时刻严格递增、累计需求非递减，没有固定长度 (`core/models.py:108-127`)。
- 环境 reset 对所有实体的所有检查点动态入队 (`env/ca_rule_selection.py:203-217`)；目标值结算也用 `enumerate(entity.checkpoints)` (`core/objective.py:37-52`)。
- 指标代码用 `range(len(entity.checkpoints))` 和 `enumerate` 构建每个 `(entity_id, checkpoint_index)` (`dispatch/ca_metrics.py:397-468`)；候选作业的多检查点价值也遍历整个 tuple (`dispatch/ca_metrics.py:242-265`)。
- 观测关系张量的 checkpoint 轴长度由 `len(metrics.checkpoints)` 动态决定 (`env/ca_observation.py:305-361`)；理论特征取当前实体所有未到时检查点，再取最早项或遍历可达项 (`env/ca_observation.py:390-439`; `dispatch/theory_features.py:118-125,167-205`)。
- 异构图从当前指标字典动态收集活跃检查点，依实际数量建立节点、索引与作业—检查点边 (`rl/graph.py:103-112,143-150,171-238,271-282`)。

### 发现的“恰好 2 个检查点”或阻止 1 个检查点的位置

| 位置 | 假定/影响 |
|---|---|
| `generation/scenario_generator.py:145-148` | 新 center/L9 生成器直接构造两个 `Checkpoint`，因而它只会生成 2 个。 |
| `generation/instance_generator.py:35-48,51-56` | `CheckpointProfile` 显式只允许 profile 长度 2 或 3，且内置 profile 也只有 2/3 两类；因而虽然 `InstanceConfig.checkpoints_per_entity` 在第 123–124 行声明允许 1–3，当前没有可生成 1 个检查点的 profile。 |
| `generation/scale_config.py:28-36` | S/M/L/X 配置强制 `checkpoints_per_entity=(2,2)` 并且只选 `balanced_2`/`early_pressure_2`。 |
| `experiments/validate_framework.py:51-60` | 旧验证实验的手工 `InstanceConfig` 强制 2 个检查点。 |
| `experiments/pre_formal_check.py:91-103` | 旧 M-balanced 对照配置强制 2 个检查点。 |
| `tests/test_scenario_generator.py:43-49` | 测试明确断言新场景每实体恰好 2 个检查点。 |
| `tests/test_scenario_generator.py:60-68` | 旧生成器可复现性 fixture 使用 `(2,2)`；这是测试数据假定，不是环境运行时限制。 |

因此，**消费检查点的环境、目标、特征和图支持 1–3 个；当前实例生成入口并没有覆盖完整的 1–3 范围**：新场景只生成 2，通用旧 profile 生成器可生成 2 或 3，但不能生成 1。

## 6. 旧 S/M/L/X 生成器的对应设置

### \(q_j\)

`generation/instance_generator.py:176-188` 将 structure、latent、actual、commitment、due 和 forecast 分成不同 RNG 流。`generation/instance_generator.py:238-256` 先按 latent 实体期望需求加权分配实体，然后从 `quantity_range` 离散均匀抽取 \(q_j\)。S/M/L/X 没有覆盖默认范围，所以同样是整数 1–5 (`generation/instance_generator.py:89`)。它与加工时间的 structure RNG 流明确分离，比新场景的“同一 RNG 串行独立抽样”更强。

### \(D_r\) 与作业交货期

旧生成器的实体最后检查点不取该实体作业交货期的最大值。它先仅用配置分布的期望最小加工量计算

\[
B=\max\!\left(3,\operatorname{round}\!\left(
\frac{J\,E[|O_j|],E[\min p_o]}{M}\times\text{checkpoint\_tightness}
\right)\right)
\]

(`generation/instance_generator.py:390-411`)，再按实体序号设

\[
D_r=\max\!\left(K_r,\operatorname{round}\!\left(B\left[0.9+0.2\frac{r}{\max(1,R-1)}\right]\right)\right)
\]

(`generation/instance_generator.py:272-278`)。随后才以该 \(D_r\) 生成作业交货期：
\(d_j=\max(r_j+P_j^{\min},\operatorname{round}(D_r U_j))\)，其中 \(U_j\) 来自难度配置的 `job_due_date_factor_range` (`generation/instance_generator.py:299-318`)。

### 需求、权重与检查点数

- 旧生成器的最终服务承诺是 \(\max(1,\bar Q_r S_r)\)：\(\bar Q_r\) 是 latent 实体期望数量，\(S_r\) 是 `service_commitment_scale_range` 内的均匀随机数 (`generation/instance_generator.py:279-286`)。它不是该实体已实现作业 \(q_j\) 的和；这是新场景没有沿用的部分。
- 旧内置 2 检查点 profile 的目标比例/权重为 `balanced_2`: `(0.45,1.00)` / `(1,2)`，`early_pressure_2`: `(0.60,1.00)` / `(1,3)` (`generation/instance_generator.py:51-55`)；实际权重再乘 `checkpoint_weight_scale` (`generation/instance_generator.py:287-295`)。当前 balanced/easy/hard 难度该系数默认为 1，service_heavy 为 2.5 (`generation/scale_config.json`)。
- S/M/L/X 入口在 `generation/scale_config.py:28-36` 固定 `checkpoints_per_entity=(2,2)` 并只使用两个 2-checkpoint profile，因此旧正式规模实例实际上也都是每实体 2 个检查点。

### 新场景对旧设置的沿用结论

新场景沿用了相同的 `Job`/`Operation`/`Checkpoint`/`ServiceEntity` 数据类、整数数量 1–5 的默认范围，以及“每实体 2 个检查点”这一当前正式配置。它没有沿用旧生成器的 latent-demand 分配、多 RNG 流、外生检查点日历、latent 承诺量、profile 随机选择和 forecast provider；新场景改为由已生成作业的 \(d_j\) 与 \(q_j\) 直接定义 \(D_r\) 和 \(Q_r\)，并固定两个检查点权重模板。
