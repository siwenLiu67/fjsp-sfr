# Paper A 代码实现规格

> 给 Claude Code 在 VS Code 中使用的参考文档。
> 请将此文件放入项目根目录，Claude Code 会自动读取。

---

## 1. 现有代码审查

### 1.1 模块映射（现有代码 → 论文）

| 论文概念 | 代码位置 | 状态 |
|---------|---------|------|
| 问题模型 (Sec 3): Job, Operation, Entity, Checkpoint | `core/models.py` | ✅ 完整 |
| 仿真状态: ScheduleState, EventQueue | `core/state.py`, `core/events.py` | ✅ 完整 |
| 信息边界: OnlineProblemView | `core/online.py` | ✅ 完整 |
| 预测: ForecastSnapshot, ForecastProvider | `core/forecast.py` | ✅ 完整 |
| 目标函数: F = αTT + βWSF | `core/objective.py` | ✅ 完整 |
| 理论量 (Sec 4): G_rkt, C^LB_j, W_j, Q^fut, Λ, Δ^fc, t^cov, S^fc | `dispatch/ca_metrics.py` | ✅ 基本完整 |
| 容量压力 (Sec 4): workload, capacity | `dispatch/ca_metrics.py` 中 `compute_optimistic_checkpoint_workload_capacity` | ✅ 有计算但未显式导出 Δ^cap |
| 8条固定规则 R1–R8 | `dispatch/ca_rules.py` | ✅ 完整 |
| 仿真环境（事件驱动） | `rl/ca_rule_selection.py` CAPPORuleSelectionEnv | ✅ 完整 |
| 奖励: r_n = -(F^cum 增量)/F_ref | `rl/ca_reward.py` ca_transition_reward | ✅ 逻辑正确，F_ref 归一化在训练端做 |
| 观测特征: shop/candidate/checkpoint tokens | `rl/ca_observation.py` | ✅ 完整 |
| 理论因子向量 (Table 5-2 的子集) | `rl/ca_observation.py` build_theory_factor_values | ⚠️ 当前5维，论文要求11维 |
| 符号规则引擎 H=⟨Target,Cond,Filter,Priority⟩ | **不存在** | ❌ 需要新建 (Module ③) |
| LLM 规则生成 (Algorithm 1) | **不存在** | ❌ 需要新建 (Module ④) |
| HGT 异构图网络 | **不存在** | ❌ 需要新建 (Module ⑤) |
| PPO 训练循环 | **不存在** | ❌ 需要新建 (Module ⑤) |

### 1.2 现有代码的正确性确认

**仿真器 (CAPPORuleSelectionEnv):**
- 事件处理顺序正确：同一时刻先处理所有事件（arrival/completion/checkpoint），再尝试派工
- work-conserving: `_advance_to_decision` 在同一时刻连续派工直到无空闲机器或无候选
- 非延迟: `next_trigger_machine` 按 machine_id 确定性排序
- checkpoint 结算是隐式的：`assessed_weighted_shortfall` 通过 `checkpoint.time > state.current_time` 判断
- `step_principle(principle_index, principle)` 已经支持可插拔规则！**这是 Module ③ 的接入点**

**理论量 (ca_metrics.py):**
- `residual_checkpoint_gap` = G_rkt ✓
- `optimistic_candidate_completion` = C^LB_{ju}(τ)，固定触发机器 ✓
- `visible_optimistic_completion` = C*_j(τ)，全局最优 ✓
- `multi_checkpoint_service_value` = W_j（论文已改名，代码匹配）✓
- `compute_future_required_quantity` = Q^fut_rkt ✓
- `compute_arrival_forecast_summary` = (d̂, σ, Λ^-) ✓
- `compute_arrival_insufficiency` = Δ^fc ✓
- `compute_release_threshold_time` = t^cov ✓
- `compute_release_slack` = S^fc ✓
- `compute_optimistic_checkpoint_workload_capacity` 返回 (required_workload, capacity) ✓

### 1.3 需要补充的理论量

以下量在 `ca_metrics.py` 中可以从现有计算推导，但需要显式导出为 `CheckpointMetrics` 字段或独立函数：

```python
# Δ^cap: 容量压力比  (Proposition 1 / Corollary 1)
# 已有 workload 和 capacity，只需:
delta_cap = workload / (capacity + epsilon)  # 或 max(0, workload - capacity)

# ι^fc: 预测不足标记
iota_fc = arrival_insufficiency > 0.0

# ι^cap: 容量压力标记
iota_cap = delta_cap > 1.0  # 工作量超过产能
```

### 1.4 关键发现：step_principle 已经是 Module ③ 的入口

现有环境的 `step_principle(principle_index, principle)` 方法：
```python
# principle 对象需要:
#   .name: str
#   .expression: str
#   .evaluate_batch(factor_rows: tuple[tuple[float,...], ...]) -> list[float]
#
# factor_rows 是 build_theory_factor_values 的输出
# 每行对应一个 candidate，返回每个 candidate 的分数
# 选分数最高的 candidate 派工
```

**这意味着 Module ③ 不需要改仿真器，只需要：**
1. 实现 `SymbolicRule` 类满足 `principle` 接口
2. 扩展 `build_theory_factor_values` 从 5 维到论文要求的特征集
3. 提供安全的表达式求值器

---

## 2. Module ③: 符号规则引擎

### 2.1 目标

实现论文 Sec 5.2 的规则表示 H = ⟨Target, Cond, Filter, Priority⟩，使 LLM 生成的规则字符串可以被安全求值，并通过现有 `step_principle` 接口接入仿真器。

### 2.2 文件结构

```
fjsp_sfr/
  dispatch/
    symbolic_rule.py      # SymbolicRule 类
    rule_library.py       # RuleLibrary 管理
    theory_features.py    # 扩展的理论特征计算（从 ca_observation.py 的 build_theory_factor_values 扩展）
    safe_eval.py          # 安全表达式求值
  dispatch/
    ca_rules.py           # 保留，R1-R8 可以包装成 SymbolicRule
```

### 2.3 接口定义

#### 2.3.1 safe_eval.py

```python
"""安全的数学表达式求值器。

只允许：算术运算 (+, -, *, /, **)、比较 (<, >, <=, >=)、
min, max, abs, math.tanh, math.log, math.exp, 条件 (if-else)。
不允许：import, exec, eval, __builtins__, attribute access, function calls beyond whitelist。
"""

def compile_expression(expr: str, variable_names: tuple[str, ...]) -> CompiledExpression:
    """将表达式字符串编译为可安全执行的对象。
    
    Args:
        expr: 数学表达式，例如 "0.5 * gap_pressure + 0.3 * tardiness_exposure"
        variable_names: 允许的变量名，例如 THEORY_FEATURE_NAMES
    
    Returns:
        CompiledExpression 对象
    
    Raises:
        ExpressionSyntaxError: 如果表达式包含不安全的构造
    """

class CompiledExpression:
    """编译后的安全表达式。"""
    
    source: str  # 原始表达式
    
    def evaluate(self, variables: dict[str, float]) -> float:
        """用给定变量值求值。"""
    
    def evaluate_batch(self, rows: tuple[tuple[float, ...], ...],
                       variable_names: tuple[str, ...]) -> list[float]:
        """对多行变量值批量求值。每行对应一个 candidate。"""
```

实现建议：用 `ast.parse` + 白名单 visitor 验证安全性，然后 `compile` 为 Python code object。

#### 2.3.2 symbolic_rule.py

```python
"""论文 Sec 5.2: H = ⟨Target, Cond, Filter, Priority⟩"""

from dataclasses import dataclass
from enum import IntEnum

class RuleType(IntEnum):
    """Table 5-3 规则分类"""
    TYPE_I = 1    # 纯优先级
    TYPE_II = 2   # 条件优先级
    TYPE_III = 3  # 加权组合 Σλ_h Φ̂_h
    TYPE_IV = 4   # 过滤 + 优先级

@dataclass(frozen=True)
class SymbolicRule:
    """一条可求值的符号规则。"""
    
    name: str                    # 规则名称，如 "R1_projected_tardiness" 或 LLM 生成的名称
    expression: str              # 人可读的完整表达式描述
    rule_type: RuleType
    priority_expr: str           # Priority 表达式，例如 "gap_pressure * 0.5 + tardiness_exposure * 0.3"
    cond_expr: str | None = None # Cond 表达式（Type II/IV），例如 "any_gap_positive"
    filter_expr: str | None = None  # Filter 表达式（Type IV），例如 "reachable_checkpoints > 0"
    
    # Type III 专用
    weights: tuple[float, ...] | None = None  # λ_h
    basis_names: tuple[str, ...] | None = None  # Φ̂_h 对应的特征名
    
    def evaluate_batch(self, factor_rows: tuple[tuple[float, ...], ...]) -> list[float]:
        """满足 step_principle 的 principle 接口。
        
        Args:
            factor_rows: 每行是一个 candidate 的理论特征向量
                         长度 = len(THEORY_FEATURE_NAMES)
        
        Returns:
            每个 candidate 的分数，分数越高越优先
        """
    
    def is_applicable(self, 
                      theory_summary: dict[str, float]) -> bool:
        """检查 Cond 是否满足（Type II/IV）。
        
        Args:
            theory_summary: 决策点级别的聚合特征
                           （例如 any_gap_positive, any_forecast_shortage 等）
        
        Returns:
            True 如果规则在当前决策点可用
        """
    
    def filter_candidates(self,
                          factor_rows: tuple[tuple[float, ...], ...]) -> tuple[bool, ...]:
        """返回每个 candidate 是否通过 Filter（Type IV）。
        
        Args:
            factor_rows: 同 evaluate_batch
        
        Returns:
            每个 candidate 的 mask，True 表示通过
        """

def wrap_fixed_rule(action: RuleAction) -> SymbolicRule:
    """将现有 R1-R8 包装为 SymbolicRule。
    
    用于基线对比和向后兼容。
    这些规则的 evaluate_batch 直接调用现有 ca_rules.py 逻辑。
    """

# 默认排序规则（论文：due-date-slack ascending）
DEFAULT_RULE = SymbolicRule(
    name="default_due_date_slack",
    expression="due_date_slack ascending",
    rule_type=RuleType.TYPE_I,
    priority_expr="-due_date_slack",  # 负号因为 max 选最优
)
```

#### 2.3.3 theory_features.py

```python
"""扩展的理论特征集，对应论文 Table 5-2。

现有 build_theory_factor_values 产出 5 维:
  [gap_pressure, forecast_deficit, checkpoint_slack_urgency, 
   multi_checkpoint_value, tardiness_exposure]

扩展到论文要求的完整特征集。所有值归一化到 [0, 1]。
"""

# 完整特征名列表 — 需要与论文 Table 5-2 对齐
# 下面是基于论文 Sec 4 接口表 (Table 4-1) 的候选特征
THEORY_FEATURE_NAMES = (
    # 来自现有 5 维
    "gap_pressure",              # ν_rk * G^+_rkt / Cap_rk(t) 的归一化
    "forecast_deficit",          # Δ^fc / G_rkt 的归一化
    "checkpoint_slack_urgency",  # C^LB_j / (τ_rk - τ) 
    "multi_checkpoint_value",    # W_j 的归一化
    "tardiness_exposure",        # tanh((C^LB_j - d_j) / horizon)
    
    # 新增
    "due_date_slack",            # S^d_j = (d_j - C^LB_j) / horizon，归一化
    "capacity_pressure",         # Δ^cap = workload / capacity，归一化
    "forecast_flag",             # ι^fc ∈ {0, 1}
    "capacity_flag",             # ι^cap ∈ {0, 1}
    "quantity_contribution",     # χ^Q_jrk = q_j / G_rkt，归一化
    "processing_ratio",          # p_{jℓm} / Σ remaining_processing，归一化
)

def build_extended_theory_features(
    problem: SchedulingProblem,
    state: ScheduleState,
    metrics: CADecisionMetrics,
    *,
    time_scale: float | None = None,
) -> tuple[tuple[float, ...], ...]:
    """构建扩展理论特征矩阵。
    
    Returns:
        tuple of tuples, shape = (num_candidates, len(THEORY_FEATURE_NAMES))
        每个值在 [0, 1] 内，与现有 build_theory_factor_values 的约定一致。
    """
```

**重要约束：** THEORY_FEATURE_NAMES 的具体内容以论文 Table 5-2 为准。上面列出的是从 Sec 4 推导的合理候选。实现时请核对论文原文，确保特征名和计算公式一一对应。

#### 2.3.4 rule_library.py

```python
"""规则库管理。对应论文 Sec 5.2.4 贪心边际增益选择。"""

@dataclass
class RuleLibrary:
    """存储和管理符号规则集合。"""
    
    rules: tuple[SymbolicRule, ...]
    
    @classmethod
    def from_json(cls, path: str) -> "RuleLibrary":
        """从 JSON 文件加载规则库。"""
    
    def to_json(self, path: str) -> None:
        """保存规则库到 JSON 文件。"""
    
    def action_size(self) -> int:
        """PPO 动作空间大小 = 规则数量。"""
        return len(self.rules)
    
    def select_and_dispatch(
        self,
        rule_index: int,
        factor_rows: tuple[tuple[float, ...], ...],
    ) -> int:
        """选择规则并返回最佳 candidate index。
        
        Args:
            rule_index: PPO 选择的规则索引
            factor_rows: 理论特征矩阵
        
        Returns:
            最佳 candidate 的 index（在排序后的 candidates 中）
        """

def evaluate_rule_objective(
    rule: SymbolicRule,
    instances: list[FJSPInstance],
    *,
    num_episodes: int = 10,
) -> float:
    """f_H: 规则在多个实例上的标准化目标值。
    
    对应论文公式：f_H = mean(F_H / F_ref)
    其中 F_ref 是某个基线（如 default rule）的目标值。
    """

def compute_behavior_similarity(
    rule_a: SymbolicRule,
    rule_b: SymbolicRule,
    instances: list[FJSPInstance],
) -> float:
    """Sim(H_a, H_b): 两条规则的行为相似度。
    
    对应论文定义：在相同决策点上选择相同 candidate 的比例。
    """

def greedy_library_selection(
    candidates: list[SymbolicRule],
    budget: int,
    instances: list[FJSPInstance],
    *,
    similarity_threshold: float = 0.8,
) -> RuleLibrary:
    """贪心边际增益选择。
    
    论文 Sec 5.2.4 算法：
    1. 初始化 L = {default_rule}
    2. 每步选择使 marginal gain 最大的规则加入 L
    3. marginal gain = f_H * (1 - max_sim(H, L))
    4. 直到 |L| = budget
    """
```

### 2.4 集成方式

Module ③ 通过现有 `step_principle` 接入仿真器，**不需要修改 `CAPPORuleSelectionEnv`**：

```python
# 使用方式
env = CAPPORuleSelectionEnv(instance)
obs = env.reset()

library = RuleLibrary.from_json("rule_library.json")
rule_index = ppo_agent.select(obs)  # Module ⑤ 选规则
rule = library.rules[rule_index]

result = env.step_principle(rule_index, rule)  # 现有接口
```

但需要一个小改动：`build_theory_factor_values` 需要被替换为 `build_extended_theory_features`。

修改位置: `ca_rule_selection.py` 中 `step_principle` 方法第 276 行：
```python
# 现有：
factor_rows = build_theory_factor_values(problem, self._state, self.current_metrics, time_scale=self.time_scale)

# 改为：
factor_rows = build_extended_theory_features(problem, self._state, self.current_metrics, time_scale=self.time_scale)
```

### 2.5 测试用例

```python
# test_symbolic_rule.py

def test_type_i_due_date_slack():
    """Type I 规则：纯优先级，选 due-date slack 最小的 candidate。"""
    rule = SymbolicRule(
        name="edd", expression="earliest due date slack",
        rule_type=RuleType.TYPE_I,
        priority_expr="-due_date_slack",
    )
    # 3 个 candidate, due_date_slack 分别是 0.8, 0.2, 0.5
    rows = ((0.1, 0.0, 0.3, 0.5, 0.1, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.1, 0.0, 0.3, 0.5, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.1, 0.0, 0.3, 0.5, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    scores = rule.evaluate_batch(rows)
    assert scores.index(max(scores)) == 1  # slack=0.2 的优先级最高（-0.2最大）

def test_type_iii_weighted_combination():
    """Type III 规则：加权组合。"""
    rule = SymbolicRule(
        name="combo", expression="0.6*gap_pressure + 0.4*tardiness_exposure",
        rule_type=RuleType.TYPE_III,
        priority_expr="0.6 * gap_pressure + 0.4 * tardiness_exposure",
        weights=(0.6, 0.4),
        basis_names=("gap_pressure", "tardiness_exposure"),
    )
    rows = ((0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 0.6*0.9+0.4*0.1=0.58
            (0.1, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))  # 0.6*0.1+0.4*0.9=0.42
    scores = rule.evaluate_batch(rows)
    assert scores[0] > scores[1]

def test_safe_eval_rejects_import():
    """表达式求值器拒绝危险构造。"""
    with pytest.raises(ExpressionSyntaxError):
        compile_expression("__import__('os').system('rm -rf /')", ("x",))

def test_safe_eval_rejects_attribute_access():
    with pytest.raises(ExpressionSyntaxError):
        compile_expression("x.__class__", ("x",))

def test_existing_rules_as_symbolic():
    """现有 R1-R8 可以包装为 SymbolicRule 且结果一致。"""
    instance = make_test_instance()  # 需要一个测试实例构造函数
    env = CAPPORuleSelectionEnv(instance)
    obs = env.reset()
    
    for action in RuleAction:
        wrapped = wrap_fixed_rule(action)
        # 用 step vs step_principle 对比
        env1 = env.clone()
        env2 = env.clone()
        
        mask = rule_applicability_mask(env1._problem_view(), env1.current_metrics)
        if mask[int(action)]:
            r1 = env1.step(action)
            r2 = env2.step_principle(int(action), wrapped)
            assert r1.decision.job_id == r2.decision.job_id

def test_rule_serialization():
    """规则可以序列化/反序列化。"""
    rule = SymbolicRule(name="test", expression="test rule",
                        rule_type=RuleType.TYPE_I, priority_expr="gap_pressure")
    library = RuleLibrary(rules=(rule,))
    library.to_json("/tmp/test_lib.json")
    loaded = RuleLibrary.from_json("/tmp/test_lib.json")
    assert loaded.rules[0].name == "test"
    assert loaded.rules[0].priority_expr == "gap_pressure"
```

---

## 3. Module ④: LLM 规则生成

### 3.1 目标

实现论文 Algorithm 1：迭代调用 LLM 生成符号规则，评估后构建规则库。

### 3.2 文件结构

```
fjsp_sfr/
  llm/
    __init__.py
    prompt_template.py    # LLM prompt 构造
    rule_parser.py        # 解析 LLM 输出为 SymbolicRule
    generator.py          # Algorithm 1 主循环
    evaluator.py          # f_H 和 Sim 计算
```

### 3.3 接口定义

#### 3.3.1 prompt_template.py

```python
def build_generation_prompt(
    theory_features: tuple[str, ...],  # THEORY_FEATURE_NAMES
    existing_rules: list[SymbolicRule],  # 已有规则（供 LLM 参考/避免重复）
    iteration: int,  # 第几轮迭代
    feedback: str | None = None,  # 上一轮生成规则的评估反馈
) -> str:
    """构造 LLM 生成规则的 prompt。
    
    prompt 包含:
    1. 问题描述（FJSP + 累积服务检查点）
    2. 理论量名称和含义
    3. 规则格式 H = ⟨Target, Cond, Filter, Priority⟩
    4. 已有规则（避免重复）
    5. 评估反馈（如有）
    6. 要求生成 N 条新规则
    
    Returns:
        完整 prompt 字符串
    """
```

#### 3.3.2 rule_parser.py

```python
def parse_llm_output(
    text: str,
    feature_names: tuple[str, ...],
) -> list[SymbolicRule]:
    """解析 LLM 输出文本为 SymbolicRule 列表。
    
    LLM 输出格式约定（在 prompt 中指定）:
    ```
    RULE: <name>
    TYPE: I | II | III | IV
    COND: <expression or NONE>
    FILTER: <expression or NONE>
    PRIORITY: <expression>
    ```
    
    对每条规则:
    1. 提取各字段
    2. 用 compile_expression 验证表达式安全性
    3. 检查表达式中的变量名都在 feature_names 中
    4. 构造 SymbolicRule
    
    Returns:
        成功解析的规则列表（跳过无法解析的）
    """
```

#### 3.3.3 generator.py (Algorithm 1)

```python
def iterative_rule_generation(
    *,
    llm_client,  # LLM API 客户端 (如 anthropic.Client)
    instances: list[FJSPInstance],  # 评估用实例
    max_iterations: int = 5,
    rules_per_iteration: int = 10,
    library_budget: int = 20,  # 最终规则库大小
    similarity_threshold: float = 0.8,
) -> RuleLibrary:
    """Algorithm 1: 迭代 LLM 规则生成。
    
    流程:
    1. 初始化候选池 = R1-R8 包装后的符号规则
    2. for i in range(max_iterations):
       a. 构造 prompt（含已有规则和上轮反馈）
       b. 调用 LLM 生成 rules_per_iteration 条规则
       c. 解析、验证
       d. 用 evaluate_rule_objective 评估每条规则的 f_H
       e. 过滤掉 f_H 差于基线的规则
       f. 构造下轮反馈
       g. 加入候选池
    3. 从候选池用 greedy_library_selection 选出 library_budget 条
    4. 返回 RuleLibrary
    """
```

### 3.4 测试用例

```python
def test_parse_valid_rule():
    text = """
    RULE: urgency_combo
    TYPE: III
    COND: NONE
    FILTER: NONE
    PRIORITY: 0.5 * gap_pressure + 0.3 * tardiness_exposure + 0.2 * forecast_deficit
    """
    rules = parse_llm_output(text, THEORY_FEATURE_NAMES)
    assert len(rules) == 1
    assert rules[0].rule_type == RuleType.TYPE_III
    assert rules[0].cond_expr is None

def test_parse_rejects_unknown_variable():
    text = """
    RULE: bad_rule
    TYPE: I
    COND: NONE
    FILTER: NONE
    PRIORITY: nonexistent_variable * 2
    """
    rules = parse_llm_output(text, THEORY_FEATURE_NAMES)
    assert len(rules) == 0  # 应被过滤掉

def test_evaluate_rule_objective():
    """f_H 在 trivial 实例上可计算且有限。"""
    rule = SymbolicRule(name="test", expression="test",
                        rule_type=RuleType.TYPE_I,
                        priority_expr="-tardiness_exposure")
    instances = [make_test_instance()]
    f_h = evaluate_rule_objective(rule, instances, num_episodes=1)
    assert 0.0 < f_h < float('inf')

def test_behavior_similarity_identical():
    """同一规则与自身的相似度为 1.0。"""
    rule = SymbolicRule(name="a", expression="a",
                        rule_type=RuleType.TYPE_I,
                        priority_expr="-tardiness_exposure")
    instances = [make_test_instance()]
    sim = compute_behavior_similarity(rule, rule, instances)
    assert sim == 1.0
```

---

## 4. Module ⑤: HGT-PPO

### 4.1 目标

实现论文 Sec 5.1 (HGT 异构图编码器) 和 Sec 5.3 (PPO 训练)。

### 4.2 文件结构

```
fjsp_sfr/
  rl/
    graph.py              # 异构图 G_n 构建
    hgt.py                # HGT 编码器网络
    ppo_agent.py          # PPO 策略 + 价值网络
    ppo_trainer.py         # 训练循环
    rollout_buffer.py     # 轨迹存储
```

### 4.3 接口定义

#### 4.3.1 graph.py

```python
"""论文 Sec 5.1: 异构图 G_n = (V, E)

节点类型:
- J: 候选工件节点 (candidate jobs)
- O: 工序节点 (operations, 当前待处理工序)
- M: 机器节点
- C: 检查点节点 (active checkpoints, G_rkt > 0)

边类型 (带数值属性):
- J → O: 工件到其当前工序
- O → M: 工序到可选机器 (边属性: processing_time)
- J → C: 工件到其关联实体的检查点 (边属性: quantity_contribution, completion_margin)
- M → M: 自连接 (或全连接)

修改点 (相对 Hu 2020):
1. 边带数值属性 (不只是类型)
2. 用 ReLU 替换 softmax attention
"""

import torch
from torch_geometric.data import HeteroData

def build_hetero_graph(
    observation: CADecisionObservation,
    theory_features: tuple[tuple[float, ...], ...],
) -> HeteroData:
    """从观测构建 PyG HeteroData。
    
    Node features:
    - 'job': candidate_tokens (6-dim) + theory_features (11-dim) = 17-dim
    - 'operation': 从 candidate 的 processing 信息提取
    - 'machine': [busy_ratio, ready_delay] (从 shop vector)
    - 'checkpoint': checkpoint_tokens (8-dim)
    
    Edge features (数值属性):
    - ('job', 'has_op', 'operation'): []
    - ('operation', 'on', 'machine'): [processing_time_ratio]
    - ('job', 'serves', 'checkpoint'): candidate_checkpoint_relations (2-dim)
    
    Edge mask:
    - ('job', 'serves', 'checkpoint'): candidate_checkpoint_mask
    
    Global features:
    - shop vector (7-dim)
    """
```

#### 4.3.2 hgt.py

```python
"""论文 Sec 5.1: 修改版 HGT (Heterogeneous Graph Transformer)

与 Hu 2020 的区别:
1. 边带数值属性: message = W_msg @ [src_feat || edge_feat]
2. attention 用 ReLU 而非 softmax (稀疏性更好)
"""

import torch
import torch.nn as nn

class HGTLayer(nn.Module):
    """单层 HGT，支持数值边属性和 ReLU attention。"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        node_types: tuple[str, ...],   # ('job', 'operation', 'machine', 'checkpoint')
        edge_types: tuple[tuple[str, str, str], ...],  # 三元组
        edge_feature_dims: dict[tuple[str, str, str], int],  # 每种边的特征维度
    ): ...
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict) -> dict[str, torch.Tensor]:
        """
        Args:
            x_dict: {node_type: (N, hidden_dim)}
            edge_index_dict: {edge_type: (2, E)}
            edge_attr_dict: {edge_type: (E, edge_feat_dim)}
        
        Returns:
            {node_type: (N, hidden_dim)}
        """


class HGTEncoder(nn.Module):
    """多层 HGT + 全局池化。"""
    
    def __init__(
        self,
        node_input_dims: dict[str, int],  # 每种节点的输入特征维度
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        node_types: tuple[str, ...] = ('job', 'operation', 'machine', 'checkpoint'),
        edge_types: tuple[tuple[str, str, str], ...] = ...,
        edge_feature_dims: dict[tuple[str, str, str], int] = ...,
    ): ...
    
    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Args:
            data: PyG HeteroData (来自 build_hetero_graph)
        
        Returns:
            z_n: (hidden_dim,) 全局状态表示
            论文: z_n = MLP(pooled || x_glob)
            pooled = mean_pool(job_embeddings) || mean_pool(checkpoint_embeddings)
            x_glob = shop_features
        """
```

#### 4.3.3 ppo_agent.py

```python
"""论文 Sec 5.3: PPO 策略"""

class HGTPPOAgent(nn.Module):
    """HGT 编码器 + 策略头 + 价值头。"""
    
    def __init__(
        self,
        encoder: HGTEncoder,
        action_size: int,  # = RuleLibrary.action_size()
        hidden_dim: int = 64,
    ): ...
    
    def forward(self, data: HeteroData, action_mask: torch.Tensor):
        """
        Returns:
            policy_logits: (action_size,) 被 mask 后的 log-prob
            value: scalar V(s)
        """
    
    def select_action(self, data: HeteroData, action_mask: torch.Tensor):
        """采样一个动作。
        
        Returns:
            action: int (规则索引)
            log_prob: float
            value: float
        """
```

#### 4.3.4 ppo_trainer.py

```python
"""论文 Sec 5.3: PPO 训练循环

损失函数 (论文一个公式):
L = L_clip + c_v * L_value + c_e * L_entropy

GAE cited, discount γ = 1 (论文 Sec 5.3)
"""

class PPOTrainer:
    def __init__(
        self,
        agent: HGTPPOAgent,
        library: RuleLibrary,
        instances: list[FJSPInstance],  # 训练实例
        *,
        lr: float = 3e-4,
        gamma: float = 1.0,       # 论文: discount = 1
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs_per_update: int = 4,
        batch_size: int = 64,
        f_ref: float | None = None,  # 奖励归一化基线
    ): ...
    
    def collect_rollout(self, num_episodes: int) -> RolloutBuffer:
        """收集 num_episodes 条轨迹。
        
        对每个 episode:
        1. env.reset()
        2. 每个决策点:
           a. 构建图: build_hetero_graph(obs, theory_features)
           b. agent.select_action(graph, action_mask) → rule_index
           c. env.step_principle(rule_index, library.rules[rule_index])
           d. reward = result.reward / f_ref
           e. 存入 buffer
        3. 直到 terminated
        """
    
    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """PPO 参数更新。
        
        Returns:
            训练统计: {policy_loss, value_loss, entropy, approx_kl}
        """
    
    def train(self, total_episodes: int, log_interval: int = 100):
        """完整训练流程。"""
```

### 4.4 测试用例

```python
def test_graph_construction_shapes():
    """图节点数和边数与观测一致。"""
    instance = make_test_instance()
    env = CAPPORuleSelectionEnv(instance)
    obs = env.reset()
    features = build_extended_theory_features(...)
    graph = build_hetero_graph(obs, features)
    
    assert graph['job'].x.shape[0] == len(obs.candidate_job_ids)
    assert graph['checkpoint'].x.shape[0] == len(obs.checkpoint_keys)

def test_hgt_forward_pass():
    """HGT 前向传播输出维度正确。"""
    encoder = HGTEncoder(node_input_dims={...}, hidden_dim=64)
    graph = make_test_graph()
    z = encoder(graph)
    assert z.shape == (64,)

def test_ppo_agent_action_mask():
    """不可用规则不会被选中。"""
    agent = HGTPPOAgent(encoder, action_size=5)
    mask = torch.tensor([True, False, True, False, True])
    graph = make_test_graph()
    for _ in range(100):
        action, _, _ = agent.select_action(graph, mask)
        assert mask[action]

def test_reward_normalization():
    """奖励正确归一化。"""
    # r_n = -(F^cum(t_{n+1}) - F^cum(t_n)) / F_ref
    # 现有 ca_transition_reward 返回 -(α*ΔTT + β*ΔWSF)
    # 训练时除以 F_ref
    instance = make_test_instance()
    env = CAPPORuleSelectionEnv(instance)
    obs = env.reset()
    result = env.step(RuleAction.PROJECTED_TARDINESS)
    raw_reward = result.reward
    f_ref = 100.0  # 基线目标值
    normalized = raw_reward / f_ref
    assert isinstance(normalized, float)

def test_gae_discount_one():
    """GAE 在 γ=1 时正确计算。"""
    rewards = [1.0, 2.0, 3.0]
    values = [5.0, 4.0, 2.0]
    next_value = 0.0
    advantages = compute_gae(rewards, values, next_value, gamma=1.0, lam=0.95)
    # γ=1 时 δ_t = r_t + V(t+1) - V(t)
    assert len(advantages) == 3
```

---

## 5. 实现顺序和依赖

```
Module ②补充 (theory_features.py)
    ↓
Module ③ (symbolic_rule.py + safe_eval.py + rule_library.py)
    ↓
Module ④ (llm/ 整个目录)     Module ⑤ (graph.py → hgt.py → ppo_agent.py → ppo_trainer.py)
    ↓                              ↓
           规则库 + 训练好的 PPO
```

**建议在 Claude Code 中的执行步骤:**

1. **Step 1:** 实现 `safe_eval.py` + 测试 → 通过后继续
2. **Step 2:** 实现 `theory_features.py`（扩展理论特征）→ 用现有测试实例验证值域 [0,1]
3. **Step 3:** 实现 `symbolic_rule.py` + `rule_library.py` → 通过 wrap_fixed_rule 测试（与现有 R1-R8 对比）
4. **Step 4a:** 实现 `llm/` 目录 → 可以跑 iterative_rule_generation（需要 LLM API key）
5. **Step 4b (并行):** 实现 `graph.py` → `hgt.py` → `ppo_agent.py` → `ppo_trainer.py`
6. **Step 5:** 端到端：生成规则库 → 训练 PPO → 评估

---

## 6. 给 Claude Code 的指令模板

在 VS Code 中，对每个 Step，你可以发送如下指令：

```
请实现 fjsp_sfr/dispatch/safe_eval.py。

规格要求：
- [粘贴上面 2.3.1 节的接口定义]

约束：
- 用 ast.parse + 白名单 NodeVisitor 实现安全检查
- 只允许：Add, Sub, Mult, Div, Pow, USub, Compare, BoolOp, IfExp, Call(仅限 min/max/abs)
- 数学函数通过 math 模块: tanh, log, exp, sqrt
- 所有变量名必须在 variable_names 中
- 不允许任何形式的 attribute access（除了 math.xxx）

测试文件：
- [粘贴上面 2.5 节的相关测试]

现有代码参考：
- 这个模块被 symbolic_rule.py 调用
- variable_names 来自 theory_features.py 的 THEORY_FEATURE_NAMES
```
