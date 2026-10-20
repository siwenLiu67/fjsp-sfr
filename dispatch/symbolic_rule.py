"""论文 Sec 5.2: 符号规则 H = ⟨Target, Cond, Filter, Priority⟩。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from fjsp_sfr.dispatch.safe_eval import CompiledExpression, compile_expression
from fjsp_sfr.dispatch.theory_features import THEORY_FEATURE_NAMES


class RuleType(IntEnum):
    TYPE_I = 1
    TYPE_II = 2
    TYPE_III = 3
    TYPE_IV = 4


@dataclass(frozen=True)
class SymbolicRule:
    name: str
    expression: str
    rule_type: RuleType
    priority_expr: str
    cond_expr: str | None = None
    filter_expr: str | None = None

    _compiled_priority: CompiledExpression | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _compiled_cond: CompiledExpression | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _compiled_filter: CompiledExpression | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_compiled_priority",
            compile_expression(self.priority_expr, THEORY_FEATURE_NAMES),
        )
        if self.cond_expr is not None:
            object.__setattr__(
                self,
                "_compiled_cond",
                compile_expression(self.cond_expr, THEORY_FEATURE_NAMES),
            )
        if self.filter_expr is not None:
            object.__setattr__(
                self,
                "_compiled_filter",
                compile_expression(self.filter_expr, THEORY_FEATURE_NAMES),
            )

    def evaluate_batch(
        self, factor_rows: tuple[tuple[float, ...], ...]
    ) -> list[float]:
        """计算每个 candidate 的优先级分数。"""

        assert self._compiled_priority is not None
        scores: list[float] = []
        for row in factor_rows:
            variables = dict(zip(THEORY_FEATURE_NAMES, row))
            if (
                self._compiled_filter is not None
                and self._compiled_filter.evaluate(variables) <= 0.0
            ):
                scores.append(float("-inf"))
                continue
            scores.append(self._compiled_priority.evaluate(variables))
        return scores

    def is_applicable(self, summary_row: tuple[float, ...]) -> bool:
        """评估决策点级 Cond；无 Cond 的规则始终可用。"""

        if self._compiled_cond is None:
            return True
        variables = dict(zip(THEORY_FEATURE_NAMES, summary_row))
        return self._compiled_cond.evaluate(variables) > 0.0

    def to_dict(self) -> dict:
        """将规则定义序列化为不含编译缓存的字典。"""

        data = {
            "name": self.name,
            "expression": self.expression,
            "rule_type": int(self.rule_type),
            "priority_expr": self.priority_expr,
        }
        if self.cond_expr is not None:
            data["cond_expr"] = self.cond_expr
        if self.filter_expr is not None:
            data["filter_expr"] = self.filter_expr
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SymbolicRule":
        """从字典恢复规则并重新安全编译表达式。"""

        return cls(
            name=data["name"],
            expression=data["expression"],
            rule_type=RuleType(data["rule_type"]),
            priority_expr=data["priority_expr"],
            cond_expr=data.get("cond_expr"),
            filter_expr=data.get("filter_expr"),
        )
