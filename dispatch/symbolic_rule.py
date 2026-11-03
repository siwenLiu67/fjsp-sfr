"""论文 Sec 5.2: 符号规则 H = ⟨Target, Cond, Filter, Priority⟩。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from fjsp_sfr.dispatch.safe_eval import CompiledExpression, compile_expression
from fjsp_sfr.dispatch.theory_features import (
    batch_context_for_rows,
    expression_names_for,
    feature_names_for,
)


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
    feature_set: str = "theory11"

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
        feature_names = expression_names_for(self.feature_set)
        object.__setattr__(
            self,
            "_compiled_priority",
            compile_expression(self.priority_expr, feature_names),
        )
        if self.cond_expr is not None:
            object.__setattr__(
                self,
                "_compiled_cond",
                compile_expression(self.cond_expr, feature_names),
            )
        if self.filter_expr is not None:
            object.__setattr__(
                self,
                "_compiled_filter",
                compile_expression(self.filter_expr, feature_names),
            )

    def _validate_rows(self, rows: tuple[tuple[float, ...], ...]) -> None:
        expected = len(feature_names_for(self.feature_set))
        if any(len(row) != expected for row in rows):
            raise ValueError(
                f"rule {self.name!r} expects {self.feature_set} ({expected} columns); "
                "the supplied feature matrix has an incompatible width"
            )

    def evaluate_batch(
        self, factor_rows: tuple[tuple[float, ...], ...]
    ) -> list[float]:
        """计算每个 candidate 的优先级分数。"""

        self._validate_rows(factor_rows)
        context = batch_context_for_rows(self.feature_set, factor_rows)
        priority_scores = self.priority_scores(factor_rows)
        scores: list[float] = []
        for row, priority_score in zip(
            factor_rows, priority_scores, strict=True
        ):
            variables = dict(zip(feature_names_for(self.feature_set), row, strict=True))
            variables.update(context)
            if (
                self._compiled_filter is not None
                and self._compiled_filter.evaluate(variables) <= 0.0
            ):
                scores.append(float("-inf"))
                continue
            scores.append(priority_score)
        return scores

    def priority_scores(
        self, factor_rows: tuple[tuple[float, ...], ...]
    ) -> list[float]:
        """Evaluate only Priority, before an optional candidate filter."""

        assert self._compiled_priority is not None
        self._validate_rows(factor_rows)
        context = batch_context_for_rows(self.feature_set, factor_rows)
        scores = []
        for row in factor_rows:
            variables = dict(zip(feature_names_for(self.feature_set), row, strict=True))
            variables.update(context)
            scores.append(self._compiled_priority.evaluate(variables))
        return scores

    def is_applicable(self, summary_row: tuple[float, ...]) -> bool:
        """评估决策点级 Cond；无 Cond 的规则始终可用。"""

        if self._compiled_cond is None:
            return True
        self._validate_rows((summary_row,))
        variables = dict(zip(feature_names_for(self.feature_set), summary_row, strict=True))
        if self.feature_set == "theory11_plus3":
            index = feature_names_for(self.feature_set).index("remaining_work")
            value = summary_row[index]
            variables["mean_remaining_work"] = value / max(1e-12, 1.0 - value)
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
        if self.feature_set != "theory11":
            data["feature_set"] = self.feature_set
        if self.feature_set == "paper14":
            data["feature_names"] = list(feature_names_for(self.feature_set))
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SymbolicRule":
        """从字典恢复规则并重新安全编译表达式。"""

        if data.get("feature_set") == "paper14":
            declared = data.get("feature_names")
            if declared != list(feature_names_for("paper14")):
                raise ValueError("paper14 symbolic rule must declare the exact ordered feature names")
        return cls(
            name=data["name"],
            expression=data["expression"],
            rule_type=RuleType(data["rule_type"]),
            priority_expr=data["priority_expr"],
            cond_expr=data.get("cond_expr"),
            filter_expr=data.get("filter_expr"),
            feature_set=data.get("feature_set", "theory11"),
        )
