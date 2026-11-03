"""规则库：存储、加载、序列化 SymbolicRule 集合。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from fjsp_sfr.dispatch.symbolic_rule import SymbolicRule
from fjsp_sfr.dispatch.theory_features import feature_names_for


@dataclass(frozen=True)
class RuleLibrary:
    rules: tuple[object, ...]
    feature_protocol: str | None = None

    def validate_protocol(self, feature_set: str, *, formal: bool = False) -> None:
        expected = feature_names_for(feature_set)
        if formal and feature_set != "paper14":
            raise ValueError("formal rules require paper14")
        if formal and self.feature_protocol != "paper14":
            raise ValueError("formal rule library lacks the paper14 protocol declaration")
        for rule in self.rules:
            if isinstance(rule, SymbolicRule):
                if rule.feature_set != feature_set:
                    raise ValueError(f"rule {rule.name!r} uses {rule.feature_set}, expected {feature_set}")
                if feature_names_for(rule.feature_set) != expected:
                    raise ValueError("rule feature ordering mismatch")

    def action_size(self) -> int:
        return len(self.rules)

    def get_rule(self, index: int) -> SymbolicRule:
        return self.rules[index]

    def select_and_dispatch(
        self,
        rule_index: int,
        factor_rows: tuple[tuple[float, ...], ...],
    ) -> int:
        """用指定规则返回得分最高的 candidate index。"""

        scores = self.rules[rule_index].evaluate_batch(factor_rows)
        best_index = 0
        best_score = float("-inf")
        for index, score in enumerate(scores):
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    @classmethod
    def from_json(cls, path: str) -> "RuleLibrary":
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
        from fjsp_sfr.dispatch.classic_rules import ClassicRule

        return cls(
            feature_protocol=data.get("feature_protocol"),
            rules=tuple(
                ClassicRule.from_dict(item)
                if item.get("kind") == "classic"
                else SymbolicRule.from_dict(item)
                for item in data["rules"]
            )
        )

    def to_json(self, path: str) -> None:
        data = {"rules": [rule.to_dict() for rule in self.rules]}
        if self.feature_protocol is not None:
            data["feature_protocol"] = self.feature_protocol
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
