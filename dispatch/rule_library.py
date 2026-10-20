"""规则库：存储、加载、序列化 SymbolicRule 集合。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from fjsp_sfr.dispatch.symbolic_rule import SymbolicRule


@dataclass(frozen=True)
class RuleLibrary:
    rules: tuple[SymbolicRule, ...]

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
        return cls(
            rules=tuple(SymbolicRule.from_dict(item) for item in data["rules"])
        )

    def to_json(self, path: str) -> None:
        data = {"rules": [rule.to_dict() for rule in self.rules]}
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
