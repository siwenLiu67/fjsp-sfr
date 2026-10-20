"""解析 LLM 生成的规则文本为 SymbolicRule 对象。"""

from __future__ import annotations

import re
import sys

from fjsp_sfr.dispatch.safe_eval import ExpressionSyntaxError, compile_expression
from fjsp_sfr.dispatch.symbolic_rule import RuleType, SymbolicRule
from fjsp_sfr.dispatch.theory_features import THEORY_FEATURE_NAMES


_TYPE_MAP = {
    "I": RuleType.TYPE_I,
    "II": RuleType.TYPE_II,
    "III": RuleType.TYPE_III,
    "IV": RuleType.TYPE_IV,
}


def _field(block: str, name: str) -> str | None:
    match = re.search(rf"(?im)^\s*{name}:\s*(.*?)\s*$", block)
    return match.group(1).strip() if match else None


def _optional_expression(value: str | None) -> str | None:
    if value is None or value.upper() == "NONE":
        return None
    return value


def parse_llm_output(
    text: str,
    feature_names: tuple[str, ...] = THEORY_FEATURE_NAMES,
) -> list[SymbolicRule]:
    """解析并安全验证 LLM 输出中的每条规则。"""

    rules: list[SymbolicRule] = []
    blocks = re.split(r"(?im)^\s*RULE:\s*", text)[1:]
    for block in blocks:
        lines = block.strip().splitlines()
        name = lines[0].strip() if lines else ""
        body = "\n".join(lines[1:])
        type_text = _field(body, "TYPE")
        priority_expr = _field(body, "PRIORITY")
        cond_expr = _optional_expression(_field(body, "COND"))
        filter_expr = _optional_expression(_field(body, "FILTER"))

        try:
            if not name or type_text is None or priority_expr is None:
                raise ValueError("RULE, TYPE, and PRIORITY are required")
            rule_type = _TYPE_MAP[type_text.upper()]
            for expression in (priority_expr, cond_expr, filter_expr):
                if expression is not None:
                    compile_expression(expression, feature_names)
            rules.append(
                SymbolicRule(
                    name=name,
                    expression=priority_expr,
                    rule_type=rule_type,
                    priority_expr=priority_expr,
                    cond_expr=cond_expr,
                    filter_expr=filter_expr,
                )
            )
        except (ExpressionSyntaxError, KeyError, TypeError, ValueError) as error:
            print(f"Skipping invalid rule {name!r}: {error}", file=sys.stderr)

    return rules
