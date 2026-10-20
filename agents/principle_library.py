"""Validated symbolic scheduling principles over the five theory factors.

The evaluator deliberately implements a tiny expression language instead of
calling :func:`eval`.  This makes generated libraries inspectable, portable,
and incapable of reading any state beyond ``G``, ``F``, ``U``, ``M``, and ``T``.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


FACTOR_SYMBOLS = ("G", "F", "U", "M", "T")
ALLOWED_FUNCTIONS = ("min", "max", "clip")
ALLOWED_CONSTANTS = frozenset({0.0, 0.25, 0.5, 1.0, 2.0})
SAFE_DIVIDE_EPSILON = 1e-9
MAX_ABS_SCORE = 1e6


class PrincipleValidationError(ValueError):
    """Raised when a principle escapes the theory-constrained grammar."""


def _is_allowed_constant(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and abs(float(value)) in ALLOWED_CONSTANTS
    )


class _ExpressionValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.factors_used: set[str] = set()
        self.node_count = 0

    def generic_visit(self, node: ast.AST) -> None:
        self.node_count += 1
        if self.node_count > 80:
            raise PrincipleValidationError("expression exceeds the 80-node limit")
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.USub,
            ast.UAdd,
        )
        if not isinstance(node, allowed):
            raise PrincipleValidationError(
                f"forbidden syntax: {type(node).__name__}"
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.node_count += 1
        if node.id in FACTOR_SYMBOLS:
            self.factors_used.add(node.id)
            return
        if node.id not in ALLOWED_FUNCTIONS:
            raise PrincipleValidationError(f"unknown factor or function: {node.id}")

    def visit_Constant(self, node: ast.Constant) -> None:
        self.node_count += 1
        if not _is_allowed_constant(node.value):
            raise PrincipleValidationError(
                f"constant {node.value!r} is not in the fixed constant set"
            )

    def visit_Call(self, node: ast.Call) -> None:
        self.node_count += 1
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise PrincipleValidationError("only min, max, and clip calls are allowed")
        if node.keywords:
            raise PrincipleValidationError("keyword arguments are forbidden")
        required = 3 if node.func.id == "clip" else 2
        if len(node.args) != required:
            raise PrincipleValidationError(
                f"{node.func.id} requires exactly {required} arguments"
            )
        for argument in node.args:
            self.visit(argument)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.node_count += 1
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise PrincipleValidationError("only +, -, *, and / are allowed")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        self.node_count += 1
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            raise PrincipleValidationError("only unary + and - are allowed")
        self.visit(node.operand)


def parse_expression(expression: str) -> tuple[ast.Expression, tuple[str, ...]]:
    if not isinstance(expression, str) or not expression.strip():
        raise PrincipleValidationError("expression must be a non-empty string")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise PrincipleValidationError(f"invalid expression syntax: {error.msg}") from error
    validator = _ExpressionValidator()
    validator.visit(tree)
    if not validator.factors_used:
        raise PrincipleValidationError("expression must use at least one theory factor")
    return tree, tuple(symbol for symbol in FACTOR_SYMBOLS if symbol in validator.factors_used)


def canonical_expression(expression: str) -> str:
    """Return a formatting-insensitive representation for duplicate detection."""

    tree, _ = parse_expression(expression)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < SAFE_DIVIDE_EPSILON:
        denominator = math.copysign(SAFE_DIVIDE_EPSILON, denominator or 1.0)
    return numerator / denominator


def _evaluate_node(node: ast.AST, values: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, values)
    if isinstance(node, ast.Name):
        return float(values[node.id])
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_node(node.operand, values)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, values)
        right = _evaluate_node(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return _safe_divide(left, right)
    if isinstance(node, ast.Call):
        function = node.func.id  # type: ignore[union-attr]
        arguments = [_evaluate_node(argument, values) for argument in node.args]
        if function == "min":
            return min(arguments)
        if function == "max":
            return max(arguments)
        return min(max(arguments[0], arguments[1]), arguments[2])
    raise PrincipleValidationError(f"unsupported validated node: {type(node).__name__}")


@dataclass(frozen=True)
class PrincipleSpec:
    name: str
    expression: str
    rationale: str
    factors_used: tuple[str, ...]
    source: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.rationale.strip():
            raise PrincipleValidationError("name and rationale must be non-empty")
        if self.source not in {"llm", "manual_debug"}:
            raise PrincipleValidationError("source must be 'llm' or 'manual_debug'")
        _tree, actual = parse_expression(self.expression)
        declared = tuple(self.factors_used)
        if len(set(declared)) != len(declared) or any(
            factor not in FACTOR_SYMBOLS for factor in declared
        ):
            raise PrincipleValidationError("factors_used contains invalid or duplicate names")
        if set(declared) != set(actual):
            raise PrincipleValidationError(
                f"factors_used {declared!r} does not match expression factors {actual!r}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PrincipleSpec":
        required = {"name", "expression", "rationale", "factors_used", "source"}
        extra = set(payload) - required
        missing = required - set(payload)
        if missing or extra:
            raise PrincipleValidationError(
                f"principle fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        factors = payload["factors_used"]
        if not isinstance(factors, list) or not all(isinstance(x, str) for x in factors):
            raise PrincipleValidationError("factors_used must be a JSON string array")
        return cls(
            name=str(payload["name"]),
            expression=str(payload["expression"]),
            rationale=str(payload["rationale"]),
            factors_used=tuple(factors),
            source=str(payload["source"]),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["factors_used"] = list(self.factors_used)
        return payload

    def evaluate(self, factors: Sequence[float] | Mapping[str, float]) -> float:
        if isinstance(factors, Mapping):
            if set(factors) != set(FACTOR_SYMBOLS):
                raise ValueError("factor mapping must contain exactly G, F, U, M, and T")
            values = {key: float(factors[key]) for key in FACTOR_SYMBOLS}
        else:
            if len(factors) != len(FACTOR_SYMBOLS):
                raise ValueError("factor row must contain exactly five values")
            values = dict(zip(FACTOR_SYMBOLS, map(float, factors)))
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("input theory factors must be finite")
        tree, _ = parse_expression(self.expression)
        score = _evaluate_node(tree, values)
        if not math.isfinite(score):
            score = math.copysign(MAX_ABS_SCORE, score if not math.isnan(score) else 1.0)
        return min(MAX_ABS_SCORE, max(-MAX_ABS_SCORE, float(score)))

    def evaluate_batch(self, factor_rows: Iterable[Sequence[float]]) -> tuple[float, ...]:
        return tuple(self.evaluate(row) for row in factor_rows)


@dataclass(frozen=True)
class PrincipleLibrary:
    principles: tuple[PrincipleSpec, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise PrincipleValidationError("only principle-library schema version 1 is supported")
        if not self.principles:
            raise PrincipleValidationError("principle library must not be empty")
        names = [principle.name for principle in self.principles]
        if len(set(names)) != len(names):
            raise PrincipleValidationError("principle names must be unique")
        expressions = [canonical_expression(item.expression) for item in self.principles]
        if len(set(expressions)) != len(expressions):
            raise PrincipleValidationError("duplicate principle expressions are forbidden")

    def __len__(self) -> int:
        return len(self.principles)

    def __getitem__(self, index: int) -> PrincipleSpec:
        return self.principles[index]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "principles": [principle.to_dict() for principle in self.principles],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PrincipleLibrary":
        if set(payload) != {"schema_version", "principles"}:
            raise PrincipleValidationError(
                "library must contain exactly schema_version and principles"
            )
        raw = payload["principles"]
        if not isinstance(raw, list):
            raise PrincipleValidationError("principles must be a JSON array")
        return cls(
            principles=tuple(PrincipleSpec.from_dict(item) for item in raw),  # type: ignore[arg-type]
            schema_version=int(payload["schema_version"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PrincipleLibrary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise PrincipleValidationError("principle library root must be an object")
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

