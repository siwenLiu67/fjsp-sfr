"""Safe compilation and evaluation of mathematical rule expressions."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from types import CodeType


class ExpressionSyntaxError(Exception):
    """Raised when an expression contains syntax outside the safe subset."""


_ALLOWED_BINARY_OPERATORS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.FloorDiv,
    ast.Mod,
)
_ALLOWED_UNARY_OPERATORS = (ast.USub, ast.UAdd, ast.Not)
_ALLOWED_COMPARISON_OPERATORS = (
    ast.Lt,
    ast.Gt,
    ast.LtE,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
)
_ALLOWED_BOOLEAN_OPERATORS = (ast.And, ast.Or)
_ALLOWED_FUNCTIONS = frozenset({"min", "max", "abs"})
_ALLOWED_MATH_FUNCTIONS = frozenset({"tanh", "log", "exp", "sqrt", "log1p"})
_SAFE_NAMES = _ALLOWED_FUNCTIONS | {"math", "True", "False"}


class _SafetyValidator(ast.NodeVisitor):
    """Validate exactly the expression grammar supported by this module."""

    def __init__(self, variable_names: tuple[str, ...]) -> None:
        self.allowed_names = frozenset(variable_names) | _SAFE_NAMES

    def generic_visit(self, node: ast.AST) -> None:
        raise ExpressionSyntaxError(
            f"unsupported expression syntax: {type(node).__name__}"
        )

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, _ALLOWED_BINARY_OPERATORS):
            raise ExpressionSyntaxError(
                f"unsupported binary operator: {type(node.op).__name__}"
            )
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, _ALLOWED_UNARY_OPERATORS):
            raise ExpressionSyntaxError(
                f"unsupported unary operator: {type(node.op).__name__}"
            )
        self.visit(node.operand)

    def visit_Compare(self, node: ast.Compare) -> None:
        if any(
            not isinstance(operator, _ALLOWED_COMPARISON_OPERATORS)
            for operator in node.ops
        ):
            raise ExpressionSyntaxError("unsupported comparison operator")
        self.visit(node.left)
        for comparator in node.comparators:
            self.visit(comparator)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not isinstance(node.op, _ALLOWED_BOOLEAN_OPERATORS):
            raise ExpressionSyntaxError(
                f"unsupported boolean operator: {type(node.op).__name__}"
            )
        for value in node.values:
            self.visit(value)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Call(self, node: ast.Call) -> None:
        if node.keywords:
            raise ExpressionSyntaxError("keyword arguments are not allowed")
        if isinstance(node.func, ast.Name):
            if node.func.id not in _ALLOWED_FUNCTIONS:
                raise ExpressionSyntaxError(
                    f"function call is not allowed: {node.func.id}"
                )
        elif isinstance(node.func, ast.Attribute):
            if (
                not isinstance(node.func.value, ast.Name)
                or node.func.value.id != "math"
                or node.func.attr not in _ALLOWED_MATH_FUNCTIONS
            ):
                raise ExpressionSyntaxError("math function call is not allowed")
        else:
            raise ExpressionSyntaxError("only approved function calls are allowed")

        for argument in node.args:
            self.visit(argument)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.allowed_names:
            raise ExpressionSyntaxError(f"unknown variable or name: {node.id}")

    def visit_Constant(self, node: ast.Constant) -> None:
        return

    def visit_Tuple(self, node: ast.Tuple) -> None:
        for element in node.elts:
            self.visit(element)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        raise ExpressionSyntaxError("attribute access is not allowed")


@dataclass(frozen=True)
class CompiledExpression:
    """A validated expression compiled to a Python code object."""

    source: str
    _code: CodeType

    def evaluate(self, variables: dict[str, float]) -> float:
        """Evaluate the expression with the supplied variable values."""

        namespace: dict[str, object] = dict(variables)
        namespace.update(
            {
                "min": min,
                "max": max,
                "abs": abs,
                "math": math,
                "__builtins__": {},
            }
        )
        return float(eval(self._code, namespace))

    def evaluate_batch(
        self,
        rows: tuple[tuple[float, ...], ...],
        variable_names: tuple[str, ...],
    ) -> list[float]:
        """Evaluate one expression score for each candidate row."""

        return [
            self.evaluate(dict(zip(variable_names, row)))
            for row in rows
        ]


def compile_expression(
    expr: str, variable_names: tuple[str, ...]
) -> CompiledExpression:
    """Compile a mathematical expression after validating its AST."""

    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, TypeError, ValueError) as error:
        raise ExpressionSyntaxError(f"invalid expression syntax: {error}") from error

    _SafetyValidator(variable_names).visit(tree)
    code = compile(tree, "<rule>", "eval")
    return CompiledExpression(source=expr, _code=code)
