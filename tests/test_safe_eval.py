import pytest

from fjsp_sfr.dispatch.safe_eval import ExpressionSyntaxError, compile_expression


def test_simple_arithmetic():
    ce = compile_expression("x + y * 2", ("x", "y"))
    assert ce.evaluate({"x": 1.0, "y": 3.0}) == 7.0


def test_math_functions():
    ce = compile_expression("max(x, y) + abs(z)", ("x", "y", "z"))
    assert ce.evaluate({"x": 1.0, "y": 5.0, "z": -3.0}) == 8.0


def test_math_module():
    ce = compile_expression("math.tanh(x)", ("x",))
    import math

    assert abs(ce.evaluate({"x": 1.0}) - math.tanh(1.0)) < 1e-10


def test_conditional():
    ce = compile_expression("x if x > 0.5 else 1.0 - x", ("x",))
    assert ce.evaluate({"x": 0.8}) == 0.8
    assert ce.evaluate({"x": 0.3}) == 0.7


def test_batch():
    ce = compile_expression("a + b", ("a", "b"))
    rows = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    result = ce.evaluate_batch(rows, ("a", "b"))
    assert result == [3.0, 7.0, 11.0]


def test_rejects_import():
    with pytest.raises(ExpressionSyntaxError):
        compile_expression("__import__('os')", ("x",))


def test_rejects_attribute_access():
    with pytest.raises(ExpressionSyntaxError):
        compile_expression("x.__class__", ("x",))


def test_rejects_unknown_variable():
    with pytest.raises(ExpressionSyntaxError):
        compile_expression("unknown_var + 1", ("x",))


def test_rejects_lambda():
    with pytest.raises(ExpressionSyntaxError):
        compile_expression("(lambda: 1)()", ("x",))


def test_rejects_subscript():
    with pytest.raises(ExpressionSyntaxError):
        compile_expression("x[0]", ("x",))


def test_rejects_dunder_builtins():
    with pytest.raises(ExpressionSyntaxError):
        compile_expression("__builtins__", ("x",))


def test_source_preserved():
    ce = compile_expression("x + 1", ("x",))
    assert ce.source == "x + 1"
