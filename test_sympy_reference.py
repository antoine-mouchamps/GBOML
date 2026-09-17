import contextlib
import io
import os
import tempfile
import uuid
import unittest

import numpy as np
import sympy

from gboml.compiler.gboml_compiler import compile_gboml

T = 4
PARAMETERS = {"a": 1.5, "b": 2.25, "p": 0.7, "q": 3.0, "r": 1.25, "s": 0.4}
TABLES = {"cap": [1.0, 2.5, 0.5, 4.0], "dem": [0.5, 1.0, 2.0, 3.0]}
X = sympy.symbols(f"x0:{T}")
Y = sympy.symbols(f"y0:{T}")
MODEL_DIRECTORY = tempfile.mkdtemp()


def compile_constraint(written: str, where: str = "") -> tuple:
    """The rows of `written <= 0` in a model over x[T] and y[T], as dense A, Q and b."""
    parameters = [f"{name} = {value};" for name, value in PARAMETERS.items()]
    parameters += [f"{name} = {{{', '.join(map(str, values))}}};" for name, values in TABLES.items()]
    text = "\n".join([
        "#TIMEHORIZON", f"T = {T};", "#NODE A", "#PARAMETERS", *parameters,
        "#VARIABLES", "internal : x[T];", "internal : y[T];",
        "#CONSTRAINTS", f"c : {written} <= 0 {where};", "#OBJECTIVES", "min : x[0];", ""])
    path = os.path.join(MODEL_DIRECTORY, f"model_{uuid.uuid4().hex}.txt")
    with open(path, "w") as file:
        file.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        result = compile_gboml(path)
    linear, quadratic, right_hand_side = result[4].toarray(), result[6], result[5]
    dense_quadratic = np.zeros(quadratic.shape)
    np.add.at(dense_quadratic, tuple(quadratic.coords), quadratic.data)
    return linear, dense_quadratic, np.asarray(right_hand_side)


def expected_row(expression, nb_columns: int) -> tuple:
    """The linear part, the degree two part and the constant of a sympy expression."""
    linear = np.zeros(nb_columns)
    quadratic = np.zeros((nb_columns, nb_columns))
    constant = 0.0
    polynomial = sympy.Poly(sympy.expand(expression), *X, *Y)
    for degrees, coefficient in polynomial.terms():
        columns = [column for column, degree in enumerate(degrees) for _ in range(degree)]
        if not columns:
            constant = float(coefficient)
        elif len(columns) == 1:
            linear[columns[0]] = float(coefficient)
        else:
            quadratic[min(columns), max(columns)] = float(coefficient)
    return linear, quadratic, constant


class SympyReferenceTests(unittest.TestCase):
    """Each constraint is compiled end to end and compared, row by row, with sympy."""

    def check(self, written: str, expression_of_row, rows=range(T), where: str = ""):
        linear, quadratic, right_hand_side = compile_constraint(written, where)
        self.assertEqual(linear.shape[0], len(rows))
        for row, t in enumerate(rows):
            expected = expected_row(expression_of_row(t), linear.shape[1])
            np.testing.assert_allclose(linear[row], expected[0], atol=1e-12, err_msg=f"row t={t}")
            np.testing.assert_allclose(quadratic[row], expected[1], atol=1e-12, err_msg=f"row t={t}")
            self.assertAlmostEqual(right_hand_side[row], -expected[2], places=12)

    def test_like_terms_with_parameter_coefficients(self):
        a, b = PARAMETERS["a"], PARAMETERS["b"]
        self.check("a*x[t] + b*x[t] - a*x[t]", lambda t: a*X[t] + b*X[t] - a*X[t])

    def test_a_parameter_expression_distributed_over_variables(self):
        p, q, r, s = (PARAMETERS[name] for name in "pqrs")
        self.check("(p + q)*(r*x[t] - s*y[t])", lambda t: (p + q)*(r*X[t] - s*Y[t]))

    def test_a_division_by_a_parameter_expression(self):
        p, q = PARAMETERS["p"], PARAMETERS["q"]
        self.check("(x[t] + 2*y[t]) / (p*q)", lambda t: (X[t] + 2*Y[t]) / (p*q))

    def test_the_square_of_a_sum_of_variables(self):
        self.check("(x[t] + y[t])**2", lambda t: (X[t] + Y[t])**2)

    def test_equivalent_affine_indexes(self):
        self.check("x[1*t+1-2] + x[t-1]", lambda t: X[t-1] + X[t-1], rows=range(1, T), where="where t >= 1")

    def test_a_product_written_in_both_orders(self):
        self.check("x[t]*y[t] - y[t]*x[t] + x[t]", lambda t: X[t])

    def test_a_product_of_two_ragged_sums(self):
        self.check("sum(x[i] for i in [0:t]) * sum(y[j] for j in [0:t])",
                   lambda t: sum(X[: t + 1]) * sum(Y[: t + 1]))

    def test_nested_ragged_sums(self):
        self.check("sum(sum(x[j] for j in [0:i]) for i in [0:t])",
                   lambda t: sum(sum(X[: i + 1]) for i in range(t + 1)))

    def test_a_sum_whose_body_ignores_its_index_times_a_sum(self):
        self.check("sum(x[t] for i in [0:2]) * sum(y[j] for j in [0:t])",
                   lambda t: 3*X[t] * sum(Y[: t + 1]))

    def test_a_sum_with_a_condition_inside_a_product(self):
        self.check("sum(x[i] for i in [0:T-1] where i != t) * y[t]",
                   lambda t: sum(X[i] for i in range(T) if i != t) * Y[t])

    def test_parameter_coefficients_inside_sums(self):
        cap, dem = TABLES["cap"], TABLES["dem"]
        self.check("sum(cap[i]*x[i] for i in [0:t]) * sum(dem[j]*y[j] for j in [0:t])",
                   lambda t: sum(cap[i]*X[i] for i in range(t + 1)) * sum(dem[j]*Y[j] for j in range(t + 1)))

    def test_the_same_expression_under_two_different_sums(self):
        cap = TABLES["cap"]
        self.check("sum(cap[i] for i in [0:t])*x[t] + sum(cap[i] for i in [0:T-1])*y[t]",
                   lambda t: sum(cap[: t + 1])*X[t] + sum(cap)*Y[t])


if __name__ == "__main__":
    unittest.main()
