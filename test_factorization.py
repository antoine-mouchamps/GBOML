import contextlib
import io
import os
import subprocess
import sys
import tempfile
import uuid
import unittest
from pathlib import Path

import numpy as np

from gboml import GbomlGraph
from gboml.compiler.classes.symbolic import Indexed, Number, ParameterSymbol, Range
from gboml.compiler.gboml_compiler import compile_gboml
from gboml.compiler.gboml_matrix_generation import get_flat_nodes_edges_ordered
from gboml.compiler.gboml_parser import parse_file
from gboml.compiler.gboml_semantic import (
    check_program_degrees,
    factorize_program,
    semantic,
)
from gboml.compiler.utils import move_to_directory

REPOSITORY = Path(__file__).resolve().parent
MODEL_DIRECTORY = tempfile.mkdtemp()


def model_text(constraints: str, parameters: str = "", horizon: int = 3) -> str:
    return "\n".join(
        [
            "#TIMEHORIZON",
            f"T = {horizon};",
            "#NODE A",
            "#PARAMETERS",
            parameters,
            "#VARIABLES",
            "internal : x[T];",
            "internal : y[T];",
            "#CONSTRAINTS",
            constraints,
            "#OBJECTIVES",
            "min : x[0];",
            "",
        ]
    )


def write_model(text: str) -> str:
    """A file of its own for every model, since parse_file caches programs by file name."""
    path = os.path.join(MODEL_DIRECTORY, f"model_{uuid.uuid4().hex}.txt")
    with open(path, "w") as file:
        file.write(text)
    return path


def compile_model(text: str) -> tuple:
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        result = compile_gboml(write_model(text))
    return result, printed.getvalue()


def factorize_model(text: str) -> tuple[dict, dict]:
    """Every constraint factor of node A once factorized, by name, and the values of the parameters."""
    previous_directory, file_name = move_to_directory(write_model(text))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            program, variables, definitions = semantic(parse_file(file_name))
            check_program_degrees(program, variables, definitions)
            factorize_program(program, variables, definitions)
    finally:
        os.chdir(previous_directory)
    elements = get_flat_nodes_edges_ordered(program.get_nodes(), program.get_links())
    factors = {
        factor.get_name(): factor
        for element in elements
        for factor in element.get_constraint_factors()
    }
    return factors, program.parameter_values


class FactorizationTests(unittest.TestCase):
    """Sections 6.5 to 6.8 of SHARED_GRAPH_PLAN.md."""

    def test_like_terms_with_equivalent_indexes_merge(self):
        factors, values = factorize_model(
            model_text(
                "c : x[t] == x[t-1] - loss*x[1*t+1-2] + eff*y[t] where t >= 1;",
                "loss = 0.05;\neff = 0.9;",
            )
        )
        block = factors["c"].block
        self.assertEqual(len(block.linear), 3)
        self.assertEqual(block.quadratic, {})
        self.assertEqual(
            sorted(
                round(term.coefficient.evalf(values), 12)
                for term in block.linear.values()
            ),
            [-0.95, -0.9, 1.0],
        )

    def test_a_product_of_two_sums_is_one_term(self):
        factors, _ = factorize_model(
            model_text("c : sum(x[i] for i in [0:t]) * sum(y[j] for j in [0:t]) <= 1;")
        )
        self.assertEqual(factors["c"].block.linear, {})
        [(ranges, variables)] = factors["c"].block.quadratic
        self.assertEqual((len(ranges), len(variables)), (2, 2))

    def test_a_product_of_two_sums_does_not_depend_on_their_order(self):
        factors, _ = factorize_model(
            model_text(
                "c : sum(x[i] for i in [0:t]) * sum(y[j] for j in [0:t])"
                " - sum(y[j] for j in [0:t]) * sum(x[i] for i in [0:t]) + x[t] <= 1;"
            )
        )
        self.assertEqual(factors["c"].block.quadratic, {})

    def test_a_variable_that_does_not_read_the_index_leaves_the_sum(self):
        factors, values = factorize_model(
            model_text("c : sum(cap[i]*x[t] for i in [0:2]) <= 1;", "cap = {1, 2, 4};")
        )
        [term] = factors["c"].block.linear.values()
        self.assertEqual(term.ranges, ())
        self.assertEqual(term.coefficient.evalf(values), 7.0)

    def test_a_square_merges_symmetric_products(self):
        factors, values = factorize_model(
            model_text("c : (sum(x[i] for i in [0:t]) + y[t])**2 <= 1;")
        )
        quadratic = factors["c"].block.quadratic
        self.assertEqual(len(quadratic), 3)
        [cross] = [
            term.coefficient for term in quadratic.values() if len(term.ranges) == 1
        ]
        self.assertEqual(cross.evalf(values), 2.0)

    def test_the_letter_of_a_sum_does_not_matter(self):
        factors, _ = factorize_model(
            model_text(
                "c : sum(x[i] for i in [0:t]) - sum(x[k] for k in [0:t]) + y[t] <= 1;"
            )
        )
        self.assertEqual(len(factors["c"].block.linear), 1)

    def test_a_derived_parameter_is_inlined(self):
        factors, _ = factorize_model(
            model_text("c : b*x[t] - 2*a*x[t] + y[t] <= 1;", "a = 2;\nb = 2*a;")
        )
        self.assertEqual(len(factors["c"].block.linear), 1)

    def test_a_derived_table_pushes_its_index(self):
        result, _ = compile_model(
            model_text("c : x[t] <= b[t];", "cap = {1, 2, 4};\nb = cap * 2;")
        )
        np.testing.assert_array_equal(result[5], [2.0, 4.0, 8.0])

    def test_an_alias_is_the_same_variable(self):
        text = "\n".join(
            [
                "#TIMEHORIZON",
                "T = 2;",
                "#NODE A",
                "#NODE B",
                "#VARIABLES",
                "internal : x[T];",
                "#CONSTRAINTS",
                "d : x[t] <= 2;",
                "#VARIABLES",
                "internal : out[T] <- B.x[T];",
                "#CONSTRAINTS",
                "c : out[t] <= 1;",
                "#OBJECTIVES",
                "min : out[0];",
                "",
            ]
        )
        factors, _ = factorize_model(text)
        [(_, (alias,))] = factors["c"].block.linear
        [(_, (member,))] = factors["d"].block.linear
        self.assertIs(alias, member)

    def test_the_number_of_rows_comes_from_the_text(self):
        result, _ = compile_model(model_text("c : x[t] - x[t] + y[0] <= 1;"))
        self.assertEqual(result[4].shape[0], 3)


class ParameterValueTests(unittest.TestCase):
    """The values of the parameters live outside the expressions, so one model evaluates with any values."""

    def test_editing_a_value_needs_no_clearing(self):
        factors, values = factorize_model(
            model_text("c : a*x[t] + b*y[t] <= 1;", "a = 3;\nb = 2*a;")
        )
        factor = factors["c"]
        factor.extend(values, {})
        np.testing.assert_array_equal(factor.sparse.toarray()[0], [3, 0, 0, 6, 0, 0])
        values["A.a"] = 5.0
        factor.extend(values, {})
        np.testing.assert_array_equal(factor.sparse.toarray()[0], [5, 0, 0, 10, 0, 0])

    def test_one_model_evaluates_with_two_sets_of_values(self):
        factors, values = factorize_model(model_text("c : a*x[t] <= 1;", "a = 3;"))
        factor = factors["c"]
        factor.extend({**values, "A.a": 7.0}, {})
        with_seven = factor.sparse.toarray()
        factor.extend(values, {})
        np.testing.assert_array_equal(with_seven.diagonal(), [7, 7, 7])
        np.testing.assert_array_equal(factor.sparse.toarray().diagonal(), [3, 3, 3])

    def test_a_value_of_another_shape_is_refused(self):
        factors, values = factorize_model(model_text("c : a*x[t] <= 1;", "a = 3;"))
        with self.assertRaises(SystemExit):
            factors["c"].extend({**values, "A.a": np.array([1.0, 2.0])}, {})

    def test_a_parameter_is_replaced(self):
        factors, values = factorize_model(model_text("c : a*x[t] <= 1;", "a = 3;"))
        block = factors["c"].block
        table = block.constant.coefficient.table
        twice_b = 2 * ParameterSymbol("A.b", (), table)
        replaced = block.replace({ParameterSymbol("A.a", (), table): twice_b})
        [term] = replaced.linear.values()
        self.assertEqual(term.coefficient.evalf({**values, "A.b": 4.0}), 8.0)

    def test_the_derivative_with_respect_to_a_parameter(self):
        factors, values = factorize_model(
            model_text("c : a*a*x[t] + b*y[t] <= 1;", "a = 3;\nb = 2;")
        )
        factor = factors["c"]
        derived = factor.derivative(
            ParameterSymbol("A.a", (), factor.block.constant.coefficient.table)
        )
        [term] = derived.linear.values()
        self.assertEqual(term.coefficient.evalf(values), 6.0)

    def test_the_derivative_keeps_the_rows_of_its_constraint(self):
        factors, values = factorize_model(model_text("c : x[t+1] == a*y[t];", "a = 2;"))
        factor = factors["c"]
        factor.block = factor.derivative(
            ParameterSymbol("A.a", (), factor.block.constant.coefficient.table)
        )
        with contextlib.redirect_stdout(io.StringIO()):
            factor.extend(values, {})
        np.testing.assert_array_equal(factor.sparse.toarray()[:, 3:], [[-1, 0], [0, -1]])

    def test_the_derivative_refuses_a_parameter_deciding_the_variables(self):
        factors, _ = factorize_model(model_text("c : x[t+d] <= 1;", "d = 0;"))
        factor = factors["c"]
        with self.assertRaises(SystemExit):
            factor.derivative(
                ParameterSymbol("A.d", (), factor.block.constant.coefficient.table)
            )

    def test_the_derivative_with_respect_to_an_entry_lands_on_its_row(self):
        for constraint in ("c : cap[t]*x[t] <= 1;", "c : sum(cap[i]*x[i] for i in [0:T-1]) <= 1;"):
            with self.subTest(constraint=constraint):
                factors, values = factorize_model(model_text(constraint, "cap = {1, 2, 3};"))
                factor = factors["c"]
                table = factor.block.constant.coefficient.table
                entry = Indexed(ParameterSymbol("A.cap", (3,), table), (Number(1, table),))
                factor.block = factor.derivative(entry)
                with contextlib.redirect_stdout(io.StringIO()):
                    factor.extend(values, {})
                self.assertEqual(factor.sparse.nnz, 1)
                self.assertEqual(factor.sparse.toarray()[:, 1].sum(), 1)

    def test_the_derivative_refuses_an_entry_deciding_the_variables(self):
        factors, _ = factorize_model(model_text("c : x[cap[0]] <= 1;", "cap = {0, 1, 2};"))
        factor = factors["c"]
        table = factor.block.constant.coefficient.table
        with self.assertRaises(SystemExit):
            factor.derivative(Indexed(ParameterSymbol("A.cap", (3,), table), (Number(0, table),)))

    def test_the_derivative_of_a_discount_factor(self):
        factors, values = factorize_model(model_text("c : (1 + r)**n * x[t] <= 1;", "r = 0.07;\nn = 25;"))
        factor = factors["c"]
        derived = factor.derivative(ParameterSymbol("A.r", (), factor.block.constant.coefficient.table))
        [term] = derived.linear.values()
        self.assertAlmostEqual(term.coefficient.evalf(values), 25 * 1.07**24)

    def test_the_derivative_refuses_a_parameter_deciding_the_rows(self):
        factors, _ = factorize_model(model_text("c : a*x[t] <= 1 where t < a;", "a = 2;"))
        factor = factors["c"]
        with self.assertRaises(SystemExit):
            factor.derivative(
                ParameterSymbol("A.a", (), factor.block.constant.coefficient.table)
            )


def build_graph(text: str) -> GbomlGraph:
    graph = GbomlGraph(3)
    with contextlib.redirect_stdout(io.StringIO()):
        nodes, edges, global_parameters = graph.import_all_nodes_and_edges(write_model(text))
        graph.add_global_parameters(global_parameters)
        graph.add_nodes_in_model(*nodes)
        graph.add_hyperedges_in_model(*edges)
        graph.build_model()
    return graph


class GraphParameterTests(unittest.TestCase):
    """The parameters of a built model are read, added, changed and replaced through GbomlGraph."""

    def setUp(self):
        self.graph = build_graph(model_text("c : a*x[t] <= b;", "a = 3;\nb = 2*a;\ncap = {1, 2, 3};"))

    def regenerate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.graph.generate_matrices()

    def test_a_replaced_parameter_follows_the_values_of_the_parameters_it_reads(self):
        a = self.graph.parameter("A.a")
        tax = self.graph.add_parameter("A.tax", 1)
        self.graph.replace_parameter("A.a", a + tax)
        self.regenerate()
        np.testing.assert_array_equal(self.graph.matrix_a_ineq.toarray()[:, :3], 4 * np.eye(3))
        self.graph.set_parameter_value("A.a", 5)
        self.regenerate()
        np.testing.assert_array_equal(self.graph.matrix_a_ineq.toarray()[:, :3], 6 * np.eye(3))
        # b = 2*a reads the replaced parameter, so it follows too
        np.testing.assert_array_equal(self.graph.matrix_b_ineq, [12, 12, 12])

    def test_a_parameter_defined_from_other_parameters_can_not_be_reached(self):
        with self.assertRaises(SystemExit):
            self.graph.parameter("A.b")

    def test_a_replacement_keeps_the_shape_of_the_parameter(self):
        with self.assertRaises(SystemExit):
            self.graph.replace_parameter("A.a", self.graph.parameter("A.cap"))
        with self.assertRaises(SystemExit):
            self.graph.set_parameter_value("A.cap", [1, 2])

    def test_a_parameter_is_replaced_by_an_expression_and_not_by_a_number(self):
        with self.assertRaises(SystemExit):
            self.graph.replace_parameter("A.a", 4)

    def test_the_time_horizon_can_not_be_replaced(self):
        with self.assertRaises(SystemExit):
            self.graph.replace_parameter("T", self.graph.parameter("A.a"))

    def test_a_parameter_fixed_at_compilation_can_not_be_replaced_or_changed(self):
        text = model_text("c : x[t]**p + z[0] <= 1;", "n = 2;\np = 2;").replace(
            "internal : y[T];", "internal : y[T];\ninternal : z[n];"
        )
        graph = build_graph(text)
        one = graph.add_parameter("A.one", 1)
        for path in ("A.n", "A.p"):
            with self.subTest(path=path), self.assertRaises(SystemExit):
                graph.replace_parameter(path, graph.parameter(path) + one)
            with self.subTest(path=path), self.assertRaises(SystemExit):
                graph.set_parameter_value(path, 3)

    def test_a_parameter_deciding_the_rows_or_the_entries_read_can_not_be_replaced(self):
        graph = build_graph(model_text("c : sum(x[i] for i in [0:n]) + y[t+k] <= 1;", "n = 1;\nk = 0;"))
        one = graph.add_parameter("A.one", 1)
        for path in ("A.n", "A.k"):
            with self.subTest(path=path), self.assertRaises(SystemExit):
                graph.replace_parameter(path, graph.parameter(path) + one)

    def test_an_added_parameter_needs_a_new_name_in_an_existing_element(self):
        with self.assertRaises(SystemExit):
            self.graph.add_parameter("A.b", 1)
        with self.assertRaises(SystemExit):
            self.graph.add_parameter("MISSING.tax", 1)


class GraphDerivativeTests(unittest.TestCase):
    """GbomlGraph.derivative returns the derivative of every array the solvers receive, and leaves the graph unchanged."""

    def setUp(self):
        text = model_text(
            "c : a*a*x[t]*y[t] + a*x[t] + b*y[t] <= a;\nd : x[t] == cap[t];",
            "a = 3;\nb = 2;\ncap = {1, 2, 3};\nn = 2;",
        ).replace("min : x[0];", "min : a*x[0] + a;").replace("internal : y[T];", "internal : y[T];\ninternal : z[n];")
        self.graph = build_graph(text)

    def derivative(self, path):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.graph.derivative(path)

    def test_every_array_keeps_the_shape_of_the_model(self):
        derivative = self.derivative("A.a")
        for key in ("matrix_a_eq", "matrix_b_eq", "quadratic_eq", "matrix_a_ineq", "matrix_b_ineq", "quadratic_ineq"):
            with self.subTest(key=key):
                self.assertEqual(np.shape(derivative[key]), np.shape(getattr(self.graph, key)))

    def test_every_degree_is_differentiated(self):
        derivative = self.derivative("A.a")
        np.testing.assert_array_equal(derivative["matrix_a_ineq"].toarray()[:, :3], np.eye(3))
        np.testing.assert_array_equal(derivative["quadratic_ineq"].data, [6, 6, 6])
        np.testing.assert_array_equal(derivative["matrix_b_ineq"], [1, 1, 1])
        self.assertEqual(derivative["vector_c"][0], 1)
        self.assertEqual(derivative["objective_offset"], 1)

    def test_the_derivative_with_respect_to_an_entry(self):
        derivative = self.derivative("A.cap[2]")
        np.testing.assert_array_equal(derivative["matrix_b_eq"], [0, 0, 1])

    def test_the_graph_is_left_unchanged(self):
        before = self.graph.matrix_a_ineq.toarray(), self.graph.program.get_nodes()[0].get_objectives_data()
        self.derivative("A.a")
        with contextlib.redirect_stdout(io.StringIO()):
            self.graph.generate_matrices()
        np.testing.assert_array_equal(self.graph.matrix_a_ineq.toarray(), before[0])
        self.assertEqual(str(self.graph.program.get_nodes()[0].get_objectives_data()), str(before[1]))

    def test_a_derivative_that_can_not_be_taken_is_refused(self):
        for path in ("A.n", "T", "A.cap", "A.cap[3]", "A.a[0]"):
            with self.subTest(path=path), self.assertRaises(SystemExit):
                self.derivative(path)


class EvaluationTests(unittest.TestCase):
    """Section 7 of SHARED_GRAPH_PLAN.md."""

    def test_ranges_match_what_head_computed(self):
        recorded = np.load(REPOSITORY / "test" / "get_range_reference.npz")
        cases = {key.split(".")[0] for key in recorded.files}
        for case in sorted(cases):
            with self.subTest(case=case):
                expected = recorded[f"{case}.range"]
                values, filled = Range.range_from_bounds(
                    recorded[f"{case}.begin"],
                    recorded[f"{case}.end"],
                    recorded[f"{case}.step"],
                )
                np.testing.assert_array_equal(filled, np.isfinite(expected))
                np.testing.assert_array_equal(
                    values[filled], expected[np.isfinite(expected)]
                )

    def test_a_sum_without_variables_can_read_the_row_in_its_range(self):
        result, _ = compile_model(
            model_text("c : x[t] <= sum(cap[i] for i in [0:t]);", "cap = {1, 2, 4};")
        )
        np.testing.assert_array_equal(result[5], [1.0, 3.0, 7.0])


class BehaviourChangeTests(unittest.TestCase):
    """Section 9 of SHARED_GRAPH_PLAN.md."""

    def test_variables_that_cancel_are_not_bound_checked(self):
        result, _ = compile_model(model_text("c : x[t+5] - x[t+5] + y[t] <= 1;"))
        self.assertEqual(result[4].shape[0], 3)

    def test_a_division_that_simplifies_is_not_evaluated(self):
        result, _ = compile_model(
            model_text("c : cap[t]/cap[t]*x[t] <= 1;", "cap = {0, 2, 4};")
        )
        np.testing.assert_array_equal(result[4].toarray()[:, :3], np.eye(3))

    def test_a_non_integer_index_is_an_error(self):
        with self.assertRaises(SystemExit):
            compile_model(model_text("c : x[t/2] <= 1;"))

    def test_a_where_inside_a_sum_without_variables_is_applied(self):
        result, _ = compile_model(
            model_text(
                "c : x[t] <= sum(cap[i] for i in [0:T-1] where i != t);",
                "cap = {1, 2, 4};",
            )
        )
        np.testing.assert_array_equal(result[5], [6.0, 5.0, 3.0])

    def test_a_sum_inside_the_where_of_a_sum_keeps_its_own_letter(self):
        result, _ = compile_model(
            model_text(
                "c : x[t] <= sum(cap[i] for i in [0:T-1] where sum(cap[j]*cap[i] for j in [0:T-1]) > 7);",
                "cap = {1, 2, 4};",
            )
        )
        np.testing.assert_array_equal(result[5], [6.0, 6.0, 6.0])

    def test_a_read_outside_a_table_in_the_range_of_a_sum_drops_the_row(self):
        result, _ = compile_model(
            model_text(
                "c : x[t] <= sum(cap[i] for i in [0:n[t+1]]);",
                "cap = {1, 2, 4};\nn = {0, 1, 2};",
            )
        )
        np.testing.assert_array_equal(result[5], [3.0, 7.0])

    def test_a_read_outside_a_table_in_the_condition_of_the_rows_is_an_error(self):
        with self.assertRaises(SystemExit):
            compile_model(model_text("c : x[t] <= 1 where n[t+1] > 0;", "n = {0, 1, 2};"))

    def test_a_variable_raised_to_zero_is_one(self):
        result, _ = compile_model(model_text("c : x[t]**0 + y[t] <= 3;"))
        np.testing.assert_array_equal(result[5], [2.0, 2.0, 2.0])

    def test_a_variable_raised_to_a_parameter_worth_zero_is_one(self):
        result, _ = compile_model(model_text("c : x[t]**p + y[t] <= 3;", "p = 0;"))
        np.testing.assert_array_equal(result[5], [2.0, 2.0, 2.0])


class ParameterEvaluationTests(unittest.TestCase):
    """Parameters are evaluated with the symbolic expressions the constraints are built from."""

    def test_a_parameter_can_read_a_parameter_declared_after_it(self):
        result, _ = compile_model(model_text("c : x[t] <= a;", "a = b + 1;\nb = 2;"))
        np.testing.assert_array_equal(result[5], [3.0, 3.0, 3.0])

    def test_a_parameter_defined_in_terms_of_itself_is_refused(self):
        with self.assertRaises(SystemExit):
            compile_model(model_text("c : x[t] <= a;", "a = b;\nb = a;"))

    def test_a_read_outside_a_table_is_refused(self):
        with self.assertRaises(SystemExit):
            compile_model(model_text("c : x[t] <= r;", "p = {1, 2};\nr = p[5];"))

    def test_a_sum_whose_condition_keeps_nothing_is_zero(self):
        result, _ = compile_model(
            model_text(
                "c : x[t] <= s;",
                "p = {1, 2, 3};\ns = sum(p[i] for i in [0:2] where i > 10);",
            )
        )
        np.testing.assert_array_equal(result[5], [0.0, 0.0, 0.0])

    def test_tables_combine_entry_by_entry(self):
        result, _ = compile_model(
            model_text("c : x[t] <= q[t];", "p = {1, 2, 3};\nr = {4, 5, 6};\nq = p*r - p;")
        )
        np.testing.assert_array_equal(result[5], [3.0, 8.0, 15.0])

    def test_a_table_can_list_other_parameters(self):
        result, _ = compile_model(
            model_text("c : x[t] <= v[t];", "s = 2;\np = {1, 2, 3};\nv = {s, p[0] + 1, 4};")
        )
        np.testing.assert_array_equal(result[5], [2.0, 2.0, 4.0])

    def test_a_table_of_expressions_follows_a_changed_value(self):
        factors, values = factorize_model(
            model_text("c : x[t] <= v[t];", "p = 1;\nv = {p, 2*p, 3};")
        )
        factor = factors["c"]
        factor.extend({**values, "A.p": 5.0}, {})
        np.testing.assert_array_equal(factor.independent_terms, [5.0, 10.0, 3.0])

    def test_a_sum_letter_naming_a_parameter_is_refused(self):
        with self.assertRaises(SystemExit):
            compile_model(
                model_text(
                    "c : x[t] <= s;",
                    "s = sum(p[i] for i in [0:2]);\np = {1, 2, 3};\ni = 4;",
                )
            )

    def test_every_parameter_keeps_its_value(self):
        result, _ = compile_model(
            model_text("c : x[t] <= q[t];", "p = {1, 2, 3};\nq = p * 2;")
        )
        parameters = result[0].get_nodes()[0].get_parameter_dict()
        self.assertEqual(parameters["q"].get_value(), [2.0, 4.0, 6.0])


class DeterminismTests(unittest.TestCase):
    def test_the_hash_seed_does_not_change_the_matrices(self):
        path = write_model(
            model_text(
                "c : sum(cap[i]*x[i] for i in [0:t]) * sum(y[j] + cap[j] for j in [0:t]) + x[t] <= 1;",
                "cap = {1.1, 2.3, 4.7};",
            )
        )
        script = (
            "import sys, io, contextlib, numpy as np; from gboml.compiler.gboml_compiler import compile_gboml\n"
            "with contextlib.redirect_stdout(io.StringIO()): r = compile_gboml(sys.argv[1])\n"
            "q = r[6]; order = np.lexsort(q.coords[::-1])\n"
            "print(r[4].toarray().tobytes().hex(), q.data[order].tobytes().hex(), r[5].tobytes().hex())"
        )
        outputs = {
            subprocess.run(
                [sys.executable, "-c", script, path],
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONHASHSEED": seed},
            ).stdout
            for seed in ("1", "2", "3")
        }
        self.assertEqual(len(outputs), 1)


if __name__ == "__main__":
    unittest.main()
