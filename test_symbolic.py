import math
import pickle
import time
import unittest
from functools import reduce
from operator import add, mul

from gboml.compiler.classes.symbolic import (
    Add,
    Idx,
    Indexed,
    Number,
    Operation,
    ParameterSymbol,
    Range,
    Sum,
    VariableSymbol,
)


class ExpressionTests(unittest.TestCase):
    """The rules applied while building an expression, section 5.2 of SHARED_GRAPH_PLAN.md."""

    def setUp(self):
        self.table = {}
        self.t = Idx("t", self.table)
        self.u = Idx("u", self.table)
        self.i = Idx(("sum", 0), self.table)
        self.p = ParameterSymbol("A.p", (), self.table)
        self.q = ParameterSymbol("A.q", (), self.table)
        self.cap = ParameterSymbol("A.cap", (10,), self.table)
        self.x = VariableSymbol(0, 3, "A.x", self.table)

    def number(self, value):
        return Number(value, self.table)

    def expression(self, operand):
        return operand if not isinstance(operand, (int, float)) else self.number(operand)

    def plus(self, *operands):
        return reduce(add, map(self.expression, operands))

    def minus(self, left, right):
        return self.expression(left) - self.expression(right)

    def times(self, *operands):
        return reduce(mul, map(self.expression, operands))

    def read(self, base, index):
        return Indexed(base, (self.expression(index),))

    def value_range(self, index, end, condition=None):
        return Range(index, self.number(0), self.expression(end), self.number(1), condition)

    def test_a_read_at_a_fixed_position_counts_its_reads_of_missing_entries_while_built(self):
        self.assertEqual(self.read(self.x, 2).missing_entry_count, self.number(0))
        self.assertEqual(self.read(self.x, 3).missing_entry_count, self.number(1))
        self.assertNotIsInstance(self.read(self.x, self.t).missing_entry_count, Number)

    def test_a_replacement_of_another_shape_is_refused(self):
        with self.assertRaises(SystemExit):
            self.times(self.p, self.q).replace({self.p: self.cap})

    def test_numbers_fold(self):
        self.assertEqual(self.plus(2, 3), self.number(5))
        self.assertEqual(self.times(2, 3), self.number(6))
        self.assertEqual(self.number(3) / self.number(2), self.number(1.5))
        self.assertEqual(self.number(2) ** self.number(3), self.number(8))
        self.assertEqual(Operation("mod", (self.number(7), self.number(3))), self.number(1))
        self.assertEqual(Operation("<", (self.number(1), self.number(2))), self.number(1))

    def test_leaves_are_compared_by_value_and_never_stored(self):
        self.assertEqual(self.number(2), self.number(2.0))
        self.assertEqual(ParameterSymbol("A.p", (), {}), self.p)
        self.assertEqual(self.plus(self.p, 0), self.p)
        self.assertEqual(self.table, {})

    def test_parameters_are_never_folded(self):
        self.assertIsInstance(self.times(self.p, 1), ParameterSymbol)
        self.assertIsInstance(self.plus(self.p, self.q), Add)

    def test_equivalent_affine_indexes_give_one_read(self):
        written = self.minus(self.plus(self.times(1, self.t), 1), 2)
        self.assertIs(self.read(self.x, written), self.read(self.x, self.minus(self.t, 1)))

    def test_operand_order_does_not_matter(self):
        self.assertIs(self.plus(self.t, 1), self.plus(1, self.t))
        self.assertIs(self.times(self.p, self.q), self.times(self.q, self.p))
        self.assertIs(self.plus(self.p, self.q, self.t), self.plus(self.t, self.q, self.p))

    def test_terms_are_in_a_fixed_order(self):
        written = self.plus(self.q, self.p)
        self.assertEqual([term for term, _ in written.terms], [self.p, self.q])

    def test_a_number_is_distributed_into_an_addition(self):
        self.assertIs(self.times(2, self.plus(self.t, 1)), self.plus(self.times(2, self.t), 2))

    def test_an_expression_is_not_distributed_into_an_addition(self):
        self.assertNotIsInstance(self.times(self.p, self.plus(self.t, 1)), Add)

    def test_number_coefficients_are_gathered(self):
        self.assertIs(self.times(2, self.times(3, self.p)), self.times(6, self.p))
        self.assertIs(self.plus(self.times(2, self.p), self.times(3, self.p)), self.times(5, self.p))

    def test_cancelled_terms_disappear(self):
        self.assertEqual(self.minus(self.p, self.p), self.number(0))
        self.assertEqual(self.minus(self.plus(self.p, self.t), self.p), self.t)

    def test_a_product_holding_zero_is_zero(self):
        self.assertEqual(self.times(0, self.p), self.number(0))

    def test_divisions_and_powers(self):
        self.assertIs(self.p / self.q, self.times(self.p, self.q ** -1))
        self.assertEqual(self.times(self.p / self.q, self.q), self.p)
        self.assertIs(self.p ** 2, self.times(self.p, self.p))
        self.assertEqual(self.p ** 0, self.number(1))
        self.assertIsInstance(self.p ** self.q, Operation)
        self.assertIs(self.p / 2, self.times(0.5, self.p))

    def test_negation(self):
        self.assertEqual(-(-self.p), self.p)
        self.assertIs(-self.p, self.times(-1, self.p))

    def test_a_written_chain_creates_one_expression(self):
        leaves = [ParameterSymbol(f"A.a{k}", (), self.table) for k in range(1000)]
        chain = Add([(leaf, 1) for leaf in leaves])
        self.assertEqual(len(chain.terms), 1000)
        self.assertEqual(len(self.table), 1)

    def test_free_indexes(self):
        self.assertEqual(self.read(self.cap, self.minus(self.t, 1)).free_indexes, {self.t})
        condition = Operation("!=", (self.i, self.t))
        value_range = self.value_range(self.i, self.t, condition)
        self.assertEqual(value_range.free_indexes, {self.t})
        self.assertEqual(Sum(self.read(self.cap, self.i), value_range).free_indexes, {self.t})
        fixed_range = self.value_range(self.i, 2)
        self.assertEqual(Sum(self.read(self.cap, self.i), fixed_range).free_indexes, set())

    def test_parameters_read(self):
        expression = self.times(self.p, self.read(self.cap, self.t))
        self.assertEqual(expression.parameters_read, {self.p, self.cap})

    def test_a_sum_moves_out_the_factors_that_do_not_read_its_index(self):
        value_range = self.value_range(self.i, self.t)
        cap_i, cap_t = self.read(self.cap, self.i), self.read(self.cap, self.t)
        summed = Sum(cap_i, value_range)
        self.assertIsInstance(summed, Sum)
        self.assertIs(Sum(self.times(cap_i, cap_t), value_range), self.times(cap_t, summed))
        self.assertIs(Sum(self.times(2, cap_i), value_range), self.times(2, summed))
        count = Sum(self.number(1), value_range)
        self.assertIs(Sum(self.p, value_range), self.times(self.p, count))

    def test_identical_sums_are_one_expression(self):
        def build():
            return Sum(self.read(self.cap, self.i), self.value_range(self.i, self.t))
        self.assertIs(build(), build())

    def test_replace_renames_an_index(self):
        expression = self.times(self.read(self.cap, self.minus(self.t, 1)), self.p)
        expected = self.times(self.read(self.cap, self.minus(self.u, 1)), self.p)
        self.assertIs(expression.replace({self.t: self.u}), expected)

    def test_replace_renames_the_index_of_a_range(self):
        j = Idx(("term", 0), self.table)

        def build(index):
            return self.value_range(index, self.t, Operation("<", (index, self.t)))

        self.assertIs(build(self.i).replace({self.i: j}), build(j))

    def test_replace_renames_the_letter_of_a_sum_everywhere(self):
        def build(letter, end):
            return Sum(self.read(self.cap, letter), self.value_range(letter, end))

        replaced = build(self.i, self.t).replace({self.i: self.u, self.t: self.p})
        self.assertIs(replaced, build(self.u, self.p))

    def test_replace_replaces_a_parameter(self):
        expression = self.times(self.p, self.read(self.cap, self.t))
        replaced = expression.replace({self.p: self.times(2, self.q)})
        self.assertIs(replaced, self.times(2, self.q, self.read(self.cap, self.t)))

    def test_replace_visits_a_shared_expression_once(self):
        def build(index):
            expression = index
            for _ in range(200):
                expression = Operation("mod", (expression, expression))
            return expression

        start = time.perf_counter()
        self.assertIs(build(self.t).replace({self.t: self.u}), build(self.u))
        self.assertLess(time.perf_counter() - start, 1.0)

    def test_derivative_follows_the_product_rule(self):
        expression = self.plus(self.times(self.p, self.p, self.q), self.times(3, self.p))
        self.assertIs(expression.derivative(self.p), self.plus(self.times(2, self.p, self.q), 3))
        self.assertEqual(expression.derivative(ParameterSymbol("A.r", (), self.table)), self.number(0))

    def test_the_derivative_of_a_power_follows_its_base_and_its_exponent(self):
        power = self.plus(1, self.p) ** self.q
        values = {"A.p": 0.07, "A.q": 25.0}
        self.assertAlmostEqual(power.derivative(self.p).evalf(values), 25 * 1.07**24)
        self.assertAlmostEqual(power.derivative(self.q).evalf(values), 1.07**25 * math.log(1.07))

    def test_the_derivative_with_respect_to_an_entry_is_one_at_its_position(self):
        entry = self.read(self.cap, 5)
        self.assertEqual(self.read(self.cap, self.t).derivative(entry), Operation("==", (self.t, self.number(5))))
        self.assertEqual(entry.derivative(entry), self.number(1))
        self.assertEqual(self.read(self.cap, 4).derivative(entry), self.number(0))

    def test_derivative_refuses_a_parameter_in_a_range(self):
        summed = Sum(self.times(self.p, self.read(self.cap, self.i)), self.value_range(self.i, self.p))
        with self.assertRaises(SystemExit):
            summed.derivative(self.p)

    def test_an_expression_survives_pickling(self):
        condition = Operation("!=", (self.i, self.t))
        expression = Sum(self.read(self.cap, self.i), self.value_range(self.i, self.t, condition))
        copied = pickle.loads(pickle.dumps(expression))
        self.assertEqual(repr(copied), repr(expression))
        self.assertEqual(copied.free_indexes, expression.free_indexes)
        self.assertEqual(repr(copied.range.unconditioned), repr(expression.range.unconditioned))


if __name__ == "__main__":
    unittest.main()
