# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).
from collections.abc import Iterable

import numpy as np  # type: ignore
from scipy.sparse import coo_array  # type: ignore

from .symbolic import (
    Add,
    Context,
    Expr,
    Idx,
    Indexed,
    Number,
    ParameterSymbol,
    Range,
    Sum,
    check_parameter_values,
)

RowMask = np.ndarray[tuple[int], np.dtype[np.bool_]]


class Term:
    """
    The sum, over every position of `ranges`, of the product of `variables`
    scaled by `coefficient`. `ranges` are outermost first, and the index of each
    is named after its position, the innermost being 0. The number of variables
    is the degree of the term, and a term without variables has no ranges either.

    During one evaluation, the term also holds its rectangle of entries over the
    rows: `values`, one array of `columns` per variable, the `widths` of its ranges,
    and `filled`, false on the entries a `where` condition rejected and on the
    ones a ragged range does not reach.
    """

    __slots__ = (
        "ranges",
        "variables",
        "coefficient",
        "values",
        "columns",
        "widths",
        "filled",
    )

    def __init__(
        self,
        ranges: tuple[Range, ...],
        variables: Iterable[Indexed],
        coefficient: Expr,
    ) -> None:
        self.ranges = ranges
        # a product does not depend on the order of its variables, so x*y and y*x are one key
        self.variables = tuple(
            sorted(variables, key=lambda variable: variable.sort_key)
        )
        self.coefficient = coefficient
        self.values = None
        self.columns = None
        self.widths = None
        self.filled = None

    @property
    def key(self) -> tuple:
        return self.ranges, self.variables

    @property
    def sort_key(self) -> tuple:
        return (
            tuple(variable.sort_key for variable in self.variables),
            tuple(value_range.sort_key for value_range in self.ranges),
        )

    def degree(self) -> int:
        """The number of variables multiplied together in every entry."""
        return len(self.variables)

    def scaled(self, factor: Expr) -> "Term":
        return Term(self.ranges, self.variables, self.coefficient * factor)

    def negated(self) -> "Term":
        return Term(self.ranges, self.variables, -self.coefficient)

    def replace(self, replacements: dict) -> "Term":
        return Term(
            tuple(value_range.replace(replacements) for value_range in self.ranges),
            (variable.replace(replacements) for variable in self.variables),
            self.coefficient.replace(replacements),
        )

    def times(self, other: "Term") -> "Term":
        """The product of two terms of degree one, the one sorting first placed outside, so their order never matters."""
        left, right = sorted((self, other), key=lambda term: term.sort_key)
        nb_right_ranges = len(right.ranges)
        if nb_right_ranges and left.ranges:
            table = left.coefficient.table
            left = left.replace(
                {
                    Idx(("term", position), table): Idx(
                        ("term", position + nb_right_ranges), table
                    )
                    for position in range(len(left.ranges))
                }
            )
        return Term(
            left.ranges + right.ranges,
            left.variables + right.variables,
            left.coefficient * right.coefficient,
        )

    def summed_over(self, value_range: Range) -> "Term":
        """The term of a sum: a term reading its index keeps its range, any other takes the sum in its coefficient."""
        index = value_range.index
        if not any(
            index in expression.free_indexes
            for expression in self.variables + self.ranges
        ):
            return Term(
                self.ranges, self.variables, Sum(self.coefficient, value_range)
            )
        widened = Term((value_range,) + self.ranges, self.variables, self.coefficient)
        return widened.replace(
            {index: Idx(("term", len(self.ranges)), value_range.table)}
        )

    @property
    def reads_out_of_range(self) -> Expr:
        """The number of reads out of range in the term, over every position of its ranges."""
        count = Add(
            [(expression.reads_out_of_range, 1) for expression in (self.coefficient, *self.variables)]
        )
        for value_range in reversed(self.ranges):
            count = value_range.reads_out_of_range + Sum(count, value_range)
        return count

    def evaluate_over(self, row_context: Context) -> None:
        """Stores the rectangle of the term over the rows of the context."""
        context = row_context
        for value_range in self.ranges:
            context = context.widened(value_range, *value_range.evaluate_over(context))

        # the coefficient keeps the dtype of what it was computed from, often integers
        self.values = self.coefficient.evaluate_over(context).astype(float, copy=False)
        self.columns = tuple(variable.evaluate_over(context) for variable in self.variables)
        self.widths = context.shape[context.nb_row_axes :]
        self.filled = (
            np.ones((1,) * self.values.ndim, dtype=bool)
            if context.filled is None
            else context.filled
        )
        assert self.values.ndim > len(self.widths) and all(
            array.ndim == self.values.ndim for array in (self.filled, *self.columns)
        )

    def free_evaluation(self) -> None:
        """Releases the rectangle of the term, once assembly has read it."""
        self.values = None
        self.columns = None
        self.widths = None
        self.filled = None


class Block:
    """
    A polynomial of degree at most two in the variables: its constant term,
    and its terms of degree one and two by key, no two of them sharing a key.
    """

    __slots__ = ("constant", "linear", "quadratic")

    def __init__(
        self,
        constant: Term,
        linear: list[Term],
        quadratic: list[Term],
    ) -> None:
        self.constant = constant
        self.linear = {}
        self.quadratic = {}
        # terms sharing a key are added up once, and those adding up to zero are left out
        for stored, input_terms in ((self.linear, linear), (self.quadratic, quadratic)):
            for term in input_terms:
                if term.key in stored:
                    term = Term(
                        term.ranges,
                        term.variables,
                        stored[term.key].coefficient + term.coefficient,
                    )
                if term.coefficient.is_zero:
                    stored.pop(term.key, None)
                else:
                    stored[term.key] = term

    @staticmethod
    def of_constant(expression: Expr) -> "Block":
        return Block(Term((), (), expression), [], [])

    @staticmethod
    def of_variable(variable: Indexed) -> "Block":
        return Block(
            Term((), (), Number(0, variable.table)),
            [Term((), (variable,), Number(1, variable.table))],
            [],
        )

    @property
    def terms(self) -> list[Term]:
        return [self.constant, *self.linear.values(), *self.quadratic.values()]

    @property
    def reads_out_of_range(self) -> Expr:
        return Add([(term.reads_out_of_range, 1) for term in self.terms])

    def degree(self) -> int:
        if self.quadratic:
            return 2
        if self.linear:
            return 1

        return 0

    def negated(self) -> "Block":

        return Block(
            self.constant.negated(),
            [term.negated() for term in self.linear.values()],
            [term.negated() for term in self.quadratic.values()],
        )

    def scaled(self, factor: Expr) -> "Block":
        """The block with every coefficient multiplied by `factor`, which holds no variable."""

        if factor.is_zero:
            return Block.of_constant(factor)
        return Block(
            self.constant.scaled(factor),
            [term.scaled(factor) for term in self.linear.values()],
            [term.scaled(factor) for term in self.quadratic.values()],
        )

    def summed_over(self, value_range: Range) -> "Block":
        """The block of a sum, whose terms can meet on one key once their ranges are renamed."""
        return Block(
            self.constant.summed_over(value_range),
            [term.summed_over(value_range) for term in self.linear.values()],
            [term.summed_over(value_range) for term in self.quadratic.values()],
        )

    def add(self, other: "Block") -> "Block":
        return Block(
            Term((), (), self.constant.coefficient + other.constant.coefficient),
            [*self.linear.values(), *other.linear.values()],
            [*self.quadratic.values(), *other.quadratic.values()],
        )

    def times(self, other: "Block") -> "Block":
        """The product: a block of degree zero scales the other, and two degree one parts give the degree two part."""
        if other.degree() == 0:
            return self.scaled(other.constant.coefficient)
        if self.degree() == 0:
            return other.scaled(self.constant.coefficient)

        left_constant = self.constant.coefficient
        right_constant = other.constant.coefficient
        return Block(
            self.constant.scaled(right_constant),
            [
                *(term.scaled(right_constant) for term in self.linear.values()),
                *(term.scaled(left_constant) for term in other.linear.values()),
            ],
            [
                left.times(right)
                for left in self.linear.values()
                for right in other.linear.values()
            ],
        )

    def replace(self, replacements: dict) -> "Block":
        """The block with every index or parameter of `replacements` replaced, which can make two keys one."""
        return Block(
            self.constant.replace(replacements),
            [term.replace(replacements) for term in self.linear.values()],
            [term.replace(replacements) for term in self.quadratic.values()],
        )

    def derivative(self, target: ParameterSymbol | Indexed) -> "Block":
        """The block whose coefficients are the derivatives of these ones with respect to `target`."""
        return Block(
            Term((), (), self.constant.coefficient.derivative(target)),
            [
                Term(term.ranges, term.variables, term.coefficient.derivative(target))
                for term in self.linear.values()
            ],
            [
                Term(term.ranges, term.variables, term.coefficient.derivative(target))
                for term in self.quadratic.values()
            ],
        )

    def evaluate_over(
        self,
        rows: Range,
        reads_out_of_range: Expr,
        parameter_values: dict,
        cache: dict,
        line: int,
    ) -> tuple[np.ndarray, RowMask]:
        """
        The values the row index takes, and the rows where `reads_out_of_range`
        is zero. Every term keeps its evaluation until `assemble` reads it.
        """
        terms = self.terms
        # checked once here, so that every read of a value in the evaluation trusts it
        check_parameter_values(
            rows.parameters_read.union(
                reads_out_of_range.parameters_read,
                *(
                    expression.parameters_read
                    for term in terms
                    for expression in (term.coefficient, *term.ranges, *term.variables)
                ),
            ),
            parameter_values,
        )
        # the rows are evaluated in a context without any axis, which they then widen
        context = Context(parameter_values, (), {}, cache, line=line)
        row_values, filled = rows.evaluate_over(context)
        row_index_values = row_values[filled]
        nb_rows = len(row_index_values)
        if nb_rows == 0:
            return row_index_values, np.zeros(0, dtype=bool)

        context = context.widened(rows, row_index_values, np.ones(nb_rows, dtype=bool))
        valid = np.broadcast_to(reads_out_of_range.evaluate_over(context) == 0, (nb_rows,))
        for term in terms:
            term.evaluate_over(context)
        return row_index_values, valid

    def assemble(self, valid: RowMask) -> tuple:
        """
        Over the valid rows, the constant and the degree one and two parts as
        sparse arrays, a part holding no entry being None.
        """
        if not valid.any():
            for term in self.terms:
                term.free_evaluation()
            return np.zeros(0), None, None

        nb_rows = len(valid)
        constant = np.broadcast_to(self.constant.values, (nb_rows,))[valid]
        self.constant.free_evaluation()
        nb_kept_rows = np.count_nonzero(valid)
        linear = _build_sparse_array(
            list(self.linear.values()), valid, nb_kept_rows, nb_rows
        )
        quadratic = _build_sparse_array(
            list(self.quadratic.values()), valid, nb_kept_rows, nb_rows
        )
        return constant, linear, quadratic


def _gather_nonzero_terms(
    terms: list[Term], valid: RowMask, nb_rows: int
) -> tuple[np.ndarray, ...]:
    """
    _gather_nonzero_terms function : turns a list of terms into flat arrays,
    keeping only the entries whose value is not zero.

    Terms are handled one at a time. For each term, its values are brought to
    the rows and widths they cover and reduced to the positions that are not
    zero; the term's columns are then read only at those same positions. Each
    term is freed once read, so that what it holds is released before the
    sparse array is built.
    INPUT:  terms -> non empty list of evaluated Term objects, all of the same
                     degree, freed by the call
            valid -> ndarray, the rows to keep
            nb_rows -> number of rows the terms were evaluated at, before
                       valid narrows them down
    OUTPUT: the rows array, then the values and one array per column of the
            terms, all of the same length
    """

    keep_every_row = bool(valid.all())
    kept_row_position = None if keep_every_row else np.cumsum(valid) - 1

    rows = [np.zeros(0, dtype=int)]
    values = [np.zeros(0, dtype=float)]
    columns = [[np.zeros(0, dtype=int)] for _ in range(terms[0].degree())]

    for term in terms:
        shape = (nb_rows,) + term.widths
        term_values = np.broadcast_to(term.values, shape)

        live = term_values != 0.0
        if not keep_every_row:
            # remove rows that should be excluded
            valid_rows = valid.reshape((nb_rows,) + (1,) * len(term.widths))
            live &= valid_rows
        if not term.filled.all():
            # remove ragged terms in the width
            filled = np.broadcast_to(term.filled, shape)
            live &= filled

        # one array per axis, the first of them being the row an entry lands on
        kept = np.nonzero(live)
        if len(kept[0]) > 0:
            rows.append(kept[0] if keep_every_row else kept_row_position[kept[0]])
            values.append(term_values[kept])
            for gathered_column, term_column in zip(columns, term.columns):
                entry_columns = np.broadcast_to(term_column, shape)[kept]
                gathered_column.append(entry_columns)
        term.free_evaluation()

    return (
        np.concatenate(rows),
        np.concatenate(values),
        *(np.concatenate(column) for column in columns),
    )


def _build_sparse_array(
    terms: list[Term], valid: RowMask, nb_kept_rows: int, nb_rows: int
) -> coo_array | None:
    """
    _build_sparse_array function : build the terms of a block as a coo_array
    whose first axis is the row and whose other axes are the columns of the
    terms, one for the degree one part and two for the degree two one. Entries
    at the same (row, *columns) tuple in more than one term are summed and
    cancelled by sum_duplicates/eliminate_zeros.
    INPUT:  terms -> list of evaluated Term objects, all of the same degree
            valid -> mask over the rows of the block
            nb_kept_rows -> number of rows the mask keeps
            nb_rows -> number of rows the terms were evaluated at
    OUTPUT: coo_array of shape (nb_kept_rows, *ncols), or None when `terms` is
            empty
    """

    if not terms:
        return None

    degree = terms[0].degree()
    assert degree > 0 and all(term.degree() == degree for term in terms)

    term_rows, values, *columns = _gather_nonzero_terms(terms, valid, nb_rows)
    if len(columns) == 2:
        # a product is commutative so x * y and y * x are the very same entry.
        columns = [np.minimum(*columns), np.maximum(*columns)]
    # indices are never negative, so a column's max is its ncols - 1
    # whenever it holds any, and the initial floor of 0 gives ncols = 1 when it
    # does not
    shape = (nb_kept_rows,) + tuple(
        int(column.max(initial=0)) + 1 for column in columns
    )
    sparse = coo_array((values, (term_rows, *columns)), shape=shape)
    sparse.sum_duplicates()  # TODO: keep it or no ??
    sparse.eliminate_zeros()

    return sparse
