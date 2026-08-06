# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).

"""
block.py : the value a factorized expression takes once the index of the
constraint is bound.

A Block is the polynomial an expression reduces to, evaluated over every row
the constraint expands to at once. It holds

    constant   the degree zero part, one value per row
    linear     the degree one part, a list of terms
    quadratic  the degree two part, a list of terms
    valid      the rows whose variable indexes all landed in bounds

A *term* is a rectangle of `width` entries per row : `values` and `columns`
(or `columns_i`/`columns_j` for a degree two term) broadcastable to
`(nb_rows, width)`. Keeping the terms in a list rather than in one array is
what lets a coefficient that does not depend on the index stay a single
python float all the way to the final assembly, where the flat arrays are
allocated once.
"""

import numpy as np  # type: ignore


def is_scalar(value) -> bool:

    return not isinstance(value, np.ndarray)


def as_rows(value, nb_rows):
    """
    as_rows function : broadcasts a value to one entry per row, leaving a
                       scalar alone so that it stays a scalar
    INPUT:  value -> scalar or ndarray
            nb_rows -> number of rows
    OUTPUT: scalar or ndarray of shape (nb_rows,)
    """

    if is_scalar(value):
        return value

    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    if array.shape == (nb_rows,):
        return array

    return np.broadcast_to(array, (nb_rows,))


def as_column(value, nb_rows):
    """
    as_column function : shapes a per-row value so that it multiplies a
                         (nb_rows, width) rectangle row-wise
    INPUT:  value -> scalar or ndarray of shape (nb_rows,)
            nb_rows -> number of rows
    OUTPUT: scalar or ndarray of shape (nb_rows, 1)
    """

    if is_scalar(value):
        return value

    return np.asarray(value, dtype=float).reshape(nb_rows, 1)


def materialize(value, nb_rows, width, dtype=float):
    """
    materialize function : turns a term entry into a full rectangle, used only
                           where the entries have to be written to
                           individually, so that a scalar entry is kept as a
                           scalar everywhere else
    INPUT:  value -> scalar or ndarray broadcastable to (nb_rows, width)
            nb_rows -> number of rows
            width -> number of entries per row
            dtype -> type of the array returned
    OUTPUT: ndarray of shape (nb_rows, width)
    """

    if is_scalar(value):
        return np.full((nb_rows, width), value, dtype=dtype)

    array = np.asarray(value, dtype=dtype)
    if array.shape == (nb_rows, width):
        return array

    return np.ascontiguousarray(np.broadcast_to(array, (nb_rows, width)), dtype=dtype)


def combine_valid(first, second):
    """
    combine_valid function : intersects two row masks, None standing for the
                             mask that keeps every row
    """

    if first is None:
        return second
    if second is None:
        return first

    return first & second


class LinearTerm:
    """One rectangle of degree one entries : `width` columns per row."""

    __slots__ = ("columns", "values", "width")
    COLUMN_FIELDS = ("columns",)

    def __init__(self, values, columns, width):

        self.values = values
        self.columns = columns
        self.width = width

    def scaled(self, factor):

        return LinearTerm(self.values * factor, self.columns, self.width)

    def negated(self):

        return LinearTerm(-self.values, self.columns, self.width)


class QuadraticTerm:
    """One rectangle of degree two entries, carrying the two columns whose
    variables are multiplied together."""

    __slots__ = ("columns_i", "columns_j", "values", "width")
    COLUMN_FIELDS = ("columns_i", "columns_j")

    def __init__(self, values, columns_i, columns_j, width):

        self.values = values
        self.columns_i = columns_i
        self.columns_j = columns_j
        self.width = width

    def scaled(self, factor):

        return QuadraticTerm(
            self.values * factor, self.columns_i, self.columns_j, self.width
        )

    def negated(self):

        return QuadraticTerm(-self.values, self.columns_i, self.columns_j, self.width)


class Block:
    """
    The polynomial an expression reduces to over `nb_rows` rows. See the module
    docstring.
    """

    __slots__ = ("nb_rows", "constant", "linear", "quadratic", "valid")

    def __init__(self, nb_rows, constant=0.0, linear=None, quadratic=None, valid=None):

        self.nb_rows = nb_rows
        self.constant = constant
        self.linear = linear if linear is not None else []
        self.quadratic = quadratic if quadratic is not None else []
        self.valid = valid

    def degree(self) -> int:
        """The degree of the expression with respect to the variables."""

        if self.quadratic:
            return 2
        if self.linear:
            return 1

        return 0

    def negated(self):

        return Block(
            self.nb_rows,
            -self.constant,
            [term.negated() for term in self.linear],
            [term.negated() for term in self.quadratic],
            self.valid,
        )

    def scaled(self, factor):
        """
        scaled function : multiplies the whole polynomial by a value which does
                          not depend on the variables, one per row at most
        INPUT:  factor -> scalar or ndarray of shape (nb_rows,)
        OUTPUT: Block
        """

        row_factor = as_column(factor, self.nb_rows)

        return Block(
            self.nb_rows,
            self.constant * factor,
            [term.scaled(row_factor) for term in self.linear],
            [term.scaled(row_factor) for term in self.quadratic],
            self.valid,
        )


def add_blocks(blocks):
    """
    add_blocks function : sums any number of blocks at once.
    INPUT:  blocks -> list of Block objects, all over the same rows
    OUTPUT: Block
    """

    if len(blocks) == 1:
        return blocks[0]

    nb_rows = blocks[0].nb_rows
    constant = 0.0
    linear = []
    quadratic = []
    valid = None
    for block in blocks:
        constant += block.constant
        linear += block.linear
        quadratic += block.quadratic
        valid = combine_valid(valid, block.valid)

    return Block(nb_rows, constant, linear, quadratic, valid)


def multiply_blocks(left, right):
    """
    multiply_blocks function : multiplies two polynomials, the degrees adding
    up. The degree two part is the outer product of the degree one parts along
    the term axis, which is where the distribution the quadratic factorization
    needs happens.
    INPUT:  left, right -> Block objects over the same rows
    OUTPUT: Block
    """

    nb_rows = left.nb_rows
    valid = combine_valid(left.valid, right.valid)

    # constant times anything only scales it
    if left.degree() == 0:
        return Block(nb_rows, *_scaled_parts(right, left.constant), valid=valid)
    if right.degree() == 0:
        return Block(nb_rows, *_scaled_parts(left, right.constant), valid=valid)

    constant = left.constant * right.constant
    linear = [term.scaled(as_column(right.constant, nb_rows)) for term in left.linear]
    linear += [term.scaled(as_column(left.constant, nb_rows)) for term in right.linear]
    # a degree two part on either side can only be scaled by the constant of
    # the other one, anything else being of degree three and refused by the
    # semantic analysis
    quadratic = [
        term.scaled(as_column(right.constant, nb_rows)) for term in left.quadratic
    ]
    quadratic += [
        term.scaled(as_column(left.constant, nb_rows)) for term in right.quadratic
    ]

    for left_term in left.linear:
        for right_term in right.linear:
            quadratic.append(_outer_product(left_term, right_term, nb_rows))

    return Block(nb_rows, constant, linear, quadratic, valid)


def _scaled_parts(block, factor):
    """The parts of a block scaled by a constant."""

    row_factor = as_column(factor, block.nb_rows)

    return (
        block.constant * factor,
        [term.scaled(row_factor) for term in block.linear],
        [term.scaled(row_factor) for term in block.quadratic],
    )


def _outer_product(left_term, right_term, nb_rows):
    """
    _outer_product function : every entry of one degree one term multiplied by
    every entry of the other, giving a degree two term of the product of both
    widths.
    INPUT:  left_term, right_term -> LinearTerm objects
            nb_rows -> number of rows
    OUTPUT: QuadraticTerm
    """

    if left_term.width == 1 and right_term.width == 1:
        return QuadraticTerm(
            left_term.values * right_term.values,
            left_term.columns,
            right_term.columns,
            1,
        )

    width = left_term.width * right_term.width
    left_values = materialize(left_term.values, nb_rows, left_term.width)
    right_values = materialize(right_term.values, nb_rows, right_term.width)
    left_columns = materialize(left_term.columns, nb_rows, left_term.width, dtype=int)
    right_columns = materialize(
        right_term.columns, nb_rows, right_term.width, dtype=int
    )

    values = left_values[:, :, None] * right_values[:, None, :]
    columns_i = np.broadcast_to(
        left_columns[:, :, None], (nb_rows, left_term.width, right_term.width)
    )
    columns_j = np.broadcast_to(
        right_columns[:, None, :], (nb_rows, left_term.width, right_term.width)
    )

    return QuadraticTerm(
        values.reshape(nb_rows, width),
        columns_i.reshape(nb_rows, width),
        columns_j.reshape(nb_rows, width),
        width,
    )
