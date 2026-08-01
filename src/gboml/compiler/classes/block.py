# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).

"""
block.py : the value a factorized expression takes once the index of the
constraint is bound.

A Block is the affine expression an expression reduces to, evaluated over every
row the constraint expands to at once. It holds

    constant   the part which does not depend on the variables, per row
    linear     the terms, each carrying values and the columns they sit on
    valid      the rows whose variable indexes all landed in bounds

A *term* is a rectangle of `width` entries per row : `values` and `columns`
broadcastable to `(nb_rows, width)`. Keeping the terms in a list rather than in
one array is what lets a coefficient that does not depend on the index stay a
single python float all the way to the final assembly, where the flat arrays
are allocated once.

Summing is not an operation here. Terms landing on the same row of the matrix
are summed by the matrix itself, so a sum over an index only has to widen the
terms it encloses -- see SumNode in factorization.py.
"""

import numpy as np  # type: ignore


class ByteArray:
    def __init__(self, length):
        self.array = np.ones(length, dtype=bool)

    def intersect_inplace(self, values):
        self.array = self.array & values

    def reset(self):
        self.array = np.ones(len(self.array), dtype=bool)

def __f_index2__(expr, all_index, name):
    max_length = len(name)
    is_in = np.array(((0 <= expr) & (expr < max_length)), dtype=bool)
    all_index.array = all_index.array & is_in
    expr = np.array(expr, dtype=int)

    if not is_in.all():
        return_value = np.array(name[np.clip(expr, 0, max_length - 1)], dtype=float)
    else:
        return_value = np.array(name[expr], dtype=float)

    return return_value

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

    return np.ascontiguousarray(np.broadcast_to(array, (nb_rows, width)),
                                dtype=dtype)


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

    __slots__ = ("values", "columns", "width")

    def __init__(self, values, columns, width):

        self.values = values
        self.columns = columns
        self.width = width

    def scaled(self, factor):

        return LinearTerm(self.values * factor, self.columns, self.width)

    def negated(self):

        return LinearTerm(-self.values, self.columns, self.width)


class Block:
    """
    The polynomial an expression reduces to over `nb_rows` rows. See the module
    docstring.
    """

    __slots__ = ("nb_rows", "constant", "linear", "valid")

    def __init__(self, nb_rows, constant=0.0, linear=None, valid=None):

        self.nb_rows = nb_rows
        self.constant = constant
        self.linear = linear if linear is not None else []
        self.valid = valid

    def is_constant(self) -> bool:
        """Whether the expression holds no variable at all."""

        return not self.linear

    def also_valid(self, valid):
        """The same block, with one more row mask intersected into its own."""

        self.valid = combine_valid(self.valid, valid)

        return self

    def negated(self):

        return Block(self.nb_rows, -self.constant,
                     [term.negated() for term in self.linear],
                     self.valid)

    def scaled(self, factor):
        """
        scaled function : multiplies the whole polynomial by a value which does
                          not depend on the variables, one per row at most
        INPUT:  factor -> scalar or ndarray of shape (nb_rows,)
        OUTPUT: Block
        """

        row_factor = as_column(factor, self.nb_rows)

        return Block(self.nb_rows, self.constant * factor,
                     [term.scaled(row_factor) for term in self.linear],
                     self.valid)

    def linear_width(self) -> int:

        return sum(term.width for term in self.linear)

def add_blocks(blocks):
    """
    add_blocks function : sums any number of blocks at once. Terms are gathered
    in a single list rather than concatenated pairwise, which would copy the
    accumulated arrays once per operand.
    INPUT:  blocks -> list of Block objects, all over the same rows
    OUTPUT: Block
    """

    if len(blocks) == 1:

        return blocks[0]

    nb_rows = blocks[0].nb_rows
    constant = 0.0
    linear = []
    valid = None
    for block in blocks:

        constant = constant + block.constant
        linear += block.linear
        valid = combine_valid(valid, block.valid)

    return Block(nb_rows, constant, linear, valid)


def flatten_linear(block):
    """
    flatten_linear function : gathers the degree one terms of a block in the
    flat (values, columns) pair a sparse matrix is built from. The result is
    allocated once, at its final size, and every term is written into its own
    slice of it.
    INPUT:  block -> Block object
    OUTPUT: values, columns of shape (nb_rows, total width)
    """

    return _flatten(block.nb_rows, block.linear,
                    [["values", float], ["columns", int]])


def _flatten(nb_rows, terms, fields):

    total_width = sum(term.width for term in terms)
    arrays = [np.empty((nb_rows, total_width), dtype=dtype)
              for _, dtype in fields]
    offset = 0
    for term in terms:

        end = offset + term.width
        for array, [name, dtype] in zip(arrays, fields):

            value = getattr(term, name)
            if is_scalar(value):
                array[:, offset:end] = value
            else:
                array[:, offset:end] = np.asarray(value, dtype=dtype)
        offset = end

    return arrays
