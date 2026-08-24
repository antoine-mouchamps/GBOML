# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).
import numpy as np  # type: ignore
from scipy.sparse import coo_array  # type: ignore


def combine_valid(first, second):
    if first is None:
        return second
    if second is None:
        return first

    return first & second


class Term:
    """
    One rectangle of entries, each of them the product of the variables at
    `columns` scaled by `values`. `columns` is a sequence of one column array
    for a degree one term, of two for a degree two one. `filled` is false on
    the entries a `where` condition rejected and on the ones a ragged range
    does not reach. `widths` holds one width per sum the term has come out of,
    outermost first : the entries of a row are laid out over all of them, so a
    row carries prod(widths) of them.
    """

    __slots__ = ("columns", "filled", "values", "widths")

    def __init__(self, values, columns, widths, filled):

        assert values.ndim > len(widths) and all(
            array.ndim == values.ndim for array in (filled, *columns)
        )
        self.values = values
        self.columns = columns
        self.widths = widths
        self.filled = filled

    def degree(self) -> int:
        """The number of variables multiplied together in every entry."""

        return len(self.columns)

    def scaled(self, factor):

        return Term(
            # a factor which does not vary along the widths, which trail
            self.values * factor.reshape(factor.shape + (1,) * len(self.widths)),
            self.columns,
            self.widths,
            self.filled,
        )

    def negated(self):

        return Term(-self.values, self.columns, self.widths, self.filled)

    def widened(self, width, filled):
        """
        The term a sum gives back when it ends. Its range was the innermost
        axis the body was evaluated over and becomes the outermost width,
        which is the very same axis : nothing is moved, and only the mask of
        the index values the sum keeps has to be brought in.
        """

        return Term(
            self.values,
            self.columns,
            (width,) + self.widths,
            self.filled
            if filled is None
            else self.filled & filled.reshape(filled.shape + (1,) * len(self.widths)),
        )


class Block:
    """
    The polynomial an expression reduces to over `nb_rows` rows. See the module
    docstring.
    """

    __slots__ = ("nb_rows", "constant", "linear", "quadratic", "valid")

    def __init__(self, nb_rows, constant, linear=None, quadratic=None, valid=None):
        self.nb_rows = nb_rows
        self.constant = constant
        self.linear = linear if linear is not None else []
        self.quadratic = quadratic if quadratic is not None else []
        self.valid = valid

    @staticmethod
    def of_constant(nb_rows, rank, value, valid=None):
        """
        of_constant function : the block a subexpression holding no variable
                               reduces to. The compiled expression gives back
                               either a single value, whatever the shape it is
                               written in, or one value per position of the
                               rank it was evaluated at
        INPUT:  nb_rows -> number of rows
                rank -> number of axes the expression is evaluated over
                value -> the value the compiled expression evaluated to
                valid -> mask of the rows still valid, or None
        OUTPUT: Block
        """
        value = np.asarray(value, dtype=float)
        assert value.ndim in (0, rank)
        if value.ndim == 0:
            value = value.reshape((1,) * rank)

        return Block(nb_rows, value, valid=valid)

    @staticmethod
    def of_variable(nb_rows, rank, columns, valid=None):
        """
        of_variable function : the block one variable of the problem reduces
                               to, that is a single degree one entry per row
        INPUT:  nb_rows -> number of rows
                rank -> number of axes the expression is evaluated over
                columns -> the column of the variable, one per position or one
                           for all of them
                valid -> mask of the rows still valid, or None
        OUTPUT: Block
        """

        columns = np.asarray(columns, dtype=int)
        assert columns.ndim in (0, rank)
        if columns.ndim == 0:
            columns = columns.reshape((1,) * rank)

        return Block(
            nb_rows,
            np.zeros((1,) * rank),
            [
                Term(
                    np.ones((1,) * rank),
                    (columns,),
                    (),
                    np.ones((1,) * rank, dtype=bool),
                )
            ],
            valid=valid,
        )

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

    def scaled(self, factor, valid=None):
        """
        scaled function : multiplies the whole polynomial by a value which does
                          not depend on the variables.
        INPUT:  factor -> ndarray of the rank the block is evaluated at
                valid -> the mask to carry instead of the block's own, for a
                         caller which has already combined it with the mask of
                         wherever the factor came from
        OUTPUT: Block
        """

        return Block(
            self.nb_rows,
            self.constant * factor,
            [term.scaled(factor) for term in self.linear],
            [term.scaled(factor) for term in self.quadratic],
            self.valid if valid is None else valid,
        )

    def summed_over(self, width, filled):
        """
        The block a sum reduces to, once its body has been evaluated over the
        range. Only the constant is really summed over the index values : the
        terms keep one entry per index value, for the matrix to add up later,
        and `valid` is a row mask already.
        """

        kept = None if bool(filled.all()) else filled

        constant = self.constant
        if constant.shape[-1] == 1:
            # the same value on every index, so summing it is counting them
            counted = width if kept is None else kept.sum(axis=-1)
            constant = constant[..., 0] * counted
        else:
            if kept is not None:
                constant = np.where(kept, constant, 0.0)
            constant = constant.sum(axis=-1)

        return Block(
            self.nb_rows,
            constant,
            [term.widened(width, kept) for term in self.linear],
            [term.widened(width, kept) for term in self.quadratic],
            self.valid,
        )

    def constant_over(self, valid):
        """
        constant_over function : the degree zero part as one value per row the
                                 mask keeps. This is the boundary the shape of
                                 the evaluation stops at, the matrices being
                                 built from a flat vector
        INPUT:  valid -> mask over the rows of the block
        OUTPUT: ndarray of shape (number of rows the mask keeps,)
        """

        return np.broadcast_to(self.constant, (self.nb_rows,))[valid]

    def assemble(self, valid):
        """
        assemble function : the degree one and the degree two parts as the
        sparse arrays the matrices of the problem are built from, over the rows
        the mask keeps. A part with no entry at all comes back as None.

        The block is left empty : its terms are released one by one as they are
        read, so that the memory they hold is gone before the sparse array is
        built. Assembling is the last thing done with a block.
        INPUT:  valid -> mask over the rows of the block
        OUTPUT: (coo_array or None, coo_array or None)
        """

        nb_kept_rows = int(np.count_nonzero(valid))

        return (
            _build_sparse_array(self.linear, valid, nb_kept_rows, self.nb_rows),
            _build_sparse_array(self.quadratic, valid, nb_kept_rows, self.nb_rows),
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
    constant = blocks[0].constant
    linear = list(blocks[0].linear)
    quadratic = list(blocks[0].quadratic)
    valid = blocks[0].valid
    for block in blocks[1:]:
        # every block was evaluated over the same axes, so the constants line
        # up without anything having to be inserted
        constant = constant + block.constant
        linear += block.linear
        quadratic += block.quadratic
        valid = combine_valid(valid, block.valid)

    return Block(nb_rows, constant, linear, quadratic, valid)


def multiply_blocks(left, right):
    """
    multiply_blocks function : multiplies two polynomials, the degrees adding
    up. The degree two part is the outer product of the degree one parts along
    the widths, which is where the distribution the quadratic factorization
    needs happens.
    INPUT:  left, right -> Block objects over the same rows
    OUTPUT: Block
    """

    nb_rows = left.nb_rows
    valid = combine_valid(left.valid, right.valid)

    if left.degree() == 0:
        return right.scaled(left.constant, valid)
    if right.degree() == 0:
        return left.scaled(right.constant, valid)

    assert not left.quadratic and not right.quadratic

    constant = left.constant * right.constant
    linear = []
    if right.constant.any():
        linear += [term.scaled(right.constant) for term in left.linear]
    if left.constant.any():
        linear += [term.scaled(left.constant) for term in right.linear]

    quadratic = []
    for left_term in left.linear:
        for right_term in right.linear:
            quadratic.append(_outer_product(left_term, right_term))

    return Block(nb_rows, constant, linear, quadratic, valid)


def _outer_product(left_term, right_term):
    """
    Every entry of one degree one term multiplied by every entry of the other.

    The widths of the two are the widths of different sums, so their entries
    are independent summands and every pair of them exists : each operand is
    given a length one axis where the other holds a width, and the product is
    then the plain one. The axes of the outer sums are shared, which is what
    makes sum(x[i]*y[i] for i) the dot product.
    """

    [left_columns] = left_term.columns
    [right_columns] = right_term.columns
    nb_left = len(left_term.widths)
    nb_right = len(right_term.widths)

    def as_left(value):
        # the widths of the right operand are new, and come after its own
        return value.reshape(value.shape + (1,) * nb_right)

    def as_right(value):
        # the widths of the left operand are new, and come before its own
        rows = value.ndim - nb_right
        return value.reshape(value.shape[:rows] + (1,) * nb_left + value.shape[rows:])

    return Term(
        as_left(left_term.values) * as_right(right_term.values),
        (as_left(left_columns), as_right(right_columns)),
        left_term.widths + right_term.widths,
        as_left(left_term.filled) & as_right(right_term.filled),
    )


def _gather_nonzero_terms(terms, valid, nb_rows):
    """
    _gather_nonzero_terms function : turns a list of terms into flat arrays,
    keeping only the entries whose value is not zero.

    Terms are handled one at a time. For each term, its values are brought to
    the rows and widths they cover and reduced to the positions that are not
    zero; the term's columns are then read only at those same positions. The
    list is emptied as it is read, so that what a term holds is released
    before the sparse array is built.
    INPUT:  terms -> non empty list of Term objects, all of the same degree,
                     emptied by the call
            valid -> ndarray, the rows to keep
            nb_rows -> number of rows the terms were evaluated at, before
                       valid narrows them down
    OUTPUT: the rows array, then the values and one array per column of the
            terms, all of the same length
    """

    keep_every_row = bool(valid.all())
    row_number = None if keep_every_row else np.cumsum(valid) - 1

    rows = [np.zeros(0, dtype=int)]
    values = [np.zeros(0, dtype=float)]
    columns = [[np.zeros(0, dtype=int)] for _ in range(terms[0].degree())]

    # popping from the end is what releases each term, and reversing first
    # keeps the entries in the order they were written in
    terms.reverse()
    while terms:
        term = terms.pop()
        shape = (nb_rows,) + term.widths
        term_values = np.broadcast_to(term.values, shape)

        live = term_values != 0.0
        if not keep_every_row:
            # remove rows that should be excluded
            valid_rows = valid.reshape((nb_rows,) + (1,) * len(term.widths))
            live &= valid_rows
        if not term.filled.all():
            # remove ragged terms in the width
            valid_padding = np.broadcast_to(term.filled, shape)
            live &= valid_padding

        # one array per axis, the first of them being the row an entry lands on
        kept = np.nonzero(live)
        if len(kept[0]) == 0:
            continue

        rows.append(kept[0] if keep_every_row else row_number[kept[0]])
        values.append(term_values[kept])
        for gathered_column, term_column in zip(columns, term.columns):
            rectangle = np.broadcast_to(term_column, shape)[kept]
            gathered_column.append(rectangle)

    return (
        np.concatenate(rows),
        np.concatenate(values),
        *(np.concatenate(column) for column in columns),
    )


def _build_sparse_array(terms, valid, nb_kept_rows, nb_rows):
    """
    _build_sparse_array function : build the terms of a block as a coo_array
    whose first axis is the row and whose other axes are the columns of the
    terms, one for the degree one part and two for the degree two one. Entries
    at the same (row, *columns) tuple in more than one term are summed and
    cancelled by sum_duplicates/eliminate_zeros.
    INPUT:  terms -> list of Term objects, all of the same degree
            valid -> mask over the rows of the block
            nb_kept_rows -> number of rows the mask keeps
            nb_rows -> number of rows the terms were evaluated at
    OUTPUT: coo_array of shape (nb_kept_rows, *ncols), or None when `terms` is
            empty
    """

    if not terms:
        return None

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
