# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).
import math

import numpy as np  # type: ignore
from scipy.sparse import coo_array  # type: ignore

RowMask = np.ndarray[tuple[int], np.dtype[np.bool_]]


def combine_valid(first: RowMask | None, second: RowMask | None) -> RowMask | None:
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
    does not reach. `widths` holds one width per sum whose index the columns
    read, outermost first : a sum they ignore is summed away when it ends. The
    entries of a row are laid out over all of the widths, so a row carries
    prod(widths) of them.
    """

    __slots__ = ("columns", "filled", "values", "widths")

    def __init__(
        self,
        values: np.ndarray,
        columns: tuple[np.ndarray, ...] = (),
        widths: tuple[int, ...] = (),
        filled: np.ndarray | None = None,
    ) -> None:

        if filled is None:
            filled = np.ones((1,) * values.ndim, dtype=bool)
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

    def scaled(self, factor: np.ndarray) -> "Term":

        return Term(
            # a factor which does not vary along the widths, which trail
            self.values * factor.reshape(factor.shape + (1,) * len(self.widths)),
            self.columns,
            self.widths,
            self.filled,
        )

    def negated(self) -> "Term":

        return Term(-self.values, self.columns, self.widths, self.filled)

    def summed_over(self, width: int, kept: np.ndarray | None) -> "Term":
        """
        The term a sum gives back when it ends. Its range was the innermost
        axis the body was evaluated over and becomes the outermost width,
        which is the very same axis : nothing is moved, and only the mask of
        the index values the sum keeps has to be brought in. A term whose
        columns do not read the index is the same entry on every index
        value, and is added up here instead ; a constant, having no column at
        all, always is.
        """

        axis = self.values.ndim - len(self.widths) - 1
        if kept is not None:
            assert kept.shape[axis] == width
            kept = kept.reshape(kept.shape + (1,) * len(self.widths))

        if any(column.shape[axis] != 1 for column in self.columns):
            return Term(
                self.values,
                self.columns,
                (width,) + self.widths,
                self.filled if kept is None else self.filled & kept,
            )

        if self.values.shape[axis] == 1:
            nb_kept_index_values = width if kept is None else kept.sum(axis=axis)
            values = self.values.squeeze(axis) * nb_kept_index_values
        else:
            values_to_sum = self.values
            if kept is not None:
                values_to_sum = np.where(kept, values_to_sum, 0.0)
            values = values_to_sum.sum(axis=axis)

        return Term(
            values,
            tuple(column.squeeze(axis) for column in self.columns),
            self.widths,
            self.filled.squeeze(axis),
        )


class Block:
    """
    The polynomial an expression reduces to over `nb_rows` rows. See the module
    docstring.
    """

    __slots__ = ("nb_rows", "constant", "linear", "quadratic", "valid")

    def __init__(
        self,
        nb_rows: int,
        constant: Term,
        linear: list[Term] | None = None,
        quadratic: list[Term] | None = None,
        valid: RowMask | None = None,
    ) -> None:
        self.nb_rows = nb_rows
        self.constant = constant
        self.linear = linear if linear is not None else []
        self.quadratic = quadratic if quadratic is not None else []
        self.valid = valid

    @staticmethod
    def of_constant(
        nb_rows: int, rank: int, value: np.ndarray, valid: RowMask | None = None
    ) -> "Block":
        """
        of_constant function : the block a subexpression holding no variable
                               reduces to. The compiled expression gives back
                               one value per position of the rank it was
                               evaluated at, an axis it does not vary along
                               being length one
        INPUT:  nb_rows -> number of rows
                rank -> number of axes the expression is evaluated over
                value -> the value the compiled expression evaluated to
                valid -> mask of the rows still valid, or None
        OUTPUT: Block
        """
        value = np.asarray(value, dtype=float)
        assert value.ndim == rank

        return Block(nb_rows, Term(value), valid=valid)

    @staticmethod
    def of_variable(
        nb_rows: int, rank: int, column: np.ndarray, valid: RowMask | None = None
    ) -> "Block":
        """
        of_variable function : the block one variable of the problem reduces
                               to, that is a single degree one entry per
                               position
        INPUT:  nb_rows -> number of rows
                rank -> number of axes the expression is evaluated over
                column -> the column the variable is read at, one per position
                          or one for all of them
                valid -> mask of the rows still valid, or None
        OUTPUT: Block
        """

        column = np.asarray(column, dtype=int)
        assert column.ndim == rank

        return Block(
            nb_rows,
            Term(np.zeros((1,) * rank)),
            [Term(np.ones((1,) * rank), (column,))],
            valid=valid,
        )

    def degree(self) -> int:
        """The degree of the expression with respect to the variables."""

        if self.quadratic:
            return 2
        if self.linear:
            return 1

        return 0

    def negated(self) -> "Block":

        return Block(
            self.nb_rows,
            self.constant.negated(),
            [term.negated() for term in self.linear],
            [term.negated() for term in self.quadratic],
            self.valid,
        )

    def scaled(self, factor: np.ndarray, valid: RowMask | None = None) -> "Block":
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
            self.constant.scaled(factor),
            [term.scaled(factor) for term in self.linear],
            [term.scaled(factor) for term in self.quadratic],
            self.valid if valid is None else valid,
        )

    def summed_over(self, width: int, filled: np.ndarray) -> "Block":
        """
        The block a sum reduces to, once its body has been evaluated over the
        range. Whatever does not read the index is actually summed over its
        values, A term which does read the index keeps one entry per index
        value, for the matrix to add up later. `valid` is a row mask already.
        """

        kept = None if bool(filled.all()) else filled

        return Block(
            self.nb_rows,
            self.constant.summed_over(width, kept),
            [term.summed_over(width, kept) for term in self.linear],
            [term.summed_over(width, kept) for term in self.quadratic],
            self.valid,
        )

    def constant_over(self, valid: RowMask) -> np.ndarray:
        """
        constant_over function : the degree zero part as one value per row the
                                 mask keeps. This is the boundary the shape of
                                 the evaluation stops at, the matrices being
                                 built from a flat vector
        INPUT:  valid -> mask over the rows of the block
        OUTPUT: ndarray of shape (number of rows the mask keeps,)
        """

        return np.broadcast_to(self.constant.values, (self.nb_rows,))[valid]

    def assemble(self, valid: RowMask) -> tuple[coo_array | None, coo_array | None]:
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

        nb_kept_rows = np.count_nonzero(valid)

        return (
            _build_sparse_array(self.linear, valid, nb_kept_rows, self.nb_rows),
            _build_sparse_array(self.quadratic, valid, nb_kept_rows, self.nb_rows),
        )


def add_blocks(blocks: list[Block]) -> Block:
    """
    add_blocks function : sums any number of blocks at once.
    INPUT:  blocks -> list of Block objects, all over the same rows
    OUTPUT: Block
    """

    if len(blocks) == 1:
        return blocks[0]

    nb_rows = blocks[0].nb_rows
    constant = blocks[0].constant.values
    linear = list(blocks[0].linear)
    quadratic = list(blocks[0].quadratic)
    valid = blocks[0].valid
    for block in blocks[1:]:
        # every block was evaluated over the same axes, so the constants line
        # up without anything having to be inserted
        constant = constant + block.constant.values
        linear += block.linear
        quadratic += block.quadratic
        valid = combine_valid(valid, block.valid)

    return Block(nb_rows, Term(constant), linear, quadratic, valid)


def multiply_blocks(left: Block, right: Block) -> Block:
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
    left_constant = left.constant.values
    right_constant = right.constant.values

    if left.degree() == 0:
        return right.scaled(left_constant, valid)
    if right.degree() == 0:
        return left.scaled(right_constant, valid)

    assert not left.quadratic and not right.quadratic

    constant = left_constant * right_constant
    linear = []
    if right_constant.any():
        linear += [term.scaled(right_constant) for term in left.linear]
    if left_constant.any():
        linear += [term.scaled(left_constant) for term in right.linear]

    quadratic = []
    for left_term in left.linear:
        for right_term in right.linear:
            quadratic.append(_outer_product(left_term, right_term))

    return Block(nb_rows, Term(constant), linear, quadratic, valid)


def _outer_product(left_term: Term, right_term: Term) -> Term:
    """
    Every entry of one degree one term multiplied by every entry of the other.

    The widths of the two are the widths of different sums, so their entries
    are independent summands and every pair of them exists : each operand is
    given a length one axis where the other holds a width, and the product is
    then the plain one. The axes of the outer sums are shared, which is what
    makes sum(x[i]*y[i] for i) the dot product.
    """

    [left_column] = left_term.columns
    [right_column] = right_term.columns
    nb_left = len(left_term.widths)
    nb_right = len(right_term.widths)

    def as_left(value: np.ndarray) -> np.ndarray:
        # the widths of the right operand are new, and come after its own
        return value.reshape(value.shape + (1,) * nb_right)

    def as_right(value: np.ndarray) -> np.ndarray:
        # the widths of the left operand are new, and come before its own,
        # but after the common axes
        rows = value.ndim - nb_right
        return value.reshape(value.shape[:rows] + (1,) * nb_left + value.shape[rows:])

    return Term(
        as_left(left_term.values) * as_right(right_term.values),
        (as_left(left_column), as_right(right_column)),
        left_term.widths + right_term.widths,
        as_left(left_term.filled) & as_right(right_term.filled),
    )


def _gather_nonzero_terms(
    terms: list[Term], valid: RowMask, nb_rows: int
) -> tuple[np.ndarray, ...]:
    """
    _gather_nonzero_terms function : turns a list of terms into flat arrays,
    keeping only the entries whose value is not zero.

    Terms are handled one at a time. For each term, its values are brought to
    the rows and widths they cover and reduced to the positions that are not
    zero; the term's columns are then read only at those same positions. The
    list is emptied as it is read, so that what a term holds is released
    before the sparse array is built. The final arrays are allocated once, at
    the total number of cells every term could contribute, and each term
    writes its entries into its own slice; the arrays are then trimmed to
    what was actually written.
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

    upper_bound = sum(nb_rows * math.prod(term.widths) for term in terms)
    rows = np.empty(upper_bound, dtype=int)
    values = np.empty(upper_bound, dtype=float)
    columns = tuple(np.empty(upper_bound, dtype=int) for _ in range(terms[0].degree()))
    written = 0
    row_range = np.arange(nb_rows)

    # popping from the end is what releases each term, and reversing first
    # keeps the entries in the order they were written in
    terms.reverse()
    while terms:
        term = terms.pop()
        shape = (nb_rows,) + term.widths
        term_values = np.broadcast_to(term.values, shape)
        fully_filled = term.filled.all()
        live = term_values != 0.0

        if keep_every_row and fully_filled and live.all():
            # every cell is a real, kept entry: ravel instead of masking
            row_indices = np.broadcast_to(
                row_range.reshape((nb_rows,) + (1,) * len(term.widths)),
                shape,
            ).ravel()
            entry_values = term_values.ravel()
            entry_columns = [
                np.broadcast_to(term_column, shape).ravel()
                for term_column in term.columns
            ]
        else:
            if not keep_every_row:
                # remove rows that should be excluded
                live &= valid.reshape((nb_rows,) + (1,) * len(term.widths))
            if not fully_filled:
                # remove ragged terms in the width
                live &= np.broadcast_to(term.filled, shape)

            kept = np.nonzero(live)
            row_indices = kept[0] if keep_every_row else row_number[kept[0]]
            entry_values = term_values[kept]
            entry_columns = [
                np.broadcast_to(term_column, shape)[kept]
                for term_column in term.columns
            ]

        count = len(row_indices)
        if count == 0:
            continue

        end = written + count
        rows[written:end] = row_indices
        values[written:end] = entry_values
        for gathered_column, entry_column in zip(columns, entry_columns):
            gathered_column[written:end] = entry_column
        written = end

    return (
        rows[:written],
        values[:written],
        *(column[:written] for column in columns),
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
    INPUT:  terms -> list of Term objects, all of the same degree
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
    if degree == 2:
        # a degree one row is consolidated by the solver's own tocsr/tocsc
        # conversion; a degree two row is read raw before that, by
        # _split_rows_by_degree, so cancelling terms must be merged here.
        sparse.sum_duplicates()
        sparse.eliminate_zeros()

    return sparse
