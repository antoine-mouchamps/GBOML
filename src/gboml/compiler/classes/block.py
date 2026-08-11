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

A *term* is a rectangle of `width` entries per row : `values` and one column
array per variable of the product, all broadcastable to `(nb_rows, width)`.
One column makes a degree one term, two a degree two term.
"""

import numpy as np  # type: ignore
from scipy.sparse import coo_array  # type: ignore


def is_scalar(value) -> bool:

    return not isinstance(value, np.ndarray)


def per_row(value, nb_rows):
    """
    per_row function : gives a value the trailing axis that lines it up with
                       one row of a (nb_rows, width) rectangle, so that it
                       applies to every entry of that row at once
    INPUT:  value -> scalar or ndarray of shape (nb_rows,)
            nb_rows -> number of rows
    OUTPUT: scalar or ndarray of shape (nb_rows, 1), keeping the type it came
            in, so that a column stays an index and does not become a float
    """

    if is_scalar(value):
        return value

    return value.reshape(nb_rows, 1)


def materialize(value, shape, dtype=float):
    """
    materialize function : turns a value into the full array it stands for
    INPUT:  value -> scalar or ndarray broadcastable to shape
            shape -> shape of the array returned, (nb_rows,) for a constant and
                     (nb_rows, width) for a term entry
            dtype -> type of the array returned
    OUTPUT: ndarray of the given shape, a read only view whenever the value had
            to be stretched
    """

    if is_scalar(value):
        return np.full(shape, value, dtype=dtype)

    array = np.asarray(value, dtype=dtype)
    if array.shape == shape:
        return array

    return np.broadcast_to(array, shape)


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


class Term:
    """
    One rectangle of entries : `width` products per row, each of them the
    product of the variables at `columns` scaled by `values`. `columns` is a
    sequence of one column array for a degree one term, of two for a degree two
    one.
    """

    __slots__ = ("columns", "values", "width")

    def __init__(self, values, columns, width):

        self.values = values
        self.columns = columns
        self.width = width

    def degree(self) -> int:
        """The number of variables multiplied together in every entry."""

        return len(self.columns)

    def scaled(self, factor):

        return Term(self.values * factor, self.columns, self.width)

    def negated(self):

        return Term(-self.values, self.columns, self.width)


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

    @staticmethod
    def of_constant(nb_rows, value, valid=None):
        """
        of_constant function : the block a subexpression holding no variable
                               reduces to. The compiled expression gives back a
                               scalar, a 0-d array for a parameter read at a
                               constant index, or one value per row.
        INPUT:  nb_rows -> number of rows
                value -> the value the compiled expression evaluated to
                valid -> mask of the rows still valid, or None
        OUTPUT: Block
        """

        if not is_scalar(value):
            array = np.asarray(value, dtype=float)
            value = float(array) if array.ndim == 0 else array

        return Block(nb_rows, value, valid=valid)

    @staticmethod
    def of_variable(nb_rows, columns, valid=None):
        """
        of_variable function : the block one variable of the problem reduces
                               to, that is a single degree one entry per row
        INPUT:  nb_rows -> number of rows
                columns -> the column of the variable, one per row or one for
                           all of them
                valid -> mask of the rows still valid, or None
        OUTPUT: Block
        """

        if not is_scalar(columns):
            columns = per_row(np.asarray(columns, dtype=int), nb_rows)

        return Block(nb_rows, 0.0, [Term(1.0, (columns,), 1)], valid=valid)

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

        row_factor = per_row(factor, self.nb_rows)

        return Block(
            self.nb_rows,
            self.constant * factor,
            [term.scaled(row_factor) for term in self.linear],
            [term.scaled(row_factor) for term in self.quadratic],
            self.valid,
        )

    def folded(self, nb_rows, nb_cells, padding):
        """
        folded function : brings a block evaluated over (row, index of the sum)
        pairs back to one over rows, by folding the index of the sum into the
        term axis. A term of `width` entries becomes a term of
        `nb_cells * width` entries. This block holds nb_rows * nb_cells rows.
        INPUT:  nb_rows, nb_cells -> shape the block is folded back to
                padding -> (nb_rows, nb_cells) mask of the inert entries
        OUTPUT: Block over nb_rows rows
        """

        has_padding = bool(padding.any())
        flat_padding = padding.reshape(-1)

        constant = self.constant
        if not is_scalar(constant):
            constant = np.where(flat_padding, 0.0, constant)
            constant = constant.reshape(nb_rows, nb_cells).sum(axis=1)
        elif has_padding:
            constant = constant * (~padding).sum(axis=1)
        else:
            constant = constant * nb_cells

        valid = self.valid
        if valid is not None:
            # a padded entry never invalidates the row it sits on
            valid = (valid.reshape(nb_rows, nb_cells) | padding).all(axis=1)
            if valid.all():
                valid = None

        def fold(terms):

            return [
                _folded_term(term, nb_rows, nb_cells, flat_padding, has_padding)
                for term in terms
            ]

        return Block(nb_rows, constant, fold(self.linear), fold(self.quadratic), valid)

    def constant_over(self, valid, nb_rows):
        """
        constant_over function : the degree zero part as one value per row the
                                 mask keeps
        INPUT:  valid -> mask over the rows of the block
                nb_rows -> number of rows the mask keeps
        OUTPUT: ndarray of shape (nb_rows,)
        """

        if is_scalar(self.constant):
            return materialize(self.constant, (nb_rows,))

        return self.constant[valid]

    def assemble(self, valid, nb_rows):
        """
        assemble function : the degree one and the degree two parts as the
        sparse arrays the matrices of the problem are built from, over the rows
        the mask keeps. A part with no entry at all comes back as None.

        The block is left empty : its terms are released one by one as they are
        read, so that the memory they hold is gone before the sparse array is
        built. Assembling is the last thing done with a block.
        INPUT:  valid -> mask over the rows of the block
                nb_rows -> number of rows the mask keeps
        OUTPUT: (coo_array or None, coo_array or None)
        """

        return (
            _build_sparse_array(self.linear, valid, nb_rows, self.nb_rows),
            _build_sparse_array(self.quadratic, valid, nb_rows, self.nb_rows),
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

    if left.degree() == 0:
        return Block(nb_rows, *_scaled_parts(right, left.constant), valid=valid)
    if right.degree() == 0:
        return Block(nb_rows, *_scaled_parts(left, right.constant), valid=valid)

    constant = left.constant * right.constant
    linear = [term.scaled(per_row(right.constant, nb_rows)) for term in left.linear]
    linear += [term.scaled(per_row(left.constant, nb_rows)) for term in right.linear]

    quadratic = []
    for left_term in left.linear:
        for right_term in right.linear:
            quadratic.append(_outer_product(left_term, right_term, nb_rows))

    return Block(nb_rows, constant, linear, quadratic, valid)


def _scaled_parts(block, factor):
    """The parts of a block scaled by a constant."""

    row_factor = per_row(factor, block.nb_rows)

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
    INPUT:  left_term, right_term -> degree one Term objects
            nb_rows -> number of rows
    OUTPUT: degree two Term
    """

    [left_columns] = left_term.columns
    [right_columns] = right_term.columns
    if left_term.width == 1 and right_term.width == 1:
        return Term(
            left_term.values * right_term.values,
            (left_columns, right_columns),
            1,
        )

    width = left_term.width * right_term.width
    left_values = materialize(left_term.values, (nb_rows, left_term.width))
    right_values = materialize(right_term.values, (nb_rows, right_term.width))
    left_columns = materialize(left_columns, (nb_rows, left_term.width), dtype=int)
    right_columns = materialize(right_columns, (nb_rows, right_term.width), dtype=int)

    values = left_values[:, :, None] * right_values[:, None, :]
    columns_i = np.broadcast_to(
        left_columns[:, :, None], (nb_rows, left_term.width, right_term.width)
    )
    columns_j = np.broadcast_to(
        right_columns[:, None, :], (nb_rows, left_term.width, right_term.width)
    )

    return Term(
        values.reshape(nb_rows, width),
        (columns_i.reshape(nb_rows, width), columns_j.reshape(nb_rows, width)),
        width,
    )


def _folded_term(term, nb_rows, nb_cells, flat_padding, has_padding):
    """
    _folded_term function : one term of a block being folded, the index of the
    sum laid out along the entries of the term. See Block.folded.
    INPUT:  term -> Term over nb_rows * nb_cells rows
            nb_rows, nb_cells -> shape the term is folded back to
            flat_padding -> the padding mask over the inner rows
            has_padding -> whether it marks anything at all
    OUTPUT: Term of nb_cells * width entries over nb_rows rows
    """

    width = nb_cells * term.width
    inner_shape = (nb_rows * nb_cells, term.width)

    values = term.values
    if has_padding:
        # zeroing the value is what makes a padded entry inert, whatever the
        # column it carries
        values = materialize(values, inner_shape)
        values = np.where(flat_padding[:, None], 0.0, values)

    def fold(value, dtype):

        if is_scalar(value):
            return value

        return materialize(value, inner_shape, dtype=dtype).reshape(nb_rows, width)

    return Term(
        fold(values, float),
        tuple(fold(columns, int) for columns in term.columns),
        width,
    )


def _gather_nonzero_terms(terms, valid, block_nb_rows):
    """
    _gather_nonzero_terms function : turns a list of terms into flat arrays,
    keeping only the entries whose value is not zero.

    Terms are handled one at a time. For each term, its values are turned into
    a (row, entry) rectangle first and reduced to the positions that are not
    zero; the term's columns are then read only at those same positions.
    The list is emptied as it is read : a term is dropped as soon as its
    entries have been copied out, so that the rectangles it holds are released
    before the sparse array is built.
    INPUT:  terms -> non empty list of Term objects, all of the same degree,
                     emptied by the call
            valid -> ndarray, the rows to keep
            block_nb_rows -> number of rows the terms were evaluated at,
                             before valid narrows them down
    OUTPUT: the rows array, then the values and one array per column of the
            terms, all of the same length
    """

    keep_every_row = bool(valid.all())
    gathered = [[np.zeros(0, dtype=int)], [np.zeros(0, dtype=float)]]
    gathered += [[np.zeros(0, dtype=int)] for _ in range(terms[0].degree())]

    # popping from the end is what releases each term, and reversing first
    # keeps the entries in the order they were written in
    terms.reverse()
    while terms:
        term = terms.pop()
        values = materialize(term.values, (block_nb_rows, term.width))
        if not keep_every_row:
            values = values[valid]

        kept_rows, kept_entries = np.nonzero(values)
        if len(kept_entries) == 0:
            continue

        gathered[0].append(kept_rows)
        gathered[1].append(values[kept_rows, kept_entries])
        for index, columns in enumerate(term.columns, start=2):
            if is_scalar(columns):
                # the same column for every entry, no rectangle to build
                gathered[index].append(np.full(len(kept_rows), columns, dtype=int))
                continue

            rectangle = materialize(columns, (block_nb_rows, term.width), dtype=int)
            if not keep_every_row:
                rectangle = rectangle[valid]
            gathered[index].append(rectangle[kept_rows, kept_entries])

    return tuple(np.concatenate(arrays) for arrays in gathered)


def _build_sparse_array(terms, valid, nb_rows, block_nb_rows):
    """
    _build_sparse_array function : build the terms of a block as a coo_array
    whose first axis is the row and whose other axes are the columns of the
    terms, one for the degree one part and two for the degree two one. Entries
    at the same (row, *columns) tuple in more than one term are summed and
    cancelled by sum_duplicates/eliminate_zeros.
    INPUT:  terms -> list of Term objects, all of the same degree
            valid, nb_rows, block_nb_rows
    OUTPUT: coo_array of shape (nb_rows, *ncols), or None when `terms` is empty
    """

    if not terms:
        return None

    term_rows, values, *columns = _gather_nonzero_terms(terms, valid, block_nb_rows)
    if len(columns) == 2:
        # a product is commutative so x * y and y * x are the very same entry.
        columns = [np.minimum(*columns), np.maximum(*columns)]
    # indices are never negative, so a column's max is its ncols - 1
    # whenever it holds any, and the initial floor of 0 gives ncols = 1 when it
    # does not
    shape = (nb_rows,) + tuple(int(column.max(initial=0)) + 1 for column in columns)
    sparse = coo_array((values, (term_rows, *columns)), shape=shape)
    sparse.sum_duplicates()  # TODO: keep it or no ??
    sparse.eliminate_zeros()

    return sparse
