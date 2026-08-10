# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).

"""
factor.py : turns a constraint or an objective into the rows of the
problem, whatever its degree with respect to the variables.

The expression is compiled once, index free, into a tree of nodes, and
every node knows how to evaluate itself into a Block. The rules are the
algebra of polynomials:

    constant     a subtree holding no variable, evaluated whole
    variable     one column of the problem
    - and +      the terms of both sides, gathered
    *            the degrees add up, the degree two part being the outer
                 product of the degree one parts
    /            a scaling, the denominator holding no variable
    **           repeated multiplication, the exponent being a constant integer
    sum          the enclosed terms, widened over the range of the sum

Terms landing on the same row are summed by the matrix, so a sum only has to
lay its range out along the term axis. Nothing is unrolled in python, and a
sum inside a sum is that same rule applied twice.

A range that depends on an enclosing index is ragged, and is squared off with
NaN padding by TimeInterval.get_range. Padded entries, and the entries a `where`
condition rejects, are marked in one `padding` mask: their value is forced to
zero, so they contribute nothing, and they are excluded from the check that
every variable index landed in bounds, so they do not invalidate their row.
"""

import numpy as np  # type: ignore
from scipy.sparse import coo_array  # type: ignore

from gboml.compiler.utils import error_
from .block import (
    Block,
    LinearTerm,
    add_blocks,
    multiply_blocks,
    as_rows,
    as_column,
    materialize,
    combine_valid,
    is_scalar,
)
from .constraint import Constraint
from .expression import Expression
from .identifier import Identifier
from .objective import Objective


class ByteArray:
    __slots__ = ("array",)

    def __init__(self, length):
        self.array = np.ones(length, dtype=bool)

    def intersect_inplace(self, values):
        self.array = self.array & values


def __f_index2__(expr, all_index, name):
    max_length = len(name)
    is_in = np.array(((0 <= expr) & (expr < max_length)), dtype=bool)
    all_index.intersect_inplace(is_in)
    expr = np.array(expr, dtype=int)

    if not is_in.all():
        return_value = np.array(name[np.clip(expr, 0, max_length - 1)], dtype=float)
    else:
        return_value = np.array(name[expr], dtype=float)

    return return_value


def is_index_dependant_expression(expression, index) -> bool:

    e_type = expression.get_type()
    children = expression.get_children()
    predicate: bool = False
    if e_type == "literal":
        seed = expression.get_name()
        if type(seed) == Identifier:
            identifier = seed
            id_type = identifier.get_type()
            id_name = identifier.get_name()
            if id_type == "assign":
                predicate = is_index_dependant_expression(
                    identifier.get_expression(), index
                )
            if id_name in index:
                predicate = True

    else:
        for child in children:
            predicate_i = is_index_dependant_expression(child, index)
            if predicate_i:
                predicate = predicate_i
                break

        if predicate is False and e_type == "sum":
            # the index may appear only in the range of the sum, as in
            # sum(x[i] for i in [0:t]), which makes the expression depend
            # on it just as much as using it in the body would
            predicate = is_index_dependant_interval(expression, index)

    return predicate


def is_index_dependant_interval(expression, index) -> bool:
    """
    is_index_dependant_interval function : whether the range a sum runs
    over, or the condition filtering it, depends on an enclosing index
    INPUT:  expression -> Expression object of the sum
            index -> name of the index, or list of names
    OUTPUT: bool
    """

    time_interval = expression.get_time_interval()
    if time_interval is None:
        return False

    parts = [time_interval.get_begin(), time_interval.get_end()]
    step = time_interval.get_step()
    if step is not None and not isinstance(step, int):
        parts.append(step)
    condition = expression.get_condition()
    if condition is not None:
        parts.append(condition)

    return any(is_index_dependant_expression(part, index) for part in parts)


class Context:
    """
    The namespace expressions are evaluated against, plus the shape they are
    evaluated at. `index_names` is the list of the indexes enclosing the
    expression, outermost first, every one of them bound to an array of one
    value per row.
    """

    __slots__ = ("definitions", "nb_rows", "index_names")

    def __init__(self, definitions, nb_rows, index_names):

        self.definitions = definitions
        self.nb_rows = nb_rows
        self.index_names = index_names

    def evaluate(self, python_expression):
        """
        evaluate function : evaluates a compiled variable free expression over
        the rows. Indexing a parameter out of its range is reported through the
        `extension_range` mask the generated code writes into, which is
        returned alongside the value rather than kept in shared state.
        INPUT:  python_expression -> compiled python expression
        OUTPUT: value (scalar or ndarray), mask of the rows still valid or None
        """

        mask = ByteArray(self.nb_rows)
        self.definitions["extension_range"] = mask
        value = eval(python_expression, self.definitions, {})
        if mask.array.all():
            return value, None

        return value, mask.array


class Node:
    """A compiled subexpression. Subclasses only implement evaluate."""

    __slots__ = ()

    def evaluate(self, context) -> Block:

        raise NotImplementedError


class ConstantNode(Node):
    """A subtree holding no variable at all, kept whole and evaluated in one
    go rather than distributed."""

    __slots__ = ("python_expression",)

    def __init__(self, expression):

        self.python_expression = expression.turn_to_python_expression()

    def evaluate(self, context) -> Block:

        value, valid = context.evaluate(self.python_expression)

        return Block(context.nb_rows, as_rows(value, context.nb_rows), valid=valid)


class VariableNode(Node):
    """One variable of the problem, giving a single column per row."""

    __slots__ = ("index", "python_offset", "size")

    def __init__(self, index, offset_expression, size):

        self.index = index
        self.python_offset = (
            None
            if offset_expression is None
            else offset_expression.turn_to_python_expression()
        )
        self.size = size

    def evaluate(self, context) -> Block:

        nb_rows = context.nb_rows
        valid = None
        if self.python_offset is None:
            offset = 0
        else:
            offset, valid = context.evaluate(self.python_offset)
            offset = np.asarray(offset)
            in_bounds = (0 <= offset) & (offset < self.size)
            if not np.all(in_bounds):
                # an index out of range invalidates its row, and is clamped so
                # that the column stays addressable until the row is dropped
                valid = combine_valid(
                    valid, np.broadcast_to(in_bounds, (nb_rows,)).copy()
                )
                offset = np.clip(offset, 0, max(self.size - 1, 0))
            if offset.ndim == 0:
                offset = int(offset)

        columns = offset + self.index
        if not is_scalar(columns):
            columns = as_column(np.asarray(columns, dtype=int), nb_rows)

        return Block(nb_rows, 0.0, [LinearTerm(1.0, columns, 1)], valid=valid)


class AdditionNode(Node):
    """A run of + and - gathered in one node, so the terms of a long sum of
    products are concatenated once instead of once per operand."""

    __slots__ = ("children",)

    def __init__(self, children):

        self.children = children

    def evaluate(self, context) -> Block:

        blocks = []
        for sign, child in self.children:
            block = child.evaluate(context)
            blocks.append(block if sign > 0 else block.negated())

        return add_blocks(blocks)


class MultiplicationNode(Node):
    __slots__ = ("left", "right")

    def __init__(self, left, right):

        self.left = left
        self.right = right

    def evaluate(self, context) -> Block:

        left = self.left.evaluate(context)
        right = self.right.evaluate(context)
        if left.degree() + right.degree() > 2:
            error_(
                "INTERNAL ERROR : a product of degree "
                + str(left.degree() + right.degree())
                + " reached the factorization, check the semantic analysis"
            )

        return multiply_blocks(left, right)


class DivisionNode(Node):
    """A division whose denominator holds no variable, which is therefore a
    scaling of the numerator."""

    __slots__ = ("numerator", "denominator")

    def __init__(self, numerator, denominator):

        self.numerator = numerator
        self.denominator = denominator

    def evaluate(self, context) -> Block:

        numerator = self.numerator.evaluate(context)
        denominator = self.denominator.evaluate(context)
        block = numerator.scaled(1.0 / denominator.constant)
        block.valid = combine_valid(block.valid, denominator.valid)

        return block


class PowerNode(Node):
    """A variable expression raised to a constant non negative integer, small
    enough for the result to stay of degree two."""

    __slots__ = ("child", "exponent")

    def __init__(self, child, exponent):

        self.child = child
        self.exponent = exponent

    def evaluate(self, context) -> Block:

        if self.exponent == 0:
            return Block(context.nb_rows, 1.0)

        block = self.child.evaluate(context)
        result = block
        for _ in range(self.exponent - 1):
            result = multiply_blocks(result, block)

        return result


class SumNode(Node):
    """
    A sum over a range. The range is laid out along the term axis, so summing
    is what the matrix does when the terms land on the same row. Nothing is
    unrolled here, and the very same rule applies to a sum inside a sum.
    """

    __slots__ = ("child", "time_interval", "index_name", "python_condition", "line")

    def __init__(self, child, time_interval, condition, line):

        self.child = child
        self.time_interval = time_interval
        self.index_name = time_interval.get_index_name()
        self.python_condition = (
            None if condition is None else condition.turn_to_python_expression()
        )
        self.line = line

    def range_of(self, context):
        """
        range_of function : the values of the index, one row per row of the
        enclosing expression. A range depending on an enclosing index is ragged
        and comes back NaN-padded, which is reported as a mask rather than
        left in the values.
        INPUT:  context -> Context object
        OUTPUT: (nb_rows, nb_cells) values, (nb_rows, nb_cells) padding mask
        """

        index_range = self.time_interval.get_range(
            context.definitions, list_indexes=context.index_names
        )

        if not isinstance(index_range, np.ndarray):
            values = np.array(list(index_range), dtype=float)
            index_range = np.broadcast_to(values, (context.nb_rows, len(values)))

        index_range = np.asarray(index_range, dtype=float)
        if index_range.ndim == 1:
            index_range = np.broadcast_to(
                index_range, (context.nb_rows, index_range.shape[0])
            )
        padding = np.isnan(index_range)

        return np.where(padding, 0.0, index_range), padding

    def evaluate(self, context) -> Block:

        nb_rows = context.nb_rows
        index_range, padding = self.range_of(context)
        nb_cells = index_range.shape[1]
        if nb_cells == 0:
            error_(
                "Error: no index in the sum at line " + str(self.line) + " is in range"
            )

        # every index enclosing the sum is repeated so that it holds one value
        # per (row, index of the sum) pair, the shape everything inside the sum
        # is evaluated at.
        enclosing = [name for name in context.index_names if name != self.index_name]
        definitions = dict(context.definitions)
        for name in enclosing:
            definitions[name] = np.repeat(definitions[name], nb_cells)
        definitions[self.index_name] = np.asarray(index_range.reshape(-1), dtype=int)

        inner = Context(definitions, nb_rows * nb_cells, enclosing + [self.index_name])
        padding = self.apply_condition(inner, padding)
        block = self.child.evaluate(inner)

        return self.fold(block, nb_rows, nb_cells, padding)

    def apply_condition(self, inner, padding):
        """
        apply_condition function : an index the `where` condition rejects is
        marked as padding, which makes it inert rather than removed. No row has
        to be re-packed for the ranges to stay rectangular.
        INPUT:  inner -> Context object inside the sum
                padding -> (nb_rows, nb_cells) mask
        OUTPUT: (nb_rows, nb_cells) mask
        """

        if self.python_condition is None:
            return padding

        try:
            kept = eval(self.python_condition, inner.definitions, {})
        except IndexError:
            error_(
                "ERROR: error while evaluating the condition of the sum "
                "at line %s" % str(self.line)
            )

        kept = np.broadcast_to(np.asarray(kept, dtype=bool), (inner.nb_rows,)).reshape(
            padding.shape
        )

        return padding | ~kept

    def fold(self, block, nb_rows, nb_cells, padding):
        """
        fold function : brings a block evaluated over (row, index of the sum)
        pairs back to one over rows, by folding the index of the sum into the
        term axis. A term of `width` entries becomes a term of
        `nb_cells * width` entries.
        INPUT:  block -> Block over nb_rows * nb_cells rows
                nb_rows, nb_cells -> shape the block is folded back to
                padding -> (nb_rows, nb_cells) mask of the inert entries
        OUTPUT: Block over nb_rows rows
        """

        has_padding = padding.any()
        flat_padding = padding.reshape(-1)

        constant = block.constant
        if not is_scalar(constant):
            constant = np.where(flat_padding, 0.0, constant)
            constant = constant.reshape(nb_rows, nb_cells).sum(axis=1)
        elif has_padding:
            constant = constant * (~padding).sum(axis=1)
        else:
            constant = constant * nb_cells

        linear = [
            self.fold_term(term, nb_rows, nb_cells, flat_padding, has_padding)
            for term in block.linear
        ]
        quadratic = [
            self.fold_term(term, nb_rows, nb_cells, flat_padding, has_padding)
            for term in block.quadratic
        ]
        valid = block.valid
        if valid is not None:
            # a padded entry never invalidates the row it sits on
            valid = (valid.reshape(nb_rows, nb_cells) | padding).all(axis=1)
            if valid.all():
                valid = None

        return Block(nb_rows, constant, linear, quadratic, valid)

    def fold_term(self, term, nb_rows, nb_cells, flat_padding, has_padding):

        fields = ["values"] + list(type(term).COLUMN_FIELDS)
        width = nb_cells * term.width
        inner_rows = nb_rows * nb_cells
        values = term.values
        if has_padding:
            # zeroing the value is what makes a padded entry inert, whatever
            # the column it carries
            values = materialize(values, inner_rows, term.width)
            values = np.where(flat_padding[:, None], 0.0, values)

        folded = []
        for name, value in zip(
            fields, [values] + [getattr(term, other) for other in fields[1:]]
        ):
            if is_scalar(value):
                folded.append(value)
            else:
                dtype = float if name == "values" else int
                folded.append(
                    materialize(value, inner_rows, term.width, dtype=dtype).reshape(
                        nb_rows, width
                    )
                )

        return type(term)(*folded, width)


def build_node(expression, variables, constants, collected_variables):
    """
    build_node function : compiles an expression into the tree the evaluation
    walks. Subtrees holding no variable are kept whole, so that a coefficient
    is evaluated once however many variables it multiplies.
    INPUT:  expression -> Expression object
            variables -> dictionary of <name,identifier> objects
            constants -> dictionary of <name,array> objects
            collected_variables -> list the variables met are appended to
    OUTPUT: Node object, True whenever the subtree holds a variable
    """

    e_type = expression.get_type()
    children = expression.get_children()

    if e_type == "literal":
        seed = expression.get_name()
        if type(seed) == Identifier:
            identifier: Identifier = seed
            node_name = identifier.get_node_name()
            name = identifier.get_name()
            if node_name in variables and name in variables[node_name]:
                collected_variables.append([node_name, name])
                variable_identifier = variables[node_name][name].get_identifier()

                return VariableNode(
                    variable_identifier.get_index(),
                    identifier.get_expression(),
                    variable_identifier.get_size(),
                ), True

        return ConstantNode(expression), False

    if e_type == "sum":
        child, has_variable = build_node(
            children[0], variables, constants, collected_variables
        )
        if not has_variable:
            return ConstantNode(expression), False

        return SumNode(
            child,
            expression.get_time_interval(),
            expression.get_condition(),
            expression.get_line(),
        ), True

    sub_nodes = [
        build_node(child, variables, constants, collected_variables)
        for child in children
    ]
    if not any(has_variable for _, has_variable in sub_nodes):
        # the whole subtree is a coefficient, it is never distributed
        return ConstantNode(expression), False

    if e_type == "u-":
        return AdditionNode([[-1, sub_nodes[0][0]]]), True

    if e_type == "+" or e_type == "-":
        return AdditionNode(_flatten_addition(sub_nodes, e_type)), True

    if e_type == "*":
        return MultiplicationNode(sub_nodes[0][0], sub_nodes[1][0]), True

    if e_type == "/":
        return DivisionNode(sub_nodes[0][0], sub_nodes[1][0]), True

    if e_type == "**":
        exponent = children[1].evaluate_as_constant_integer(constants)
        if exponent is None or exponent < 0:
            error_(
                "INTERNAL ERROR : the exponent at line "
                + str(expression.get_line())
                + " is not a constant non "
                "negative integer, check the semantic analysis"
            )

        return PowerNode(sub_nodes[0][0], exponent), True

    error_(
        "The operator '" + str(e_type) + "' is not allowed on variables at "
        "line " + str(expression.get_line())
    )


def _flatten_addition(sub_nodes, e_type):
    """
    _flatten_addition function : gathers a run of + and - into one list of
    signed operands, so that a long sum of terms is concatenated once
    INPUT:  sub_nodes -> the compiled children with their variable predicate
            e_type -> '+' or '-'
    OUTPUT: list of [sign, Node]
    """

    signs = [1, 1 if e_type == "+" else -1]
    flattened = []
    for sign, [node, _] in zip(signs, sub_nodes):
        if isinstance(node, AdditionNode) and sign > 0:
            flattened += node.children
        elif isinstance(node, AdditionNode):
            flattened += [
                [-inner_sign, inner_node] for inner_sign, inner_node in node.children
            ]
        else:
            flattened.append([sign, node])

    return flattened


def gather_nonzero_terms(terms, valid, rows, block_nb_rows):
    """
    gather_nonzero_terms function : turns a list of terms into a flat
    arrays, keeping
    only the entries whose value is not zero.

    Terms are handled one at a time. For each term, its values are turned into
    a (row, entry) rectangle first and reduced to the positions that are not
    zero; the term's columns are then read only at those same positions.
    INPUT:  terms -> non empty list of LinearTerm or QuadraticTerm objects
            valid -> ndarray, the rows to keep
            rows -> ndarray, the row number each kept entry lands on, one
                    entry per row valid keeps
            block_nb_rows -> number of rows the terms were evaluated at,
                             before valid narrows them down -- the shape valid
                             itself indexes against
    OUTPUT: the rows array, then the values and one array per column of the
            term type, all of the same length -- one entry per kept
            (row, term-column) pair
    """

    fields = [("values", float)] + [
        (name, int) for name in type(terms[0]).COLUMN_FIELDS
    ]
    keep_every_row = bool(valid.all())
    gathered = [[np.zeros(0, dtype=int)]] + [
        [np.zeros(0, dtype=dtype)] for _, dtype in fields
    ]

    def entries(term, name, dtype):
        rectangle = materialize(
            getattr(term, name), block_nb_rows, term.width, dtype=dtype
        )

        return rectangle if keep_every_row else rectangle[valid]

    for term in terms:
        values = entries(term, *fields[0])
        kept_rows, kept_entries = np.nonzero(values)
        if len(kept_entries) == 0:
            continue

        gathered[0].append(rows[kept_rows])
        gathered[1].append(values[kept_rows, kept_entries])
        for index, field in enumerate(fields[1:], start=2):
            gathered[index].append(entries(term, *field)[kept_rows, kept_entries])

    return tuple(np.concatenate(arrays) for arrays in gathered)


def build_sparse_array(terms, valid, rows, nb_rows, block_nb_rows):
    """
    build_sparse_array function : build the terms of a block as a coo_array
    whose first axis is the row and whose other axes are the term's own
    COLUMN_FIELDS. Entries at the same (row, *columns) tuple in more than
    one term are summed and cancelled by sum_duplicates/eliminate_zeros.
    INPUT:  terms -> list of LinearTerm or QuadraticTerm objects
            valid, rows, nb_rows, block_nb_rows
    OUTPUT: coo_array of shape (nb_rows, *ncols), or None when `terms` is empty
    """

    if not terms:
        return None

    term_rows, values, *columns = gather_nonzero_terms(
        terms, valid, rows, block_nb_rows
    )
    if len(columns) == 2:
        # a product is commutative so x * y and y * x are the very same entry.
        columns = [np.minimum(*columns), np.maximum(*columns)]
    # indices are never negative, so a column's max is its ncols - 1
    # whenever it holds any, and the initial floor of 0 gives ncols = 1 when it
    # does not
    shape = (nb_rows,) + tuple(int(column.max(initial=0)) + 1 for column in columns)
    sparse = coo_array((values, (term_rows, *columns)), shape=shape)
    sparse.sum_duplicates()
    sparse.eliminate_zeros()

    return sparse


class Factorize:
    """
    Factorize object factorizes a constraint or an objective of any
    degree accepted by the language, that is zero, one or two with respect to
    the variables.
    """

    def __init__(self, obj):

        self.name = None
        self.obj = obj
        self.node = None
        self.variables = []
        self.line = obj.get_line()
        self.type_fact = "objective" if type(obj) == Objective else "constraint"
        self.is_quadratic = False
        self.extension_type = ""
        self.sparse = None
        self.independent_terms = []
        self.quadratic_terms = None

    def get_name(self):

        return self.name

    def get_line(self):

        return self.line

    def free(self):

        self.node = None
        self.obj = None

    def free_extension(self):
        """
        free_extension function : releases what the extension produced, once it
                                  has been gathered in the matrices
        """

        self.sparse = None
        self.independent_terms = None
        self.quadratic_terms = None

    def has_variables(self) -> bool:
        return bool(self.variables)

    def nb_rows(self) -> int:
        """
        nb_rows function : the number of rows the factor expanded to. Every
        degree lands in those same rows, and every one of them carries an
        independent term, so counting them is counting the rows.
        OUTPUT: int
        """

        return len(self.independent_terms)

    def factorize_constraint(self, variables, constants):
        """
        factorize_constraint function : compiles both sides of the constraint
        into a single expression, the left hand side minus the right hand side
        INPUT:  variables -> dictionary of <name,identifier> objects
                constants -> dictionary of <name,array> objects
        OUTPUT: None
        """

        constraint = self.obj
        self.name = constraint.name
        left, _ = build_node(constraint.get_lhs(), variables, constants, self.variables)
        right, _ = build_node(
            constraint.get_rhs(), variables, constants, self.variables
        )
        self.node = AdditionNode([[1, left], [-1, right]])
        self.is_quadratic = constraint.is_quadratic()

    def factorize_objective(self, variables, constants):
        """
        factorize_objective function : compiles the expression of the objective
        INPUT:  variables -> dictionary of <name,identifier> objects
                constants -> dictionary of <name,array> objects
        OUTPUT: None
        """

        objective = self.obj
        self.name = objective.name
        self.node, _ = build_node(
            objective.get_expression(), variables, constants, self.variables
        )
        self.is_quadratic = objective.is_quadratic()

    def row_range(self, gl):
        """
        row_range function : the values the index of the constraint takes, that
        is one per row. An explicit range is used as written; without one, an
        expression which does not depend on the index gives a single row and
        any other one gives a row per time step. A `where` condition then
        filters them.
        INPUT:  gl -> namespace the expressions are evaluated against
        OUTPUT: list of the values of the index, True whenever it was written
                explicitly
        """

        obj = self.obj
        name_index = obj.get_index_var()
        gl["extension_range"] = ByteArray(1)
        time_range = obj.get_time_range(gl)
        explicit = time_range is not None

        if not explicit:
            if self.type_fact == "constraint":
                expressions = [obj.get_rhs(), obj.get_lhs()]
            else:
                expressions = [obj.get_expression()]
            condition = obj.get_condition()
            if condition is not None:
                expressions.append(condition)

            depends = any(
                is_index_dependant_expression(expression, [name_index])
                for expression in expressions
            )
            time_range = range(gl["T"] if depends else 1)

        condition = obj.get_condition()
        if condition is not None:
            gl[name_index] = np.array(time_range)
            gl["extension_range"] = ByteArray(len(gl[name_index]))
            try:
                kept = eval(condition.turn_to_python_expression(), gl, {})
            except IndexError:
                error_(
                    "ERROR: error while evaluating the condition at line %s"
                    % (str(obj.get_line()))
                )
            time_range = [value for value, keep in zip(time_range, kept) if keep]

        return time_range, explicit

    def extend(self, definitions):
        """
        extend function : evaluates the compiled expression over every row of
        the constraint at once and gathers the result in the arrays the
        matrices of the problem are built from
        INPUT:  definitions -> dictionary of <name,array> objects
        OUTPUT: None
        """

        obj = self.obj
        gl = {"sum": sum, "range": range, "len": len, "__f_index__": __f_index2__}
        gl.update(definitions)

        if self.type_fact == "constraint":
            self.extension_type = obj.get_sign()
        else:
            self.extension_type = obj.get_type()

        time_range, explicit = self.row_range(gl)
        name_index = obj.get_index_var()
        nb_rows = len(time_range)
        index_values = np.array(time_range)
        gl[name_index] = index_values

        if nb_rows == 0:
            # a part is None exactly when it holds no entry, which is what
            # build_sparse_array returns for a degree the block never met
            self.sparse = None
            self.quadratic_terms = None
            self.independent_terms = np.zeros(0)

            return

        block = self.node.evaluate(Context(gl, nb_rows, [name_index]))
        valid = block.valid
        if valid is None:
            valid = np.ones(nb_rows, dtype=bool)

        if not valid.any():
            dropped = str(index_values[~valid])
            if explicit:
                error_(
                    "Constraint : "
                    + str(obj)
                    + " at line "
                    + str(obj.get_line())
                    + " has a time range ill-defined "
                    "as a variable goes out of bounds for "
                    + str(name_index)
                    + " equals "
                    + dropped
                )
            print(
                "Warning constraint : %s at line %s is ignored for %s "
                "equal to %s"
                % (str(obj), str(obj.get_line()), str(name_index), dropped)
            )
            self.sparse = None
            self.quadratic_terms = None
            self.independent_terms = np.zeros(0)

            return

        nb_rows = int(np.count_nonzero(valid))
        rows = np.arange(nb_rows)

        constant = as_rows(block.constant, block.nb_rows)
        if is_scalar(constant):
            constant = np.full(nb_rows, constant, dtype=float)
        else:
            constant = constant[valid]

        if self.type_fact == "constraint":
            # the matrix holds the coefficients of the left hand side minus the
            # right hand side, so the independent term moves to the other side
            self.independent_terms = -constant
        else:
            self.independent_terms = constant

        self.sparse = build_sparse_array(
            block.linear, valid, rows, nb_rows, block.nb_rows
        )
        self.quadratic_terms = build_sparse_array(
            block.quadratic, valid, rows, nb_rows, block.nb_rows
        )
