# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).

"""
factorization.py : turns a constraint or an objective into the rows of the
problem.

The expression is compiled once, index free, into a tree of nodes, and
every node knows how to evaluate itself into a Block. The rules are the
algebra of affine expressions:

    constant     a subtree holding no variable, evaluated whole
    variable     one column of the problem
    - and +      the terms of both sides, gathered
    *            a scaling, one of the two sides holding no variable
    /            a scaling, the denominator holding no variable
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
from scipy.sparse import coo_matrix  # type: ignore

from gboml.compiler.utils import error_
from .block import (
    Block,
    LinearTerm,
    add_blocks,
    as_rows,
    as_column,
    materialize,
    combine_valid,
    flatten_linear,
    is_scalar,
)
from .constraint import Constraint
from .expression import Expression
from .identifier import Identifier
from .objective import Objective
from .block import ByteArray, __f_index2__


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


class NegationNode(Node):
    __slots__ = ("child",)

    def __init__(self, child):

        self.child = child

    def evaluate(self, context) -> Block:

        return self.child.evaluate(context).negated()


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
        # check_linear refuses a product of two expressions holding variables,
        # so one of the two sides is a coefficient and the product is a scaling
        if left.is_constant():
            return right.scaled(left.constant).also_valid(left.valid)
        if right.is_constant():
            return left.scaled(right.constant).also_valid(right.valid)

        error_(
            "INTERNAL ERROR : a product of two expressions holding variables "
            "reached the factorization, check the semantic analysis"
        )


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

        return numerator.scaled(1.0 / denominator.constant)


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
        # is evaluated at. The index of the sum may shadow an enclosing one, in
        # which case the enclosing binding is hidden inside the sum and put
        # back untouched on the way out
        definitions = context.definitions
        enclosing = [name for name in context.index_names if name != self.index_name]
        saved = [
            [name, definitions.get(name)] for name in enclosing + [self.index_name]
        ]

        for name in enclosing:
            definitions[name] = np.repeat(definitions[name], nb_cells)
        definitions[self.index_name] = np.asarray(index_range.reshape(-1), dtype=int)

        inner = Context(definitions, nb_rows * nb_cells, enclosing + [self.index_name])
        padding = self.apply_condition(inner, padding)

        try:
            block = self.child.evaluate(inner)
        finally:
            for name, value in saved:
                if value is None:
                    definitions.pop(name, None)
                else:
                    definitions[name] = value

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
            self.fold_term(
                term,
                nb_rows,
                nb_cells,
                flat_padding,
                has_padding,
                ["values", "columns"],
            )
            for term in block.linear
        ]
        valid = block.valid
        if valid is not None:
            # a padded entry never invalidates the row it sits on
            valid = (valid.reshape(nb_rows, nb_cells) | padding).all(axis=1)
            if valid.all():
                valid = None

        return Block(nb_rows, constant, linear, valid)

    def fold_term(self, term, nb_rows, nb_cells, flat_padding, has_padding, fields):

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
        return NegationNode(sub_nodes[0][0]), True

    if e_type == "+" or e_type == "-":
        return AdditionNode(flatten_addition(expression, sub_nodes, e_type)), True

    if e_type == "*":
        return MultiplicationNode(sub_nodes[0][0], sub_nodes[1][0]), True

    if e_type == "/":
        return DivisionNode(sub_nodes[0][0], sub_nodes[1][0]), True

    error_(
        "The operator '" + str(e_type) + "' is not allowed on variables at "
        "line " + str(expression.get_line())
    )


def flatten_addition(expression, sub_nodes, e_type):
    """
    flatten_addition function : gathers a run of + and - into one list of
    signed operands, so that a long sum of terms is concatenated once
    INPUT:  expression -> Expression object of the addition
            sub_nodes -> the compiled children with their variable predicate
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


class Factorize:
    """
    Factorize object factorizes a constraint or an objective of any
    degree accepted by the language, that is zero, one or two with respect to
    the variables.

    factorize_* compiles the expression once, index free. extend binds the
    index of the constraint to every row at once and evaluates the compiled
    tree into a Block, from which the sparse matrix, the independent terms and,
    when there are any, the products of two variables are read off.
    """

    def __init__(self, obj):

        self.name = None
        self.obj = obj
        self.node = None
        self.variables = []
        self.line = obj.get_line()
        self.type_fact = "objective" if type(obj) == Objective else "constraint"
        self.extension_type = ""
        self.sparse = None
        self.independent_terms = []

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
            self.set_empty()

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
            self.set_empty()

            return

        self.gather(block, valid)

    def set_empty(self):

        self.sparse = None
        self.independent_terms = []

    def gather(self, block, valid):
        """
        gather function : reads the rows of the problem off the block, keeping
        only the rows whose variable indexes all landed in bounds
        INPUT:  block -> Block object
                valid -> ndarray of the rows to keep
        OUTPUT: None
        """

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

        self.sparse = self.build_sparse(block, valid, rows, nb_rows)

        if self.sparse is None and self.type_fact == "constraint":
            # a constraint always holds a variable, the semantic analysis
            # refuses one which does not
            self.sparse = coo_matrix(
                (np.zeros(0), (np.zeros(0), np.zeros(0))), shape=(nb_rows, 1)
            )

    def build_sparse(self, block, valid, rows, nb_rows):
        """
        build_sparse function : the degree one part of every row, as a sparse
        matrix. Entries of a same row landing on a same column are summed,
        which is how the terms of a sum add up.
        """

        if not block.linear:
            return None

        values, columns = flatten_linear(block)
        values = values[valid]
        columns = columns[valid]
        width = values.shape[1]
        sparse = coo_matrix(
            (values.reshape(-1), (np.repeat(rows, width), columns.reshape(-1))),
            shape=(nb_rows, int(columns.max()) + 1),
        )
        sparse.sum_duplicates()
        sparse.eliminate_zeros()

        return sparse
