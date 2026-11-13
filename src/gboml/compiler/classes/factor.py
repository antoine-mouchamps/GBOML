# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).
from types import CodeType

import numpy as np  # type: ignore

from gboml.compiler.utils import error_
from .block import (
    Block,
    RowMask,
    add_blocks,
    combine_valid,
    multiply_blocks,
)
from .constraint import Constraint
from .expression import Expression
from .identifier import Identifier
from .objective import Objective
from .variable import Variable


class ByteArray:
    """The mask the generated code writes the positions it read a parameter
    out of range at into. It starts at a single position along every axis and
    widens to whatever is read, so it can not be intersected in place."""

    __slots__ = ("array",)

    def __init__(self, rank: int) -> None:
        self.array = np.ones((1,) * rank, dtype=bool)

    def intersect(self, values: np.ndarray) -> None:
        self.array = self.array & values


def __f_index2__(
    indexes: tuple[np.ndarray, ...], all_index: ByteArray, name: np.ndarray
) -> np.ndarray:
    shape = np.shape(name)
    assert len(indexes) == len(shape), (
        "Internal error: a parameter read with the wrong number of indexes"
    )
    clipped = []
    for index, extent in zip(indexes, shape):
        is_in = np.array(((0 <= index) & (index < extent)), dtype=bool)
        all_index.intersect(is_in)
        index = np.array(index, dtype=int)
        if not is_in.all():
            index = np.clip(index, 0, max(extent - 1, 0))
        clipped.append(index)

    return np.array(name[tuple(clipped)], dtype=float)


class Context:
    """
    The namespace an expression is evaluated against, and the shape it is
    evaluated at. `shape` is the rows of the constraint, one axis per index
    they run over, then one axis per sum currently open, outermost first.
    `nb_row_axes` is the limit between both. `index_names` are the indexes in
    scope, each bound to an array of that rank. `filled` is false where the
    open sums do not reach, None meaning they reach everywhere.
    """

    __slots__ = ("definitions", "filled", "index_names", "nb_row_axes", "shape")

    def __init__(
        self,
        definitions: dict,
        shape: tuple[int, ...],
        index_names: list[str],
        nb_row_axes: int,
        filled: np.ndarray | None = None,
    ) -> None:

        self.definitions = definitions
        self.shape = shape
        self.index_names = index_names
        self.nb_row_axes = nb_row_axes
        self.filled = filled

    @property
    def row_shape(self) -> tuple[int, ...]:
        return self.shape[: self.nb_row_axes]

    @property
    def rank(self) -> int:
        return len(self.shape)

    def valid_rows(self, in_bounds: np.ndarray) -> RowMask:
        """
        Which rows survive, given where an index landed in range.
        """

        if self.filled is not None:
            in_bounds = in_bounds | np.logical_not(self.filled)
        in_bounds = np.broadcast_to(in_bounds, self.shape)
        if self.rank > self.nb_row_axes:
            in_bounds = in_bounds.all(axis=tuple(range(self.nb_row_axes, self.rank)))

        return np.array(in_bounds)

    def evaluate(
        self, python_expression: CodeType
    ) -> tuple[np.ndarray, RowMask | None]:
        """
        evaluate function : evaluates a compiled variable free expression over
        the rows. The value comes back at the rank it was evaluated at, an
        expression reading no index at all being the one which is rankless and
        has to be lifted. Indexing a parameter out of its range is reported
        through the `extension_range` mask the generated code writes into,
        which is returned alongside the value rather than kept in shared state.
        INPUT:  python_expression -> compiled python expression
        OUTPUT: value of rank axes, mask of the rows still valid or None
        """

        mask = ByteArray(self.rank)
        self.definitions["extension_range"] = mask
        value = np.asarray(eval(python_expression, self.definitions, {}))
        if value.ndim == 0:
            value = value.reshape((1,) * self.rank)
        if mask.array.all():
            return value, None

        valid = self.valid_rows(mask.array)

        return value, None if valid.all() else valid


class Node:
    """A compiled subexpression. Subclasses only implement evaluate."""

    __slots__ = ()

    def evaluate(self, context: Context) -> Block:

        raise NotImplementedError


class ConstantNode(Node):
    """A subtree holding no variable at all, kept whole and evaluated in one
    go rather than distributed."""

    __slots__ = ("python_expression",)

    def __init__(self, expression: Expression) -> None:

        self.python_expression = expression.turn_to_python_expression()

    def evaluate(self, context: Context) -> Block:

        value, valid = context.evaluate(self.python_expression)

        return Block.of_constant(context.row_shape, context.rank, value, valid)


class VariableNode(Node):
    """
    One variable of the problem, giving one column per position. Its
    `nb_block_indexes` leading indexes choose one block of consecutive
    columns and the rest walk inside it, an ordinary variable being one block.
    """

    __slots__ = ("block_first_columns", "columns_per_index", "index_base",
                 "nb_block_indexes", "python_offsets", "shape")

    def __init__(
        self,
        index_base: int,
        offset_expressions: list[Expression],
        shape: tuple[int, ...],
        block_first_columns: np.ndarray | None = None,
        nb_block_indexes: int = 0,
    ) -> None:

        assert len(offset_expressions) == len(shape), (
            "Internal error: a variable read with the wrong number of indexes"
        )
        # a gathered variable carries its position in its first columns
        self.index_base = 0 if block_first_columns is not None else index_base
        self.block_first_columns = block_first_columns
        self.nb_block_indexes = nb_block_indexes
        self.python_offsets = tuple(
            expression.turn_to_python_expression() for expression in offset_expressions
        )
        self.shape = shape
        # how many columns one unit of each index stands for
        columns_per_index = []
        columns = 1
        for extent in reversed(shape):
            columns_per_index.append(columns)
            columns *= extent
        self.columns_per_index = tuple(reversed(columns_per_index))

    def evaluate(self, context: Context) -> Block:

        valid = None
        block_number = None
        # a variable read with no index at all has no offset to add, so the
        # base is lifted to the rank the block is evaluated at right away
        column = np.full((1,) * context.rank, self.index_base, dtype=int)
        for axis, (python_offset, extent, columns) in enumerate(
            zip(self.python_offsets, self.shape, self.columns_per_index)
        ):
            index_offset, offset_valid = context.evaluate(python_offset)
            valid = combine_valid(valid, offset_valid)
            index_offset = np.asarray(index_offset)
            in_bounds = (0 <= index_offset) & (index_offset < extent)
            if not np.all(in_bounds):
                # an index out of range invalidates its row, and is clamped so
                # that the column stays addressable until the row is dropped
                valid = combine_valid(valid, context.valid_rows(in_bounds))
                index_offset = np.clip(index_offset, 0, max(extent - 1, 0))
            if axis < self.nb_block_indexes:
                block_number = (
                    index_offset if block_number is None
                    else block_number * extent + index_offset
                )
            else:
                column = column + index_offset * columns

        if block_number is not None:
            column = column + self.block_first_columns[block_number]

        return Block.of_variable(context.row_shape, context.rank, column, valid)


class AdditionNode(Node):
    """Signed operands added together, one of them only for a unary minus."""

    __slots__ = ("children",)

    def __init__(self, children: list[tuple[int, Node]]) -> None:

        self.children = children

    def evaluate(self, context: Context) -> Block:

        blocks = []
        for sign, child in self.children:
            block = child.evaluate(context)
            blocks.append(block if sign > 0 else block.negated())

        return add_blocks(blocks)


class MultiplicationNode(Node):
    __slots__ = ("left", "right")

    def __init__(self, left: Node, right: Node) -> None:

        self.left = left
        self.right = right

    def evaluate(self, context: Context) -> Block:

        left = self.left.evaluate(context)
        right = self.right.evaluate(context)

        return multiply_blocks(left, right)


class DivisionNode(Node):
    """A division whose denominator holds no variable, which is therefore a
    scaling of the numerator."""

    __slots__ = ("numerator", "denominator")

    def __init__(self, numerator: Node, denominator: Node) -> None:

        self.numerator = numerator
        self.denominator = denominator

    def evaluate(self, context: Context) -> Block:

        numerator = self.numerator.evaluate(context)
        denominator = self.denominator.evaluate(context)

        return numerator.scaled(
            1.0 / denominator.constant.values,
            combine_valid(numerator.valid, denominator.valid),
        )


class PowerNode(Node):
    """A variable expression raised to a constant non negative integer, small
    enough for the result to stay of degree two."""

    __slots__ = ("child", "exponent")

    def __init__(self, child: Node, exponent: int) -> None:

        self.child = child
        self.exponent = exponent

    def evaluate(self, context: Context) -> Block:

        if self.exponent == 0:
            return Block.of_constant(
                context.row_shape, context.rank, np.ones((1,) * context.rank)
            )

        result = self.child.evaluate(context)
        for _ in range(self.exponent - 1):
            result = multiply_blocks(result, self.child.evaluate(context))

        return result


class SumNode(Node):
    """
    A sum over a range. The range is laid out along the term axis, so summing
    is what the matrix does when the terms land on the same row. Nothing is
    unrolled here, and the very same rule applies to a sum inside a sum.
    """

    __slots__ = (
        "child",
        "expression",
        "index_name",
        "line",
        "python_condition",
        "time_interval",
    )

    def __init__(self, child: Node, expression: Expression) -> None:

        self.child = child
        self.expression = expression
        self.time_interval = expression.get_time_interval()
        self.index_name = self.time_interval.get_index_name()
        condition = expression.get_condition()
        self.python_condition = (
            None if condition is None else condition.turn_to_python_expression()
        )
        self.line = expression.get_line()

    def _index_values(self, context: Context) -> tuple[np.ndarray, np.ndarray]:
        """
        The values of the index, laid out along one new axis after the ones
        the outer sums are evaluated over, and the mask of the ones the range
        reaches. A ragged range has rows of different lengths, which get_range
        brings to the longest, marking with NaN what it leaves.
        """

        bound_index_names = [
            index_name
            for index_name in context.index_names
            if self.time_interval.depends_on_index([index_name])
        ]
        bound_index_shapes = [
            context.definitions[index_name].shape for index_name in bound_index_names
        ]
        outer_shape = np.broadcast_shapes((1,) * context.rank, *bound_index_shapes)

        flat_definitions = dict(context.definitions)
        for index_name in bound_index_names:
            flat_definitions[index_name] = np.broadcast_to(
                context.definitions[index_name], outer_shape
            ).ravel()

        index_range = np.asarray(
            self.time_interval.get_range(
                flat_definitions, list_indexes=bound_index_names
            ),
            dtype=float,
        )

        range_width = index_range.shape[-1]
        index_range = index_range.reshape(outer_shape + (range_width,))

        filled = np.isfinite(index_range)
        index_values = np.where(filled, index_range, 0).astype(int)

        return index_values, filled

    def evaluate(self, context: Context) -> Block:
        index_values, filled = self._index_values(context)
        width = index_values.shape[-1]
        if width == 0:
            error_(
                "Error: no index in the sum at line " + str(self.line) + " is in range"
            )

        outer_index_names = [
            index_name
            for index_name in context.index_names
            if index_name != self.index_name
        ]
        definitions = dict(context.definitions)

        for index_name in outer_index_names:
            definition: np.ndarray = definitions[index_name]
            definitions[index_name] = definition.reshape(definition.shape + (1,))
        definitions[self.index_name] = index_values

        if self.python_condition is not None:
            kept = eval(self.python_condition, definitions, {})
            filled = filled & np.asarray(kept, dtype=bool)

        if context.filled is None:
            inner_filled = filled
        else:
            outer_filled = context.filled.reshape(context.filled.shape + (1,))
            inner_filled = outer_filled & filled

        inner_filled = None if inner_filled.all() else inner_filled

        block = self.child.evaluate(
            Context(
                definitions,
                context.shape + (width,),
                outer_index_names + [self.index_name],
                context.nb_row_axes,
                inner_filled,
            )
        )

        return block.summed_over(width, filled)


def build_node(
    expression: Expression,
    variables: dict,
    constants: dict,
    collected_variables: list[Variable],
) -> tuple[Node, bool]:
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
    children: list[Expression] = expression.get_children()

    if e_type == "literal":
        seed = expression.get_name()
        if type(seed) == Identifier:
            identifier: Identifier = seed
            node_name = identifier.get_node_name()
            name = identifier.get_name()
            if node_name in variables and name in variables[node_name]:
                collected_variables.append([node_name, name])
                variable: Variable = variables[node_name][name]

                return VariableNode(
                    variable.get_identifier().get_index(),
                    identifier.get_expressions(),
                    variable.get_shape(),
                    variable.get_block_first_columns(),
                    variable.get_nb_block_indexes(),
                ), True

        return ConstantNode(expression), False

    if e_type == "sum":
        child, has_variable = build_node(
            children[0], variables, constants, collected_variables
        )
        if not has_variable:
            return ConstantNode(expression), False

        return SumNode(child, expression), True

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
        sign = 1 if e_type == "+" else -1
        return AdditionNode([(1, sub_nodes[0][0]), (sign, sub_nodes[1][0])]), True

    if e_type == "*":
        return MultiplicationNode(sub_nodes[0][0], sub_nodes[1][0]), True

    if e_type == "/":
        return DivisionNode(sub_nodes[0][0], sub_nodes[1][0]), True

    if e_type == "**":
        exponent = children[1].evaluate_as_constant_integer(constants)
        return PowerNode(sub_nodes[0][0], exponent), True

    error_(
        "The operator '" + str(e_type) + "' is not allowed on variables at "
        "line " + str(expression.get_line())
    )


class Factorize:
    """
    Factorize object factorizes a constraint or an objective of any
    degree accepted by the language, that is zero, one or two with respect to
    the variables.
    """

    def __init__(self, obj: Constraint | Objective, indexes: dict) -> None:

        self.obj = obj
        self.name = obj.get_name()
        self.node = None
        self.variables = []
        self.line = obj.get_line()
        self.extension_type = ""
        if type(obj) == Objective:
            self.type_fact = "objective"
            read = [obj.get_expression()]
        else:
            self.type_fact = "constraint"
            read = [obj.get_lhs(), obj.get_rhs()]
        condition = obj.get_condition()
        if condition is not None:
            read.append(condition)
        self.row_indexes = tuple(
            (name.lower(), parameter.get_extent())
            for name, parameter in indexes.items()
            if any(expression.depends_on_index([name.lower()]) for expression in read)
        )
        self.sparse = None
        self.independent_terms = []
        self.quadratic_terms = None

    def get_name(self) -> str:

        return self.name

    def get_line(self) -> int:

        return self.line

    def free_extension(self) -> None:
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

    def factorize_constraint(self, variables: dict, constants: dict) -> None:
        """
        factorize_constraint function : compiles both sides of the constraint
        into a single expression, the left hand side minus the right hand side
        INPUT:  variables -> dictionary of <name,identifier> objects
                constants -> dictionary of <name,array> objects
        OUTPUT: None
        """

        constraint = self.obj
        left, _ = build_node(constraint.get_lhs(), variables, constants, self.variables)
        right, _ = build_node(
            constraint.get_rhs(), variables, constants, self.variables
        )
        self.node = AdditionNode([[1, left], [-1, right]])

    def factorize_objective(self, variables: dict, constants: dict) -> None:
        """
        factorize_objective function : compiles the expression of the objective
        INPUT:  variables -> dictionary of <name,identifier> objects
                constants -> dictionary of <name,array> objects
        OUTPUT: None
        """

        self.node, _ = build_node(
            self.obj.get_expression(), variables, constants, self.variables
        )

    def _row_axes(
        self, definitions: dict
    ) -> tuple[list[tuple[str, np.ndarray]], RowMask | None, bool]:
        """
        _row_axes function : one axis per index the rows run over, the values
        that index takes along it, and the mask a `where` leaves of them. Each
        axis is bound in the definitions, at the rank the rows have.

        INPUT:  definitions -> namespace the expressions are evaluated against
        OUTPUT: list of (name, values), the mask of the rows a `where` keeps
                or None, True whenever a range was written explicitly
        """

        obj = self.obj
        definitions["extension_range"] = ByteArray(1)
        time_range = obj.get_time_range(definitions)
        range_is_written = time_range is not None

        axes = []
        if range_is_written:
            axes.append((obj.get_index_var(), np.array(time_range)))
        axes += [(name, np.arange(extent)) for name, extent in self.row_indexes]

        if not axes:
            # nothing the rows could run over, so the constraint holds once
            axes.append((obj.get_index_var(), np.arange(1)))

        for axis, (name, values) in enumerate(axes):
            shape = [1] * len(axes)
            shape[axis] = len(values)
            definitions[name] = values.reshape(shape)

        condition = obj.get_condition()
        if condition is None:
            return axes, None, range_is_written

        kept = eval(condition.turn_to_python_expression(), definitions, {})
        # a condition reading no index at all gives a single bool
        row_shape = tuple(len(values) for _, values in axes)
        kept = np.array(np.broadcast_to(np.asarray(kept, dtype=bool), row_shape))

        if len(axes) == 1:
            # one axis may lose a row, a grid of two axes or more may not
            name, values = axes[0]
            axes = [(name, values[kept])]
            definitions[name] = axes[0][1]

            return axes, None, range_is_written

        return axes, kept, range_is_written

    def extend(self, definitions: dict) -> None:
        """
        extend function : evaluates the compiled expression over every row of
        the constraint at once and gathers the result in the arrays the
        matrices of the problem are built from
        INPUT:  definitions -> dictionary of <name,array> objects
        OUTPUT: None
        """

        obj = self.obj
        definitions = {
            "sum": sum,
            "range": range,
            "len": len,
            "__f_index__": __f_index2__,
            **definitions,
        }

        if self.type_fact == "constraint":
            self.extension_type = obj.get_sign()
        else:
            self.extension_type = obj.get_type()

        axes, row_condition, range_is_written = self._row_axes(definitions)
        row_shape = tuple(len(values) for _, values in axes)

        if 0 in row_shape:
            # a part is None exactly when it holds no entry
            self.sparse = None
            self.quadratic_terms = None
            self.independent_terms = np.zeros(0)

            return

        block = self.node.evaluate(
            Context(
                definitions,
                row_shape,
                [name for name, _ in axes],
                len(row_shape),
            )
        )
        valid = block.valid
        if valid is None:
            valid = np.ones(row_shape, dtype=bool)

        if not valid.any():
            dropped = str(
                [
                    tuple(
                        int(values[position])
                        for (_, values), position in zip(axes, positions)
                    )
                    for positions in zip(*np.nonzero(~valid))
                ]
            )
            row_index_names = ", ".join(name for name, _ in axes)
            if range_is_written:
                error_(
                    "Constraint : "
                    + str(obj)
                    + " at line "
                    + str(obj.get_line())
                    + " has a time range ill-defined "
                    "as a variable goes out of bounds for "
                    + str(row_index_names)
                    + " equals "
                    + dropped
                )
            print(
                "Warning constraint : %s at line %s is ignored for %s "
                "equal to %s"
                % (str(obj), str(obj.get_line()), row_index_names, dropped)
            )
            self.sparse = None
            self.quadratic_terms = None
            self.independent_terms = np.zeros(0)

            return

        if row_condition is not None:
            valid = valid & row_condition
            if not valid.any():
                self.sparse = None
                self.quadratic_terms = None
                self.independent_terms = np.zeros(0)

                return

        if self.type_fact == "constraint":
            # the matrix holds the coefficients of the left hand side minus the
            # right hand side, so the independent term moves to the other side
            self.independent_terms = -block.constant_over(valid)
        else:
            self.independent_terms = block.constant_over(valid)

        self.sparse, self.quadratic_terms = block.assemble(valid)
