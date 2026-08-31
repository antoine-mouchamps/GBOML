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
    expr: np.ndarray, all_index: ByteArray, name: np.ndarray
) -> np.ndarray:
    max_length = len(name)
    is_in = np.array(((0 <= expr) & (expr < max_length)), dtype=bool)
    all_index.intersect(is_in)
    expr = np.array(expr, dtype=int)

    if not is_in.all():
        return_value = np.array(name[np.clip(expr, 0, max_length - 1)], dtype=float)
    else:
        return_value = np.array(name[expr], dtype=float)

    return return_value


class Context:
    """
    The namespace expressions are evaluated against, plus the shape they are
    evaluated at. `shape` holds the rows of the constraint, then the range of
    every outer sum, outermost first. `index_names` are the indexes in scope,
    each one bound to an array of the rank of `shape`.
    `filled` is false on the positions the outer sums do not read, None
    standing for the mask that reads them all.
    """

    __slots__ = ("definitions", "filled", "index_names", "shape")

    def __init__(
        self,
        definitions: dict,
        shape: tuple[int, ...],
        index_names: list[str],
        filled: np.ndarray | None = None,
    ) -> None:

        self.definitions = definitions
        self.shape = shape
        self.index_names = index_names
        self.filled = filled

    @property
    def nb_rows(self) -> int:
        return self.shape[0]

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
        if self.rank > 1:
            in_bounds = in_bounds.all(axis=tuple(range(1, self.rank)))

        return np.array(in_bounds)

    def evaluate(
        self, python_expression: CodeType
    ) -> tuple[float | np.ndarray, RowMask | None]:
        """
        evaluate function : evaluates a compiled variable free expression over
        the rows. Indexing a parameter out of its range is reported through the
        `extension_range` mask the generated code writes into, which is
        returned alongside the value rather than kept in shared state.
        INPUT:  python_expression -> compiled python expression
        OUTPUT: value (scalar or ndarray), mask of the rows still valid or None
        """

        mask = ByteArray(self.rank)
        self.definitions["extension_range"] = mask
        value = eval(python_expression, self.definitions, {})
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

        return Block.of_constant(context.nb_rows, context.rank, value, valid)


class VariableNode(Node):
    """One variable of the problem, giving a single column per row."""

    __slots__ = ("index_base", "python_offset", "size")

    def __init__(
        self, index_base: int, offset_expression: Expression | None, size: int
    ) -> None:

        self.index_base = index_base
        self.python_offset = (
            None
            if offset_expression is None
            else offset_expression.turn_to_python_expression()
        )
        self.size = size

    def evaluate(self, context: Context) -> Block:

        valid = None
        if self.python_offset is None:
            index_offset = 0
        else:
            index_offset, valid = context.evaluate(self.python_offset)
            index_offset = np.asarray(index_offset)
            in_bounds = (0 <= index_offset) & (index_offset < self.size)
            if not np.all(in_bounds):
                # an index out of range invalidates its row, and is clamped so
                # that the column stays addressable until the row is dropped
                valid = combine_valid(valid, context.valid_rows(in_bounds))
                index_offset = np.clip(index_offset, 0, max(self.size - 1, 0))

        return Block.of_variable(
            context.nb_rows, context.rank, self.index_base + index_offset, valid
        )


class AdditionNode(Node):
    """A run of + and - gathered in one node, so the terms of a long sum of
    products are concatenated once instead of once per operand."""

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
            1.0 / denominator.constant,
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
            return Block.of_constant(context.nb_rows, context.rank, 1.0)

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

        read = [
            index_name
            for index_name in context.index_names
            if self.time_interval.depends_on_index([index_name])
        ]
        # the range varies exactly where the indexes it reads do, and nowhere
        # else, so it is laid out over those axes alone
        outer_shape = (1,) * context.rank
        for index_name in read:
            outer_shape = tuple(
                map(max, outer_shape, context.definitions[index_name].shape)
            )

        if read:
            # get_range reads one value per row of a flat layout, so those
            # indexes are handed to it flattened over the axes they span and
            # its answer is folded back
            flattened = dict(context.definitions)
            for index_name in read:
                flattened[index_name] = np.broadcast_to(
                    context.definitions[index_name], outer_shape
                ).reshape(-1)
            index_range = self.time_interval.get_range(flattened, list_indexes=read[:1])
        else:
            index_range = self.time_interval.get_range(
                context.definitions, list_indexes=[]
            )

        index_range = np.asarray(index_range, dtype=float)
        if index_range.ndim == 1:
            # the same range on every row, so it is held once and not once per
            # row, whatever the layout it was asked at
            outer_shape = (1,) * context.rank
        index_range = index_range.reshape(outer_shape + index_range.shape[-1:])

        filled = np.isfinite(index_range)
        index_values = np.zeros(index_range.shape, dtype=int)
        np.copyto(index_values, index_range, where=filled, casting="unsafe")

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
            definitions[index_name] = definitions[index_name][..., None]
        definitions[self.index_name] = index_values

        if self.python_condition is not None:
            kept = eval(self.python_condition, definitions, {})
            filled = filled & np.asarray(kept, dtype=bool)

        # an entry is really there when this sum keeps it and every outer one
        # already did
        inner_filled = filled
        if context.filled is not None:
            inner_filled = context.filled[..., None] & inner_filled
        inner_filled = None if inner_filled.all() else inner_filled

        block = self.child.evaluate(
            Context(
                definitions,
                context.shape + (width,),
                outer_index_names + [self.index_name],
                inner_filled,
            )
        )

        return block.summed_over(width, filled)


def build_node(
    expression: Expression,
    variables: dict[str, dict[str, Variable]],
    constants: dict,
    collected_variables: list[list[str]],
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

        return SumNode(child, expression), True

    sub_nodes = [
        build_node(child, variables, constants, collected_variables)
        for child in children
    ]
    if not any(has_variable for _, has_variable in sub_nodes):
        # the whole subtree is a coefficient, it is never distributed
        return ConstantNode(expression), False

    if e_type == "u-":
        return AdditionNode([(-1, sub_nodes[0][0])]), True

    if e_type == "+" or e_type == "-":
        return AdditionNode(_flatten_addition(sub_nodes, e_type)), True

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


def _flatten_addition(
    sub_nodes: list[tuple[Node, bool]], e_type: str
) -> list[tuple[int, Node]]:
    """
    _flatten_addition function : gathers a run of + and - into one list of
    signed operands, so that a long sum of terms is concatenated once
    INPUT:  sub_nodes -> the compiled children with their variable predicate
            e_type -> '+' or '-'
    OUTPUT: list of (sign, Node)
    """

    signs = [1, 1 if e_type == "+" else -1]
    flattened = []
    for sign, (node, _) in zip(signs, sub_nodes):
        if isinstance(node, AdditionNode) and sign > 0:
            flattened += node.children
        elif isinstance(node, AdditionNode):
            flattened += [
                (-inner_sign, inner_node) for inner_sign, inner_node in node.children
            ]
        else:
            flattened.append((sign, node))

    return flattened


class Factorize:
    """
    Factorize object factorizes a constraint or an objective of any
    degree accepted by the language, that is zero, one or two with respect to
    the variables.
    """

    def __init__(self, obj: Constraint | Objective) -> None:

        self.name = None
        self.obj = obj
        self.node = None
        self.variables = []
        self.line = obj.get_line()
        self.type_fact = "objective" if type(obj) == Objective else "constraint"
        self.extension_type = ""
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

    def factorize_constraint(
        self, variables: dict[str, dict[str, Variable]], constants: dict
    ) -> None:
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
        self.node = AdditionNode([(1, left), (-1, right)])

    def factorize_objective(
        self, variables: dict[str, dict[str, Variable]], constants: dict
    ) -> None:
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

    def _row_index_values(
        self, definitions: dict
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.int_]], bool]:
        """
        _row_index_values function : the values the index of the constraint
        takes, that is one per row -- they are the rows.
        INPUT:  definitions -> namespace the expressions are evaluated against
        OUTPUT: the values of the index, True whenever the range was written
                explicitly
        """

        obj = self.obj
        index_name = obj.get_index_var()
        condition = obj.get_condition()
        definitions["extension_range"] = ByteArray(1)
        index_values = obj.get_time_range(definitions)
        explicit = index_values is not None

        if not explicit:
            if self.type_fact == "constraint":
                expressions = [obj.get_rhs(), obj.get_lhs()]
            else:
                expressions = [obj.get_expression()]
            if condition is not None:
                expressions.append(condition)

            depends = any(
                expression.depends_on_index([index_name])
                for expression in expressions
            )
            index_values = range(definitions["T"] if depends else 1)

        index_values = np.array(index_values)

        if condition is not None:
            definitions[index_name] = index_values
            definitions["extension_range"] = ByteArray(1)
            try:
                kept = eval(condition.turn_to_python_expression(), definitions, {})
            except IndexError:
                error_(
                    "ERROR: error while evaluating the condition at line %s"
                    % (str(obj.get_line()))
                )
            # a condition which does not read the index gives a single bool,
            # keeping either every row or none of them
            kept = np.broadcast_to(np.asarray(kept, dtype=bool), index_values.shape)
            index_values = index_values[kept]

        return index_values, explicit

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

        row_index_values, explicit = self._row_index_values(definitions)
        row_index_name = obj.get_index_var()
        nb_rows = len(row_index_values)
        definitions[row_index_name] = row_index_values

        if nb_rows == 0:
            # a part is None exactly when it holds no entry
            self.sparse = None
            self.quadratic_terms = None
            self.independent_terms = np.zeros(0)

            return

        block = self.node.evaluate(Context(definitions, (nb_rows,), [row_index_name]))
        valid = block.valid
        if valid is None:
            valid = np.ones(nb_rows, dtype=bool)

        if not valid.any():
            dropped = str(row_index_values[~valid])
            if explicit:
                error_(
                    "Constraint : "
                    + str(obj)
                    + " at line "
                    + str(obj.get_line())
                    + " has a time range ill-defined "
                    "as a variable goes out of bounds for "
                    + str(row_index_name)
                    + " equals "
                    + dropped
                )
            print(
                "Warning constraint : %s at line %s is ignored for %s "
                "equal to %s"
                % (str(obj), str(obj.get_line()), str(row_index_name), dropped)
            )
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
