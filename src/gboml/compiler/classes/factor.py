# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).
from typing import TYPE_CHECKING

import numpy as np

from gboml.compiler.utils import error_
from .block import Block
from .constraint import Constraint
from .expression import Expression, Scope
from .identifier import Identifier
from .objective import Objective
from .symbolic import Idx

if TYPE_CHECKING:
    from .symbolic import Expr, Indexed, ParameterSymbol


class Node:
    """A part of an expression holding variables. Subclasses only implement factorize."""

    __slots__ = ()

    def factorize(self) -> Block:

        raise NotImplementedError


class ConstantLeaf(Node):
    """A subtree holding no variable at all, kept whole as one symbolic
    expression rather than distributed."""

    __slots__ = ("symbolic",)

    def __init__(self, symbolic) -> None:

        self.symbolic = symbolic

    def factorize(self) -> Block:

        return Block.of_constant(self.symbolic)


class VariableLeaf(Node):
    """One variable of the problem read at one index."""

    __slots__ = ("symbolic",)

    def __init__(self, symbolic) -> None:

        self.symbolic = symbolic

    def factorize(self) -> Block:

        return Block.of_variable(self.symbolic)


class AdditionNode(Node):
    """Signed operands added together."""

    __slots__ = ("children",)

    def __init__(self, children: list[tuple[int, Node]]) -> None:

        self.children = children

    def factorize(self) -> Block:
        total = None
        for sign, child in self.children:
            block = child.factorize()
            block = block if sign == 1 else block.negated()
            total = block if total is None else total.add(block)

        return total


class MultiplicationNode(Node):
    __slots__ = ("left", "right")

    def __init__(self, left: Node, right: Node) -> None:

        self.left = left
        self.right = right

    def factorize(self) -> Block:

        left = self.left.factorize()
        right = self.right.factorize()
        return left.times(right)


class DivisionNode(Node):
    """A division whose denominator holds no variable, which is therefore a
    scaling of the numerator."""

    __slots__ = ("numerator", "denominator")

    def __init__(self, numerator: Node, denominator: Node) -> None:

        self.numerator = numerator
        self.denominator = denominator

    def factorize(self) -> Block:

        numerator = self.numerator.factorize()
        denominator = self.denominator.factorize()

        return numerator.scaled(1.0 / denominator.constant.coefficient)


class PowerNode(Node):
    """A variable expression raised to a constant integer, 1 or 2, so that the
    result stays of degree two. The degree check makes an exponent of 0 a constant."""

    __slots__ = ("child", "exponent")

    def __init__(self, child: Node, exponent: int) -> None:

        self.child = child
        self.exponent = exponent

    def factorize(self) -> Block:

        child = self.child.factorize()
        result = child
        for _ in range(self.exponent - 1):
            result = result.times(child)

        return result


class SumNode(Node):
    """A sum over a range whose body holds variables."""

    __slots__ = ("child", "range")

    def __init__(self, child: Node, value_range) -> None:

        self.child = child
        self.range = value_range

    def factorize(self) -> Block:

        return self.child.factorize().summed_over(self.range)


def build_node(
    expression: Expression,
    scope: Scope,
    collected_variables: list[list[str]],
) -> Node:
    """
    build_node function : compiles an expression into the tree the
    factorization walks. Subtrees holding no variable are kept whole.
    INPUT:  expression -> Expression object, its degree already checked
            scope -> Scope object the names are looked up in
            collected_variables -> list the variables met are appended to
    OUTPUT: Node object
    """

    if expression.degree == 0:
        return ConstantLeaf(expression.to_symbolic(scope))

    def build(child: Expression, child_scope: Scope = scope) -> Node:
        return build_node(child, child_scope, collected_variables)

    e_type = expression.get_type()
    children: list[Expression] = expression.get_children()

    if e_type == "literal":
        identifier: Identifier = expression.get_name()
        collected_variables.append([identifier.get_node_name(), identifier.get_name()])
        return VariableLeaf(scope.resolve_variable(identifier))

    if e_type == "sum":
        interval = expression.get_time_interval()
        index = Idx(("sum", interval.get_index_name()), scope.table)
        value_range, body_scope = scope.enter_index(index, interval, expression.get_condition())
        return SumNode(build(children[0], body_scope), value_range)

    if e_type == "u-":
        return AdditionNode([(-1, build(children[0]))])

    if e_type in ("+", "-"):
        sign = 1 if e_type == "+" else -1
        return AdditionNode([(1, build(children[0])), (sign, build(children[1]))])

    if e_type == "*":
        return MultiplicationNode(build(children[0]), build(children[1]))

    if e_type == "/":
        return DivisionNode(build(children[0]), build(children[1]))

    if e_type == "**":
        # the degree check gave a power the degree of its base times its exponent
        return PowerNode(build(children[0]), expression.degree // children[0].degree)


class Factorize:
    """
    Factorize object factorizes a constraint or an objective of any
    degree accepted by the language, that is zero, one or two with respect to
    the variables.
    """

    def __init__(self, obj: Constraint | Objective) -> None:

        self.name = obj.name
        self.obj = obj
        self.block = None
        self.rows = None
        self.variables = []
        self.line = obj.get_line()
        if type(obj) == Objective:
            self.type_fact = "objective"
            self.extension_type = obj.get_type()
        else:
            self.type_fact = "constraint"
            self.extension_type = obj.get_sign()
        self.sparse = None
        self.independent_terms = np.zeros(0)
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
        self.independent_terms = np.zeros(0)
        self.quadratic_terms = None

    def has_variables(self) -> bool:
        return self.block.degree() > 0

    def nb_rows(self) -> int:
        """
        nb_rows function : the number of rows the factor expanded to. Every
        degree lands in those same rows, and every one of them carries an
        independent term, so counting them is counting the rows.
        OUTPUT: int
        """

        return len(self.independent_terms)

    def factorize_constraint(self, scope: Scope) -> None:
        """
        factorize_constraint function : writes the left hand side minus the
        right hand side of the constraint as a block
        """

        constraint = self.obj
        self.rows, scope = scope.enter_index(
            Idx("t", scope.table), constraint.get_rows_interval(), constraint.get_condition()
        )
        left = build_node(constraint.get_lhs(), scope, self.variables)
        right = build_node(constraint.get_rhs(), scope, self.variables)
        self.block = AdditionNode([(1, left), (-1, right)]).factorize()

    def factorize_objective(self, scope: Scope) -> None:
        """
        factorize_objective function : writes the expression of the objective as a block
        """

        objective = self.obj
        self.rows, scope = scope.enter_index(
            Idx("t", scope.table), objective.get_rows_interval(), objective.get_condition()
        )
        self.block = build_node(
            objective.get_expression(), scope, self.variables
        ).factorize()

    def derivative(self, target: "ParameterSymbol | Indexed") -> Block:
        """The block with every coefficient differentiated with respect to a parameter or to one entry of a table."""
        self.rows.check_does_not_read(target, self.line)
        self.block.kept_rows.check_does_not_read(target, self.line)
        return self.block.derivative(target)

    def replace(self, replacements: dict["ParameterSymbol", "Expr"]) -> None:
        """Replaces every parameter of `replacements` in the rows and in the block."""
        self.rows = self.rows.replace(replacements)
        self.block = self.block.replace(replacements)

    def extend(self, parameter_values: dict, cache: dict) -> None:
        """
        extend function : evaluates the block over every row of the
        constraint at once, with the given parameter values, and gathers the
        result in the arrays the matrices of the problem are built from
        """

        obj = self.obj
        row_index_values, valid = self.block.evaluate_over(
            self.rows, parameter_values, cache, self.line
        )

        if len(row_index_values) > 0 and not valid.any():
            row_index_name = obj.get_index_var()
            dropped = str(row_index_values[~valid])
            if obj.get_time_interval() is not None:
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

        constant, self.sparse, self.quadratic_terms = self.block.assemble(valid)

        if self.type_fact == "constraint":
            # the matrix holds the coefficients of the left hand side minus the
            # right hand side, so the independent term moves to the other side
            self.independent_terms = -constant
        else:
            self.independent_terms = constant
