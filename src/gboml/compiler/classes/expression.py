# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).
import numpy as np

from .parent import Symbol
from gboml.compiler.utils import error_, list_to_string
from .identifier import Identifier
from .error import WrongUsage
from .parameter import Parameter
from .symbolic import (
    Array,
    Expr,
    Idx,
    Indexed,
    Number,
    Operation,
    ParameterSymbol,
    Range,
    Sum,
    VariableSymbol,
)
import copy


class Scope:
    """What the names written in one constraint, or in one parameter definition, stand for."""

    def __init__(self, program, variables: dict, definitions: dict) -> None:
        # shared by the whole model: its composite expressions and the value of each parameter;
        # the variables and the definitions are those of the element being translated
        self.table = program.expression_table
        self.parameter_values = program.parameter_values
        self.variables = variables
        self.definitions = definitions
        self.enclosing_element_names = [
            name for name in definitions if name not in ("T", "global")
        ]
        self.bound_indexes = {}
        self.local_parameters = None
        # the paths of the parameters whose definition is being translated, shared by every copy of the scope
        self.parameters_in_translation = set()

    def with_bound_index(self, name: str, index: Idx) -> "Scope":
        """The same scope, with `name` bound to `index`, as a sum or a `for` does."""
        scope = copy.copy(self)
        scope.bound_indexes = {**self.bound_indexes, name: index}
        return scope

    def for_parameter_definition(self, parameters: dict, element_path: str) -> "Scope":
        """The scope of a parameter definition, where a bare name is a parameter of the same element."""
        scope = copy.copy(self)
        scope.bound_indexes = {}
        scope.local_parameters = (parameters, element_path)
        return scope

    def enter_index(
        self, index: Idx, interval, written_condition
    ) -> tuple[Range, "Scope"]:
        """The range an interval gives an index, and the scope in which its letter is bound."""
        letter = interval.get_index_name()
        local_names = () if self.local_parameters is None else self.local_parameters[0]
        if (
            letter in self.bound_indexes
            or letter in self.definitions
            or letter in local_names
        ):
            error_(
                "ERROR: index "
                + letter
                + " for loop already defined. Redefinition at line "
                + str(interval.line)
            )
        value_range: Range = interval.to_range(index, written_condition, self)
        return value_range, self.with_bound_index(letter, index)

    def resolve_variable(self, identifier: Identifier) -> Indexed:
        """A variable read at its written index, its columns taken from its declaration."""
        node_name, name = identifier.get_node_name(), identifier.get_name()
        declaration = self.variables[node_name][name].get_identifier()
        variable = VariableSymbol(
            declaration.get_index(),
            declaration.get_size(),
            node_name + "." + name,
            self.table,
        )
        written_index = identifier.get_expression()
        index = (
            Number(0, self.table)
            if written_index is None
            else written_index.to_symbolic(self)
        )
        return Indexed(variable, (index,))

    def resolve_constant(self, identifier: Identifier) -> Expr:
        """A name holding no variable: an index bound by a sum or the rows, or a parameter read at its written index."""
        node_name, name = identifier.get_node_name(), identifier.get_name()
        if node_name == "" and name in self.bound_indexes:
            return self.bound_indexes[name]
        parameter = self.resolve_parameter(node_name, name, identifier.get_line())
        index = identifier.get_expression()
        return (
            parameter
            if index is None
            else parameter.read_at((index.to_symbolic(self),))
        )

    def resolve_parameter(self, node_name: str, name: str, line: int = 0) -> Expr:
        """The expression a parameter stands for: its definition when that reads other parameters, else a symbol whose value is recorded."""
        parameters, element_path = self.locate_parameter(node_name, name, line)
        path = element_path + "." + name if element_path else name
        parameter = parameters[name]
        if path in self.parameters_in_translation:
            error_(
                "ERROR: the parameter "
                + path
                + " is defined in terms of itself at line "
                + str(parameter.get_line())
            )
        self.parameters_in_translation.add(path)
        definition_scope = self.for_parameter_definition(parameters, element_path)
        if parameter.read_from_file:
            definition = np.array(parameter.get_value(), dtype=float)
        elif parameter.get_expression() is not None:
            definition = parameter.get_expression().to_symbolic(definition_scope)
        else:
            entries = tuple(
                entry.to_symbolic(definition_scope) for entry in parameter.get_vector()
            )
            numbers = [entry.value for entry in entries if isinstance(entry, Number)]
            definition = (
                np.array(numbers, dtype=float)
                if len(numbers) == len(entries)
                else Array(entries)
            )
        self.parameters_in_translation.remove(path)
        if isinstance(definition, Number):
            definition = float(definition.value)
        if isinstance(definition, Expr):
            return definition
        self.parameter_values[path] = definition
        return ParameterSymbol(path, np.shape(definition), self.table)

    def locate_parameter(
        self, node_name: str, name: str, line: int
    ) -> tuple[dict, str]:
        """The parameters a written name is looked up in, and the path of the element holding them."""
        if node_name == "" and name == "T" and "T" in self.definitions:
            parameters, element_path = {"T": self.definitions["T"]}, ""
        elif node_name == "":
            parameters, element_path = self.local_parameters or ({}, "")
        elif node_name == "global":
            parameters, element_path = self.definitions.get("global", {}), "global"
        elif node_name in self.enclosing_element_names:
            # the innermost enclosing element of that name, whose path is every element name down to it
            names = self.enclosing_element_names
            depth = len(names) - names[::-1].index(node_name)
            parameters, element_path = (
                self.definitions[node_name],
                ".".join(names[:depth]),
            )
        else:
            parameters, element_path = {}, ""
        if not isinstance(parameters.get(name), Parameter):
            written_name = name if node_name == "" else node_name + "." + name
            error_(
                "ERROR: Unknown identifier " + written_name + " at line " + str(line)
            )
        return parameters, element_path


class Expression(Symbol):
    """
    Expression object is a tree like structure : 
    Its internal nodes are made up of 
    - an operator as type 
    - Children nodes of type Expression
    Its leafs are made up of 
    - a fixed type -> "literal"
    - no children nodes
    - a name field -> containing an evaluable unit 
    (number, identifier, etc)
    """

    def __init__(self, node_type: str, name=None, line: int = 0):

        assert type(node_type) == str, \
            "Internal error: expected string for Expression type"
        assert node_type in \
               ["+", "-", "/", "*", "**", "u-", "mod", "literal", "sum"], \
            "Internal error: unknown type for Expression"
        Symbol.__init__(self, name, node_type, line)
        self.parent = None
        self.children: list = []
        self.leafs = None
        self.condition = None
        self.time_interval = None
        self.degree = None

    def __str__(self) -> str:

        if self.type != "literal":

            string = '[' + str(self.type)
            if self.name != "":
                string += " , " + str(self.name)
            if len(self.children) == 0:

                string += ']'
            else:

                string += "[" + list_to_string(self.children) + "]]"
        else:

            string = str(self.name)

        return string

    def free(self):

        for child in self.children:
            child.free()
        self.children = []
        if self.time_interval is not None:
            self.time_interval = None
        self.time_interval = None
        self.condition = None
        self.name = None
        self.parent = None
        self.type = None
        self.line = None

    def get_children(self):

        return self.children

    def set_children(self, children):

        self.children = children

    def get_nb_children(self):

        return len(self.children)

    def get_parent(self):

        return self.parent

    def add_parent(self, p):

        self.parent = p

    def add_child(self, child):

        child.add_parent(self)
        self.children.append(child)

    def set_leafs(self, leaves):

        self.leafs = leaves

    def get_leafs(self):

        if self.leafs is None:
            self.leafs = self.find_leafs()

        return self.leafs

    def set_time_interval(self, time_interv):

        self.time_interval = time_interv

    def get_time_interval(self):

        return self.time_interval

    def set_condition(self, cond):

        self.condition = cond

    def get_condition(self):

        return self.condition

    def depends_on_index(self, names) -> bool:
        """Whether the expression reads one of those indexes without
        binding it, which is what makes what it evaluates to vary along
        them."""

        if self.type == "literal":
            seed = self.name
            return type(seed) == Identifier and (
                seed.get_name() in names
                or (
                    seed.get_type() == "assign"
                    and seed.get_expression().depends_on_index(names)
                )
            )

        if any(child.depends_on_index(names) for child in self.children):
            return True

        if self.type == "sum":
            # the index may appear in the range of the sum, as in
            # sum(x[i] for i in [0:t]), which makes the expression depend
            # on it
            return (
                self.time_interval is not None
                and self.time_interval.depends_on_index(names)
            ) or (self.condition is not None and self.condition.depends_on_index(names))

        return False

    def to_symbolic(self, scope):
        """This expression, which holds no variable, as a symbolic expression."""
        if self.type == "literal":
            if isinstance(self.name, Identifier):
                return self.name.to_symbolic(scope)
            return Number(self.name, scope.table)
        if self.type == "u-":
            return -self.children[0].to_symbolic(scope)
        if self.type == "sum":
            index = Idx(("sum", self.time_interval.get_index_name()), scope.table)
            value_range, body_scope = scope.enter_index(
                index, self.time_interval, self.condition
            )
            return Sum(self.children[0].to_symbolic(body_scope), value_range)

        if self.type == "**":
            if self.degree == 0 and self.children[0].degree:
                # the base holds a variable, so the degree check found an exponent of 0
                return Number(1, scope.table)
            return self.children[0].to_symbolic(scope) ** self.children[1].to_symbolic(scope)

        left = self.children[0].to_symbolic(scope)
        right = self.children[1].to_symbolic(scope)
        if self.type == "+":
            return left + right
        if self.type == "-":
            return left - right
        if self.type == "*":
            return left * right
        if self.type == "/":
            return left / right
        return Operation(self.type, (left, right))

    def find_leafs(self):

        all_children = []
        if self.type == "literal" or self.type == "sum":

            all_children = [self]
        else:

            children = self.get_children()
            for child in children:
                all_children += child.get_leafs()

        return all_children

    def __copy__(self):

        copy_name = copy.copy(self.get_name())
        copy_type = copy.copy(self.get_type())

        expr_copy = Expression(copy_type, copy_name, self.get_line())
        children = self.get_children()
        new_children = [copy.copy(child) for child in children]
        time_interval = copy.copy(self.time_interval)
        expr_copy.set_children(new_children)
        expr_copy.set_time_interval(time_interval)

        return expr_copy

    def replace_basic_parameters(self, definitions):

        expr_type = self.get_type()
        name = self.get_name()

        if expr_type == "literal":

            if type(name) == Identifier:

                identifier = name
                id_node_name = identifier.get_node_name()
                id_name = identifier.get_name()
                id_type = identifier.get_type()
                parameter_type = ""
                to_replace = False
                value = 0

                if id_type != "basic":
                    WrongUsage("function replace basic parameters can only be "
                               "used for basic parameters")

                if id_node_name == "" and id_name in definitions:
                    parameter = definitions[id_name]
                    parameter_type = parameter.get_type()
                    value = parameter.get_value()
                    to_replace = True

                elif id_node_name in definitions and \
                        id_name in definitions[id_node_name]:
                    parameter = definitions[id_node_name][id_name]
                    parameter_type = parameter.get_type()
                    value = parameter.get_value()
                    to_replace = True

                if to_replace:
                    if parameter_type != 'expression':
                        WrongUsage("function replace basic parameters can "
                                   "only be used for basic parameters")

                    value = value[0]
                    self.name = value

        else:

            children = self.get_children()
            for child in children:
                child.replace_basic_parameters(definitions)

    def rename_node_inside(self, new_name, old_name):
        leafs = self.get_leafs()
        for leaf in leafs:
            type_id = leaf.get_type()
            if type_id == "literal":
                term = leaf.get_name()
                if isinstance(term, Identifier):
                    node_name = term.get_node_name()
                    if node_name == old_name:
                        term.set_node_name(new_name)
            if type_id == "sum":
                expr_sum = leaf
                expr_sum.get_time_interval().rename_inside_expressions(new_name, old_name)
                if expr_sum.get_condition() is not None:
                    expr_sum.get_condition().rename_inside_expressions(new_name, old_name)
                expr_sub_sum = expr_sum.get_children()[0]
                expr_sub_sum.rename_node_inside(new_name, old_name)
