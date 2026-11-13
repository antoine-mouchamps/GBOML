# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).

import numpy as np  # type: ignore

from .parent import Symbol
from .identifier import Identifier
from .error import RedefinitionError


def identifiers_of_gathered(gathered):
    """Every identifier an assignment names, however deeply it is nested."""

    if gathered is None:
        return

    if isinstance(gathered, list):
        for member in gathered:
            yield from identifiers_of_gathered(member)
    else:
        yield gathered


class Variable(Symbol):
    """
    Variable object is composed of: 
    - a type 
    - an Identifier object

    Its columns are consecutive from the identifier's index. One gathering
    other variables instead holds one block per gathered variable, listed in
    `block_first_columns` and picked by the `nb_block_indexes` first indexes.
    """

    def __init__(self, identifier: Identifier,
                 v_type: str, v_option: str = "",
                 child_variable=None, line=0):
        assert type(v_type) == str, \
            "Internal error: expected string for variable type"
        assert v_type == "internal" or v_type == "external", \
            "Internal error: unknown variable type "+str(v_type)
        assert type(identifier) == Identifier, \
            "Internal error: identifier must be an Identifier object"
        Symbol.__init__(self, identifier, v_type, line)
        self.option = v_option
        self.dependencies = []
        self.lower_constraint = None
        self.upper_constraint = None
        self.initial_constraint = None
        self.dynamics = None
        self.assignment = None
        self.child_variable = child_variable
        self.block_first_columns = None
        self.nb_block_indexes = 0

    def __str__(self):
        string = "[" + str(self.name) + ' , ' + str(self.type)
        string += ']'

        return string

    def get_child_assignment(self):
        return self.child_variable

    def get_lower_constraint(self):
        return self.lower_constraint

    def set_lower_constraint(self, expression):
        if self.lower_constraint is None:
            self.lower_constraint = expression
        else:
            raise RedefinitionError

    def get_upper_constraint(self):
        return self.upper_constraint

    def set_upper_constraint(self, expression):
        if self.upper_constraint is None:
            self.upper_constraint = expression
        else:
            raise RedefinitionError

    def get_dynamic(self):
        return self.dynamics

    def set_dynamic(self, dynamic):
        if self.dynamics is None:
            self.dynamics = dynamic
        else:
            raise RedefinitionError

    def get_assignment(self):
        return self.assignment

    def set_assignment(self, expression):
        if self.assignment is None:
            self.assignment = expression
        else:
            raise RedefinitionError

    def get_initial_constraint(self):
        return self.initial_constraint

    def set_initial_constraint(self, expression):
        if self.initial_constraint is None:
            self.initial_constraint = expression
        else:
            raise RedefinitionError

    def get_dependencies(self):
        return self.dependencies

    def set_dependencies(self, dependencies):
        self.dependencies = dependencies

    def get_option(self):
        return self.option

    def get_size(self):
        return self.name.get_size()

    def get_shape(self):
        return self.name.get_shape()

    def set_blocks(self, block_first_columns, nb_block_indexes):
        self.block_first_columns = block_first_columns
        self.nb_block_indexes = nb_block_indexes

    def get_block_first_columns(self):
        return self.block_first_columns

    def get_nb_block_indexes(self):
        return self.nb_block_indexes

    def get_columns(self):
        """Where the variable sits, one slice when it is a single block."""

        if self.block_first_columns is None:
            index = self.name.get_index()

            return slice(index, index + self.get_size())

        block_size = self.get_size() // len(self.block_first_columns)

        return (self.block_first_columns[:, None]
                + np.arange(block_size)).ravel()

    def get_identifier(self):
        return self.get_name()

    def reset_type(self, vtype):
        assert vtype == "internal" or vtype == "external", \
            "Internal error: unknown variable type"
        self.type = vtype

    def gathered_identifiers(self):
        return identifiers_of_gathered(self.child_variable)

    def rename_inside_expressions(self, new_name, old_name):
        identifier = self.get_identifier()
        for expr in identifier.get_expressions():
            expr.rename_node_inside(new_name, old_name)

        for gathered in self.gathered_identifiers():
            for expr in gathered.get_expressions():
                expr.rename_node_inside(new_name, old_name)