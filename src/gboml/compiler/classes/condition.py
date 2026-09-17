# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).

from .parent import Type
from .expression import Expression
from .symbolic import Operation


class Condition(Type):
    """
    Condition object is a tree like structure where:
    - each node is defined by an operator 
    - internal nodes have children of Condition objects
    - leaf nodes have children of Expressions type
    """

    def __init__(self, type_id: str, children: list, line: int = 0) -> None:
    
        assert type(type_id) == str, \
            "Internal error: expected string for condition type"
        assert type_id in \
               ["and", "not", "or", "==", "!=", "<", ">", "<=", ">="], \
               "Internal error: unknown type for condition"
        Type.__init__(self, type_id, line)
        self.type = type_id
        self.children = children

    def __str__(self) -> str:
    
        string = "["+str(self.type)
        for child in self.children:
            string += ','+str(child)
        string += ']'
    
        return string

    def get_children(self) -> list:
    
        return self.children

    def depends_on_index(self, names) -> bool:
        """A condition depends on an index when one of its children
        does."""

        return any(child.depends_on_index(names) for child in self.children)

    def rename_inside_expressions(self, new_name, old_name):
        children = self.children
        for child in children:
            if isinstance(child, Expression):
                child.rename_node_inside(new_name, old_name)
            elif isinstance(child, Condition):
                child.rename_inside_expressions(new_name, old_name)

    def to_symbolic(self, scope):
        """This condition as a symbolic expression."""
        return Operation(
            self.type, tuple(child.to_symbolic(scope) for child in self.children)
        )