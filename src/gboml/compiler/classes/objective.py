# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).

from .parent import Type
from .expression import Expression
from .time_obj import TimeInterval


class Objective(Type):
    """
    Objective object is composed of:
    - a type either min or max
    - an expression
    - a quadratic predicate
    """

    def __init__(self, o_type, expression, time_interval=None,
                 condition=None, name=None, line=0):

        assert o_type == "min" or o_type == "max", \
            "Internal error: unknown objective type"
        assert type(expression) == Expression, \
            "Internal error: expected expression type expression in objective"
        Type.__init__(self, o_type, line)
        self.expression = expression
        self.time_interval = time_interval
        self.condition = condition
        self.name = name

    def __str__(self):

        string = "["+str(self.type)+','+str(self.expression)+']'

        return string

    def get_name(self):
        return self.name

    def get_expression(self):
        
        return self.expression

    def get_index_var(self):

        var_name = "t"
        if self.time_interval is not None:

            var_name = self.time_interval.get_index_name()

        return var_name

    def reads_index(self) -> bool:
        """Whether the expression or the `where` condition reads the index."""
        written = [self.expression] + ([self.condition] if self.condition is not None else [])
        return any(part.depends_on_index([self.get_index_var()]) for part in written)

    def get_rows_interval(self) -> TimeInterval:
        """The written `for`, or the one implied: every time step when the index is read, else one step."""
        if self.time_interval is not None:
            return self.time_interval
        if self.reads_index():
            return TimeInterval("t", line=self.line, explicit=False)
        return TimeInterval("t", end=Expression("literal", 0, self.line), line=self.line, explicit=False)

    def get_condition(self):

        return self.condition

    def get_time_interval(self):

        return self.time_interval

    def rename_inside_expressions(self, new_name, old_name):
        self.expression.rename_node_inside(new_name, old_name)
        if self.condition is not None:
            self.condition.rename_inside_expressions(new_name, old_name)
        if self.time_interval is not None:
            self.time_interval.rename_inside_expressions(new_name, old_name)
