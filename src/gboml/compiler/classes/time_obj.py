# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).

from gboml.compiler.utils import error_
from .expression import Expression
from .symbolic import Number, Range


class Time:
    """
    Time object is a structure composed of 
    - a variable name 
    - a right handside expression
    - a value (evaluation of the right handside)
    """

    def __init__(self, time_var: str, expr: Expression, line: int = None):

        assert type(time_var) == str, \
            "Internal error: expected string for Time identifier"
        assert type(expr) == Expression, \
            "Internal error: unknown type for expression in Time object"
        if time_var != "T":
            error_("Semantic error:" + str(line) + ": Use \"T\"" +
                   " as a symbol for the time horizon. \"" + str(time_var) +
                   "\" is not allowed")
        self.time = time_var
        self.expr = expr
        # evaluated by the semantic analysis, which holds the scope of the model
        self.value = None
        self.line = line

    def __str__(self) -> str:

        string = 'Time Horizon: ' + str(self.time) + '\texpr: ' + str(self.expr)

        return string

    def get_line(self):
        return self.line

    def get_name(self):

        return self.time

    def set_value(self, value):

        self.value = value

    def get_value(self) -> float:

        return self.value

    def get_expression(self) -> Expression:

        return self.expr


class TimeInterval:
    """
    Time Interval object is a structure composed of 
    - a variable name 
    - a begin expression
    - an end expression 
    - a step expression or int if not defined
    """

    def __init__(self, time_var: str, begin: Expression,
                 end: Expression, step, line: int):

        assert type(time_var) == str, \
            "Internal error: expected string for TimeInterval identifier"
        assert type(begin) == Expression, \
            "Internal error: unknown type for begin in TimeInterval object"
        assert type(end) == Expression, \
            "Internal error: unknown type for end in TimeInterval object"
        assert type(step) == Expression or type(step) == int, \
            "Internal error: unknown type for step in TimeInterval object"
        self.name = time_var
        if time_var == "t":
            error_("ERROR: t is a reserved keyword "
                   "and can not be overwritten at line " + str(line))

        self.begin = begin
        if type(step) == int:

            self.step = step
        else:

            self.step = step
        self.end = end
        self.line = line

    def __copy__(self):

        time_int = TimeInterval(self.name, self.begin,
                                self.end, self.step, self.line)

        return time_int

    def get_begin(self):

        return self.begin

    def get_step(self):

        return self.step

    def get_end(self):

        return self.end

    def get_index_name(self):

        return self.name

    def depends_on_index(self, names) -> bool:
        """Whether the bounds of the range depend on an outer index, which
        is what makes it ragged.
        """

        bounds = [self.begin, self.end]
        if self.step is not None and not isinstance(self.step, int):
            bounds.append(self.step)

        return any(bound.depends_on_index(names) for bound in bounds)

    def get_interval(self) -> list:

        return [self.begin, self.step, self.end]

    def to_range(self, index, written_condition, scope):
        """The symbolic range of this interval over `index`, kept where `written_condition` holds."""
        step = Number(self.step, scope.table) if isinstance(self.step, int) else self.step.to_symbolic(scope)
        condition = None if written_condition is None else written_condition.to_symbolic(scope.with_bound_index(self.name, index))
        return Range(index, self.begin.to_symbolic(scope), self.end.to_symbolic(scope), step, condition)

    def rename_inside_expressions(self, new_name, old_name):
        self.begin.rename_node_inside(new_name, old_name)
        self.end.rename_node_inside(new_name, old_name)
        if self.step is not None:
            self.step.rename_node_inside(new_name, old_name)
