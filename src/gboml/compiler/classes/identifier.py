# Copyright (C) 2020 - 2022
# Bardhyl Miftari, Mathias Berger, Hatim Djelassi, Damien Ernst,
# University of Liege .
# Licensed under the MIT License (see LICENSE file).

from .parent import Symbol
from gboml.compiler.utils import error_
import copy
import math


class Identifier(Symbol):
    """
    Identifier object is a structure composed of
    - a type id
    - a name
    - one index expression per axis, in the order they were written
    - one extent per axis
    - an unitialized index field for knowing its position
      if it is a variable

    A variable occupies `size` columns, the product of the extents,
    consecutive from `index` unless it gathers other variables.
    """

    def __init__(self, type_id: str, name_id: str, node_name: str = '',
                 expressions=(), line: int = 0):

        assert type(name_id) == str, \
            "Internal error: expected string for identifier name"
        assert type_id == "basic" or type_id == "assign", \
            "Internal error: unknown type for identifier"
        Symbol.__init__(self, name_id, type_id, line)
        self.expressions = tuple(expressions)
        self.node = None
        self.node_name = node_name
        self.index = 0  # GLOBAL INDEX inside the Ax <= b matrix
        self.shape = None
        self.size = None
        self.option = None

    def __str__(self):

        string = str(self.name)
        if self.node_name != "":
            string = str(self.node_name) + "," + string

        for expression in self.expressions:
            string += str(expression)

        string = '['+string+']'

        return string

    def __copy__(self):

        expressions = tuple(copy.copy(e) for e in self.expressions)
        identifier = Identifier(self.type, self.name,
                                expressions=expressions)
        if self.node is not None:
            identifier.set_node(self.node)
        elif self.node_name != "":
            identifier.node_name = self.node_name

        return identifier

    def set_node(self, node):
        self.node = node
        self.node_name = node.get_name()

    def get_node(self):
        return self.node

    def get_node_name(self):
        return self.node_name

    def set_node_name(self, new_name):
        self.node_name = new_name

    def set_option(self, option):

        self.option = option

    def get_option(self):

        return self.option

    def set_size(self, dictionary):
        shape = tuple(expression.evaluate_expression(dictionary)
                      for expression in self.expressions)
        if any(extent <= 0 for extent in shape):
            error_('ERROR : wrong size for variable : '
                   + str(self) + " at line : " + str(self.get_line()))

        self.shape = shape
        self.size = math.prod(shape)

    def get_size(self):

        return self.size

    def get_shape(self):

        return self.shape

    def name_compare(self, identifier_i) -> bool:

        # a == a[t] ! NO ANYMORE
        equal = False
        if type(identifier_i) == type(self):

            if self.name == identifier_i.name:
                equal = True
        elif type(identifier_i) == str:

            if self.name == identifier_i:
                equal = True

        return equal

    def set_expressions(self, expressions):

        self.expressions = tuple(expressions)

    def get_expressions(self):

        return self.expressions

    def set_expression(self, expr):

        assert len(self.expressions) <= 1, \
            "Internal error: set_expression on a multi axis identifier"
        self.expressions = () if expr is None else (expr,)

    def get_expression(self):

        assert len(self.expressions) <= 1, \
            "Internal error: get_expression on a multi axis identifier"

        return self.expressions[0] if self.expressions else None

    def set_index(self, value):

        self.index = value
        return value + self.size

    def get_index(self):

        return self.index

    def to_python_ast(self):
        import ast
        type_id = self.get_type()
        identifier_name = self.get_name()
        node_name = self.get_node_name()

        if node_name != "":
            identifier_name = node_name+"."+identifier_name

        output = None
        if type_id == "basic":
            output = ast.Name(id=identifier_name, ctx=ast.Load())
        elif type_id == "assign":
            output = ast.Call(func=ast.Name(id="__f_index__", ctx=ast.Load()),
                              args=[
                                  ast.Tuple(
                                      elts=[expression.to_python_ast()
                                            for expression in self.expressions],
                                      ctx=ast.Load()),
                                  ast.Name(id='extension_range', ctx=ast.Load()),
                                  ast.Name(id=identifier_name, ctx=ast.Load())
                              ],
                              keywords=[])

            """
            output = ast.Subscript(value=ast.Name(id=identifier_name, ctx=ast.Load()),
                                   slice=ast.Index(
                                       value=ast.Call(
                                            func=ast.Name(id='__f_index__', ctx=ast.Load()),
                                            args=[
                                                self.expressions[0].to_python_ast(),
                                                ast.Name(id='extension_range', ctx=ast.Load()),
                                                ast.Call(
                                                    func=ast.Name(id='len', ctx=ast.Load()),
                                                    args=[
                                                        ast.Name(id=identifier_name, ctx=ast.Load())],
                                                    keywords=[])],
                                            keywords=[]),
                                       ctx=ast.Load()), ctx=ast.Load())
            """
        return output
