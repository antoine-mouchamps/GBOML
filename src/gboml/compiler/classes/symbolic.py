from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from gboml.compiler.utils import error_

OPERATIONS: dict[str, Callable[..., np.ndarray]] = {
    "**": np.float_power,
    "mod": np.mod,
    "==": np.equal,
    "!=": np.not_equal,
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "and": np.logical_and,
    "or": np.logical_or,
    "not": np.logical_not,
    # never written in a model, only built by the derivative of a power
    "log": np.log,
}


def check_parameter_values(
    parameters: frozenset["ParameterSymbol"],
    parameter_values: dict[str, float | np.ndarray],
) -> None:
    """Refuses to evaluate when a parameter has no value, or a value of another shape than the one it was compiled with."""
    for parameter in parameters:
        if parameter.path not in parameter_values:
            error_(
                "INTERNAL ERROR: no value was given for the parameter "
                + parameter.path
            )
        if np.shape(parameter_values[parameter.path]) != parameter.shape:
            error_(
                "ERROR: the parameter "
                + parameter.path
                + " was given a value whose shape differs from the one "
                "it was compiled with, which requires compiling the model again"
            )


class Context:
    """
    The axes an expression is evaluated over: the row axes first, then one
    width per open range. Every array has the rank of the context, an axis it
    does not vary along being of length one. The values of the parameters, and
    what was already evaluated with them, travel along.
    """

    __slots__ = (
        "shape",
        "index_values",
        "filled",
        "ranges",
        "line",
        "nb_row_axes",
        "parameter_values",
        "cache",
    )

    def __init__(
        self,
        parameter_values: dict[str, float | np.ndarray],
        shape: tuple[int, ...],
        index_values: dict["Idx", np.ndarray],
        cache: dict[tuple["Expr", tuple["Range", ...]], tuple[np.ndarray | None, ...]],
        *,
        filled: np.ndarray | None = None,
        ranges: tuple["Range", ...] = (),
        line: int = 0,
        nb_row_axes: int = 1,
    ) -> None:
        # one range per axis, the rows being the values their range keeps
        assert len(ranges) == len(shape)
        self.shape = shape
        self.index_values = index_values
        self.filled = filled
        self.ranges = ranges
        self.line = line
        self.nb_row_axes = nb_row_axes
        self.parameter_values = parameter_values
        self.cache = cache

    @property
    def rank(self) -> int:
        return len(self.shape)

    def widened(
        self, added_range: "Range", values: np.ndarray, filled: np.ndarray
    ) -> "Context":
        """The context with one more axis, along which the index of `added_range` takes `values`, kept where `filled`."""
        index_values = {
            index: array.reshape(array.shape + (1,))
            for index, array in self.index_values.items()
        }
        index_values[added_range.index] = values
        if self.filled is not None:
            filled = self.filled.reshape(self.filled.shape + (1,)) & filled
        return Context(
            self.parameter_values,
            self.shape + (values.shape[-1],),
            index_values,
            self.cache,
            filled=None if filled.all() else filled,
            ranges=self.ranges + (added_range,),
            line=self.line,
            nb_row_axes=self.nb_row_axes,
        )

    def other_axes(self, read_ranges: tuple["Range", ...]) -> tuple[int, ...]:
        """The axes of the ranges outside `read_ranges`."""
        return tuple(
            axis
            for axis, value_range in enumerate(self.ranges)
            if value_range not in read_ranges
        )

    def without_other_axes(
        self, arrays: np.ndarray | tuple[np.ndarray, ...], read_ranges: tuple["Range", ...]
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """The array, or each array of a tuple, without the axes outside `read_ranges`, which are of length one."""
        axes = self.other_axes(read_ranges)
        if isinstance(arrays, tuple):
            return tuple(np.squeeze(array, axis=axes) for array in arrays)
        return np.squeeze(arrays, axis=axes)

    def with_other_axes(
        self, arrays: np.ndarray | tuple[np.ndarray, ...], read_ranges: tuple["Range", ...]
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """The array, or each array of a tuple, given back a length one axis for every range outside `read_ranges`."""
        axes = self.other_axes(read_ranges)
        if isinstance(arrays, tuple):
            return tuple(np.expand_dims(array, axes) for array in arrays)
        return np.expand_dims(arrays, axes)

    def ranges_read_by(self, expression: "Expr") -> tuple["Range", ...]:
        """The ranges of the context whose axes the values of `expression` vary along."""
        needed, read_ranges = set(expression.free_indexes), []
        for value_range in reversed(self.ranges):
            if value_range.index in needed:
                read_ranges.append(value_range)
                # the values of an index vary along whatever its bounds read
                needed |= value_range.free_indexes
        return tuple(reversed(read_ranges))

    def read_positions(self, index: "Expr", size: int) -> np.ndarray:
        """The positions `index` reads in a table of `size` entries, clipped into it."""
        values = np.asarray(index.evaluate_over(self))
        if values.dtype.kind == "f":
            not_integer = values != np.round(values)
            if self.filled is not None:
                not_integer = not_integer & self.filled
            if not_integer.any():
                error_(
                    "ERROR: an index takes a value which is not an integer at line "
                    + str(self.line)
                )
        # a row reading a missing entry is dropped through missing_entry_count, whatever is read here
        return np.clip(values.astype(int), 0, max(size - 1, 0))


class Expr:
    """
    A symbolic expression of a model. A leaf is compared by value, a composite
    expression is created once in the table of its model, and every expression
    keeps that table, which is where combining it with others creates the result.
    """

    free_indexes: frozenset["Idx"] = frozenset()
    parameters_read: frozenset["ParameterSymbol"] = frozenset()

    def __getstate__(self) -> dict[str, object]:
        # the table stays behind, since its keys hold the very expressions being unpickled
        return {name: value for name, value in self.__dict__.items() if name != "table"}

    @property
    def is_zero(self) -> bool:
        return False

    def __add__(self, other: "Expr | float") -> "Expr":
        return Add([(self, 1), (other, 1)])

    __radd__ = __add__

    def __sub__(self, other: "Expr | float") -> "Expr":
        return Add([(self, 1), (other, -1)])

    def __rsub__(self, other: "Expr | float") -> "Expr":
        return Add([(other, 1), (self, -1)])

    def __neg__(self) -> "Expr":
        return Add([(self, -1)])

    def __mul__(self, other: "Expr | float") -> "Expr":
        return Mul([(self, 1), (other, 1)])

    __rmul__ = __mul__

    def __truediv__(self, other: "Expr | float") -> "Expr":
        return Mul([(self, 1), (other, -1)])

    def __rtruediv__(self, other: "Expr | float") -> "Expr":
        return Mul([(other, 1), (self, -1)])

    def __pow__(self, exponent: "Expr | float") -> "Expr":
        exponent = _as_expression(exponent, self.table)
        if isinstance(exponent, Number) and float(exponent.value).is_integer():
            return Mul([(self, int(exponent.value))])
        return Operation("**", (self, exponent))

    def as_coeff_mul(self) -> tuple[float, tuple[tuple["Expr", int], ...]]:
        """The number this expression is scaled by, and the factors of the rest with their exponents."""
        return 1, ((self, 1),)

    def read_at(self, indexes: tuple["Expr", ...]) -> "Expr":
        """A table valued expression read at `indexes`, the read moved onto every table it is built from."""
        return self

    def replace(self, replacements: dict["Idx | ParameterSymbol", "Expr"]) -> "Expr":
        """The expression with every index or parameter of `replacements` replaced, a letter bound by a sum included."""
        for name, replacement in replacements.items():
            if replacement.value_shape() != name.value_shape():
                error_(
                    "ERROR: "
                    + repr(name)
                    + " can not be replaced by "
                    + repr(replacement)
                    + ", whose value has another shape"
                )
        replaced = frozenset(replacements)
        results = {}

        def visit(expression: Expr) -> Expr:
            if expression in replacements:
                return replacements[expression]
            if not expression.reads_any(replaced):
                return expression
            if expression not in results:
                results[expression] = expression._replace(visit)
            return results[expression]

        return visit(self)

    def reads_any(self, names: frozenset["Idx | ParameterSymbol"]) -> bool:
        """Whether the expression reads one of `names`, which replacing them has to rebuild."""
        return not (
            names.isdisjoint(self.free_indexes)
            and names.isdisjoint(self.parameters_read)
        )

    def _replace(self, visit: Callable[["Expr"], "Expr"]) -> "Expr":
        """This expression rebuilt from its arguments, each of them replaced through `visit`."""
        error_("INTERNAL ERROR: " + repr(self) + " can not be replaced")

    def check_does_not_read(self, target: "ParameterSymbol | Indexed", line: int = 0) -> None:
        """Refuses a derivative with respect to `target` when it decides which entries this expression reads or which rows it keeps."""
        if not target.parameters_read.isdisjoint(self.parameters_read):
            error_(
                "ERROR: the derivative with respect to "
                + repr(target)
                + " can not be taken, since it decides which entries are read or which rows are kept"
                + (" at line " + str(line) if line else "")
            )

    def derivative(self, target: "ParameterSymbol | Indexed") -> "Expr":
        """The derivative with respect to a parameter holding one number, or to one entry `table[k]` of a table parameter."""
        if isinstance(target, ParameterSymbol) and target.ndim > 0:
            error_(
                "ERROR: the derivative with respect to the table "
                + target.path
                + " can not be taken, only with respect to one of its entries"
            )
        if target.parameters_read.isdisjoint(self.parameters_read):
            return Number(0, self.table)
        return self._derivative(target)

    @property
    def missing_entry_count(self) -> "Expr":
        """The number of reads of a table or a variable outside of its entries, one number per value of the indexes read."""
        return Number(0, self.table)

    def _derivative(self, target: "ParameterSymbol | Indexed") -> "Expr":
        """The derivative of an expression reading the parameter of `target`."""
        error_(
            "ERROR: the derivative of "
            + repr(self)
            + " with respect to "
            + repr(target)
            + " can not be taken"
        )

    def evaluate_over(self, context: Context) -> np.ndarray:
        """The values of the expression over the context, computed once per evaluation."""
        read_ranges = context.ranges_read_by(self)
        key = (self, read_ranges)
        if key not in context.cache:
            context.cache[key] = context.without_other_axes(
                self.compute(context), read_ranges
            )
        return context.with_other_axes(context.cache[key], read_ranges)

    def compute(self, context: Context) -> np.ndarray:
        """The values of the expression over the context."""
        error_("INTERNAL ERROR: " + repr(self) + " can not be evaluated")

    def value_shape(self, line: int = 0) -> tuple[int, ...]:
        """The shape of the value of this expression, `()` for a number."""
        entry = Idx("entry", self.table)
        read = self.read_at((entry,))
        if read is self:
            return ()
        lengths = _lengths_read_at(read, entry)
        if len(lengths) != 1:
            error_(
                "ERROR: tables of different lengths are combined entry by entry"
                + (" at line " + str(line) if line else "")
            )
        return (lengths.pop(),)

    def evalf(
        self, parameter_values: dict[str, float | np.ndarray], line: int = 0
    ) -> float | np.ndarray:
        """The value of this expression, which reads no index: a number, or a table when it reads a whole table."""
        if self.free_indexes:
            error_("INTERNAL ERROR: " + repr(self) + " reads an index, so it has no value")
        check_parameter_values(self.parameters_read, parameter_values)
        entry = Idx("entry", self.table)
        read = self.read_at((entry,))
        if read is self:
            context = Context(parameter_values, (), {}, {}, line=line)
        else:
            [length] = self.value_shape(line)
            entries = Range(
                entry,
                Number(0, self.table),
                Number(length - 1, self.table),
                Number(1, self.table),
            )
            context = Context(
                parameter_values,
                (length,),
                {entry: np.arange(length)},
                {},
                ranges=(entries,),
                line=line,
            )
        if np.any(read.missing_entry_count.evaluate_over(context)):
            error_("ERROR: a table is read outside of its entries at line " + str(line))
        values = read.evaluate_over(context)
        if read is self:
            return float(values)
        return np.broadcast_to(values, context.shape).astype(float)


class Atom(Expr):
    """An expression holding no other expression, compared by value: a number, an index, a parameter or a variable."""

    @property
    def sort_key(self) -> tuple[int, str, str]:
        return 0, type(self).__name__, repr(self)


class Composite(Expr):
    """An expression made of other expressions, which its constructor creates only once in the table of its model."""

    @property
    def args(self) -> tuple[Expr, ...]:
        return ()

    def find_free_indexes(self) -> frozenset["Idx"]:
        """The indexes the expression reads without binding them."""
        return frozenset().union(*(arg.free_indexes for arg in self.args))

    def find_parameters_read(self) -> frozenset["ParameterSymbol"]:
        return frozenset().union(*(arg.parameters_read for arg in self.args))

    @cached_property
    def missing_entry_count(self) -> Expr:
        return self._missing_entry_count()

    def _missing_entry_count(self) -> Expr:
        """The reads of missing entries of the arguments, added up."""
        return Add(
            [(Number(0, self.table), 1)]
            + [(arg.missing_entry_count, 1) for arg in self.args]
        )

    def __reduce__(
        self,
    ) -> tuple[Callable[[type], "Composite"], tuple[type], dict[str, object]]:
        # unpickling must not go through the constructor, which looks the expression up in a table
        return _uninitialized, (type(self),), self.__getstate__()


@dataclass(frozen=True)
class Number(Atom):
    """A number."""

    value: int | float
    table: dict[tuple[object, ...], Composite] = field(compare=False, repr=False)

    @property
    def is_zero(self) -> bool:
        return self.value == 0

    def as_coeff_mul(self) -> tuple[float, tuple[tuple[Expr, int], ...]]:
        return self.value, ()

    def compute(self, context: Context) -> np.ndarray:
        return np.full((1,) * context.rank, self.value)

    def __repr__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Idx(Atom):
    """An index. The row index is named `t`, the letter of a sum `("sum", written letter)`, and a range of a term `("term", position)`."""

    name: str | tuple[str | int, ...]
    table: dict[tuple[object, ...], Composite] = field(compare=False, repr=False)

    @property
    def free_indexes(self) -> frozenset["Idx"]:
        return frozenset((self,))

    def compute(self, context: Context) -> np.ndarray:
        return context.index_values[self]

    def __repr__(self) -> str:
        return (
            self.name if isinstance(self.name, str) else "_".join(map(str, self.name))
        )


@dataclass(frozen=True)
class ParameterSymbol(Atom):
    """A parameter, whose value is looked up in the values an evaluation is given."""

    path: str
    # fixed at compilation, as the rows a constraint keeps depend on the length of the tables it reads
    shape: tuple[int, ...]
    table: dict[tuple[object, ...], Composite] = field(compare=False, repr=False)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def length(self) -> int:
        return self.shape[0]

    @property
    def parameters_read(self) -> frozenset["ParameterSymbol"]:
        return frozenset((self,))

    def read_at(self, indexes: tuple[Expr, ...]) -> Expr:
        return Indexed(self, indexes) if self.ndim > 0 else self

    def _derivative(self, target: "ParameterSymbol | Indexed") -> Expr:
        if self == target:
            return Number(1, self.table)
        return super()._derivative(target)

    def compute(self, context: Context) -> np.ndarray:
        return np.full((1,) * context.rank, context.parameter_values[self.path])

    def entry_values(self, context: Context) -> np.ndarray:
        """The entries of this table parameter."""
        return np.asarray(context.parameter_values[self.path], dtype=float)

    def __repr__(self) -> str:
        return self.path


@dataclass(frozen=True)
class VariableSymbol(Atom):
    """A variable, whose entries are the columns from `first_column` on."""

    first_column: int
    size: int
    # left out of the comparison, so that an alias and the variable it names are one variable
    path: str = field(compare=False)
    table: dict[tuple[object, ...], Composite] = field(compare=False, repr=False)

    @property
    def length(self) -> int:
        return self.size

    def read_at(self, indexes: tuple[Expr, ...]) -> Expr:
        return Indexed(self, indexes)

    def entry_values(self, context: Context) -> np.ndarray:
        """The columns of the entries of this variable."""
        return self.first_column + np.arange(self.size)

    def __repr__(self) -> str:
        return self.path


class Indexed(Composite):
    """A variable, a parameter table or an array read at `indexes`."""

    base: "VariableSymbol | ParameterSymbol | Array"
    indexes: tuple[Expr, ...]

    def __new__(
        cls,
        base: "VariableSymbol | ParameterSymbol | Array",
        indexes: tuple[Expr, ...],
    ) -> "Indexed":
        return _shared(cls, base.table, base=base, indexes=indexes)

    @property
    def args(self) -> tuple[Expr, ...]:
        return (self.base, *self.indexes)

    def _replace(self, visit: Callable[[Expr], Expr]) -> Expr:
        return visit(self.base).read_at(tuple(map(visit, self.indexes)))

    def _derivative(self, target: "ParameterSymbol | Indexed") -> Expr:
        for index in self.indexes:
            index.check_does_not_read(target)
        if isinstance(self.base, Array):
            entries = tuple(entry.derivative(target) for entry in self.base.entries)
            return Array(entries).read_at(self.indexes)
        if isinstance(target, Indexed) and target.base == self.base:
            # the read is the target entry only where its position is the position of the target
            return Operation("==", (self.indexes[0], target.indexes[0]))
        return super()._derivative(target)

    def _missing_entry_count(self) -> Expr:
        [index] = self.indexes
        reads_missing_entry = Operation(
            "or",
            (Operation("<", (index, 0)), Operation(">=", (index, self.base.length))),
        )
        return super()._missing_entry_count() + reads_missing_entry

    def compute(self, context: Context) -> np.ndarray:
        """The entries of the base at the positions its indexes read: columns for a variable, values for a table."""
        table = self.base.entry_values(context)
        row_index = Idx("t", self.table)
        if all(value_range.index != row_index for value_range in context.ranges_read_by(self)):
            # a position that does not depend on the row has no row to drop when its entry does not exist
            positions = np.asarray(self.indexes[0].evaluate_over(context))
            missing = (positions < 0) | (positions >= len(table))
            if context.filled is not None:
                missing = missing & context.filled
            if missing.any():
                error_(
                    "ERROR: " + repr(self) + " reads an entry that does not exist at line " + str(context.line)
                )
        return table[context.read_positions(self.indexes[0], len(table))]

    def __repr__(self) -> str:
        return f"{self.base}[{', '.join(map(repr, self.indexes))}]"


class Array(Composite):
    """A table written entry by entry, where at least one entry is more than a number."""

    entries: tuple[Expr, ...]

    def __new__(cls, entries: tuple[Expr, ...]) -> "Array":
        table = entries[0].table
        probe = Idx("entry", table)
        if any(entry.read_at((probe,)) is not entry for entry in entries):
            error_(
                "ERROR: the table {"
                + ", ".join(map(repr, entries))
                + "} has an entry that is a whole table"
            )
        return _shared(cls, table, entries=entries)

    @property
    def args(self) -> tuple[Expr, ...]:
        return self.entries

    @property
    def length(self) -> int:
        return len(self.entries)

    def read_at(self, indexes: tuple[Expr, ...]) -> Expr:
        return Indexed(self, indexes)

    def _replace(self, visit: Callable[[Expr], Expr]) -> Expr:
        return Array(tuple(map(visit, self.entries)))

    def entry_values(self, context: Context) -> np.ndarray:
        """The number each entry evaluates to, since no entry reads an index."""
        return np.array(
            [
                np.asarray(entry.evaluate_over(context), dtype=float).reshape(-1)[0]
                for entry in self.entries
            ]
        )

    def __repr__(self) -> str:
        return "{" + ", ".join(map(repr, self.entries)) + "}"


class Add(Composite):
    """Expressions added together, each with a number coefficient, plus a number."""

    terms: tuple[tuple[Expr, float], ...]
    constant: float

    def __new__(
        cls, terms: Iterable[tuple[Expr | float, float]], constant: float = 0
    ) -> Expr:
        terms = list(terms)
        table = _table_of(expression for expression, _ in terms)
        coefficients = defaultdict(int)
        for expression, coefficient in terms:
            if not isinstance(expression, Expr):
                constant += coefficient * expression
            elif isinstance(expression, Number):
                constant += coefficient * expression.value
            elif isinstance(expression, Add):
                constant += coefficient * expression.constant
                for term, term_coefficient in expression.terms:
                    coefficients[term] += coefficient * term_coefficient
            else:
                coefficients[expression] += coefficient
        kept = sorted(
            (item for item in coefficients.items() if item[1] != 0),
            key=lambda item: item[0].sort_key,
        )
        if not kept:
            return Number(constant, table)
        if len(kept) == 1 and kept[0][1] == 1 and constant == 0:
            return kept[0][0]
        return _shared(cls, table, terms=tuple(kept), constant=constant)

    @property
    def args(self) -> tuple[Expr, ...]:
        return tuple(term for term, _ in self.terms)

    def as_coeff_mul(self) -> tuple[float, tuple[tuple[Expr, int], ...]]:
        if len(self.terms) == 1 and self.constant == 0:
            [(term, coefficient)] = self.terms
            return coefficient, term.as_coeff_mul()[1]
        return super().as_coeff_mul()

    def read_at(self, indexes: tuple[Expr, ...]) -> Expr:
        return Add(
            [(term.read_at(indexes), coefficient) for term, coefficient in self.terms],
            self.constant,
        )

    def _replace(self, visit: Callable[[Expr], Expr]) -> Expr:
        return Add(
            [(visit(term), coefficient) for term, coefficient in self.terms],
            self.constant,
        )

    def _derivative(self, target: "ParameterSymbol | Indexed") -> Expr:
        return Add(
            [
                (term.derivative(target), coefficient)
                for term, coefficient in self.terms
            ]
        )

    def compute(self, context: Context) -> np.ndarray:
        values = self.constant
        for term, coefficient in self.terms:
            values = values + coefficient * term.evaluate_over(context)
        return values

    def __repr__(self) -> str:
        written = [
            repr(term) if coefficient == 1 else f"{coefficient}*{term!r}"
            for term, coefficient in self.terms
        ]
        if self.constant != 0:
            written.append(str(self.constant))
        return "(" + " + ".join(written) + ")"


class Mul(Composite):
    """Expressions multiplied together, each with an integer exponent."""

    factors: tuple[tuple[Expr, int], ...]

    def __new__(cls, factors: Iterable[tuple[Expr | float, int]]) -> Expr:
        factors = list(factors)
        table = _table_of(expression for expression, _ in factors)
        coefficient = 1
        exponents = defaultdict(int)
        for expression, exponent in factors:
            number, expression_factors = _as_expression(
                expression, table
            ).as_coeff_mul()
            # zero raised to a negative exponent is a division by zero
            coefficient *= (
                float("inf") if exponent < 0 and number == 0 else number**exponent
            )
            for factor, factor_exponent in expression_factors:
                exponents[factor] += exponent * factor_exponent
        kept = sorted(
            (item for item in exponents.items() if item[1] != 0),
            key=lambda item: item[0].sort_key,
        )
        if coefficient == 0 or not kept:
            return Number(coefficient, table)
        if len(kept) == 1 and kept[0][1] == 1:
            rest = kept[0][0]
        else:
            rest = _shared(cls, table, factors=tuple(kept))
        # an addition scaled by a number gets the number distributed into its terms
        return Add([(rest, coefficient)])

    @property
    def args(self) -> tuple[Expr, ...]:
        return tuple(factor for factor, _ in self.factors)

    def as_coeff_mul(self) -> tuple[float, tuple[tuple[Expr, int], ...]]:
        return 1, self.factors

    def read_at(self, indexes: tuple[Expr, ...]) -> Expr:
        return Mul(
            [(factor.read_at(indexes), exponent) for factor, exponent in self.factors]
        )

    def _replace(self, visit: Callable[[Expr], Expr]) -> Expr:
        return Mul([(visit(factor), exponent) for factor, exponent in self.factors])

    def _derivative(self, target: "ParameterSymbol | Indexed") -> Expr:
        terms = []
        for position, (factor, exponent) in enumerate(self.factors):
            lowered = [
                (
                    other,
                    other_exponent - 1 if other_position == position else other_exponent,
                )
                for other_position, (other, other_exponent) in enumerate(self.factors)
            ]
            terms.append((Mul(lowered + [(factor.derivative(target), 1)]), exponent))
        return Add(terms)

    def compute(self, context: Context) -> np.ndarray:
        values = 1
        for factor, exponent in self.factors:
            factor_values = factor.evaluate_over(context)
            if exponent < 0:
                factor_values = np.asarray(factor_values, dtype=float)
            values = values * factor_values**exponent
        return values

    def __repr__(self) -> str:
        return "*".join(
            repr(factor) if exponent == 1 else f"{factor!r}**{exponent}"
            for factor, exponent in self.factors
        )


class Operation(Composite):
    """An operator the factorization never looks inside, such as `mod` or a comparison."""

    operator: str
    operands: tuple[Expr, ...]

    def __new__(cls, operator: str, operands: tuple[Expr | float, ...]) -> Expr:
        table = _table_of(operands)
        operands = tuple(_as_expression(operand, table) for operand in operands)
        if all(isinstance(operand, Number) for operand in operands):
            # numpy gives back its own scalar types, turned into Python numbers so that they print plainly
            result = OPERATIONS[operator](
                *(operand.value for operand in operands)
            ).item()
            return Number(int(result) if isinstance(result, bool) else result, table)
        return _shared(cls, table, operator=operator, operands=operands)

    @property
    def args(self) -> tuple[Expr, ...]:
        return self.operands

    def read_at(self, indexes: tuple[Expr, ...]) -> Expr:
        return Operation(
            self.operator, tuple(operand.read_at(indexes) for operand in self.operands)
        )

    def _replace(self, visit: Callable[[Expr], Expr]) -> Expr:
        return Operation(self.operator, tuple(map(visit, self.operands)))

    def _derivative(self, target: "ParameterSymbol | Indexed") -> Expr:
        """The derivative of a power, the only operator that has one."""
        if self.operator != "**":
            return super()._derivative(target)
        base, exponent = self.operands
        return exponent * base ** (exponent - 1) * base.derivative(target) + self * Operation(
            "log", (base,)
        ) * exponent.derivative(target)

    def compute(self, context: Context) -> np.ndarray:
        return OPERATIONS[self.operator](
            *(operand.evaluate_over(context) for operand in self.operands)
        )

    def __repr__(self) -> str:
        if len(self.operands) == 1:
            return f"({self.operator} {self.operands[0]!r})"
        return f"({self.operands[0]!r} {self.operator} {self.operands[1]!r})"


class Range(Composite):
    """The values `index` takes from `begin` to `end`, kept only where `condition` holds."""

    index: Idx
    begin: Expr
    end: Expr
    step: Expr
    condition: Expr | None
    # the same range without its condition, over which the condition itself is evaluated
    unconditioned: "Range | None"

    def __new__(
        cls,
        index: Idx,
        begin: Expr,
        end: Expr,
        step: Expr,
        condition: Expr | None = None,
    ) -> "Range":
        value_range = _shared(
            cls,
            index.table,
            index=index,
            begin=begin,
            end=end,
            step=step,
            condition=condition,
        )
        value_range.unconditioned = (
            None if condition is None else Range(index, begin, end, step)
        )
        return value_range

    @property
    def args(self) -> tuple[Expr, ...]:
        return (self.begin, self.end, self.step) + (
            () if self.condition is None else (self.condition,)
        )

    def find_free_indexes(self) -> frozenset[Idx]:
        free_indexes = (
            self.begin.free_indexes | self.end.free_indexes | self.step.free_indexes
        )
        if self.condition is None:
            return free_indexes
        return free_indexes | (self.condition.free_indexes - {self.index})

    def reads_any(self, names: frozenset[Idx | ParameterSymbol]) -> bool:
        # a range defines its index, so renaming that index rebuilds the range
        return self.index in names or super().reads_any(names)

    def _missing_entry_count(self) -> Expr:
        """The reads of missing entries of the bounds, and of the condition over every value of the index."""
        bounds = Add([(bound.missing_entry_count, 1) for bound in (self.begin, self.end, self.step)])
        if self.condition is None:
            return bounds
        return bounds + Sum(self.condition.missing_entry_count, self.unconditioned)

    def _replace(self, visit: Callable[[Expr], Expr]) -> Expr:
        condition = None if self.condition is None else visit(self.condition)
        return Range(
            visit(self.index),
            visit(self.begin),
            visit(self.end),
            visit(self.step),
            condition,
        )

    @staticmethod
    def range_from_bounds(
        begin: np.ndarray, end: np.ndarray, step: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The values from `begin` to `end` by `step` along a new last axis, and where they exist."""
        begin, end, step = np.broadcast_arrays(
            *(np.asarray(bound).astype(int) for bound in (begin, end, step))
        )
        count = np.maximum((end - begin) // step + 1, 0)
        offsets = np.arange(int(count.max()))
        values = (
            begin.reshape(begin.shape + (1,)) + step.reshape(step.shape + (1,)) * offsets
        )
        return values, offsets < count.reshape(count.shape + (1,))

    def compute(self, context: Context) -> tuple[np.ndarray, np.ndarray]:
        """The values the index takes along a new last axis, and where they are kept."""
        bounds = []
        for name, bound in (
            ("begin", self.begin),
            ("end", self.end),
            ("step", self.step),
        ):
            values = np.asarray(bound.evaluate_over(context))
            if values.size == 1 and not float(values.item()).is_integer():
                values = np.round(values)
                print(
                    "WARNING: in for loop, "
                    + name
                    + " value  is of type float and was rounded to "
                    + str(int(values.item()))
                    + " at line "
                    + str(context.line)
                )
            bounds.append(values)
        begin, end, step = bounds
        if (step < 1).any():
            error_(
                "ERROR: in for loop, the step value: "
                + repr(self.step)
                + " is negative or null at line "
                + str(context.line)
            )
        if (end < begin).all():
            error_(
                "ERROR: in for loop, the end_value: "
                + repr(self.end)
                + " is smaller than the begin value "
                + repr(self.begin)
                + " at line "
                + str(context.line)
            )
        values, filled = self.range_from_bounds(begin, end, step)
        if self.condition is not None:
            # the condition sees every value of the index, which is what the range without it holds
            kept = self.condition.evaluate_over(
                context.widened(self.unconditioned, values, filled)
            )
            filled = filled & np.asarray(kept, dtype=bool)
        return values, filled

    def __repr__(self) -> str:
        written = f"{self.index!r} in [{self.begin!r}:{self.end!r}:{self.step!r}]"
        return (
            written if self.condition is None else f"{written} where {self.condition!r}"
        )


class Sum(Composite):
    """The sum of `body` over the values `range` keeps."""

    body: Expr
    range: Range

    def __new__(cls, body: Expr, value_range: Range) -> Expr:
        table = value_range.table
        coefficient, factors = body.as_coeff_mul()
        if coefficient == 0:
            return Number(0, table)
        index = value_range.index
        inside = [factor for factor in factors if index in factor[0].free_indexes]
        outside = [factor for factor in factors if index not in factor[0].free_indexes]
        summed = _shared(
            cls,
            table,
            body=Mul(inside) if inside else Number(1, table),
            range=value_range,
        )
        # the factors that do not read the index of the sum are moved out of it
        return Mul(outside + [(summed, 1), (coefficient, 1)])

    @property
    def args(self) -> tuple[Expr, ...]:
        return self.body, self.range

    def find_free_indexes(self) -> frozenset[Idx]:
        return (self.body.free_indexes - {self.range.index}) | self.range.free_indexes

    def _replace(self, visit: Callable[[Expr], Expr]) -> Expr:
        return Sum(visit(self.body), visit(self.range))

    def _missing_entry_count(self) -> Expr:
        return self.range.missing_entry_count + Sum(self.body.missing_entry_count, self.range)

    def _derivative(self, target: "ParameterSymbol | Indexed") -> Expr:
        self.range.check_does_not_read(target)
        return Sum(self.body.derivative(target), self.range)

    def compute(self, context: Context) -> np.ndarray:
        values, filled = self.range.evaluate_over(context)
        body = self.body.evaluate_over(context.widened(self.range, values, filled))
        if body.shape[-1] == 1:
            return body.reshape(body.shape[:-1]) * filled.sum(axis=-1)
        return np.where(filled, body, 0).sum(axis=-1)

    def __repr__(self) -> str:
        return f"sum({self.body!r} for {self.range!r})"


def _as_expression(value: Expr | float, table: dict[tuple[object, ...], Composite]) -> Expr:
    return value if isinstance(value, Expr) else Number(value, table)


def _table_of(operands: Iterable[Expr | float]) -> dict[tuple[object, ...], Composite]:
    """The table of the first expression among `operands`, which always holds one."""
    return next(operand.table for operand in operands if isinstance(operand, Expr))


def _shared(
    kind: type, table: dict[tuple[object, ...], Composite], **fields: object
) -> Composite:
    """The composite expression of `kind` made of `fields`, created only once in `table`."""
    key = (kind, *fields.values())
    expression = table.get(key)
    if expression is None:
        expression = object.__new__(kind)
        expression.__dict__.update(fields)
        expression.table = table
        expression.sort_key = (1, len(table))
        expression.free_indexes = expression.find_free_indexes()
        expression.parameters_read = expression.find_parameters_read()
        table[key] = expression
    return expression


def _uninitialized(kind: type) -> Composite:
    """An empty expression of `kind`, whose fields pickle restores."""
    return object.__new__(kind)


def _lengths_read_at(expression: Expr, entry: Idx) -> set[int]:
    """The number of entries of every table that `expression` reads at `entry`."""
    if isinstance(expression, Indexed) and expression.indexes == (entry,):
        return {expression.base.length}
    if isinstance(expression, Composite):
        return set().union(*(_lengths_read_at(arg, entry) for arg in expression.args))
    return set()
