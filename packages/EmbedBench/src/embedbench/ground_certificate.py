"""Exact ground-state certificate primitives for the hard/OOD corpus.

Every floating-point coefficient is treated as the exact dyadic rational represented by
its IEEE-754 binary64 bits.  All energy arithmetic below uses Python's unbounded integers;
floating-point values are emitted only as descriptive views of exact values.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from embedbench.hard_ood_schema import canonical_sha256, validate_versioned_object

CERTIFICATE_SCHEMA = "embedbench.ground-state-certificate"
CERTIFICATE_SCHEMA_VERSION = 1
PLANTED_PROOF_SCHEMA = "embedbench.ground-state-planted-proof"
PLANTED_PROOF_SCHEMA_VERSION = 1
EXHAUSTIVE_PROOF_SCHEMA = "embedbench.ground-state-exhaustive-proof"
EXHAUSTIVE_PROOF_SCHEMA_VERSION = 1
CERTIFIED_OPTIMAL_PROOF_SCHEMA = "embedbench.ground-state-certified-optimal-proof"
CERTIFIED_OPTIMAL_PROOF_SCHEMA_VERSION = 1
UNCERTIFIED_REFERENCE_PROOF_SCHEMA = "embedbench.ground-state-uncertified-reference-proof"
UNCERTIFIED_REFERENCE_PROOF_SCHEMA_VERSION = 1
CHECKER_ACCEPTANCE_SCHEMA = "embedbench.ground-state-checker-acceptance"
CHECKER_ACCEPTANCE_SCHEMA_VERSION = 1
SPIN_ASSIGNMENT_SCHEMA = "embedbench.spin-assignment"
SPIN_ASSIGNMENT_SCHEMA_VERSION = 1
GRAY_CODE_ENUMERATION = "binary_reflected_gray_code_lsb_first_v1"
MAX_EXHAUSTIVE_VARIABLES = 24
MAX_REGISTERED_BNB_NODES = 100_000_000

GroundStatus = Literal[
    "planted_proof",
    "exact_enumeration",
    "certified_optimal",
    "uncertified_reference",
]
_GROUND_STATUSES = frozenset(
    {"planted_proof", "exact_enumeration", "certified_optimal", "uncertified_reference"}
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _require_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _require_finite_binary64(value: object, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be a binary64 float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return 0.0 if value == 0.0 else value


def _require_nonnegative_int(value: object, name: str) -> int:
    integer = _require_int(value, name)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{name} contains a Unicode surrogate")
    return value


def _require_command(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    command = tuple(_require_text(part, f"{name}[{index}]") for index, part in enumerate(value))
    if not command:
        raise ValueError(f"{name} must not be empty")
    return command


def _require_exact_fields(
    value: object,
    expected: Collection[str],
    name: str,
) -> dict[str, Any]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise TypeError(f"{name} must be an exact JSON object with string keys")
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        unknown = sorted(set(value) - set(expected))
        raise ValueError(f"{name} schema fields differ: missing={missing}, unknown={unknown}")
    return value


@dataclass(frozen=True, slots=True)
class Dyadic:
    """Canonical exact value ``integer * 2**power_of_two``."""

    integer: int
    power_of_two: int

    def __post_init__(self) -> None:
        integer = _require_int(self.integer, "dyadic integer")
        power = _require_int(self.power_of_two, "dyadic power_of_two")
        if integer == 0:
            object.__setattr__(self, "power_of_two", 0)
            return
        magnitude = abs(integer)
        trailing_zeroes = (magnitude & -magnitude).bit_length() - 1
        if trailing_zeroes:
            object.__setattr__(self, "integer", integer >> trailing_zeroes)
            object.__setattr__(self, "power_of_two", power + trailing_zeroes)

    def __add__(self, other: object) -> Dyadic:
        if not isinstance(other, Dyadic):
            return NotImplemented
        common_power = min(self.power_of_two, other.power_of_two)
        left = self.integer << (self.power_of_two - common_power)
        right = other.integer << (other.power_of_two - common_power)
        return Dyadic(left + right, common_power)

    def __sub__(self, other: object) -> Dyadic:
        if not isinstance(other, Dyadic):
            return NotImplemented
        return self + Dyadic(-other.integer, other.power_of_two)

    def __mul__(self, other: object) -> Dyadic:
        if type(other) is not int:
            return NotImplemented
        return Dyadic(self.integer * other, self.power_of_two)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Dyadic):
            return NotImplemented
        common_power = min(self.power_of_two, other.power_of_two)
        return (self.integer << (self.power_of_two - common_power)) < (
            other.integer << (other.power_of_two - common_power)
        )

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Dyadic):
            return NotImplemented
        return self == other or self < other

    def to_float(self) -> float:
        if self.integer == 0:
            return 0.0
        top_exponent = abs(self.integer).bit_length() - 1 + self.power_of_two
        if top_exponent > 1023:
            raise ValueError("exact dyadic has no finite binary64 descriptive value")
        if top_exponent < -1075:
            return 0.0
        try:
            exact = (
                Fraction(self.integer << self.power_of_two, 1)
                if self.power_of_two >= 0
                else Fraction(self.integer, 1 << -self.power_of_two)
            )
            result = float(exact)
        except OverflowError as error:
            raise ValueError("exact dyadic has no finite binary64 descriptive value") from error
        if not math.isfinite(result):
            raise ValueError("exact dyadic has no finite binary64 descriptive value")
        return 0.0 if result == 0.0 else result

    def to_dict(self) -> dict[str, int]:
        return {"integer": self.integer, "power_of_two": self.power_of_two}


def _dyadic_from_dict(value: object, name: str) -> Dyadic:
    document = _require_exact_fields(value, {"integer", "power_of_two"}, name)
    raw_integer = _require_int(document["integer"], f"{name}.integer")
    raw_power = _require_int(document["power_of_two"], f"{name}.power_of_two")
    result = Dyadic(raw_integer, raw_power)
    if result.integer != raw_integer or result.power_of_two != raw_power:
        raise ValueError(f"{name} is not a canonical dyadic")
    return result


def _require_energy_dyadic(value: object, name: str) -> Dyadic:
    if not isinstance(value, Dyadic):
        raise TypeError(f"{name} must be a Dyadic")
    if value.integer != 0:
        top_exponent = abs(value.integer).bit_length() - 1 + value.power_of_two
        if value.power_of_two < -1074 or top_exponent > 1023:
            raise ValueError(f"{name} lies outside the finite binary64 energy domain")
    return value


def binary64_to_dyadic(value: object) -> Dyadic:
    """Lift one finite binary64 value without decimal or floating-point arithmetic."""

    number = _require_finite_binary64(value, "value")
    numerator, denominator = number.as_integer_ratio()
    if denominator & (denominator - 1):  # pragma: no cover - Python float contract guard
        raise RuntimeError("binary64 denominator is not a power of two")
    return Dyadic(numerator, -(denominator.bit_length() - 1))


def _validate_linear(variables: tuple[int, ...], value: object) -> tuple[tuple[int, float], ...]:
    if type(value) is not tuple:
        raise TypeError("linear must be an immutable tuple")
    output: list[tuple[int, float]] = []
    for index, raw in enumerate(value):
        if type(raw) is not tuple or len(raw) != 2:
            raise TypeError(f"linear[{index}] must be a two-item tuple")
        variable = _require_int(raw[0], f"linear[{index}] variable")
        coefficient = _require_finite_binary64(raw[1], f"linear[{index}] coefficient")
        output.append((variable, coefficient))
    if tuple(variable for variable, _ in output) != variables:
        raise ValueError("linear must contain every variable exactly once in sorted order")
    return tuple(output)


def _validate_quadratic(
    variables: tuple[int, ...], value: object
) -> tuple[tuple[int, int, float], ...]:
    if type(value) is not tuple:
        raise TypeError("quadratic must be an immutable tuple")
    variable_set = set(variables)
    output: list[tuple[int, int, float]] = []
    for index, raw in enumerate(value):
        if type(raw) is not tuple or len(raw) != 3:
            raise TypeError(f"quadratic[{index}] must be a three-item tuple")
        left = _require_int(raw[0], f"quadratic[{index}] left endpoint")
        right = _require_int(raw[1], f"quadratic[{index}] right endpoint")
        coefficient = _require_finite_binary64(raw[2], f"quadratic[{index}] coefficient")
        if left >= right:
            raise ValueError("quadratic endpoints must be strictly increasing")
        if left not in variable_set or right not in variable_set:
            raise ValueError("quadratic endpoint is not a declared variable")
        if coefficient == 0.0:
            raise ValueError("zero quadratic coefficients must be omitted")
        output.append((left, right, coefficient))
    if output != sorted(output, key=lambda item: (item[0], item[1])):
        raise ValueError("quadratic terms must be lexicographically sorted")
    pairs = [(left, right) for left, right, _ in output]
    if len(pairs) != len(set(pairs)):
        raise ValueError("quadratic terms must be duplicate-free")
    return tuple(output)


@dataclass(frozen=True, slots=True)
class IsingProblem:
    """Canonical hard/OOD Ising problem identity payload."""

    variables: tuple[int, ...]
    linear: tuple[tuple[int, float], ...]
    quadratic: tuple[tuple[int, int, float], ...]

    @classmethod
    def from_logical_problem(cls, value: object) -> IsingProblem:
        """Take an immutable, canonical snapshot of the package's legacy problem type."""

        # Keep the certificate core's serialized schema independent of the mutable
        # NetworkX object while still providing an explicit compatibility boundary.
        from embedbench.embedding import LogicalProblem

        if not isinstance(value, LogicalProblem):
            raise TypeError("value must be a LogicalProblem")
        if value.graph.is_directed() or value.graph.is_multigraph():
            raise ValueError("LogicalProblem.graph must be a simple undirected graph")

        raw_linear = tuple(value.h.items())
        variables = tuple(
            sorted(
                _require_int(variable, f"LogicalProblem.h key {index}")
                for index, (variable, _) in enumerate(raw_linear)
            )
        )
        linear_by_variable: dict[int, float] = {}
        for index, (variable, coefficient) in enumerate(raw_linear):
            checked_variable = _require_int(variable, f"LogicalProblem.h key {index}")
            linear_by_variable[checked_variable] = _require_finite_binary64(
                coefficient,
                f"LogicalProblem.h[{checked_variable}]",
            )

        variable_set = set(variables)
        raw_quadratic = tuple(value.j.items())
        quadratic_by_pair: dict[tuple[int, int], float] = {}
        declared_pairs: set[tuple[int, int]] = set()
        for index, (raw_pair, coefficient) in enumerate(raw_quadratic):
            if type(raw_pair) is not tuple or len(raw_pair) != 2:
                raise TypeError(f"LogicalProblem.j key {index} must be a two-item tuple")
            raw_left, raw_right = raw_pair
            left = _require_int(raw_left, f"LogicalProblem.j key {index} left endpoint")
            right = _require_int(raw_right, f"LogicalProblem.j key {index} right endpoint")
            if left == right:
                raise ValueError("LogicalProblem.j must not contain self-couplings")
            pair = (left, right) if left < right else (right, left)
            if pair[0] not in variable_set or pair[1] not in variable_set:
                raise ValueError("LogicalProblem.j endpoint is not declared in h")
            if pair in declared_pairs:
                raise ValueError("LogicalProblem.j contains a duplicate undirected edge")
            declared_pairs.add(pair)
            checked_coefficient = _require_finite_binary64(
                coefficient,
                f"LogicalProblem.j[{pair}]",
            )
            if checked_coefficient != 0.0:
                quadratic_by_pair[pair] = checked_coefficient

        graph_nodes = {
            _require_int(node, f"LogicalProblem.graph node {index}")
            for index, node in enumerate(value.graph.nodes)
        }
        if graph_nodes != variable_set:
            raise ValueError("LogicalProblem.graph nodes must exactly match h")
        graph_pairs: set[tuple[int, int]] = set()
        for index, raw_edge in enumerate(value.graph.edges):
            if len(raw_edge) != 2:  # pragma: no cover - NetworkX Graph contract guard
                raise TypeError(f"LogicalProblem.graph edge {index} must have two endpoints")
            raw_left, raw_right = raw_edge
            left = _require_int(raw_left, f"LogicalProblem.graph edge {index} left endpoint")
            right = _require_int(raw_right, f"LogicalProblem.graph edge {index} right endpoint")
            if left == right:
                raise ValueError("LogicalProblem.graph must not contain self-loops")
            graph_pairs.add((left, right) if left < right else (right, left))
        if graph_pairs != declared_pairs:
            raise ValueError("LogicalProblem.graph edges must exactly match j")

        return cls(
            variables=variables,
            linear=tuple((variable, linear_by_variable[variable]) for variable in variables),
            quadratic=tuple(
                (left, right, quadratic_by_pair[(left, right)])
                for left, right in sorted(quadratic_by_pair)
            ),
        )

    def __post_init__(self) -> None:
        if type(self.variables) is not tuple:
            raise TypeError("variables must be an immutable tuple")
        variables = tuple(
            _require_int(variable, f"variables[{index}]")
            for index, variable in enumerate(self.variables)
        )
        if not variables:
            raise ValueError("problem must declare at least one variable")
        if variables != tuple(sorted(set(variables))):
            raise ValueError("variables must be sorted and duplicate-free")
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "linear", _validate_linear(variables, self.linear))
        object.__setattr__(self, "quadratic", _validate_quadratic(variables, self.quadratic))

    def identity_payload(self) -> dict[str, object]:
        return {
            "variables": list(self.variables),
            "h": [[variable, coefficient] for variable, coefficient in self.linear],
            "J": [list(term) for term in self.quadratic],
        }

    @property
    def problem_sha256(self) -> str:
        return canonical_sha256(self.identity_payload())


def _validate_spin_pairs(value: object, name: str) -> tuple[tuple[int, int], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    checked: list[tuple[int, int]] = []
    for index, raw in enumerate(value):
        if type(raw) is not tuple or len(raw) != 2:
            raise TypeError(f"{name}[{index}] must be a two-item tuple")
        variable = _require_int(raw[0], f"{name}[{index}] variable")
        spin = _require_int(raw[1], f"{name}[{index}] spin")
        if spin not in {-1, 1}:
            raise ValueError(f"{name}[{index}] spin must be -1 or +1")
        checked.append((variable, spin))
    variables = tuple(variable for variable, _ in checked)
    if variables != tuple(sorted(set(variables))):
        raise ValueError(f"{name} variables must be sorted and occur exactly once")
    return tuple(checked)


def _validate_assignment(problem: IsingProblem, assignment: object) -> tuple[tuple[int, int], ...]:
    checked = _validate_spin_pairs(assignment, "assignment")
    if tuple(variable for variable, _ in checked) != problem.variables:
        raise ValueError("assignment must contain every variable exactly once in sorted order")
    return checked


def exact_energy(problem: IsingProblem, assignment: object) -> Dyadic:
    """Evaluate an Ising assignment exactly with unbounded integer arithmetic."""

    if not isinstance(problem, IsingProblem):
        raise TypeError("problem must be an IsingProblem")
    checked = _validate_assignment(problem, assignment)
    spins = dict(checked)
    result = Dyadic(0, 0)
    for variable, coefficient in problem.linear:
        result += binary64_to_dyadic(coefficient) * spins[variable]
    for left, right, coefficient in problem.quadratic:
        result += binary64_to_dyadic(coefficient) * spins[left] * spins[right]
    return result


def spin_assignment_sha256(
    problem: IsingProblem,
    assignment: object,
) -> str:
    """Bind a canonical complete spin assignment to one problem identity."""

    checked = _validate_assignment(problem, assignment)
    return canonical_sha256(
        {
            "schema": SPIN_ASSIGNMENT_SCHEMA,
            "schema_version": SPIN_ASSIGNMENT_SCHEMA_VERSION,
            "problem_sha256": problem.problem_sha256,
            "spins": [list(item) for item in checked],
        }
    )


def _validate_exact_linear_terms(
    value: object,
    name: str,
) -> tuple[tuple[int, Dyadic], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    output: list[tuple[int, Dyadic]] = []
    for index, raw in enumerate(value):
        if type(raw) is not tuple or len(raw) != 2:
            raise TypeError(f"{name}[{index}] must be a two-item tuple")
        variable = _require_int(raw[0], f"{name}[{index}] variable")
        coefficient = _require_energy_dyadic(raw[1], f"{name}[{index}] coefficient")
        if coefficient.integer == 0:
            raise ValueError(f"{name}[{index}] zero coefficient must be omitted")
        output.append((variable, coefficient))
    if output != sorted(output, key=lambda term: term[0]):
        raise ValueError(f"{name} must be sorted")
    variables = [variable for variable, _ in output]
    if len(variables) != len(set(variables)):
        raise ValueError(f"{name} must be duplicate-free")
    return tuple(output)


def _validate_exact_quadratic_terms(
    value: object,
    name: str,
) -> tuple[tuple[int, int, Dyadic], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    output: list[tuple[int, int, Dyadic]] = []
    for index, raw in enumerate(value):
        if type(raw) is not tuple or len(raw) != 3:
            raise TypeError(f"{name}[{index}] must be a three-item tuple")
        left = _require_int(raw[0], f"{name}[{index}] left endpoint")
        right = _require_int(raw[1], f"{name}[{index}] right endpoint")
        if left >= right:
            raise ValueError(f"{name}[{index}] endpoints must be strictly increasing")
        coefficient = _require_energy_dyadic(raw[2], f"{name}[{index}] coefficient")
        if coefficient.integer == 0:
            raise ValueError(f"{name}[{index}] zero coefficient must be omitted")
        output.append((left, right, coefficient))
    if output != sorted(output, key=lambda term: (term[0], term[1])):
        raise ValueError(f"{name} must be lexicographically sorted")
    pairs = [(left, right) for left, right, _ in output]
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"{name} must be duplicate-free")
    return tuple(output)


@dataclass(frozen=True, slots=True)
class PlantedTerm:
    """One exactly represented additive term in a planted Hamiltonian proof."""

    term_index: int
    linear: tuple[tuple[int, Dyadic], ...]
    quadratic: tuple[tuple[int, int, Dyadic], ...]
    lower_bound: Dyadic

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "term_index",
            _require_nonnegative_int(self.term_index, "term_index"),
        )
        object.__setattr__(
            self,
            "linear",
            _validate_exact_linear_terms(self.linear, "planted term linear"),
        )
        object.__setattr__(
            self,
            "quadratic",
            _validate_exact_quadratic_terms(self.quadratic, "planted term quadratic"),
        )
        if not self.linear and not self.quadratic:
            raise ValueError("planted term must contain a nonzero coefficient")
        _require_energy_dyadic(self.lower_bound, "planted term lower_bound")

    @property
    def variables(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {variable for variable, _ in self.linear}
                | {endpoint for left, right, _ in self.quadratic for endpoint in (left, right)}
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "term_index": self.term_index,
            "linear": [[variable, coefficient.to_dict()] for variable, coefficient in self.linear],
            "quadratic": [
                [left, right, coefficient.to_dict()] for left, right, coefficient in self.quadratic
            ],
            "lower_bound_integer": self.lower_bound.integer,
            "lower_bound_power_of_two": self.lower_bound.power_of_two,
        }

    @classmethod
    def from_dict(cls, value: object) -> PlantedTerm:
        document = _require_exact_fields(
            value,
            {
                "term_index",
                "linear",
                "quadratic",
                "lower_bound_integer",
                "lower_bound_power_of_two",
            },
            "planted term",
        )
        raw_linear = document["linear"]
        if type(raw_linear) is not list:
            raise TypeError("planted term linear must be a JSON array")
        linear: list[tuple[int, Dyadic]] = []
        for index, raw in enumerate(raw_linear):
            if type(raw) is not list or len(raw) != 2:
                raise TypeError(f"planted term linear[{index}] must be a two-item array")
            linear.append(
                (
                    _require_int(raw[0], f"planted term linear[{index}] variable"),
                    _dyadic_from_dict(raw[1], f"planted term linear[{index}] coefficient"),
                )
            )
        raw_quadratic = document["quadratic"]
        if type(raw_quadratic) is not list:
            raise TypeError("planted term quadratic must be a JSON array")
        quadratic: list[tuple[int, int, Dyadic]] = []
        for index, raw in enumerate(raw_quadratic):
            if type(raw) is not list or len(raw) != 3:
                raise TypeError(f"planted term quadratic[{index}] must be a three-item array")
            quadratic.append(
                (
                    _require_int(raw[0], f"planted term quadratic[{index}] left endpoint"),
                    _require_int(raw[1], f"planted term quadratic[{index}] right endpoint"),
                    _dyadic_from_dict(raw[2], f"planted term quadratic[{index}] coefficient"),
                )
            )
        return cls(
            term_index=document["term_index"],
            linear=tuple(linear),
            quadratic=tuple(quadratic),
            lower_bound=Dyadic(
                _require_int(document["lower_bound_integer"], "planted term lower bound integer"),
                _require_int(
                    document["lower_bound_power_of_two"],
                    "planted term lower bound power",
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class CheckerRun:
    """Persisted checker invocation metadata; mathematical trust is handled separately."""

    source_sha256: str
    environment_sha256: str
    command: tuple[str, ...]
    exit_code: int
    log_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_sha256", _require_sha256(self.source_sha256, "checker source")
        )
        object.__setattr__(
            self,
            "environment_sha256",
            _require_sha256(self.environment_sha256, "checker environment"),
        )
        object.__setattr__(self, "command", _require_command(self.command, "checker command"))
        object.__setattr__(self, "exit_code", _require_int(self.exit_code, "checker exit_code"))
        object.__setattr__(self, "log_sha256", _require_sha256(self.log_sha256, "checker log"))


@dataclass(frozen=True, slots=True)
class SolverRun:
    """Deterministic exact-solver provenance required by Section 7.2."""

    name: str
    version: str
    command: tuple[str, ...]
    seed: int | None
    deterministic_work_limit: int
    safety_timeout_seconds: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, "solver name"))
        object.__setattr__(self, "version", _require_text(self.version, "solver version"))
        object.__setattr__(self, "command", _require_command(self.command, "solver command"))
        if self.seed is not None:
            object.__setattr__(self, "seed", _require_nonnegative_int(self.seed, "solver seed"))
        object.__setattr__(
            self,
            "deterministic_work_limit",
            _require_nonnegative_int(
                self.deterministic_work_limit,
                "deterministic work limit",
            ),
        )
        if self.safety_timeout_seconds is not None:
            timeout = _require_finite_binary64(
                self.safety_timeout_seconds,
                "safety timeout seconds",
            )
            if timeout <= 0.0:
                raise ValueError("safety timeout seconds must be positive")
            object.__setattr__(self, "safety_timeout_seconds", timeout)


@dataclass(frozen=True, slots=True)
class _EnumerationResult:
    energy: Dyadic
    assignment: tuple[tuple[int, int], ...]
    minimum_gray_step: int
    checked_state_count: int
    final_gray_word: int


def _gray_code_minimum(
    variables: tuple[int, ...],
    linear: tuple[tuple[int, Dyadic], ...],
    quadratic: tuple[tuple[int, int, Dyadic], ...],
) -> _EnumerationResult:
    if not variables:
        raise ValueError("exact enumeration requires at least one variable")
    if len(variables) > MAX_EXHAUSTIVE_VARIABLES:
        raise ValueError(f"exact enumeration is limited to {MAX_EXHAUSTIVE_VARIABLES} variables")
    variable_set = set(variables)
    if any(variable not in variable_set for variable, _ in linear) or any(
        left not in variable_set or right not in variable_set for left, right, _ in quadratic
    ):
        raise ValueError("term coefficient references an undeclared enumeration variable")
    coefficients = [coefficient for _, coefficient in linear] + [
        coefficient for _, _, coefficient in quadratic
    ]
    if not coefficients:
        return _EnumerationResult(
            energy=Dyadic(0, 0),
            assignment=tuple((variable, -1) for variable in variables),
            minimum_gray_step=0,
            checked_state_count=1 << len(variables),
            final_gray_word=(1 << (len(variables) - 1)),
        )
    common_power = min(coefficient.power_of_two for coefficient in coefficients)
    h_units = {variable: 0 for variable in variables}
    for variable, coefficient in linear:
        h_units[variable] += coefficient.integer << (coefficient.power_of_two - common_power)
    adjacency: dict[int, list[tuple[int, int]]] = {variable: [] for variable in variables}
    j_units: list[tuple[int, int, int]] = []
    for left, right, coefficient in quadratic:
        units = coefficient.integer << (coefficient.power_of_two - common_power)
        adjacency[left].append((right, units))
        adjacency[right].append((left, units))
        j_units.append((left, right, units))
    spins = {variable: -1 for variable in variables}
    energy_units = sum(h_units[variable] * spins[variable] for variable in variables)
    energy_units += sum(units * spins[left] * spins[right] for left, right, units in j_units)
    best_units = energy_units
    best_spins = dict(spins)
    best_step = 0
    state_count = 1 << len(variables)
    for step in range(1, state_count):
        position = (step & -step).bit_length() - 1
        variable = variables[position]
        old_spin = spins[variable]
        local_units = h_units[variable] + sum(
            units * spins[neighbour] for neighbour, units in adjacency[variable]
        )
        energy_units += -2 * old_spin * local_units
        spins[variable] = -old_spin
        if energy_units < best_units:
            best_units = energy_units
            best_spins = dict(spins)
            best_step = step
    return _EnumerationResult(
        energy=Dyadic(best_units, common_power),
        assignment=tuple((variable, best_spins[variable]) for variable in variables),
        minimum_gray_step=best_step,
        checked_state_count=state_count,
        final_gray_word=(state_count - 1) ^ ((state_count - 1) >> 1),
    )


def _term_energy(term: PlantedTerm, assignment: dict[int, int]) -> Dyadic:
    energy = Dyadic(0, 0)
    for variable, coefficient in term.linear:
        energy += coefficient * assignment[variable]
    for left, right, coefficient in term.quadratic:
        energy += coefficient * assignment[left] * assignment[right]
    return energy


def _verify_planted_terms(
    problem: IsingProblem,
    assignment: tuple[tuple[int, int], ...],
    terms: tuple[PlantedTerm, ...],
) -> Dyadic:
    if not terms:
        raise ValueError("planted proof requires at least one term")
    if tuple(term.term_index for term in terms) != tuple(range(len(terms))):
        raise ValueError("planted term indices must be consecutive from zero")
    variable_set = set(problem.variables)
    spins = dict(assignment)
    accumulated_h = {variable: Dyadic(0, 0) for variable in problem.variables}
    accumulated_j: dict[tuple[int, int], Dyadic] = {}
    lower_sum = Dyadic(0, 0)
    for term in terms:
        if not set(term.variables) <= variable_set:
            raise ValueError("planted term references an undeclared variable")
        enumerated = _gray_code_minimum(term.variables, term.linear, term.quadratic)
        if enumerated.energy != term.lower_bound:
            raise ValueError("planted term lower bound is not its exact minimum")
        if _term_energy(term, spins) != term.lower_bound:
            raise ValueError("planted assignment does not attain every term lower bound")
        lower_sum += term.lower_bound
        for variable, coefficient in term.linear:
            accumulated_h[variable] += coefficient
        for left, right, coefficient in term.quadratic:
            pair = (left, right)
            accumulated_j[pair] = accumulated_j.get(pair, Dyadic(0, 0)) + coefficient
    expected_h = {
        variable: binary64_to_dyadic(coefficient) for variable, coefficient in problem.linear
    }
    expected_j = {
        (left, right): binary64_to_dyadic(coefficient)
        for left, right, coefficient in problem.quadratic
    }
    if accumulated_h != expected_h:
        raise ValueError("planted terms do not reconstruct the problem linear coefficients")
    all_pairs = set(accumulated_j) | set(expected_j)
    if any(
        accumulated_j.get(pair, Dyadic(0, 0)) != expected_j.get(pair, Dyadic(0, 0))
        for pair in all_pairs
    ):
        raise ValueError("planted terms do not reconstruct the problem quadratic coefficients")
    if exact_energy(problem, assignment) != lower_sum:
        raise ValueError("planted assignment energy does not equal the sum of term bounds")
    return lower_sum


@dataclass(frozen=True, slots=True)
class PlantedProof:
    problem_sha256: str
    assignment: tuple[tuple[int, int], ...]
    spin_assignment_sha256: str
    terms: tuple[PlantedTerm, ...]
    assignment_energy: Dyadic
    sum_lower_bound: Dyadic

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "problem_sha256",
            _require_sha256(self.problem_sha256, "problem_sha256"),
        )
        object.__setattr__(
            self,
            "assignment",
            _validate_spin_pairs(self.assignment, "planted assignment"),
        )
        object.__setattr__(
            self,
            "spin_assignment_sha256",
            _require_sha256(self.spin_assignment_sha256, "spin_assignment_sha256"),
        )
        if type(self.terms) is not tuple or not all(
            isinstance(term, PlantedTerm) for term in self.terms
        ):
            raise TypeError("planted terms must be an immutable tuple of PlantedTerm values")
        if not self.terms:
            raise ValueError("planted proof requires at least one term")
        if tuple(term.term_index for term in self.terms) != tuple(range(len(self.terms))):
            raise ValueError("planted term indices must be consecutive from zero")
        _require_energy_dyadic(self.assignment_energy, "planted assignment energy")
        _require_energy_dyadic(self.sum_lower_bound, "planted sum lower bound")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PLANTED_PROOF_SCHEMA,
            "schema_version": PLANTED_PROOF_SCHEMA_VERSION,
            "problem_sha256": self.problem_sha256,
            "assignment": [list(item) for item in self.assignment],
            "spin_assignment_sha256": self.spin_assignment_sha256,
            "terms": [term.to_dict() for term in self.terms],
            "assignment_energy_integer": self.assignment_energy.integer,
            "assignment_energy_power_of_two": self.assignment_energy.power_of_two,
            "sum_lower_bound_integer": self.sum_lower_bound.integer,
            "sum_lower_bound_power_of_two": self.sum_lower_bound.power_of_two,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, problem: IsingProblem, value: object) -> PlantedProof:
        document = validate_versioned_object(
            value,
            schema=PLANTED_PROOF_SCHEMA,
            schema_version=PLANTED_PROOF_SCHEMA_VERSION,
            fields={
                "problem_sha256",
                "assignment",
                "spin_assignment_sha256",
                "terms",
                "assignment_energy_integer",
                "assignment_energy_power_of_two",
                "sum_lower_bound_integer",
                "sum_lower_bound_power_of_two",
            },
            name="planted proof",
        )
        assignment = _assignment_from_json(problem, document["assignment"])
        raw_terms = document["terms"]
        if type(raw_terms) is not list:
            raise TypeError("planted proof terms must be a JSON array")
        proof = cls(
            problem_sha256=_require_sha256(document["problem_sha256"], "problem_sha256"),
            assignment=assignment,
            spin_assignment_sha256=_require_sha256(
                document["spin_assignment_sha256"],
                "spin_assignment_sha256",
            ),
            terms=tuple(PlantedTerm.from_dict(term) for term in raw_terms),
            assignment_energy=Dyadic(
                _require_int(document["assignment_energy_integer"], "assignment energy integer"),
                _require_int(
                    document["assignment_energy_power_of_two"],
                    "assignment energy power",
                ),
            ),
            sum_lower_bound=Dyadic(
                _require_int(document["sum_lower_bound_integer"], "sum lower bound integer"),
                _require_int(
                    document["sum_lower_bound_power_of_two"],
                    "sum lower bound power",
                ),
            ),
        )
        if proof.to_dict() != value:
            raise ValueError("planted proof is not canonical")
        return proof


@dataclass(frozen=True, slots=True)
class ExhaustiveProof:
    """A replayable transcript header for full binary-reflected Gray-code enumeration."""

    problem_sha256: str
    n_vars: int
    variable_order: tuple[int, ...]
    enumeration_order: str
    initial_spin: int
    checked_state_count: int
    minimum_gray_step: int
    final_gray_word: int
    assignment: tuple[tuple[int, int], ...]
    spin_assignment_sha256: str
    minimum_energy: Dyadic

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "problem_sha256",
            _require_sha256(self.problem_sha256, "problem_sha256"),
        )
        n_vars = _require_nonnegative_int(self.n_vars, "n_vars")
        if not 1 <= n_vars <= MAX_EXHAUSTIVE_VARIABLES:
            raise ValueError(f"n_vars must be in [1, {MAX_EXHAUSTIVE_VARIABLES}]")
        if type(self.variable_order) is not tuple:
            raise TypeError("variable_order must be an immutable tuple")
        variable_order = tuple(
            _require_int(variable, f"variable_order[{index}]")
            for index, variable in enumerate(self.variable_order)
        )
        if variable_order != tuple(sorted(set(variable_order))) or len(variable_order) != n_vars:
            raise ValueError("variable_order must contain n_vars sorted unique variables")
        object.__setattr__(self, "variable_order", variable_order)
        if self.enumeration_order != GRAY_CODE_ENUMERATION:
            raise ValueError("exhaustive proof has an unregistered enumeration order")
        if type(self.initial_spin) is not int or self.initial_spin != -1:
            raise ValueError("exhaustive Gray-code initial spin must be -1")
        expected_count = 1 << n_vars
        if (
            _require_nonnegative_int(self.checked_state_count, "checked_state_count")
            != expected_count
        ):
            raise ValueError("checked_state_count must equal 2**n_vars")
        minimum_step = _require_nonnegative_int(self.minimum_gray_step, "minimum_gray_step")
        if minimum_step >= expected_count:
            raise ValueError("minimum_gray_step is outside the exhaustive transcript")
        expected_final = (expected_count - 1) ^ ((expected_count - 1) >> 1)
        if _require_nonnegative_int(self.final_gray_word, "final_gray_word") != expected_final:
            raise ValueError("final_gray_word disagrees with the complete Gray-code traversal")
        assignment = _validate_spin_pairs(self.assignment, "exhaustive assignment")
        if tuple(variable for variable, _ in assignment) != variable_order:
            raise ValueError("exhaustive assignment does not follow variable_order")
        object.__setattr__(self, "assignment", assignment)
        object.__setattr__(
            self,
            "spin_assignment_sha256",
            _require_sha256(self.spin_assignment_sha256, "spin_assignment_sha256"),
        )
        _require_energy_dyadic(self.minimum_energy, "minimum_energy")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": EXHAUSTIVE_PROOF_SCHEMA,
            "schema_version": EXHAUSTIVE_PROOF_SCHEMA_VERSION,
            "problem_sha256": self.problem_sha256,
            "n_vars": self.n_vars,
            "variable_order": list(self.variable_order),
            "enumeration_order": self.enumeration_order,
            "initial_spin": self.initial_spin,
            "checked_state_count": self.checked_state_count,
            "minimum_gray_step": self.minimum_gray_step,
            "final_gray_word": self.final_gray_word,
            "assignment": [list(item) for item in self.assignment],
            "spin_assignment_sha256": self.spin_assignment_sha256,
            "minimum_energy_integer": self.minimum_energy.integer,
            "minimum_energy_power_of_two": self.minimum_energy.power_of_two,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, problem: IsingProblem, value: object) -> ExhaustiveProof:
        document = validate_versioned_object(
            value,
            schema=EXHAUSTIVE_PROOF_SCHEMA,
            schema_version=EXHAUSTIVE_PROOF_SCHEMA_VERSION,
            fields={
                "problem_sha256",
                "n_vars",
                "variable_order",
                "enumeration_order",
                "initial_spin",
                "checked_state_count",
                "minimum_gray_step",
                "final_gray_word",
                "assignment",
                "spin_assignment_sha256",
                "minimum_energy_integer",
                "minimum_energy_power_of_two",
            },
            name="exhaustive proof",
        )
        raw_order = document["variable_order"]
        if type(raw_order) is not list:
            raise TypeError("exhaustive variable_order must be a JSON array")
        variable_order = tuple(
            _require_int(variable, f"variable_order[{index}]")
            for index, variable in enumerate(raw_order)
        )
        proof = cls(
            problem_sha256=_require_sha256(document["problem_sha256"], "problem_sha256"),
            n_vars=_require_nonnegative_int(document["n_vars"], "n_vars"),
            variable_order=variable_order,
            enumeration_order=_require_text(
                document["enumeration_order"],
                "enumeration_order",
            ),
            initial_spin=_require_int(document["initial_spin"], "initial_spin"),
            checked_state_count=_require_nonnegative_int(
                document["checked_state_count"],
                "checked_state_count",
            ),
            minimum_gray_step=_require_nonnegative_int(
                document["minimum_gray_step"],
                "minimum_gray_step",
            ),
            final_gray_word=_require_nonnegative_int(
                document["final_gray_word"],
                "final_gray_word",
            ),
            assignment=_assignment_from_json(problem, document["assignment"]),
            spin_assignment_sha256=_require_sha256(
                document["spin_assignment_sha256"],
                "spin_assignment_sha256",
            ),
            minimum_energy=Dyadic(
                _require_int(document["minimum_energy_integer"], "minimum_energy_integer"),
                _require_int(
                    document["minimum_energy_power_of_two"],
                    "minimum_energy_power_of_two",
                ),
            ),
        )
        if proof.to_dict() != value:
            raise ValueError("exhaustive proof is not canonical")
        return proof


@dataclass(frozen=True, slots=True)
class CheckerTrust:
    """Independently registered checker identity, never sourced from a solver artifact."""

    proof_schema: str
    source_sha256: str
    environment_sha256: str
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "proof_schema", _require_text(self.proof_schema, "proof schema"))
        object.__setattr__(
            self, "source_sha256", _require_sha256(self.source_sha256, "checker source")
        )
        object.__setattr__(
            self,
            "environment_sha256",
            _require_sha256(self.environment_sha256, "checker environment"),
        )
        object.__setattr__(self, "command", _require_command(self.command, "checker command"))


@dataclass(frozen=True, slots=True)
class CertifiedOptimalProof:
    """Persisted solver proof metadata; acceptance is a separate trust boundary."""

    problem_sha256: str
    assignment: tuple[tuple[int, int], ...]
    spin_assignment_sha256: str
    lower_bound: Dyadic
    upper_bound: Dyadic
    solver_status: str
    solver: SolverRun
    proof_schema: str
    proof_file_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "problem_sha256",
            _require_sha256(self.problem_sha256, "problem_sha256"),
        )
        object.__setattr__(
            self,
            "assignment",
            _validate_spin_pairs(self.assignment, "certified-optimal assignment"),
        )
        object.__setattr__(
            self,
            "spin_assignment_sha256",
            _require_sha256(self.spin_assignment_sha256, "spin_assignment_sha256"),
        )
        _require_energy_dyadic(self.lower_bound, "certified-optimal lower bound")
        _require_energy_dyadic(self.upper_bound, "certified-optimal upper bound")
        if self.upper_bound < self.lower_bound:
            raise ValueError("certified-optimal lower bound cannot exceed upper bound")
        object.__setattr__(
            self,
            "solver_status",
            _require_text(self.solver_status, "solver_status"),
        )
        if not isinstance(self.solver, SolverRun):
            raise TypeError("certified-optimal solver must be a SolverRun")
        if self.solver.deterministic_work_limit > MAX_REGISTERED_BNB_NODES:
            raise ValueError("registered branch-and-bound work limit exceeds 100,000,000 nodes")
        object.__setattr__(
            self,
            "proof_schema",
            _require_text(self.proof_schema, "proof_schema"),
        )
        object.__setattr__(
            self,
            "proof_file_sha256",
            _require_sha256(self.proof_file_sha256, "proof_file_sha256"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": CERTIFIED_OPTIMAL_PROOF_SCHEMA,
            "schema_version": CERTIFIED_OPTIMAL_PROOF_SCHEMA_VERSION,
            "problem_sha256": self.problem_sha256,
            "assignment": [list(item) for item in self.assignment],
            "spin_assignment_sha256": self.spin_assignment_sha256,
            "lower_bound_integer": self.lower_bound.integer,
            "lower_bound_power_of_two": self.lower_bound.power_of_two,
            "upper_bound_integer": self.upper_bound.integer,
            "upper_bound_power_of_two": self.upper_bound.power_of_two,
            "solver_status": self.solver_status,
            "solver": self.solver.name,
            "solver_version": self.solver.version,
            "solver_command": list(self.solver.command),
            "solver_seed": self.solver.seed,
            "deterministic_work_limit": self.solver.deterministic_work_limit,
            "safety_timeout_seconds": self.solver.safety_timeout_seconds,
            "proof_schema": self.proof_schema,
            "proof_file_sha256": self.proof_file_sha256,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, problem: IsingProblem, value: object) -> CertifiedOptimalProof:
        document = validate_versioned_object(
            value,
            schema=CERTIFIED_OPTIMAL_PROOF_SCHEMA,
            schema_version=CERTIFIED_OPTIMAL_PROOF_SCHEMA_VERSION,
            fields={
                "problem_sha256",
                "assignment",
                "spin_assignment_sha256",
                "lower_bound_integer",
                "lower_bound_power_of_two",
                "upper_bound_integer",
                "upper_bound_power_of_two",
                "solver_status",
                "solver",
                "solver_version",
                "solver_command",
                "solver_seed",
                "deterministic_work_limit",
                "safety_timeout_seconds",
                "proof_schema",
                "proof_file_sha256",
            },
            name="certified-optimal proof",
        )
        solver = _solver_from_flat(document)
        if solver is None:
            raise ValueError("certified-optimal proof requires complete solver metadata")
        result = cls(
            problem_sha256=_require_sha256(document["problem_sha256"], "problem_sha256"),
            assignment=_assignment_from_json(problem, document["assignment"]),
            spin_assignment_sha256=_require_sha256(
                document["spin_assignment_sha256"],
                "spin_assignment_sha256",
            ),
            lower_bound=Dyadic(
                _require_int(document["lower_bound_integer"], "lower_bound_integer"),
                _require_int(document["lower_bound_power_of_two"], "lower_bound_power_of_two"),
            ),
            upper_bound=Dyadic(
                _require_int(document["upper_bound_integer"], "upper_bound_integer"),
                _require_int(document["upper_bound_power_of_two"], "upper_bound_power_of_two"),
            ),
            solver_status=_require_text(document["solver_status"], "solver_status"),
            solver=solver,
            proof_schema=_require_text(document["proof_schema"], "proof_schema"),
            proof_file_sha256=_require_sha256(
                document["proof_file_sha256"],
                "proof_file_sha256",
            ),
        )
        if result.to_dict() != value:
            raise ValueError("certified-optimal proof is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class CheckerAcceptance:
    """Content-addressed receipt for one prior registered-checker execution.

    This receipt binds identities and outputs.  It does not validate a branch-and-bound tree
    by itself; the publication gate must independently execute the registered checker.
    """

    problem_sha256: str
    proof_artifact_sha256: str
    proof_file_sha256: str
    proof_schema: str
    lower_bound: Dyadic
    upper_bound: Dyadic
    checker: CheckerRun
    outcome: str = "accepted"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "problem_sha256", _require_sha256(self.problem_sha256, "problem_sha256")
        )
        object.__setattr__(
            self,
            "proof_artifact_sha256",
            _require_sha256(self.proof_artifact_sha256, "proof_artifact_sha256"),
        )
        object.__setattr__(
            self,
            "proof_file_sha256",
            _require_sha256(self.proof_file_sha256, "proof_file_sha256"),
        )
        object.__setattr__(self, "proof_schema", _require_text(self.proof_schema, "proof_schema"))
        _require_energy_dyadic(self.lower_bound, "checker lower bound")
        _require_energy_dyadic(self.upper_bound, "checker upper bound")
        if self.upper_bound < self.lower_bound:
            raise ValueError("checker lower bound cannot exceed upper bound")
        if not isinstance(self.checker, CheckerRun) or self.checker.exit_code != 0:
            raise ValueError("checker acceptance requires a successful checker run")
        if self.outcome != "accepted":
            raise ValueError("checker acceptance outcome must be 'accepted'")

    def _payload(self) -> dict[str, object]:
        return {
            "schema": CHECKER_ACCEPTANCE_SCHEMA,
            "schema_version": CHECKER_ACCEPTANCE_SCHEMA_VERSION,
            "problem_sha256": self.problem_sha256,
            "proof_artifact_sha256": self.proof_artifact_sha256,
            "proof_file_sha256": self.proof_file_sha256,
            "proof_schema": self.proof_schema,
            "lower_bound_integer": self.lower_bound.integer,
            "lower_bound_power_of_two": self.lower_bound.power_of_two,
            "upper_bound_integer": self.upper_bound.integer,
            "upper_bound_power_of_two": self.upper_bound.power_of_two,
            "checker_source_sha256": self.checker.source_sha256,
            "checker_environment_sha256": self.checker.environment_sha256,
            "checker_command": list(self.checker.command),
            "checker_exit_code": self.checker.exit_code,
            "checker_log_sha256": self.checker.log_sha256,
            "outcome": self.outcome,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "checker_acceptance_sha256": self.digest}

    @classmethod
    def from_dict(cls, value: object) -> CheckerAcceptance:
        document = validate_versioned_object(
            value,
            schema=CHECKER_ACCEPTANCE_SCHEMA,
            schema_version=CHECKER_ACCEPTANCE_SCHEMA_VERSION,
            fields={
                "problem_sha256",
                "proof_artifact_sha256",
                "proof_file_sha256",
                "proof_schema",
                "lower_bound_integer",
                "lower_bound_power_of_two",
                "upper_bound_integer",
                "upper_bound_power_of_two",
                "checker_source_sha256",
                "checker_environment_sha256",
                "checker_command",
                "checker_exit_code",
                "checker_log_sha256",
                "outcome",
                "checker_acceptance_sha256",
            },
            name="checker acceptance",
        )
        checker = _checker_from_flat(document)
        if checker is None:
            raise ValueError("checker acceptance requires complete checker metadata")
        result = cls(
            problem_sha256=_require_sha256(document["problem_sha256"], "problem_sha256"),
            proof_artifact_sha256=_require_sha256(
                document["proof_artifact_sha256"],
                "proof_artifact_sha256",
            ),
            proof_file_sha256=_require_sha256(
                document["proof_file_sha256"],
                "proof_file_sha256",
            ),
            proof_schema=_require_text(document["proof_schema"], "proof_schema"),
            lower_bound=Dyadic(
                _require_int(document["lower_bound_integer"], "lower_bound_integer"),
                _require_int(document["lower_bound_power_of_two"], "lower_bound_power_of_two"),
            ),
            upper_bound=Dyadic(
                _require_int(document["upper_bound_integer"], "upper_bound_integer"),
                _require_int(document["upper_bound_power_of_two"], "upper_bound_power_of_two"),
            ),
            checker=checker,
            outcome=_require_text(document["outcome"], "outcome"),
        )
        supplied_digest = _require_sha256(
            document["checker_acceptance_sha256"],
            "checker_acceptance_sha256",
        )
        if supplied_digest != result.digest:
            raise ValueError("checker acceptance digest is invalid")
        if result.to_dict() != value:
            raise ValueError("checker acceptance is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class UncertifiedReferenceProof:
    """A verified assignment and upper bound with no claim of global optimality."""

    problem_sha256: str
    assignment: tuple[tuple[int, int], ...]
    spin_assignment_sha256: str
    reference_energy: Dyadic
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "problem_sha256",
            _require_sha256(self.problem_sha256, "problem_sha256"),
        )
        object.__setattr__(
            self,
            "assignment",
            _validate_spin_pairs(self.assignment, "uncertified assignment"),
        )
        object.__setattr__(
            self,
            "spin_assignment_sha256",
            _require_sha256(self.spin_assignment_sha256, "spin_assignment_sha256"),
        )
        _require_energy_dyadic(self.reference_energy, "reference_energy")
        object.__setattr__(self, "reason", _require_text(self.reason, "reason"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": UNCERTIFIED_REFERENCE_PROOF_SCHEMA,
            "schema_version": UNCERTIFIED_REFERENCE_PROOF_SCHEMA_VERSION,
            "problem_sha256": self.problem_sha256,
            "assignment": [list(item) for item in self.assignment],
            "spin_assignment_sha256": self.spin_assignment_sha256,
            "reference_energy_integer": self.reference_energy.integer,
            "reference_energy_power_of_two": self.reference_energy.power_of_two,
            "reason": self.reason,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, problem: IsingProblem, value: object) -> UncertifiedReferenceProof:
        document = validate_versioned_object(
            value,
            schema=UNCERTIFIED_REFERENCE_PROOF_SCHEMA,
            schema_version=UNCERTIFIED_REFERENCE_PROOF_SCHEMA_VERSION,
            fields={
                "problem_sha256",
                "assignment",
                "spin_assignment_sha256",
                "reference_energy_integer",
                "reference_energy_power_of_two",
                "reason",
            },
            name="uncertified-reference proof",
        )
        result = cls(
            problem_sha256=_require_sha256(document["problem_sha256"], "problem_sha256"),
            assignment=_assignment_from_json(problem, document["assignment"]),
            spin_assignment_sha256=_require_sha256(
                document["spin_assignment_sha256"],
                "spin_assignment_sha256",
            ),
            reference_energy=Dyadic(
                _require_int(document["reference_energy_integer"], "reference_energy_integer"),
                _require_int(
                    document["reference_energy_power_of_two"],
                    "reference_energy_power_of_two",
                ),
            ),
            reason=_require_text(document["reason"], "reason"),
        )
        if result.to_dict() != value:
            raise ValueError("uncertified-reference proof is not canonical")
        return result


def _assignment_from_json(
    problem: IsingProblem,
    value: object,
) -> tuple[tuple[int, int], ...]:
    if type(value) is not list:
        raise TypeError("assignment must be a JSON array")
    items: list[tuple[int, int]] = []
    for index, raw in enumerate(value):
        if type(raw) is not list or len(raw) != 2:
            raise TypeError(f"assignment[{index}] must be a two-item JSON array")
        items.append(
            (
                _require_int(raw[0], f"assignment[{index}] variable"),
                _require_int(raw[1], f"assignment[{index}] spin"),
            )
        )
    return _validate_assignment(problem, tuple(items))


_CERTIFICATE_FIELDS = {
    "problem_sha256",
    "status",
    "energy",
    "spin_assignment_sha256",
    "method",
    "solver",
    "solver_version",
    "solver_command",
    "solver_seed",
    "deterministic_work_limit",
    "safety_timeout_seconds",
    "lower_bound",
    "upper_bound",
    "relative_gap",
    "absolute_gap",
    "energy_integer",
    "energy_power_of_two",
    "lower_bound_integer",
    "lower_bound_power_of_two",
    "upper_bound_integer",
    "upper_bound_power_of_two",
    "proof_artifact_sha256",
    "checker_source_sha256",
    "checker_environment_sha256",
    "checker_command",
    "checker_exit_code",
    "checker_log_sha256",
    "checker_acceptance_sha256",
    "certificate_sha256",
}


def _diagnostic_gap(lower: Dyadic, upper: Dyadic) -> tuple[float, float | None]:
    gap = upper - lower
    absolute = gap.to_float()
    upper_float = upper.to_float()
    relative = None if upper.integer == 0 else absolute / abs(upper_float)
    if relative is not None and not math.isfinite(relative):
        raise ValueError("relative gap is not finite")
    return absolute, relative


@dataclass(frozen=True, slots=True)
class GroundStateCertificate:
    problem_sha256: str
    status: GroundStatus
    energy: Dyadic
    spin_assignment_sha256: str
    method: str
    solver: SolverRun | None
    lower_bound: Dyadic | None
    upper_bound: Dyadic
    proof_artifact_sha256: str
    checker: CheckerRun | None
    checker_acceptance_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "problem_sha256", _require_sha256(self.problem_sha256, "problem_sha256")
        )
        if type(self.status) is not str or self.status not in _GROUND_STATUSES:
            raise ValueError(f"status must be one of {sorted(_GROUND_STATUSES)}")
        _require_energy_dyadic(self.energy, "energy")
        _require_energy_dyadic(self.upper_bound, "upper_bound")
        if self.lower_bound is not None:
            _require_energy_dyadic(self.lower_bound, "lower_bound")
        if self.energy != self.upper_bound:
            raise ValueError("energy must equal the exact upper bound")
        object.__setattr__(
            self,
            "spin_assignment_sha256",
            _require_sha256(self.spin_assignment_sha256, "spin_assignment_sha256"),
        )
        object.__setattr__(self, "method", _require_text(self.method, "method"))
        if self.solver is not None and not isinstance(self.solver, SolverRun):
            raise TypeError("solver must be SolverRun or None")
        object.__setattr__(
            self,
            "proof_artifact_sha256",
            _require_sha256(self.proof_artifact_sha256, "proof_artifact_sha256"),
        )
        if self.checker is not None and not isinstance(self.checker, CheckerRun):
            raise TypeError("checker must be CheckerRun or None")
        if self.checker_acceptance_sha256 is not None:
            object.__setattr__(
                self,
                "checker_acceptance_sha256",
                _require_sha256(
                    self.checker_acceptance_sha256,
                    "checker_acceptance_sha256",
                ),
            )
        if self.status == "uncertified_reference":
            if self.lower_bound is not None or self.checker_acceptance_sha256 is not None:
                raise ValueError("uncertified reference cannot carry a certified lower bound")
        else:
            if self.lower_bound is None:
                raise ValueError("certified status requires an exact lower bound")
            if self.upper_bound < self.lower_bound:
                raise ValueError("lower bound cannot exceed upper bound")
            if self.status != "exact_enumeration" and (
                self.checker is None or self.checker.exit_code != 0
            ):
                raise ValueError("certified status requires a successful checker run")
            if (
                self.status == "exact_enumeration"
                and self.checker is not None
                and self.checker.exit_code != 0
            ):
                raise ValueError("exact-enumeration checker metadata reports failure")
        if self.status == "planted_proof" and self.solver is not None:
            raise ValueError("planted proof must not claim an exact solver run")
        if self.status == "certified_optimal" and self.checker_acceptance_sha256 is None:
            raise ValueError("certified_optimal requires an independently bound checker acceptance")
        if self.status != "certified_optimal" and self.checker_acceptance_sha256 is not None:
            raise ValueError("only certified_optimal uses an external checker acceptance")

    def _payload(self) -> dict[str, object]:
        lower = self.lower_bound
        if lower is None:
            lower_float = None
            lower_integer = None
            lower_power = None
            absolute_gap = None
            relative_gap = None
        else:
            lower_float = lower.to_float()
            lower_integer = lower.integer
            lower_power = lower.power_of_two
            absolute_gap, relative_gap = _diagnostic_gap(lower, self.upper_bound)
        solver = self.solver
        checker = self.checker
        return {
            "schema": CERTIFICATE_SCHEMA,
            "schema_version": CERTIFICATE_SCHEMA_VERSION,
            "problem_sha256": self.problem_sha256,
            "status": self.status,
            "energy": self.energy.to_float(),
            "spin_assignment_sha256": self.spin_assignment_sha256,
            "method": self.method,
            "solver": None if solver is None else solver.name,
            "solver_version": None if solver is None else solver.version,
            "solver_command": None if solver is None else list(solver.command),
            "solver_seed": None if solver is None else solver.seed,
            "deterministic_work_limit": (
                None if solver is None else solver.deterministic_work_limit
            ),
            "safety_timeout_seconds": None if solver is None else solver.safety_timeout_seconds,
            "lower_bound": lower_float,
            "upper_bound": self.upper_bound.to_float(),
            "relative_gap": relative_gap,
            "absolute_gap": absolute_gap,
            "energy_integer": self.energy.integer,
            "energy_power_of_two": self.energy.power_of_two,
            "lower_bound_integer": lower_integer,
            "lower_bound_power_of_two": lower_power,
            "upper_bound_integer": self.upper_bound.integer,
            "upper_bound_power_of_two": self.upper_bound.power_of_two,
            "proof_artifact_sha256": self.proof_artifact_sha256,
            "checker_source_sha256": None if checker is None else checker.source_sha256,
            "checker_environment_sha256": (None if checker is None else checker.environment_sha256),
            "checker_command": None if checker is None else list(checker.command),
            "checker_exit_code": None if checker is None else checker.exit_code,
            "checker_log_sha256": None if checker is None else checker.log_sha256,
            "checker_acceptance_sha256": self.checker_acceptance_sha256,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "certificate_sha256": self.digest}

    @classmethod
    def from_dict(cls, value: object) -> GroundStateCertificate:
        document = validate_versioned_object(
            value,
            schema=CERTIFICATE_SCHEMA,
            schema_version=CERTIFICATE_SCHEMA_VERSION,
            fields=_CERTIFICATE_FIELDS - {"schema", "schema_version"},
            name="ground-state certificate",
        )
        status = document["status"]
        if type(status) is not str or status not in _GROUND_STATUSES:
            raise ValueError(f"status must be one of {sorted(_GROUND_STATUSES)}")
        energy = Dyadic(
            _require_int(document["energy_integer"], "energy_integer"),
            _require_int(document["energy_power_of_two"], "energy_power_of_two"),
        )
        upper = Dyadic(
            _require_int(document["upper_bound_integer"], "upper_bound_integer"),
            _require_int(document["upper_bound_power_of_two"], "upper_bound_power_of_two"),
        )
        raw_lower_integer = document["lower_bound_integer"]
        raw_lower_power = document["lower_bound_power_of_two"]
        if raw_lower_integer is None and raw_lower_power is None:
            lower = None
        elif raw_lower_integer is None or raw_lower_power is None:
            raise ValueError("exact lower-bound fields must both be null or both be integers")
        else:
            lower = Dyadic(
                _require_int(raw_lower_integer, "lower_bound_integer"),
                _require_int(raw_lower_power, "lower_bound_power_of_two"),
            )
        solver = _solver_from_flat(document)
        checker = _checker_from_flat(document)
        acceptance_digest = document["checker_acceptance_sha256"]
        if acceptance_digest is not None:
            acceptance_digest = _require_sha256(
                acceptance_digest,
                "checker_acceptance_sha256",
            )
        result = cls(
            problem_sha256=_require_sha256(document["problem_sha256"], "problem_sha256"),
            status=status,
            energy=energy,
            spin_assignment_sha256=_require_sha256(
                document["spin_assignment_sha256"],
                "spin_assignment_sha256",
            ),
            method=_require_text(document["method"], "method"),
            solver=solver,
            lower_bound=lower,
            upper_bound=upper,
            proof_artifact_sha256=_require_sha256(
                document["proof_artifact_sha256"],
                "proof_artifact_sha256",
            ),
            checker=checker,
            checker_acceptance_sha256=acceptance_digest,
        )
        _validate_certificate_descriptive_fields(document, result)
        supplied_digest = _require_sha256(document["certificate_sha256"], "certificate_sha256")
        if supplied_digest != result.digest:
            raise ValueError("certificate_sha256 does not match the canonical certificate payload")
        if result.to_dict() != value:
            raise ValueError("ground-state certificate is not canonical")
        return result


def _solver_from_flat(document: dict[str, Any]) -> SolverRun | None:
    names = (
        "solver",
        "solver_version",
        "solver_command",
        "solver_seed",
        "deterministic_work_limit",
        "safety_timeout_seconds",
    )
    values = tuple(document[name] for name in names)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values[:3]) or values[4] is None:
        raise ValueError("solver metadata must be complete or wholly null")
    raw_command = values[2]
    if type(raw_command) is not list:
        raise TypeError("solver_command must be a JSON array")
    raw_seed = values[3]
    if raw_seed is not None:
        raw_seed = _require_nonnegative_int(raw_seed, "solver_seed")
    raw_timeout = values[5]
    if raw_timeout is not None:
        raw_timeout = _require_finite_binary64(raw_timeout, "safety_timeout_seconds")
    return SolverRun(
        name=values[0],
        version=values[1],
        command=tuple(raw_command),
        seed=raw_seed,
        deterministic_work_limit=_require_nonnegative_int(
            values[4],
            "deterministic_work_limit",
        ),
        safety_timeout_seconds=raw_timeout,
    )


def _checker_from_flat(document: dict[str, Any]) -> CheckerRun | None:
    names = (
        "checker_source_sha256",
        "checker_environment_sha256",
        "checker_command",
        "checker_exit_code",
        "checker_log_sha256",
    )
    values = tuple(document[name] for name in names)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("checker metadata must be complete or wholly null")
    raw_command = values[2]
    if type(raw_command) is not list:
        raise TypeError("checker_command must be a JSON array")
    return CheckerRun(
        source_sha256=values[0],
        environment_sha256=values[1],
        command=tuple(raw_command),
        exit_code=_require_int(values[3], "checker_exit_code"),
        log_sha256=values[4],
    )


def _validate_optional_descriptive_float(
    supplied: object,
    expected: float | None,
    name: str,
) -> None:
    if expected is None:
        if supplied is not None:
            raise ValueError(f"{name} must be null")
        return
    number = _require_finite_binary64(supplied, name)
    if number != expected:
        raise ValueError(f"{name} disagrees with the exact dyadic value")


def _validate_certificate_descriptive_fields(
    document: dict[str, Any],
    certificate: GroundStateCertificate,
) -> None:
    _validate_optional_descriptive_float(
        document["energy"], certificate.energy.to_float(), "energy"
    )
    _validate_optional_descriptive_float(
        document["upper_bound"],
        certificate.upper_bound.to_float(),
        "upper_bound",
    )
    if certificate.lower_bound is None:
        expected_lower = expected_absolute = expected_relative = None
    else:
        expected_lower = certificate.lower_bound.to_float()
        expected_absolute, expected_relative = _diagnostic_gap(
            certificate.lower_bound,
            certificate.upper_bound,
        )
    _validate_optional_descriptive_float(document["lower_bound"], expected_lower, "lower_bound")
    _validate_optional_descriptive_float(
        document["absolute_gap"],
        expected_absolute,
        "absolute_gap",
    )
    _validate_optional_descriptive_float(
        document["relative_gap"],
        expected_relative,
        "relative_gap",
    )


@dataclass(frozen=True, slots=True)
class CertificateBundle:
    certificate: GroundStateCertificate
    proof: PlantedProof | ExhaustiveProof | CertifiedOptimalProof | UncertifiedReferenceProof
    checker_acceptance: CheckerAcceptance | None = None


@dataclass(frozen=True, slots=True)
class VerifiedGroundState:
    status: GroundStatus
    energy: Dyadic
    lower_bound: Dyadic | None
    upper_bound: Dyadic
    quality_eligible: bool
    certificate_sha256: str
    proof_artifact_sha256: str


def build_planted_certificate(
    problem: IsingProblem,
    *,
    assignment: object,
    terms: object,
    method: str,
    checker: CheckerRun,
) -> CertificateBundle:
    """Build a planted certificate only after independently proving every term bound."""

    if not isinstance(problem, IsingProblem):
        raise TypeError("problem must be an IsingProblem")
    checked_assignment = _validate_assignment(problem, assignment)
    if type(terms) is not tuple or not all(isinstance(term, PlantedTerm) for term in terms):
        raise TypeError("terms must be an immutable tuple of PlantedTerm values")
    if not isinstance(checker, CheckerRun) or checker.exit_code != 0:
        raise ValueError("planted proof requires a successful checker run")
    energy = _verify_planted_terms(problem, checked_assignment, terms)
    assignment_digest = spin_assignment_sha256(problem, checked_assignment)
    proof = PlantedProof(
        problem_sha256=problem.problem_sha256,
        assignment=checked_assignment,
        spin_assignment_sha256=assignment_digest,
        terms=terms,
        assignment_energy=energy,
        sum_lower_bound=energy,
    )
    certificate = GroundStateCertificate(
        problem_sha256=problem.problem_sha256,
        status="planted_proof",
        energy=energy,
        spin_assignment_sha256=assignment_digest,
        method=_require_text(method, "method"),
        solver=None,
        lower_bound=energy,
        upper_bound=energy,
        proof_artifact_sha256=proof.digest,
        checker=checker,
    )
    return CertificateBundle(certificate=certificate, proof=proof)


def _enumerate_problem(problem: IsingProblem) -> _EnumerationResult:
    return _gray_code_minimum(
        problem.variables,
        tuple(
            (variable, binary64_to_dyadic(coefficient)) for variable, coefficient in problem.linear
        ),
        tuple(
            (left, right, binary64_to_dyadic(coefficient))
            for left, right, coefficient in problem.quadratic
        ),
    )


def build_exact_enumeration_certificate(
    problem: IsingProblem,
    *,
    solver: SolverRun,
    checker: CheckerRun | None = None,
) -> CertificateBundle:
    """Enumerate every state and persist a header replayable without an external checker."""

    if not isinstance(problem, IsingProblem):
        raise TypeError("problem must be an IsingProblem")
    if len(problem.variables) > MAX_EXHAUSTIVE_VARIABLES:
        raise ValueError(f"exact enumeration is limited to {MAX_EXHAUSTIVE_VARIABLES} variables")
    if not isinstance(solver, SolverRun):
        raise TypeError("solver must be a SolverRun")
    expected_state_count = 1 << len(problem.variables)
    if solver.deterministic_work_limit != expected_state_count:
        raise ValueError("exact enumeration solver work limit must equal the full state count")
    if checker is not None and (
        not isinstance(checker, CheckerRun) or checker.exit_code != 0
    ):
        raise ValueError("exact-enumeration checker metadata must report success")
    result = _enumerate_problem(problem)
    assignment_digest = spin_assignment_sha256(problem, result.assignment)
    proof = ExhaustiveProof(
        problem_sha256=problem.problem_sha256,
        n_vars=len(problem.variables),
        variable_order=problem.variables,
        enumeration_order=GRAY_CODE_ENUMERATION,
        initial_spin=-1,
        checked_state_count=result.checked_state_count,
        minimum_gray_step=result.minimum_gray_step,
        final_gray_word=result.final_gray_word,
        assignment=result.assignment,
        spin_assignment_sha256=assignment_digest,
        minimum_energy=result.energy,
    )
    certificate = GroundStateCertificate(
        problem_sha256=problem.problem_sha256,
        status="exact_enumeration",
        energy=result.energy,
        spin_assignment_sha256=assignment_digest,
        method="gray_code_exhaustive_enumeration",
        solver=solver,
        lower_bound=result.energy,
        upper_bound=result.energy,
        proof_artifact_sha256=proof.digest,
        checker=checker,
    )
    return CertificateBundle(certificate=certificate, proof=proof)


def build_certified_optimal_certificate(
    problem: IsingProblem,
    *,
    assignment: object,
    lower_bound: Dyadic,
    upper_bound: Dyadic,
    solver_status: str,
    solver: SolverRun,
    proof_schema: str,
    proof_file_sha256: str,
    checker: CheckerRun,
) -> CertificateBundle:
    """Record solver bounds plus a receipt for a prior external checker execution.

    ``solver_status`` is retained as provenance and is deliberately not consulted for
    eligibility.  This function does not inspect the proof-file semantics or run a checker;
    the caller must supply metadata from that execution.  Only exact bound equality after the
    independently registered checker boundary can make this certificate quality eligible.
    """

    if not isinstance(problem, IsingProblem):
        raise TypeError("problem must be an IsingProblem")
    if len(problem.variables) <= MAX_EXHAUSTIVE_VARIABLES:
        raise ValueError(
            "non-planted problems with at most 24 variables require Gray-code enumeration"
        )
    checked_assignment = _validate_assignment(problem, assignment)
    _require_energy_dyadic(lower_bound, "solver lower bound")
    _require_energy_dyadic(upper_bound, "solver upper bound")
    if upper_bound < lower_bound:
        raise ValueError("solver lower bound cannot exceed upper bound")
    if exact_energy(problem, checked_assignment) != upper_bound:
        raise ValueError("solver assignment does not attain the exact upper bound")
    if not isinstance(solver, SolverRun):
        raise TypeError("solver must be a SolverRun")
    if solver.deterministic_work_limit > MAX_REGISTERED_BNB_NODES:
        raise ValueError("registered branch-and-bound work limit exceeds 100,000,000 nodes")
    if not isinstance(checker, CheckerRun) or checker.exit_code != 0:
        raise ValueError("certified-optimal proof requires a successful checker run")
    checked_status = _require_text(solver_status, "solver_status")
    checked_proof_schema = _require_text(proof_schema, "proof_schema")
    checked_proof_file = _require_sha256(proof_file_sha256, "proof_file_sha256")
    assignment_digest = spin_assignment_sha256(problem, checked_assignment)
    proof = CertifiedOptimalProof(
        problem_sha256=problem.problem_sha256,
        assignment=checked_assignment,
        spin_assignment_sha256=assignment_digest,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        solver_status=checked_status,
        solver=solver,
        proof_schema=checked_proof_schema,
        proof_file_sha256=checked_proof_file,
    )
    acceptance = CheckerAcceptance(
        problem_sha256=problem.problem_sha256,
        proof_artifact_sha256=proof.digest,
        proof_file_sha256=checked_proof_file,
        proof_schema=checked_proof_schema,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        checker=checker,
    )
    certificate = GroundStateCertificate(
        problem_sha256=problem.problem_sha256,
        status="certified_optimal",
        energy=upper_bound,
        spin_assignment_sha256=assignment_digest,
        method=checked_proof_schema,
        solver=solver,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        proof_artifact_sha256=proof.digest,
        checker=checker,
        checker_acceptance_sha256=acceptance.digest,
    )
    return CertificateBundle(
        certificate=certificate,
        proof=proof,
        checker_acceptance=acceptance,
    )


def build_uncertified_reference_certificate(
    problem: IsingProblem,
    *,
    assignment: object,
    method: str,
    reason: str,
    solver: SolverRun,
    checker: CheckerRun | None = None,
) -> CertificateBundle:
    """Persist a checked upper-bound witness without making an optimality claim."""

    if not isinstance(problem, IsingProblem):
        raise TypeError("problem must be an IsingProblem")
    checked_assignment = _validate_assignment(problem, assignment)
    if not isinstance(solver, SolverRun):
        raise TypeError("solver must be a SolverRun")
    if checker is not None and not isinstance(checker, CheckerRun):
        raise TypeError("checker must be CheckerRun or None")
    energy = exact_energy(problem, checked_assignment)
    assignment_digest = spin_assignment_sha256(problem, checked_assignment)
    proof = UncertifiedReferenceProof(
        problem_sha256=problem.problem_sha256,
        assignment=checked_assignment,
        spin_assignment_sha256=assignment_digest,
        reference_energy=energy,
        reason=_require_text(reason, "reason"),
    )
    certificate = GroundStateCertificate(
        problem_sha256=problem.problem_sha256,
        status="uncertified_reference",
        energy=energy,
        spin_assignment_sha256=assignment_digest,
        method=_require_text(method, "method"),
        solver=solver,
        lower_bound=None,
        upper_bound=energy,
        proof_artifact_sha256=proof.digest,
        checker=checker,
    )
    return CertificateBundle(certificate=certificate, proof=proof)


def _verify_checker_acceptance(
    *,
    problem: IsingProblem,
    certificate: GroundStateCertificate,
    proof: CertifiedOptimalProof,
    proof_digest: str,
    acceptance_document: object,
    expected_acceptance_digest: object,
    trust: object,
) -> None:
    if expected_acceptance_digest is None:
        raise ValueError("certified_optimal requires an independent checker acceptance digest")
    expected_digest = _require_sha256(
        expected_acceptance_digest,
        "expected_checker_acceptance_sha256",
    )
    if not isinstance(trust, CheckerTrust):
        raise ValueError("certified_optimal requires independently registered checker trust")
    if acceptance_document is None:
        raise ValueError("certified_optimal requires a checker acceptance artifact")
    acceptance = CheckerAcceptance.from_dict(acceptance_document)
    if acceptance.digest != expected_digest:
        raise ValueError("checker acceptance disagrees with its independently supplied digest")
    if certificate.checker_acceptance_sha256 != acceptance.digest:
        raise ValueError("certificate does not bind the checker acceptance")
    expected_binding = (
        problem.problem_sha256,
        proof_digest,
        proof.proof_file_sha256,
        proof.proof_schema,
        proof.lower_bound,
        proof.upper_bound,
    )
    supplied_binding = (
        acceptance.problem_sha256,
        acceptance.proof_artifact_sha256,
        acceptance.proof_file_sha256,
        acceptance.proof_schema,
        acceptance.lower_bound,
        acceptance.upper_bound,
    )
    if supplied_binding != expected_binding:
        raise ValueError("checker acceptance does not bind the exact proof and bounds")
    if certificate.checker != acceptance.checker:
        raise ValueError("certificate checker metadata disagrees with its acceptance")
    trusted_identity = (
        trust.proof_schema,
        trust.source_sha256,
        trust.environment_sha256,
        trust.command,
    )
    accepted_identity = (
        acceptance.proof_schema,
        acceptance.checker.source_sha256,
        acceptance.checker.environment_sha256,
        acceptance.checker.command,
    )
    if accepted_identity != trusted_identity:
        raise ValueError("checker acceptance does not match the independently registered checker")


def verify_ground_state_certificate(
    problem: IsingProblem,
    certificate_document: object,
    proof_document: object,
    *,
    expected_certificate_sha256: str,
    expected_proof_artifact_sha256: str,
    checker_acceptance: object | None = None,
    expected_checker_acceptance_sha256: str | None = None,
    checker_trust: object | None = None,
) -> VerifiedGroundState:
    """Verify bindings and replayable proofs against independent content identities.

    For ``certified_optimal``, this verifies a separately supplied checker receipt and trusted
    checker identity, but deliberately does not interpret the external proof-file format.
    Callers must first run the registered semantic checker on those exact proof bytes.
    """

    if not isinstance(problem, IsingProblem):
        raise TypeError("problem must be an IsingProblem")
    expected_certificate_digest = _require_sha256(
        expected_certificate_sha256,
        "expected_certificate_sha256",
    )
    expected_proof_digest = _require_sha256(
        expected_proof_artifact_sha256,
        "expected_proof_artifact_sha256",
    )
    certificate = GroundStateCertificate.from_dict(certificate_document)
    if certificate.digest != expected_certificate_digest:
        raise ValueError("certificate digest disagrees with the independently supplied digest")
    if certificate.problem_sha256 != problem.problem_sha256:
        raise ValueError("certificate problem_sha256 does not match the problem")
    if type(proof_document) is not dict:
        raise TypeError("proof artifact must be a JSON object")
    proof_digest = canonical_sha256(proof_document)
    if proof_digest != expected_proof_digest or proof_digest != certificate.proof_artifact_sha256:
        raise ValueError("proof artifact digest does not match its independent bindings")
    if certificate.status == "planted_proof":
        if any(
            value is not None
            for value in (
                checker_acceptance,
                expected_checker_acceptance_sha256,
                checker_trust,
            )
        ):
            raise ValueError("planted proof must not carry external checker acceptance inputs")
        proof = PlantedProof.from_dict(problem, proof_document)
        if proof.problem_sha256 != problem.problem_sha256:
            raise ValueError("planted proof problem_sha256 does not match the problem")
        assignment_digest = spin_assignment_sha256(problem, proof.assignment)
        if proof.spin_assignment_sha256 != assignment_digest:
            raise ValueError("planted proof assignment digest is invalid")
        if certificate.spin_assignment_sha256 != assignment_digest:
            raise ValueError("certificate assignment digest disagrees with the planted proof")
        energy = _verify_planted_terms(problem, proof.assignment, proof.terms)
        if proof.assignment_energy != energy or proof.sum_lower_bound != energy:
            raise ValueError("planted proof exact energy fields are invalid")
        if certificate.energy != energy or certificate.lower_bound != energy:
            raise ValueError("certificate energy does not match the planted proof")
    elif certificate.status == "exact_enumeration":
        if any(
            value is not None
            for value in (
                checker_acceptance,
                expected_checker_acceptance_sha256,
                checker_trust,
            )
        ):
            raise ValueError("exact enumeration must not carry external checker acceptance inputs")
        proof = ExhaustiveProof.from_dict(problem, proof_document)
        if proof.problem_sha256 != problem.problem_sha256:
            raise ValueError("exhaustive proof problem_sha256 does not match the problem")
        if len(problem.variables) > MAX_EXHAUSTIVE_VARIABLES:
            raise ValueError("exhaustive proof exceeds the registered variable limit")
        expected_count = 1 << len(problem.variables)
        if (
            certificate.solver is None
            or certificate.solver.deterministic_work_limit != expected_count
        ):
            raise ValueError("exact enumeration solver work limit is not the full state count")
        replay = _enumerate_problem(problem)
        assignment_digest = spin_assignment_sha256(problem, replay.assignment)
        expected_header = (
            len(problem.variables),
            problem.variables,
            GRAY_CODE_ENUMERATION,
            -1,
            replay.checked_state_count,
            replay.minimum_gray_step,
            replay.final_gray_word,
            replay.assignment,
            assignment_digest,
            replay.energy,
        )
        supplied_header = (
            proof.n_vars,
            proof.variable_order,
            proof.enumeration_order,
            proof.initial_spin,
            proof.checked_state_count,
            proof.minimum_gray_step,
            proof.final_gray_word,
            proof.assignment,
            proof.spin_assignment_sha256,
            proof.minimum_energy,
        )
        if supplied_header != expected_header:
            raise ValueError("exhaustive Gray-code transcript does not replay exactly")
        if certificate.spin_assignment_sha256 != assignment_digest:
            raise ValueError("certificate assignment digest disagrees with exhaustive proof")
        energy = replay.energy
        if certificate.energy != energy or certificate.lower_bound != energy:
            raise ValueError("certificate energy does not match exhaustive enumeration")
    elif certificate.status == "certified_optimal":
        proof = CertifiedOptimalProof.from_dict(problem, proof_document)
        if proof.problem_sha256 != problem.problem_sha256:
            raise ValueError("certified-optimal proof problem_sha256 does not match the problem")
        if len(problem.variables) <= MAX_EXHAUSTIVE_VARIABLES:
            raise ValueError(
                "non-planted problems with at most 24 variables require Gray-code enumeration"
            )
        if proof.solver.deterministic_work_limit > MAX_REGISTERED_BNB_NODES:
            raise ValueError("registered branch-and-bound work limit exceeds 100,000,000 nodes")
        assignment_digest = spin_assignment_sha256(problem, proof.assignment)
        if proof.spin_assignment_sha256 != assignment_digest:
            raise ValueError("certified-optimal proof assignment digest is invalid")
        if certificate.spin_assignment_sha256 != assignment_digest:
            raise ValueError("certificate assignment digest disagrees with solver proof")
        energy = exact_energy(problem, proof.assignment)
        if energy != proof.upper_bound:
            raise ValueError("certified-optimal assignment does not attain its upper bound")
        if proof.upper_bound < proof.lower_bound:
            raise ValueError("certified-optimal proof has reversed bounds")
        if certificate.solver != proof.solver:
            raise ValueError("certificate solver metadata disagrees with solver proof")
        if certificate.method != proof.proof_schema:
            raise ValueError("certificate method disagrees with the persisted proof schema")
        if (
            certificate.energy != proof.upper_bound
            or certificate.lower_bound != proof.lower_bound
            or certificate.upper_bound != proof.upper_bound
        ):
            raise ValueError("certificate bounds disagree with the certified-optimal proof")
        _verify_checker_acceptance(
            problem=problem,
            certificate=certificate,
            proof=proof,
            proof_digest=proof_digest,
            acceptance_document=checker_acceptance,
            expected_acceptance_digest=expected_checker_acceptance_sha256,
            trust=checker_trust,
        )
    elif certificate.status == "uncertified_reference":
        if any(
            value is not None
            for value in (
                checker_acceptance,
                expected_checker_acceptance_sha256,
                checker_trust,
            )
        ):
            raise ValueError("uncertified reference must not carry checker acceptance inputs")
        proof = UncertifiedReferenceProof.from_dict(problem, proof_document)
        if proof.problem_sha256 != problem.problem_sha256:
            raise ValueError("uncertified proof problem_sha256 does not match the problem")
        assignment_digest = spin_assignment_sha256(problem, proof.assignment)
        if proof.spin_assignment_sha256 != assignment_digest:
            raise ValueError("uncertified proof assignment digest is invalid")
        if certificate.spin_assignment_sha256 != assignment_digest:
            raise ValueError("certificate assignment digest disagrees with uncertified proof")
        energy = exact_energy(problem, proof.assignment)
        if proof.reference_energy != energy or certificate.energy != energy:
            raise ValueError("uncertified reference energy does not match its assignment")
        if certificate.lower_bound is not None:
            raise ValueError("uncertified reference cannot carry a lower bound")
    else:
        raise ValueError("ground-state certificate status is not implemented")
    quality_eligible = (
        certificate.status != "uncertified_reference"
        and certificate.lower_bound is not None
        and certificate.lower_bound == certificate.upper_bound
    )
    return VerifiedGroundState(
        status=certificate.status,
        energy=energy,
        lower_bound=certificate.lower_bound,
        upper_bound=certificate.upper_bound,
        quality_eligible=quality_eligible,
        certificate_sha256=certificate.digest,
        proof_artifact_sha256=proof_digest,
    )
