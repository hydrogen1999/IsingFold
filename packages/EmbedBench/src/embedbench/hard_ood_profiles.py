"""Exact registered bands and data-driven hard/OOD cell predicates.

This module intentionally does not embed release-specific profile identifiers, problem sizes,
or fill sub-bands.  Those values must arrive in a strict, frozen cell-profile object.  Only
the four Section 8.2 band thresholds are code-level registered constants.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Collection
from dataclasses import dataclass, field

import networkx as nx

from embedbench.ground_certificate import IsingProblem, verify_ground_state_certificate
from embedbench.hard_ood_schema import (
    SeedRequest,
    VerifiedSeedResolver,
    canonical_sha256,
    validate_seed_resolver,
)
from embedbench.hardness import (
    PartialContext,
    VerifiedHardness,
    VerifiedHost,
    VerifiedProblem,
    VerifiedStartingEmbedding,
    validate_hardness,
)

CELL_PROFILE_SCHEMA = "embedbench.hard-ood-cell-profile"
CELL_PROFILE_SCHEMA_VERSION = 1
PROFILE_SET_SCHEMA = "embedbench.hard-ood-profile-set"
PROFILE_SET_SCHEMA_VERSION = 1
HARD_OOD_RELEASE_ID = "embedbench-hard-ood-v1.0.0"
REGISTERED_MAXIMUM_ORDINAL = 10_000

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_PARTITIONS = frozenset(
    {"hard_dev", "mechanism_dev", "mechanism_locked", "composed_locked", "locked_ood"}
)
_CELL_PROFILE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "profile_id",
        "cell_id",
        "partition",
        "panel",
        "quota",
        "ratio_predicates",
        "integer_predicates",
        "value_predicates",
        "band_predicates",
    }
)
_RATIO_PREDICATE_FIELDS = frozenset({"field", "minimum", "maximum"})
_RATIO_BOUND_FIELDS = frozenset({"numerator", "denominator", "inclusive"})
_INTEGER_PREDICATE_FIELDS = frozenset({"field", "minimum", "maximum"})
_INTEGER_BOUND_FIELDS = frozenset({"value", "inclusive"})
_VALUE_PREDICATE_FIELDS = frozenset({"field", "allowed_values"})
_BAND_PREDICATE_FIELDS = frozenset({"axis", "allowed_bands"})
_CELL_FACT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "release_id",
        "profile_set_sha256",
        "plan_row_sha256",
        "cell_id",
        "partition",
        "panel",
        "ordinal",
        "fill_subband",
        "hardness_sha256",
        "hardness",
        "problem_sha256",
        "host_artifact_sha256",
        "host_sha256",
        "starting_embedding_sha256",
        "partial_context_sha256",
        "candidate_protocol_result_sha256",
        "topology",
        "size",
        "logical_source",
        "logical_family",
        "chain_size",
        "l_cap",
        "max_window_free",
        "q_cap_slack",
        "requested_qubit_numerator",
        "requested_qubit_denominator",
        "requested_qubit_fraction",
        "requested_coupler_numerator",
        "requested_coupler_denominator",
        "requested_coupler_fraction",
        "host_largest_component_numerator",
        "host_largest_component_denominator",
        "host_largest_component_fraction",
        "ground_certificate_status",
        "ground_certificate_sha256",
        "ground_proof_artifact_sha256",
        "ground_state_eligible",
    }
)
QUOTA_ENTRY_SCHEMA = "embedbench.hard-ood-quota-entry"
QUOTA_ENTRY_SCHEMA_VERSION = 1
_QUOTA_ENTRY_SEAL = object()
_QUOTA_ENTRY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "release_id",
        "profile_set_sha256",
        "plan_row_sha256",
        "cell_id",
        "partition",
        "panel",
        "ordinal",
        "problem_group_id",
        "problem_sha256",
        "problem_iso_sha256",
        "state_sha256",
        "host_sha256",
        "mechanism_parent_id",
        "mechanism_iso_sha256",
        "mechanism_certificate_sha256",
        "mechanism_ablation_sha256",
        "cell_facts_sha256",
        "quota_entry_sha256",
    }
)

_RATIO_FIELDS = frozenset(
    {
        "hardness.host.defect_qubit",
        "hardness.host.defect_coupler",
        "hardness.logical.average_degree",
        "hardness.logical.density",
        "hardness.starting_embedding.host_fill",
        "hardness.starting_embedding.mean_chain_length",
        "hardness.reference_candidate.residual.free_node",
        "hardness.reference_candidate.residual.free_edge",
        "hardness.reference_candidate.residual.largest_free_component_node",
        "hardness.reference_candidate.residual.largest_free_component_edge",
        "hardness.decision.exact_completion",
        "requested_qubit_fraction",
        "requested_coupler_fraction",
        "host_largest_component_fraction",
    }
)
_INTEGER_FIELDS = frozenset(
    {
        "size",
        "chain_size",
        "l_cap",
        "max_window_free",
        "q_cap_slack",
        "hardness.host.nodes",
        "hardness.host.edges",
        "hardness.host.component_count",
        "hardness.logical.variables",
        "hardness.logical.edges",
        "hardness.logical.maximum_degree",
        "hardness.logical.component_count",
        "hardness.starting_embedding.total_qubits",
        "hardness.starting_embedding.maximum_chain_length",
        "hardness.reference_candidate.residual.largest_free_component_edge_connectivity",
        "hardness.reference_candidate.residual.articulation_count",
        "hardness.reference_candidate.residual.largest_free_component_articulation_count",
        "hardness.decision.focus_degree",
        "hardness.decision.window_nodes",
        "hardness.decision.window_edges",
        "hardness.decision.full_candidate_count",
        "hardness.decision.offered_candidate_count",
    }
)
_VALUE_FIELD_TYPES: dict[str, type] = {
    "topology": str,
    "logical_source": str,
    "logical_family": str,
    "ground_state_eligible": bool,
    **{field: int for field in _INTEGER_FIELDS},
}
_REGISTERED_BANDS = {
    "host_fill": frozenset({"low", "medium", "high", "extreme"}),
    "logical_degree": frozenset({"sparse", "moderate", "dense", "very_dense"}),
    "residual_lcc": frozenset({"open", "constrained", "critical"}),
    "defects": frozenset({"none", "light", "heavy"}),
}

CELL_FACTS_SCHEMA = "embedbench.hard-ood-cell-facts"
CELL_FACTS_SCHEMA_VERSION = 1
_CELL_FACTS_SEAL = object()
PLAN_ROW_SCHEMA = "embedbench.hard-ood-plan-row"
PLAN_ROW_SCHEMA_VERSION = 1
_PLAN_ROW_SEAL = object()
_PLAN_ROW_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "release_id",
        "profile_set_sha256",
        "cell_id",
        "partition",
        "panel",
        "ordinal",
        "topology",
        "size",
        "logical_source",
        "logical_family",
        "n_vars",
        "chain_size",
        "l_cap",
        "max_window_free",
        "q_cap_slack",
        "requested_qubit_numerator",
        "requested_qubit_denominator",
        "requested_coupler_numerator",
        "requested_coupler_denominator",
        "fill_subband",
        "starting_source",
        "generator_arguments_sha256",
        "problem_seed_request",
        "problem_seed_key_sha256",
        "state_slot",
        "focus_quartile",
        "plan_row_sha256",
    }
)
_STARTING_SOURCES = frozenset({"witness", "minorminer", "cpp_baseline", "witness_perturbation"})


@dataclass(frozen=True, slots=True)
class CellRequirement:
    """One code-registered release cell, independent of profile JSON bytes."""

    cell_id: str
    partition: str
    panel: str
    quota: int
    topology: str | None = None
    size: int | None = None
    logical_family: str | None = None


def _registered_cell_requirements() -> tuple[CellRequirement, ...]:
    requirements: list[CellRequirement] = []
    base_hosts = ("chimera-5", "pegasus-3", "zephyr-2")
    for host in base_hosts:
        topology, raw_size = host.split("-")
        for panel in ("capacity_transition", "logical_stress", "long_context"):
            requirements.append(
                CellRequirement(
                    f"hard_dev/{host}/{panel}",
                    "hard_dev",
                    panel,
                    40,
                    topology,
                    int(raw_size),
                )
            )
    mechanisms = (
        "dead_end",
        "articulation",
        "cut_capacity",
        "multi_neighbour",
        "high_degree_trap",
        "short_chain_trap",
        "ordering",
    )
    for partition in ("mechanism_dev", "mechanism_locked"):
        for family in mechanisms:
            requirements.append(
                CellRequirement(
                    f"{partition}/{family}",
                    partition,
                    family,
                    50,
                    logical_family=family,
                )
            )
    for composition in ("two_trap", "three_trap"):
        requirements.append(
            CellRequirement(
                f"composed_locked/{composition}",
                "composed_locked",
                composition,
                50,
                logical_family=composition,
            )
        )
    for host in base_hosts:
        topology, raw_size = host.split("-")
        requirements.append(
            CellRequirement(
                f"locked_ood/extreme_capacity/{host}",
                "locked_ood",
                "extreme_capacity",
                30,
                topology,
                int(raw_size),
            )
        )
    for host in ("chimera-6", "pegasus-4", "zephyr-3"):
        topology, raw_size = host.split("-")
        requirements.append(
            CellRequirement(
                f"locked_ood/host_scale/{host}",
                "locked_ood",
                "host_scale",
                30,
                topology,
                int(raw_size),
            )
        )
    for family in ("jobshop", "oct_friendly", "oct_adversarial"):
        requirements.append(
            CellRequirement(
                f"locked_ood/unseen_logical_family/{family}",
                "locked_ood",
                "unseen_logical_family",
                30,
                logical_family=family,
            )
        )
    for host in base_hosts:
        topology, raw_size = host.split("-")
        requirements.append(
            CellRequirement(
                f"locked_ood/host_faults/{host}",
                "locked_ood",
                "host_faults",
                30,
                topology,
                int(raw_size),
            )
        )
    return tuple(requirements)


REGISTERED_CELL_REQUIREMENTS = _registered_cell_requirements()
_REQUIREMENT_BY_CELL = {item.cell_id: item for item in REGISTERED_CELL_REQUIREMENTS}
if len(_REQUIREMENT_BY_CELL) != len(REGISTERED_CELL_REQUIREMENTS):  # pragma: no cover
    raise RuntimeError("registered hard/OOD cell identifiers are not unique")
if sum(item.quota for item in REGISTERED_CELL_REQUIREMENTS) != 1_520:  # pragma: no cover
    raise RuntimeError("registered hard/OOD release total is not 1,520 groups")

_PROFILE_SET_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "release_id",
        "generator_source_sha256",
        "maximum_ordinal",
        "cells",
    }
)
_PROFILE_SET_SEAL = object()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedProfileSet:
    """Detached, content-addressed profile set with an exact registered cell layout."""

    _serialized: str
    profile_set_sha256: str
    _seal: object = field(repr=False, compare=False)

    @classmethod
    def _from_document(cls, document: dict[str, object]) -> VerifiedProfileSet:
        serialized = json.dumps(
            copy.deepcopy(document),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "_serialized", serialized)
        object.__setattr__(instance, "profile_set_sha256", canonical_sha256(document))
        object.__setattr__(instance, "_seal", _PROFILE_SET_SEAL)
        return instance

    def to_dict(self) -> dict[str, object]:
        document = json.loads(self._serialized)
        if type(document) is not dict:  # pragma: no cover - constructor invariant
            raise RuntimeError("verified profile set payload is not an object")
        return document

    def cell(self, cell_id: str) -> dict[str, object]:
        for cell in self.to_dict()["cells"]:
            if cell["cell_id"] == cell_id:
                return cell
        raise KeyError(cell_id)


@dataclass(frozen=True, slots=True, init=False)
class VerifiedCellFacts:
    """Immutable predicate facts derived only from authenticated scientific objects."""

    _serialized: str
    cell_facts_sha256: str
    _seal: object = field(repr=False, compare=False)

    @classmethod
    def _from_document(cls, document: dict[str, object]) -> VerifiedCellFacts:
        serialized = json.dumps(
            copy.deepcopy(document),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "_serialized", serialized)
        object.__setattr__(instance, "cell_facts_sha256", canonical_sha256(document))
        object.__setattr__(instance, "_seal", _CELL_FACTS_SEAL)
        return instance

    def to_dict(self) -> dict[str, object]:
        document = json.loads(self._serialized)
        if type(document) is not dict:  # pragma: no cover - constructor invariant
            raise RuntimeError("verified cell facts payload is not an object")
        return document


@dataclass(frozen=True, slots=True, init=False)
class VerifiedPlanRow:
    """Immutable pre-label plan row authenticated against one frozen profile set."""

    _serialized: str
    plan_row_sha256: str
    _seal: object = field(repr=False, compare=False)

    @classmethod
    def _from_document(cls, document: dict[str, object]) -> VerifiedPlanRow:
        serialized = json.dumps(
            copy.deepcopy(document),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "_serialized", serialized)
        object.__setattr__(instance, "plan_row_sha256", document["plan_row_sha256"])
        object.__setattr__(instance, "_seal", _PLAN_ROW_SEAL)
        return instance

    def to_dict(self) -> dict[str, object]:
        document = json.loads(self._serialized)
        if type(document) is not dict:  # pragma: no cover - constructor invariant
            raise RuntimeError("verified plan row payload is not an object")
        return document


@dataclass(frozen=True, slots=True, init=False)
class VerifiedQuotaEntry:
    """Immutable accepted-group identity used by release quota accounting."""

    _serialized: str
    quota_entry_sha256: str
    _seal: object = field(repr=False, compare=False)

    @classmethod
    def _from_document(cls, document: dict[str, object]) -> VerifiedQuotaEntry:
        serialized = json.dumps(
            copy.deepcopy(document),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "_serialized", serialized)
        object.__setattr__(instance, "quota_entry_sha256", document["quota_entry_sha256"])
        object.__setattr__(instance, "_seal", _QUOTA_ENTRY_SEAL)
        return instance

    def to_dict(self) -> dict[str, object]:
        document = json.loads(self._serialized)
        if type(document) is not dict:  # pragma: no cover - constructor invariant
            raise RuntimeError("verified quota entry payload is not an object")
        return document


def registered_cell_requirements() -> tuple[CellRequirement, ...]:
    """Return the immutable, code-registered 37-cell release layout."""

    return REGISTERED_CELL_REQUIREMENTS


def _require_exact_keys(value: object, expected: Collection[str], name: str) -> dict[str, object]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise TypeError(f"{name} must be an exact JSON object with string keys")
    expected_keys = set(expected)
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unknown = sorted(actual_keys - expected_keys)
        raise ValueError(f"{name} schema fields differ: missing={missing}, unknown={unknown}")
    return value


def _require_list(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact JSON array")
    return value


def _require_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{name} contains a Unicode surrogate code point")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise TypeError(f"{name} must be a positive integer")
    return value


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_fraction_pair(
    numerator: object,
    denominator: object,
    *,
    name: str,
    unit_interval: bool,
) -> tuple[int, int]:
    checked_numerator = _require_nonnegative_int(numerator, f"{name} numerator")
    checked_denominator = _require_positive_int(denominator, f"{name} denominator")
    if unit_interval and checked_numerator > checked_denominator:
        raise ValueError(f"{name} numerator must not exceed its denominator")
    return checked_numerator, checked_denominator


def _compare(
    left_numerator: int,
    left_denominator: int,
    right_numerator: int,
    right_denominator: int,
) -> int:
    difference = left_numerator * right_denominator - right_numerator * left_denominator
    return (difference > 0) - (difference < 0)


def host_fill_band(numerator: object, denominator: object) -> str:
    """Classify host fill at the exact Section 8.2 boundaries."""

    n, d = _require_fraction_pair(numerator, denominator, name="host fill", unit_interval=True)
    if _compare(n, d, 9, 20) < 0:
        return "low"
    if _compare(n, d, 3, 5) < 0:
        return "medium"
    if _compare(n, d, 18, 25) < 0:
        return "high"
    return "extreme"


def logical_degree_band(numerator: object, denominator: object) -> str:
    """Classify average logical degree at exact rational boundaries."""

    n, d = _require_fraction_pair(
        numerator, denominator, name="logical degree", unit_interval=False
    )
    if _compare(n, d, 3, 1) < 0:
        return "sparse"
    if _compare(n, d, 9, 2) < 0:
        return "moderate"
    if _compare(n, d, 6, 1) < 0:
        return "dense"
    return "very_dense"


def residual_lcc_band(numerator: object, denominator: object) -> str:
    """Classify the residual largest-component node fraction exactly."""

    n, d = _require_fraction_pair(numerator, denominator, name="residual LCC", unit_interval=True)
    if _compare(n, d, 1, 2) >= 0:
        return "open"
    if _compare(n, d, 1, 4) >= 0:
        return "constrained"
    return "critical"


def defect_band(
    qubit_numerator: object,
    qubit_denominator: object,
    coupler_numerator: object,
    coupler_denominator: object,
) -> str:
    """Classify realized qubit/coupler defects at exact Section 8.2 bounds."""

    qn, qd = _require_fraction_pair(
        qubit_numerator,
        qubit_denominator,
        name="qubit defect",
        unit_interval=True,
    )
    cn, cd = _require_fraction_pair(
        coupler_numerator,
        coupler_denominator,
        name="coupler defect",
        unit_interval=True,
    )
    if qn == 0 and cn == 0:
        return "none"
    if _compare(qn, qd, 1, 50) <= 0 and _compare(cn, cd, 1, 20) <= 0:
        return "light"
    return "heavy"


def _reference_candidate(document: dict[str, object]) -> dict[str, object]:
    reference_index = document["reference_candidate_index"]
    matches = [
        candidate
        for candidate in document["candidates"]
        if candidate["canonical_index"] == reference_index
    ]
    if len(matches) != 1:  # pragma: no cover - VerifiedHardness invariant
        raise RuntimeError("hardness reference candidate is not uniquely represented")
    return matches[0]


def _classify_hardness_document(document: dict[str, object]) -> dict[str, str]:
    host = document["host"]
    logical = document["logical"]
    embedding = document["starting_embedding"]
    residual = _reference_candidate(document)["residual"]
    return {
        "host_fill": host_fill_band(
            embedding["host_fill_numerator"], embedding["host_fill_denominator"]
        ),
        "logical_degree": logical_degree_band(
            logical["average_degree_numerator"], logical["average_degree_denominator"]
        ),
        "residual_lcc": residual_lcc_band(
            residual["largest_free_component_node_numerator"],
            residual["largest_free_component_node_denominator"],
        ),
        "defects": defect_band(
            host["defect_qubit_numerator"],
            host["defect_qubit_denominator"],
            host["defect_coupler_numerator"],
            host["defect_coupler_denominator"],
        ),
    }


def classify_registered_bands(hardness: object) -> dict[str, str]:
    """Return exact registered bands from a sealed hardness computation."""

    return _classify_hardness_document(validate_hardness(hardness).to_dict())


def _require_verified_sources(
    *,
    hardness: object,
    problem: object,
    host: object,
    starting_embedding: object,
    partial_context: object,
) -> tuple[
    VerifiedHardness,
    VerifiedProblem,
    VerifiedHost,
    VerifiedStartingEmbedding,
    PartialContext,
    dict[str, object],
]:
    checked_hardness = validate_hardness(hardness)
    if type(problem) is not VerifiedProblem:
        raise TypeError("problem must be an exact VerifiedProblem value")
    if type(host) is not VerifiedHost:
        raise TypeError("host must be an exact VerifiedHost value")
    if type(starting_embedding) is not VerifiedStartingEmbedding:
        raise TypeError("starting_embedding must be an exact VerifiedStartingEmbedding value")
    if type(partial_context) is not PartialContext:
        raise TypeError("partial_context must be an exact PartialContext value")
    document = checked_hardness.to_dict()
    sources = document["sources"]
    expected = (
        problem.problem_sha256,
        host.host_artifact_sha256,
        host.host_sha256,
        starting_embedding.starting_embedding_sha256,
        partial_context.partial_context_sha256,
    )
    actual = (
        sources["problem_sha256"],
        sources["host_artifact_sha256"],
        sources["host_sha256"],
        sources["starting_embedding_sha256"],
        sources["partial_context_sha256"],
    )
    if actual != expected:
        raise ValueError("hardness sources do not match the authenticated source objects")
    if (
        starting_embedding.problem_sha256 != problem.problem_sha256
        or starting_embedding.host_sha256 != host.host_sha256
        or partial_context.problem_sha256 != problem.problem_sha256
        or partial_context.host_sha256 != host.host_sha256
        or partial_context.starting_embedding_sha256 != starting_embedding.starting_embedding_sha256
    ):
        raise ValueError("authenticated problem, host, embedding, and context are not aligned")
    return (
        checked_hardness,
        problem,
        host,
        starting_embedding,
        partial_context,
        document,
    )


def recompute_cell_facts(
    *,
    hardness: object,
    problem: object,
    host: object,
    starting_embedding: object,
    partial_context: object,
    plan_row: object,
    profile_set: object,
    ising_problem: object,
    ground_certificate: object | None,
    ground_proof: object | None,
    expected_plan_row_sha256: object,
    expected_ground_certificate_sha256: object | None,
    expected_ground_proof_artifact_sha256: object | None,
) -> VerifiedCellFacts:
    """Derive quota facts from sealed sources; no eligibility boolean is accepted."""

    (
        checked_hardness,
        checked_problem,
        checked_host,
        checked_starting,
        checked_context,
        hardness_document,
    ) = _require_verified_sources(
        hardness=hardness,
        problem=problem,
        host=host,
        starting_embedding=starting_embedding,
        partial_context=partial_context,
    )
    checked_plan = validate_plan_row(
        plan_row,
        expected_plan_row_sha256=_require_sha256(
            expected_plan_row_sha256,
            "expected_plan_row_sha256",
        ),
    )
    checked_profiles = validate_registered_profile_set(profile_set)
    plan = checked_plan.to_dict()
    if plan["profile_set_sha256"] != checked_profiles.profile_set_sha256:
        raise ValueError("plan row does not belong to the verified profile set")
    profile = checked_profiles.cell(plan["cell_id"])
    if hardness_document["profile_id"] != profile["profile_id"]:
        raise ValueError("hardness profile_id does not match the registered cell profile")
    if type(ising_problem) is not IsingProblem:
        raise TypeError("ising_problem must be an exact immutable IsingProblem")
    if ising_problem.problem_sha256 != checked_problem.problem_sha256:
        raise ValueError("Ising problem does not match the authenticated problem identity")
    if plan["topology"] != checked_host.topology or plan["size"] != checked_host.size:
        raise ValueError("plan topology/size do not match the authenticated host")
    if plan["n_vars"] != len(checked_problem.variables):
        raise ValueError("plan n_vars does not match the authenticated problem")
    if plan["starting_source"] != checked_starting.source:
        raise ValueError("plan starting_source does not match the authenticated embedding")
    if plan["l_cap"] != checked_context.l_cap:
        raise ValueError("plan l_cap does not match the authenticated partial context")
    if plan["q_cap_slack"] != checked_context.q_cap_slack:
        raise ValueError("plan q_cap_slack does not match the authenticated partial context")
    if len(checked_context.window_nodes) > plan["max_window_free"]:
        raise ValueError("authenticated window exceeds the plan's max_window_free")

    requested_qubit = checked_host.requested_qubit_fraction.as_integer_ratio()
    requested_coupler = checked_host.requested_coupler_fraction.as_integer_ratio()
    if requested_qubit != (
        plan["requested_qubit_numerator"],
        plan["requested_qubit_denominator"],
    ):
        raise ValueError("plan qubit-defect request does not match the authenticated host")
    if requested_coupler != (
        plan["requested_coupler_numerator"],
        plan["requested_coupler_denominator"],
    ):
        raise ValueError("plan coupler-defect request does not match the authenticated host")

    ground_inputs = (
        ground_certificate,
        ground_proof,
        expected_ground_certificate_sha256,
        expected_ground_proof_artifact_sha256,
    )
    if all(value is None for value in ground_inputs):
        ground_status = None
        ground_certificate_sha256 = None
        ground_proof_sha256 = None
        ground_eligible = False
    elif any(value is None for value in ground_inputs):
        raise ValueError("ground-state verification inputs must be supplied together")
    else:
        verified_ground = verify_ground_state_certificate(
            ising_problem,
            ground_certificate,
            ground_proof,
            expected_certificate_sha256=_require_sha256(
                expected_ground_certificate_sha256,
                "expected_ground_certificate_sha256",
            ),
            expected_proof_artifact_sha256=_require_sha256(
                expected_ground_proof_artifact_sha256,
                "expected_ground_proof_artifact_sha256",
            ),
        )
        ground_status = verified_ground.status
        ground_certificate_sha256 = verified_ground.certificate_sha256
        ground_proof_sha256 = verified_ground.proof_artifact_sha256
        ground_eligible = verified_ground.quality_eligible and ground_status in {
            "planted_proof",
            "exact_enumeration",
        }

    graph = checked_host.graph()
    components = [tuple(sorted(component)) for component in nx.connected_components(graph)]
    largest_count = max(map(len, components), default=0)
    host_count = len(checked_host.nodes)
    sources = hardness_document["sources"]
    document: dict[str, object] = {
        "schema": CELL_FACTS_SCHEMA,
        "schema_version": CELL_FACTS_SCHEMA_VERSION,
        "release_id": plan["release_id"],
        "profile_set_sha256": checked_profiles.profile_set_sha256,
        "plan_row_sha256": checked_plan.plan_row_sha256,
        "cell_id": plan["cell_id"],
        "partition": plan["partition"],
        "panel": plan["panel"],
        "ordinal": plan["ordinal"],
        "fill_subband": plan["fill_subband"],
        "hardness_sha256": checked_hardness.hardness_sha256,
        "hardness": hardness_document,
        "problem_sha256": checked_problem.problem_sha256,
        "host_artifact_sha256": checked_host.host_artifact_sha256,
        "host_sha256": checked_host.host_sha256,
        "starting_embedding_sha256": checked_starting.starting_embedding_sha256,
        "partial_context_sha256": checked_context.partial_context_sha256,
        "candidate_protocol_result_sha256": sources["candidate_protocol_result_sha256"],
        "topology": checked_host.topology,
        "size": checked_host.size,
        "logical_source": plan["logical_source"],
        "logical_family": plan["logical_family"],
        "chain_size": plan["chain_size"],
        "l_cap": checked_context.l_cap,
        "max_window_free": plan["max_window_free"],
        "q_cap_slack": checked_context.q_cap_slack,
        "requested_qubit_numerator": requested_qubit[0],
        "requested_qubit_denominator": requested_qubit[1],
        "requested_qubit_fraction": requested_qubit[0] / requested_qubit[1],
        "requested_coupler_numerator": requested_coupler[0],
        "requested_coupler_denominator": requested_coupler[1],
        "requested_coupler_fraction": requested_coupler[0] / requested_coupler[1],
        "host_largest_component_numerator": largest_count,
        "host_largest_component_denominator": host_count,
        "host_largest_component_fraction": largest_count / host_count,
        "ground_certificate_status": ground_status,
        "ground_certificate_sha256": ground_certificate_sha256,
        "ground_proof_artifact_sha256": ground_proof_sha256,
        "ground_state_eligible": ground_eligible,
    }
    return VerifiedCellFacts._from_document(document)


def validate_cell_facts(value: object) -> VerifiedCellFacts:
    """Accept only facts produced from authenticated profile, plan, and science objects."""

    if type(value) is not VerifiedCellFacts or value._seal is not _CELL_FACTS_SEAL:
        raise TypeError("facts must be an exact VerifiedCellFacts value")
    document = value.to_dict()
    _require_exact_keys(document, _CELL_FACT_FIELDS, "cell facts")
    if document["schema"] != CELL_FACTS_SCHEMA or document["schema_version"] != 1:
        raise ValueError("verified cell-facts schema identity is invalid")
    if canonical_sha256(document) != value.cell_facts_sha256:
        raise ValueError("verified cell-facts content digest is invalid")
    return value


def verify_cell_facts(
    stored: object,
    *,
    expected_cell_facts_sha256: str,
    recomputed: VerifiedCellFacts,
) -> VerifiedCellFacts:
    """Bind persisted facts to a fresh authenticated recomputation."""

    if type(stored) is not dict:
        raise TypeError("stored cell facts must be an exact JSON object")
    snapshot = copy.deepcopy(stored)
    expected = _require_sha256(expected_cell_facts_sha256, "expected_cell_facts_sha256")
    if canonical_sha256(snapshot) != expected:
        raise ValueError("stored cell-facts digest disagrees with its independent binding")
    checked = validate_cell_facts(recomputed)
    if snapshot != checked.to_dict() or expected != checked.cell_facts_sha256:
        raise ValueError("stored cell facts do not equal authenticated recomputed facts")
    return checked


def _validate_ratio_bound(value: object, name: str) -> tuple[int, int, bool] | None:
    if value is None:
        return None
    document = _require_exact_keys(value, _RATIO_BOUND_FIELDS, name)
    numerator, denominator = _require_fraction_pair(
        document["numerator"],
        document["denominator"],
        name=name,
        unit_interval=False,
    )
    return numerator, denominator, _require_bool(document["inclusive"], f"{name}.inclusive")


def _validate_integer_bound(value: object, name: str) -> tuple[int, bool] | None:
    if value is None:
        return None
    document = _require_exact_keys(value, _INTEGER_BOUND_FIELDS, name)
    integer = _require_nonnegative_int(document["value"], f"{name}.value")
    return integer, _require_bool(document["inclusive"], f"{name}.inclusive")


def _ensure_nonempty_interval(
    lower: tuple[int, int, bool] | None,
    upper: tuple[int, int, bool] | None,
    name: str,
) -> None:
    if lower is None and upper is None:
        raise ValueError(f"{name} requires at least one bound")
    if lower is not None and upper is not None:
        comparison = _compare(lower[0], lower[1], upper[0], upper[1])
        if comparison > 0 or (comparison == 0 and not (lower[2] and upper[2])):
            raise ValueError(f"{name} describes an empty interval")


def validate_cell_profile(value: object) -> dict[str, object]:
    """Validate a strict data-driven profile for one exact quota cell."""

    if type(value) is not dict:
        raise TypeError("cell profile must be an exact JSON object")
    profile = _require_exact_keys(copy.deepcopy(value), _CELL_PROFILE_FIELDS, "cell profile")
    if type(profile["schema"]) is not str or profile["schema"] != CELL_PROFILE_SCHEMA:
        raise ValueError(f"cell profile schema must equal {CELL_PROFILE_SCHEMA!r}")
    if (
        type(profile["schema_version"]) is not int
        or profile["schema_version"] != CELL_PROFILE_SCHEMA_VERSION
    ):
        raise ValueError(f"cell profile schema_version must equal {CELL_PROFILE_SCHEMA_VERSION}")
    _require_text(profile["profile_id"], "profile_id")
    _require_text(profile["cell_id"], "cell_id")
    partition = profile["partition"]
    if type(partition) is not str or partition not in _PARTITIONS:
        raise ValueError(f"partition must be one of {sorted(_PARTITIONS)}")
    _require_text(profile["panel"], "panel")
    _require_positive_int(profile["quota"], "quota")

    ratio_fields: set[str] = set()
    for index, raw_predicate in enumerate(
        _require_list(profile["ratio_predicates"], "ratio_predicates")
    ):
        predicate = _require_exact_keys(
            raw_predicate, _RATIO_PREDICATE_FIELDS, f"ratio_predicates[{index}]"
        )
        field = predicate["field"]
        if type(field) is not str or field not in _RATIO_FIELDS:
            raise ValueError(f"ratio_predicates[{index}] uses an unregistered ratio field")
        if field in ratio_fields:
            raise ValueError(f"duplicate ratio predicate for field {field!r}")
        ratio_fields.add(field)
        lower = _validate_ratio_bound(predicate["minimum"], f"ratio_predicates[{index}].minimum")
        upper = _validate_ratio_bound(predicate["maximum"], f"ratio_predicates[{index}].maximum")
        _ensure_nonempty_interval(lower, upper, f"ratio_predicates[{index}]")

    integer_fields: set[str] = set()
    for index, raw_predicate in enumerate(
        _require_list(profile["integer_predicates"], "integer_predicates")
    ):
        predicate = _require_exact_keys(
            raw_predicate, _INTEGER_PREDICATE_FIELDS, f"integer_predicates[{index}]"
        )
        field = predicate["field"]
        if type(field) is not str or field not in _INTEGER_FIELDS:
            raise ValueError(f"integer_predicates[{index}] uses an unregistered integer field")
        if field in integer_fields:
            raise ValueError(f"duplicate integer predicate for field {field!r}")
        integer_fields.add(field)
        lower = _validate_integer_bound(
            predicate["minimum"], f"integer_predicates[{index}].minimum"
        )
        upper = _validate_integer_bound(
            predicate["maximum"], f"integer_predicates[{index}].maximum"
        )
        if lower is None and upper is None:
            raise ValueError(f"integer_predicates[{index}] requires at least one bound")
        if (
            lower is not None
            and upper is not None
            and (lower[0] > upper[0] or (lower[0] == upper[0] and not (lower[1] and upper[1])))
        ):
            raise ValueError(f"integer_predicates[{index}] describes an empty interval")

    value_fields: set[str] = set()
    for index, raw_predicate in enumerate(
        _require_list(profile["value_predicates"], "value_predicates")
    ):
        predicate = _require_exact_keys(
            raw_predicate, _VALUE_PREDICATE_FIELDS, f"value_predicates[{index}]"
        )
        field = predicate["field"]
        if type(field) is not str or field not in _VALUE_FIELD_TYPES:
            raise ValueError(f"value_predicates[{index}] uses an unregistered value field")
        if field in value_fields:
            raise ValueError(f"duplicate value predicate for field {field!r}")
        value_fields.add(field)
        allowed = _require_list(
            predicate["allowed_values"], f"value_predicates[{index}].allowed_values"
        )
        if not allowed:
            raise ValueError("allowed_values must not be empty")
        expected_type = _VALUE_FIELD_TYPES[field]
        if any(type(item) is not expected_type for item in allowed):
            raise TypeError(
                f"allowed values for {field!r} must have exact type {expected_type.__name__}"
            )
        if expected_type is str:
            for item in allowed:
                _require_text(item, f"allowed value for {field!r}")
        if len({item for item in allowed}) != len(allowed):
            raise ValueError(f"allowed values for {field!r} must be duplicate-free")

    band_axes: set[str] = set()
    for index, raw_predicate in enumerate(
        _require_list(profile["band_predicates"], "band_predicates")
    ):
        predicate = _require_exact_keys(
            raw_predicate, _BAND_PREDICATE_FIELDS, f"band_predicates[{index}]"
        )
        axis = predicate["axis"]
        if type(axis) is not str or axis not in _REGISTERED_BANDS:
            raise ValueError(f"band_predicates[{index}] uses an unregistered band axis")
        if axis in band_axes:
            raise ValueError(f"duplicate band predicate for axis {axis!r}")
        band_axes.add(axis)
        bands = _require_list(predicate["allowed_bands"], f"band_predicates[{index}].allowed_bands")
        if not bands:
            raise ValueError("allowed_bands must not be empty")
        if any(type(band) is not str or band not in _REGISTERED_BANDS[axis] for band in bands):
            raise ValueError(f"band_predicates[{index}] contains an unregistered band")
        if len(set(bands)) != len(bands):
            raise ValueError("allowed_bands must be duplicate-free")
    return profile


def verify_registered_profile_set(
    value: object,
    *,
    expected_profile_set_sha256: str,
) -> VerifiedProfileSet:
    """Authenticate a complete profile set and its exact release-cell registry.

    The expected digest is deliberately independent of the stored document.  Callers must
    obtain it from the frozen release manifest, never from a field inside ``value``.
    """

    if type(value) is not dict:
        raise TypeError("profile set must be an exact JSON object")
    snapshot = copy.deepcopy(value)
    document = _require_exact_keys(snapshot, _PROFILE_SET_FIELDS, "profile set")
    if type(document["schema"]) is not str or document["schema"] != PROFILE_SET_SCHEMA:
        raise ValueError(f"profile set schema must equal {PROFILE_SET_SCHEMA!r}")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != PROFILE_SET_SCHEMA_VERSION
    ):
        raise ValueError(f"profile set schema_version must equal {PROFILE_SET_SCHEMA_VERSION}")
    if type(document["release_id"]) is not str or document["release_id"] != HARD_OOD_RELEASE_ID:
        raise ValueError(f"profile set release_id must equal {HARD_OOD_RELEASE_ID!r}")
    _require_sha256(document["generator_source_sha256"], "generator_source_sha256")
    if (
        type(document["maximum_ordinal"]) is not int
        or document["maximum_ordinal"] != REGISTERED_MAXIMUM_ORDINAL
    ):
        raise ValueError(f"profile set maximum_ordinal must equal {REGISTERED_MAXIMUM_ORDINAL}")
    expected_digest = _require_sha256(expected_profile_set_sha256, "expected_profile_set_sha256")
    actual_digest = canonical_sha256(document)
    if actual_digest != expected_digest:
        raise ValueError("profile set digest disagrees with its independent binding")

    raw_cells = _require_list(document["cells"], "profile set cells")
    if not raw_cells:
        raise ValueError("profile set cells must not be empty")
    checked_cells = [validate_cell_profile(cell) for cell in raw_cells]
    cells_by_id: dict[str, dict[str, object]] = {}
    for cell in checked_cells:
        cell_id = cell["cell_id"]
        if cell_id in cells_by_id:
            raise ValueError(f"duplicate cell profile {cell_id!r}")
        cells_by_id[cell_id] = cell
        predicate_count = sum(
            len(cell[field])
            for field in (
                "ratio_predicates",
                "integer_predicates",
                "value_predicates",
                "band_predicates",
            )
        )
        if predicate_count == 0:
            raise ValueError(f"cell profile {cell_id!r} has no acceptance predicate")

    expected_cells = set(_REQUIREMENT_BY_CELL)
    actual_cells = set(cells_by_id)
    if actual_cells != expected_cells:
        missing = sorted(expected_cells - actual_cells)
        unknown = sorted(actual_cells - expected_cells)
        raise ValueError(f"profile set cell registry differs: missing={missing}, unknown={unknown}")
    for cell_id, requirement in _REQUIREMENT_BY_CELL.items():
        cell = cells_by_id[cell_id]
        actual = (cell["partition"], cell["panel"], cell["quota"])
        expected = (requirement.partition, requirement.panel, requirement.quota)
        if actual != expected:
            raise ValueError(
                f"cell {cell_id!r} registration differs: actual={actual}, expected={expected}"
            )
    return VerifiedProfileSet._from_document(document)


def validate_registered_profile_set(value: object) -> VerifiedProfileSet:
    """Accept only a sealed profile set created by independent digest verification."""

    if type(value) is not VerifiedProfileSet or value._seal is not _PROFILE_SET_SEAL:
        raise TypeError("profile_set must be an exact VerifiedProfileSet value")
    if canonical_sha256(value.to_dict()) != value.profile_set_sha256:
        raise ValueError("verified profile-set content digest is invalid")
    return value


def verify_plan_row(
    value: object,
    *,
    expected_plan_row_sha256: str,
    profile_set: VerifiedProfileSet,
    seed_resolver: VerifiedSeedResolver,
    expected_seed_registry_terminal_root_sha256: str,
) -> VerifiedPlanRow:
    """Verify a plan row against independently authenticated profiles and seed registry."""

    checked_profiles = validate_registered_profile_set(profile_set)
    checked_seed_resolver = validate_seed_resolver(
        seed_resolver,
        expected_terminal_root_sha256=expected_seed_registry_terminal_root_sha256,
    )
    if type(value) is not dict:
        raise TypeError("plan row must be an exact JSON object")
    document = _require_exact_keys(copy.deepcopy(value), _PLAN_ROW_FIELDS, "plan row")
    if document["schema"] != PLAN_ROW_SCHEMA:
        raise ValueError(f"plan row schema must equal {PLAN_ROW_SCHEMA!r}")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("plan row schema_version must equal 1")
    if document["release_id"] != HARD_OOD_RELEASE_ID:
        raise ValueError(f"plan row release_id must equal {HARD_OOD_RELEASE_ID!r}")
    if document["profile_set_sha256"] != checked_profiles.profile_set_sha256:
        raise ValueError("plan row profile_set_sha256 does not match the verified profile set")
    cell_id = _require_text(document["cell_id"], "cell_id")
    if cell_id not in _REQUIREMENT_BY_CELL:
        raise ValueError(f"plan row references unregistered cell {cell_id!r}")
    requirement = _REQUIREMENT_BY_CELL[cell_id]
    if (document["partition"], document["panel"]) != (
        requirement.partition,
        requirement.panel,
    ):
        raise ValueError("plan row partition/panel disagree with the registered cell")
    ordinal = _require_nonnegative_int(document["ordinal"], "ordinal")
    if ordinal >= REGISTERED_MAXIMUM_ORDINAL:
        raise ValueError("plan row ordinal is outside the registered finite prefix")
    topology = _require_text(document["topology"], "topology")
    for field_name in ("size", "n_vars", "chain_size", "l_cap", "max_window_free"):
        _require_positive_int(document[field_name], field_name)
    _require_text(document["logical_source"], "logical_source")
    logical_family = _require_text(document["logical_family"], "logical_family")
    if requirement.topology is not None and topology != requirement.topology:
        raise ValueError("plan row topology disagrees with the registered cell")
    if requirement.size is not None and document["size"] != requirement.size:
        raise ValueError("plan row size disagrees with the registered cell")
    if requirement.logical_family is not None and logical_family != requirement.logical_family:
        raise ValueError("plan row logical_family disagrees with the registered cell")
    q_cap_slack = _require_nonnegative_int(document["q_cap_slack"], "q_cap_slack")
    if q_cap_slack not in {0, 1}:
        raise ValueError("q_cap_slack must be one of the registered values 0 or 1")
    _require_fraction_pair(
        document["requested_qubit_numerator"],
        document["requested_qubit_denominator"],
        name="requested qubit fraction",
        unit_interval=True,
    )
    _require_fraction_pair(
        document["requested_coupler_numerator"],
        document["requested_coupler_denominator"],
        name="requested coupler fraction",
        unit_interval=True,
    )
    fill_subband = document["fill_subband"]
    if requirement.partition in {"hard_dev", "locked_ood"}:
        if type(fill_subband) is not int or fill_subband not in range(5):
            raise ValueError("non-mechanism plan rows require fill_subband in {0,1,2,3,4}")
    elif fill_subband is not None:
        raise ValueError("mechanism plan rows require null fill_subband")
    starting_source = _require_text(document["starting_source"], "starting_source")
    if starting_source not in _STARTING_SOURCES:
        raise ValueError(f"starting_source must be one of {sorted(_STARTING_SOURCES)}")
    _require_sha256(document["generator_arguments_sha256"], "generator_arguments_sha256")
    problem_seed_request = SeedRequest.from_dict(document["problem_seed_request"])
    expected_problem_seed_request = SeedRequest(
        release_id=HARD_OOD_RELEASE_ID,
        purpose="problem",
        partition=requirement.partition,
        panel=requirement.panel,
        cell=cell_id,
        task_type=None,
        problem_sha256=None,
        state_sha256=None,
        candidate_index=None,
        strength_index=None,
        label_stage=None,
        replicate=ordinal,
    )
    if (
        problem_seed_request.canonical_preimage()
        != expected_problem_seed_request.canonical_preimage()
    ):
        raise ValueError("problem_seed_request disagrees with the registered plan identity")
    registry_entry = checked_seed_resolver.resolve(problem_seed_request)
    if registry_entry.seed32_required:
        raise ValueError("problem seed requests must use native PCG words, not seed32")
    problem_seed_key_sha256 = _require_sha256(
        document["problem_seed_key_sha256"],
        "problem_seed_key_sha256",
    )
    if (
        problem_seed_key_sha256 != problem_seed_request.seed_key_hex
        or problem_seed_key_sha256 != registry_entry.seed_key_sha256
    ):
        raise ValueError("problem_seed_key_sha256 disagrees with the verified seed registry")
    state_slot = _require_nonnegative_int(document["state_slot"], "state_slot")
    focus_quartile = _require_nonnegative_int(document["focus_quartile"], "focus_quartile")
    if state_slot not in range(4):
        raise ValueError("state_slot must be one of the four registered slots")
    if focus_quartile not in range(4):
        raise ValueError("focus_quartile must be one of the four registered quartiles")
    expected_digest = _require_sha256(expected_plan_row_sha256, "expected_plan_row_sha256")
    stored_digest = _require_sha256(document["plan_row_sha256"], "plan_row_sha256")
    payload = {key: item for key, item in document.items() if key != "plan_row_sha256"}
    computed_digest = canonical_sha256(payload)
    if stored_digest != computed_digest or stored_digest != expected_digest:
        raise ValueError("plan row digest disagrees with its payload or independent binding")
    return VerifiedPlanRow._from_document(document)


def validate_plan_row(
    value: object,
    *,
    expected_plan_row_sha256: str,
) -> VerifiedPlanRow:
    """Replay a sealed plan row against an independently supplied digest."""

    if type(value) is not VerifiedPlanRow or value._seal is not _PLAN_ROW_SEAL:
        raise TypeError("plan_row must be an exact VerifiedPlanRow value")
    document = value.to_dict()
    _require_exact_keys(document, _PLAN_ROW_FIELDS, "verified plan row")
    payload = {key: item for key, item in document.items() if key != "plan_row_sha256"}
    external = _require_sha256(expected_plan_row_sha256, "expected_plan_row_sha256")
    stored = _require_sha256(document["plan_row_sha256"], "plan_row_sha256")
    if canonical_sha256(payload) != stored or stored != value.plan_row_sha256 or stored != external:
        raise ValueError("verified plan-row content digest is invalid")
    return value


def _hardness_container_and_name(
    facts: dict[str, object], field: str
) -> tuple[dict[str, object], str]:
    parts = field.split(".")
    if parts[0] != "hardness" or len(parts) < 3:
        raise ValueError(f"invalid registered hardness field {field!r}")
    hardness = facts["hardness"]
    if parts[1] == "reference_candidate":
        container: object = _reference_candidate(hardness)
        remaining = parts[2:]
    else:
        container = hardness
        remaining = parts[1:]
    for name in remaining[:-1]:
        if type(container) is not dict or name not in container:
            raise ValueError(f"registered hardness field {field!r} is not present")
        container = container[name]
    if type(container) is not dict:
        raise ValueError(f"registered hardness field {field!r} has no object container")
    return container, remaining[-1]


def _ratio_from_facts(facts: dict[str, object], field: str) -> tuple[int | None, int]:
    if field.startswith("hardness."):
        section, prefix = _hardness_container_and_name(facts, field)
        return section[f"{prefix}_numerator"], section[f"{prefix}_denominator"]
    prefix = field.removesuffix("_fraction")
    return facts[f"{prefix}_numerator"], facts[f"{prefix}_denominator"]


def _integer_from_facts(facts: dict[str, object], field: str) -> int | None:
    if field.startswith("hardness."):
        section, value_name = _hardness_container_and_name(facts, field)
        return section[value_name]
    return facts[field]


def _value_from_facts(facts: dict[str, object], field: str) -> object:
    if field in _INTEGER_FIELDS:
        return _integer_from_facts(facts, field)
    return facts[field]


def _ratio_satisfies(
    ratio: tuple[int | None, int],
    minimum: object,
    maximum: object,
) -> bool:
    numerator, denominator = ratio
    if numerator is None:
        return False
    lower = _validate_ratio_bound(minimum, "minimum")
    upper = _validate_ratio_bound(maximum, "maximum")
    if lower is not None:
        comparison = _compare(numerator, denominator, lower[0], lower[1])
        if comparison < 0 or (comparison == 0 and not lower[2]):
            return False
    if upper is not None:
        comparison = _compare(numerator, denominator, upper[0], upper[1])
        if comparison > 0 or (comparison == 0 and not upper[2]):
            return False
    return True


def _integer_satisfies(value: int | None, minimum: object, maximum: object) -> bool:
    if value is None:
        return False
    lower = _validate_integer_bound(minimum, "minimum")
    upper = _validate_integer_bound(maximum, "maximum")
    if lower is not None and (value < lower[0] or (value == lower[0] and not lower[1])):
        return False
    return not (upper is not None and (value > upper[0] or (value == upper[0] and not upper[1])))


def cell_predicate_failures(profile: object, facts: object) -> tuple[str, ...]:
    """Return deterministic machine-readable failures for one profile/fact pair."""

    checked_profile = validate_cell_profile(profile)
    checked_facts = validate_cell_facts(facts).to_dict()
    failures: list[str] = []
    if checked_facts["hardness"]["profile_id"] != checked_profile["profile_id"]:
        failures.append("profile_id")
    for predicate in checked_profile["ratio_predicates"]:
        if not _ratio_satisfies(
            _ratio_from_facts(checked_facts, predicate["field"]),
            predicate["minimum"],
            predicate["maximum"],
        ):
            failures.append(f"ratio:{predicate['field']}")
    for predicate in checked_profile["integer_predicates"]:
        if not _integer_satisfies(
            _integer_from_facts(checked_facts, predicate["field"]),
            predicate["minimum"],
            predicate["maximum"],
        ):
            failures.append(f"integer:{predicate['field']}")
    for predicate in checked_profile["value_predicates"]:
        if _value_from_facts(checked_facts, predicate["field"]) not in predicate["allowed_values"]:
            failures.append(f"value:{predicate['field']}")
    bands = _classify_hardness_document(checked_facts["hardness"])
    for predicate in checked_profile["band_predicates"]:
        if bands[predicate["axis"]] not in predicate["allowed_bands"]:
            failures.append(f"band:{predicate['axis']}")
    return tuple(failures)


def cell_matches(profile: object, facts: object) -> bool:
    """Return whether one validated fact set satisfies every exact cell predicate."""

    return not cell_predicate_failures(profile, facts)


def build_quota_entry(
    *,
    facts: object,
    profile_set: object,
    expected_problem_iso_sha256: object,
    expected_state_sha256: object,
    mechanism_parent_id: object | None = None,
    expected_mechanism_iso_sha256: object | None = None,
    expected_mechanism_certificate_sha256: object | None = None,
    expected_mechanism_ablation_sha256: object | None = None,
) -> VerifiedQuotaEntry:
    """Bind one accepted group to independently computed identity-layer digests."""

    checked_facts = validate_cell_facts(facts)
    checked_profiles = validate_registered_profile_set(profile_set)
    fact_document = checked_facts.to_dict()
    if fact_document["profile_set_sha256"] != checked_profiles.profile_set_sha256:
        raise ValueError("cell facts do not belong to the verified profile set")
    cell_id = fact_document["cell_id"]
    profile = checked_profiles.cell(cell_id)
    if not cell_matches(profile, checked_facts):
        raise ValueError(f"cell facts do not satisfy registered cell {cell_id!r}")
    problem_sha256 = _require_sha256(fact_document["problem_sha256"], "problem_sha256")
    problem_iso_sha256 = _require_sha256(expected_problem_iso_sha256, "expected_problem_iso_sha256")
    state_sha256 = _require_sha256(expected_state_sha256, "expected_state_sha256")
    partition = fact_document["partition"]
    is_mechanism = partition in {"mechanism_dev", "mechanism_locked", "composed_locked"}
    if is_mechanism:
        parent = _require_text(mechanism_parent_id, "mechanism_parent_id")
        mechanism_iso = _require_sha256(
            expected_mechanism_iso_sha256, "expected_mechanism_iso_sha256"
        )
        mechanism_certificate = _require_sha256(
            expected_mechanism_certificate_sha256,
            "expected_mechanism_certificate_sha256",
        )
        if partition == "composed_locked":
            mechanism_ablation = _require_sha256(
                expected_mechanism_ablation_sha256,
                "expected_mechanism_ablation_sha256",
            )
        elif expected_mechanism_ablation_sha256 is not None:
            raise ValueError("single-mechanism entries cannot carry a composed ablation digest")
        else:
            mechanism_ablation = None
    else:
        if any(
            value is not None
            for value in (
                mechanism_parent_id,
                expected_mechanism_iso_sha256,
                expected_mechanism_certificate_sha256,
                expected_mechanism_ablation_sha256,
            )
        ):
            raise ValueError("non-mechanism entries cannot carry mechanism identities")
        parent = mechanism_iso = mechanism_certificate = mechanism_ablation = None
    payload: dict[str, object] = {
        "schema": QUOTA_ENTRY_SCHEMA,
        "schema_version": QUOTA_ENTRY_SCHEMA_VERSION,
        "release_id": fact_document["release_id"],
        "profile_set_sha256": checked_profiles.profile_set_sha256,
        "plan_row_sha256": fact_document["plan_row_sha256"],
        "cell_id": cell_id,
        "partition": partition,
        "panel": fact_document["panel"],
        "ordinal": fact_document["ordinal"],
        "problem_group_id": f"problem-sha256:{problem_sha256}",
        "problem_sha256": problem_sha256,
        "problem_iso_sha256": problem_iso_sha256,
        "state_sha256": state_sha256,
        "host_sha256": fact_document["host_sha256"],
        "mechanism_parent_id": parent,
        "mechanism_iso_sha256": mechanism_iso,
        "mechanism_certificate_sha256": mechanism_certificate,
        "mechanism_ablation_sha256": mechanism_ablation,
        "cell_facts_sha256": checked_facts.cell_facts_sha256,
    }
    document = {**payload, "quota_entry_sha256": canonical_sha256(payload)}
    return VerifiedQuotaEntry._from_document(document)


def validate_quota_entry(value: object) -> VerifiedQuotaEntry:
    """Accept only a sealed quota entry built from verified cell facts."""

    if type(value) is not VerifiedQuotaEntry or value._seal is not _QUOTA_ENTRY_SEAL:
        raise TypeError("quota entry must be an exact VerifiedQuotaEntry value")
    document = value.to_dict()
    _require_exact_keys(document, _QUOTA_ENTRY_FIELDS, "quota entry")
    payload = {key: item for key, item in document.items() if key != "quota_entry_sha256"}
    if canonical_sha256(payload) != value.quota_entry_sha256:
        raise ValueError("verified quota-entry content digest is invalid")
    return value


def verify_quota_entry(
    stored: object,
    *,
    expected_quota_entry_sha256: str,
    recomputed: VerifiedQuotaEntry,
) -> VerifiedQuotaEntry:
    """Bind a persisted quota row to a fresh source-derived entry."""

    if type(stored) is not dict:
        raise TypeError("stored quota entry must be an exact JSON object")
    snapshot = copy.deepcopy(stored)
    expected = _require_sha256(expected_quota_entry_sha256, "expected_quota_entry_sha256")
    payload = {key: item for key, item in snapshot.items() if key != "quota_entry_sha256"}
    if canonical_sha256(payload) != expected or snapshot.get("quota_entry_sha256") != expected:
        raise ValueError("stored quota-entry digest disagrees with its independent binding")
    checked = validate_quota_entry(recomputed)
    if snapshot != checked.to_dict() or expected != checked.quota_entry_sha256:
        raise ValueError("stored quota entry does not equal authenticated recomputed entry")
    return checked


def _validated_profiles(
    profiles: object,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    profile_set = validate_registered_profile_set(profiles)
    raw_profiles = profile_set.to_dict()["cells"]
    checked = [validate_cell_profile(profile) for profile in raw_profiles]
    by_cell: dict[str, dict[str, object]] = {}
    for profile in checked:
        cell_id = profile["cell_id"]
        if cell_id in by_cell:
            raise ValueError(f"duplicate cell profile {cell_id!r}")
        by_cell[cell_id] = profile
    return checked, by_cell


def expected_quota_total(profiles: object) -> int:
    """Return the registered total target quota from a verified profile set."""

    checked, _ = _validated_profiles(profiles)
    return sum(profile["quota"] for profile in checked)


def _quota_counts(entries: object, profiles: object) -> tuple[dict[str, int], dict[str, int]]:
    checked_profiles, by_cell = _validated_profiles(profiles)
    checked_profile_set = validate_registered_profile_set(profiles)
    counts = {profile["cell_id"]: 0 for profile in checked_profiles}
    quotas = {profile["cell_id"]: profile["quota"] for profile in checked_profiles}
    if type(entries) is not tuple:
        raise TypeError("quota entries must be an immutable tuple of VerifiedQuotaEntry values")
    seen: dict[str, dict[str, str]] = {
        "plan_row_sha256": {},
        "problem_group_id": {},
        "problem_sha256": {},
        "problem_iso_sha256": {},
        "state_sha256": {},
        "mechanism_parent_id": {},
        "mechanism_iso_sha256": {},
    }
    seen_fault_hosts: dict[str, str] = {}
    for raw_entry in entries:
        entry = validate_quota_entry(raw_entry).to_dict()
        if entry["release_id"] != HARD_OOD_RELEASE_ID:
            raise ValueError("quota entry release_id is not the registered release")
        if entry["profile_set_sha256"] != checked_profile_set.profile_set_sha256:
            raise ValueError("quota entry does not belong to the verified profile set")
        cell_id = entry["cell_id"]
        if cell_id not in by_cell:
            raise ValueError(f"quota entry references unknown quota cell {cell_id!r}")
        requirement = _REQUIREMENT_BY_CELL[cell_id]
        if (entry["partition"], entry["panel"]) != (
            requirement.partition,
            requirement.panel,
        ):
            raise ValueError("quota entry partition/panel disagree with its cell")
        expected_group_id = f"problem-sha256:{entry['problem_sha256']}"
        if entry["problem_group_id"] != expected_group_id:
            raise ValueError("quota entry problem_group_id is not derived from problem_sha256")
        for identity_name in (
            "problem_group_id",
            "problem_sha256",
            "problem_iso_sha256",
            "state_sha256",
            "plan_row_sha256",
        ):
            identity = entry[identity_name]
            if identity in seen[identity_name]:
                raise ValueError(f"{identity_name} occurs in multiple quota entries: {identity!r}")
            seen[identity_name][identity] = cell_id
        for identity_name in ("mechanism_parent_id", "mechanism_iso_sha256"):
            identity = entry[identity_name]
            if identity is None:
                continue
            if identity in seen[identity_name]:
                raise ValueError(f"{identity_name} occurs in multiple quota entries: {identity!r}")
            seen[identity_name][identity] = cell_id
        if entry["panel"] == "host_faults":
            host_sha256 = entry["host_sha256"]
            if host_sha256 in seen_fault_hosts:
                raise ValueError("fault-OOD host_sha256 mask is shared by multiple groups")
            seen_fault_hosts[host_sha256] = cell_id
        counts[cell_id] += 1
        if counts[cell_id] > quotas[cell_id]:
            raise ValueError(f"quota cell {cell_id!r} is over quota")
    return counts, quotas


def quota_deficits(entries: object, profiles: object) -> dict[str, int]:
    """Return nonnegative remaining quotas after validating every supplied entry."""

    counts, quotas = _quota_counts(entries, profiles)
    return {cell_id: quotas[cell_id] - counts[cell_id] for cell_id in sorted(quotas)}


def validate_exact_quotas(entries: object, profiles: object) -> dict[str, int]:
    """Validate exact per-cell counts and distinct group/isomorphism union cardinality."""

    counts, quotas = _quota_counts(entries, profiles)
    mismatches = {
        cell_id: {"actual": counts[cell_id], "expected": quotas[cell_id]}
        for cell_id in sorted(quotas)
        if counts[cell_id] != quotas[cell_id]
    }
    if mismatches:
        raise ValueError(f"quota mismatch: {mismatches}")
    return {cell_id: counts[cell_id] for cell_id in sorted(counts)}


__all__ = [
    "CELL_FACTS_SCHEMA",
    "CELL_FACTS_SCHEMA_VERSION",
    "CELL_PROFILE_SCHEMA",
    "CELL_PROFILE_SCHEMA_VERSION",
    "HARD_OOD_RELEASE_ID",
    "PLAN_ROW_SCHEMA",
    "PLAN_ROW_SCHEMA_VERSION",
    "PROFILE_SET_SCHEMA",
    "PROFILE_SET_SCHEMA_VERSION",
    "QUOTA_ENTRY_SCHEMA",
    "QUOTA_ENTRY_SCHEMA_VERSION",
    "CellRequirement",
    "VerifiedCellFacts",
    "VerifiedPlanRow",
    "VerifiedProfileSet",
    "VerifiedQuotaEntry",
    "build_quota_entry",
    "cell_matches",
    "cell_predicate_failures",
    "classify_registered_bands",
    "defect_band",
    "expected_quota_total",
    "host_fill_band",
    "logical_degree_band",
    "quota_deficits",
    "recompute_cell_facts",
    "registered_cell_requirements",
    "residual_lcc_band",
    "validate_cell_facts",
    "validate_cell_profile",
    "validate_exact_quotas",
    "validate_plan_row",
    "validate_quota_entry",
    "validate_registered_profile_set",
    "verify_cell_facts",
    "verify_plan_row",
    "verify_quota_entry",
    "verify_registered_profile_set",
]
