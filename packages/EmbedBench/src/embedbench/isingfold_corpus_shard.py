"""Deterministic, independently sharded CandidateBank source generation.

This module keeps proposal, measurement, and authority roles separate.  Logical
problems and realized hosts are fixed before labels exist; Minorminer proposes
embeddings only; and simulated annealing is evaluated against an internally derived,
exact-dyadic enumeration certificate.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import math
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import minorminer
import networkx as nx

from embedbench.apps import FAMILIES, graphcut_mrf, jobshop, portfolio
from embedbench.candidate_bank import (
    CandidateGroup,
    CandidateRecord,
    EvaluationCurve,
    InstanceRecord,
    RepairAttempt,
    assign_split,
    canonical_json_bytes,
    content_digest,
    derive_attempt_id,
    derive_candidate_id,
    derive_generation_digest,
    derive_group_id,
    derive_split_unit_id,
    stable_seed,
    validate_group_against_instance,
)
from embedbench.embedding import Embedding, LogicalProblem
from embedbench.ground_certificate import (
    Dyadic,
    ExhaustiveProof,
    IsingProblem,
    SolverRun,
    VerifiedGroundState,
    build_exact_enumeration_certificate,
    verify_ground_state_certificate,
)
from embedbench.inkdrop import MODES, ink_drop
from embedbench.isingfold_design import LineageFact, TaskFact
from embedbench.realized_host import (
    build_realized_host_artifact,
    load_realized_host,
    validate_minor_embedding,
)
from embedbench.surrogate import BETA_RANGE, solve_probability_at

Origin = Literal["application-derived", "synthetic-ink-drop"]
Partition = Literal["train", "val", "test"]
Regime = Literal["iid", "ood"]
_TOPOLOGIES = frozenset({"chimera", "pegasus", "zephyr"})
_APPLICATION_FAMILIES = frozenset(FAMILIES)
_HEX = frozenset("0123456789abcdef")
_EXACT_AUTHORITY_SCHEMA = "embedbench.isingfold-exact-reference-authority"
_EXACT_AUTHORITY_SCHEMA_VERSION = 2
_EXACT_REPLAY_ALGORITHM = "binary-reflected-gray-code-exact-dyadic-v1"
_EXACT_REPLAY_ENTRY_POINT = "embedbench.ground_certificate.verify_ground_state_certificate"
_SOURCE_MODULES = (
    "apps",
    "candidate_bank",
    "embedding",
    "ground_certificate",
    "hard_ood_schema",
    "inkdrop",
    "isingfold_corpus_shard",
    "isingfold_design",
    "programming",
    "realized_host",
    "structural",
    "surrogate",
)
_DEPENDENCIES = (
    "dimod",
    "dwave-networkx",
    "dwave-samplers",
    "embedbench",
    "minorminer",
    "networkx",
    "numpy",
    "scipy",
)
_THREAD_CONTROLS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be non-empty text")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{name} contains a Unicode surrogate")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact_mapping(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an exact object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{name} schema fields differ: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return dict(value)


def _generation_provenance() -> dict[str, object]:
    source_modules: dict[str, str] = {}
    for name in _SOURCE_MODULES:
        module = importlib.import_module(f"embedbench.{name}")
        path = getattr(module, "__file__", None)
        if type(path) is not str:
            raise RuntimeError(f"source module {name!r} has no filesystem identity")
        source_modules[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {
        "dependency_versions": {name: importlib.metadata.version(name) for name in _DEPENDENCIES},
        "protocol": "embedbench.isingfold-corpus-generator-v4",
        "python_version": platform.python_version(),
        "source_modules": source_modules,
        "thread_controls": {name: os.environ.get(name) for name in _THREAD_CONTROLS},
    }


@dataclass(frozen=True, slots=True)
class ShardConfig:
    root_seed: int
    shard_index: int
    shard_count: int
    attempt_slots: int
    incumbent_search_slots: int
    max_split_search: int
    strengths: tuple[float, ...]
    reads: int
    sweeps: int

    def __post_init__(self) -> None:
        if type(self.root_seed) is not int or self.root_seed < 0:
            raise ValueError("root_seed must be a non-negative integer")
        _positive_int(self.shard_count, "shard_count")
        if type(self.shard_index) is not int or not 0 <= self.shard_index < self.shard_count:
            raise ValueError("shard_index must lie in [0, shard_count)")
        if _positive_int(self.attempt_slots, "attempt_slots") < 2:
            raise ValueError("attempt_slots must allow two alternatives")
        _positive_int(self.incumbent_search_slots, "incumbent_search_slots")
        _positive_int(self.max_split_search, "max_split_search")
        _positive_int(self.reads, "reads")
        _positive_int(self.sweeps, "sweeps")
        strengths = tuple(_finite(value, "strength") for value in self.strengths)
        if not strengths or len(set(strengths)) != len(strengths) or any(x <= 0 for x in strengths):
            raise ValueError("strengths must be unique positive finite values")
        object.__setattr__(self, "strengths", strengths)

    @classmethod
    def smoke(cls, *, root_seed: int, shard_index: int, shard_count: int) -> ShardConfig:
        """Construct the explicitly test-only, low-budget fixture profile."""

        return cls(
            root_seed=root_seed,
            shard_index=shard_index,
            shard_count=shard_count,
            attempt_slots=10,
            incumbent_search_slots=4,
            max_split_search=128,
            strengths=(1.0,),
            reads=2,
            sweeps=5,
        )


@dataclass(frozen=True, slots=True)
class LineageRequest:
    lineage_id: str
    application_family: str
    origin: Origin
    partition: Partition
    distribution_regime: Regime
    topology: str
    host_size: int
    n_variables: int
    qubit_fraction: float = 0.0
    coupler_fraction: float = 0.0
    ink_chain_size: int = 1
    ink_drop_mode: str = "compact"

    def __post_init__(self) -> None:
        _text(self.lineage_id, "lineage_id")
        _text(self.application_family, "application_family")
        if self.origin not in {"application-derived", "synthetic-ink-drop"}:
            raise ValueError("origin must be application-derived or synthetic-ink-drop")
        if (
            self.origin == "application-derived"
            and self.application_family not in _APPLICATION_FAMILIES
        ):
            raise ValueError(
                "application-derived origin requires a registered application family: "
                f"{sorted(_APPLICATION_FAMILIES)}"
            )
        if self.origin == "synthetic-ink-drop" and self.application_family != "ink-drop-quotient":
            raise ValueError(
                "synthetic-ink-drop origin requires application_family='ink-drop-quotient'"
            )
        if self.partition not in {"train", "val", "test"}:
            raise ValueError("partition must be train, val, or test")
        if self.distribution_regime not in {"iid", "ood"}:
            raise ValueError("distribution_regime must be iid or ood")
        if self.distribution_regime == "ood" and self.partition != "test":
            raise ValueError("OOD lineages are allowed only in test")
        if self.topology not in _TOPOLOGIES:
            raise ValueError(f"topology must be one of {sorted(_TOPOLOGIES)}")
        _positive_int(self.host_size, "host_size")
        if type(self.n_variables) is not int or not 2 <= self.n_variables <= 22:
            raise ValueError("n_variables must lie in [2, 22]")
        for name in ("qubit_fraction", "coupler_fraction"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a binary64 fraction in [0, 1]")
        _positive_int(self.ink_chain_size, "ink_chain_size")
        if self.ink_drop_mode not in MODES:
            raise ValueError(f"ink_drop_mode must be one of {sorted(MODES)}")


@dataclass(frozen=True, slots=True)
class InProcessExactReplay:
    """Typed receipt for the exact verifier call executed in the current process."""

    problem_sha256: str
    source_sha256: str
    environment_sha256: str
    variable_count: int
    checked_state_count: int
    energy: Dyadic
    certificate_sha256: str
    proof_artifact_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.problem_sha256, "exact-replay problem SHA-256")
        _sha256(self.source_sha256, "exact-replay source SHA-256")
        _sha256(self.environment_sha256, "exact-replay environment SHA-256")
        _sha256(self.certificate_sha256, "exact-replay certificate SHA-256")
        _sha256(self.proof_artifact_sha256, "exact-replay proof SHA-256")
        if type(self.variable_count) is not int or not 0 <= self.variable_count <= 22:
            raise ValueError("exact-replay variable_count must lie in [0, 22]")
        if (
            type(self.checked_state_count) is not int
            or self.checked_state_count != 1 << self.variable_count
        ):
            raise ValueError("exact-replay checked state count is not the full state space")
        if not isinstance(self.energy, Dyadic):
            raise TypeError("exact-replay energy must be an exact Dyadic")

    @classmethod
    def from_verified_replay(
        cls,
        *,
        problem: IsingProblem,
        proof: ExhaustiveProof,
        verified: VerifiedGroundState,
        source_sha256: str,
        environment_sha256: str,
    ) -> InProcessExactReplay:
        """Bind only values returned by a successful exhaustive semantic replay."""

        if verified.status != "exact_enumeration" or not verified.quality_eligible:
            raise ValueError("exact replay did not return a quality-eligible exact result")
        if (
            proof.problem_sha256 != problem.problem_sha256
            or proof.digest != verified.proof_artifact_sha256
            or proof.minimum_energy != verified.energy
        ):
            raise ValueError("exact replay result differs from its exhaustive proof")
        expected_count = 1 << len(problem.variables)
        if proof.checked_state_count != expected_count:
            raise ValueError("exact replay did not check the full state space")
        return cls(
            problem_sha256=problem.problem_sha256,
            source_sha256=source_sha256,
            environment_sha256=environment_sha256,
            variable_count=len(problem.variables),
            checked_state_count=proof.checked_state_count,
            energy=verified.energy,
            certificate_sha256=verified.certificate_sha256,
            proof_artifact_sha256=verified.proof_artifact_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": _EXACT_REPLAY_ALGORITHM,
            "certificate_sha256": self.certificate_sha256,
            "checked_state_count": self.checked_state_count,
            "energy": self.energy.to_dict(),
            "environment_sha256": self.environment_sha256,
            "execution_mode": "in_process",
            "problem_sha256": self.problem_sha256,
            "proof_artifact_sha256": self.proof_artifact_sha256,
            "quality_eligible": True,
            "replay_entry_point": _EXACT_REPLAY_ENTRY_POINT,
            "schema": _EXACT_AUTHORITY_SCHEMA,
            "schema_version": _EXACT_AUTHORITY_SCHEMA_VERSION,
            "source_sha256": self.source_sha256,
            "status": "exact_enumeration",
            "variable_count": self.variable_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> InProcessExactReplay:
        fields = {
            "algorithm",
            "certificate_sha256",
            "checked_state_count",
            "energy",
            "environment_sha256",
            "execution_mode",
            "problem_sha256",
            "proof_artifact_sha256",
            "quality_eligible",
            "replay_entry_point",
            "schema",
            "schema_version",
            "source_sha256",
            "status",
            "variable_count",
        }
        document = _exact_mapping(value, fields, "exact-replay authority")
        if (
            document["schema"] != _EXACT_AUTHORITY_SCHEMA
            or document["schema_version"] != _EXACT_AUTHORITY_SCHEMA_VERSION
            or document["algorithm"] != _EXACT_REPLAY_ALGORITHM
            or document["execution_mode"] != "in_process"
            or document["replay_entry_point"] != _EXACT_REPLAY_ENTRY_POINT
            or document["status"] != "exact_enumeration"
            or document["quality_eligible"] is not True
        ):
            raise ValueError("exact-replay authority has unsupported semantics")
        energy_document = _exact_mapping(
            document["energy"], {"integer", "power_of_two"}, "exact-replay energy"
        )
        result = cls(
            problem_sha256=document["problem_sha256"],
            source_sha256=document["source_sha256"],
            environment_sha256=document["environment_sha256"],
            variable_count=document["variable_count"],
            checked_state_count=document["checked_state_count"],
            energy=Dyadic(energy_document["integer"], energy_document["power_of_two"]),
            certificate_sha256=document["certificate_sha256"],
            proof_artifact_sha256=document["proof_artifact_sha256"],
        )
        if result.to_dict() != document:
            raise ValueError("exact-replay authority is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ExactReference:
    energy: float
    energy_integer: int
    energy_power_of_two: int
    problem_digest: str
    authority_sha256: str
    certificate_sha256: str
    proof_artifact_sha256: str
    evaluator_protocol: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy", _finite(self.energy, "exact reference energy"))
        if type(self.energy_integer) is not int or type(self.energy_power_of_two) is not int:
            raise ValueError("exact reference dyadic fields must be integers")
        exact_energy = Dyadic(self.energy_integer, self.energy_power_of_two)
        if exact_energy.to_float() != self.energy:
            raise ValueError("exact reference float differs from its exact dyadic energy")
        _sha256(self.problem_digest, "exact-reference problem digest")
        _sha256(self.authority_sha256, "exact-reference authority SHA-256")
        _sha256(self.certificate_sha256, "exact-reference certificate SHA-256")
        _sha256(self.proof_artifact_sha256, "exact-reference proof SHA-256")
        if not isinstance(self.evaluator_protocol, Mapping):
            raise ValueError("exact-reference evaluator protocol must be an object")
        replay = InProcessExactReplay.from_dict(self.evaluator_protocol)
        if (
            replay.problem_sha256 != self.problem_digest
            or replay.certificate_sha256 != self.certificate_sha256
            or replay.proof_artifact_sha256 != self.proof_artifact_sha256
            or replay.energy != exact_energy
        ):
            raise ValueError("exact-reference fields differ from their verified replay")
        if content_digest(self.evaluator_protocol) != self.authority_sha256:
            raise ValueError(
                "exact-reference authority does not authenticate its evaluator protocol"
            )


@dataclass(frozen=True, slots=True)
class ProspectiveLineage:
    request: LineageRequest
    prospective_seed: int
    prospective_slot: int
    split_unit_id: str
    logical_nodes: tuple[int, ...]
    logical_edges: tuple[tuple[int, int], ...]
    h: tuple[tuple[int, float], ...]
    j: tuple[tuple[int, int, float], ...]
    origin_metadata: Mapping[str, object]
    realized_host_artifact: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class AuthenticatedReferenceSource:
    row: Mapping[str, object]
    canonical_bytes: bytes
    record_sha256: str
    authority_sha256: str
    problem_digest: str


@dataclass(frozen=True, slots=True)
class GeneratedLineage:
    request: LineageRequest
    prospect: ProspectiveLineage
    exact_reference: ExactReference
    instance: InstanceRecord
    group: CandidateGroup
    lineage_fact: LineageFact
    task_fact: TaskFact
    reference_source: AuthenticatedReferenceSource


@dataclass(frozen=True, slots=True)
class CorpusShard:
    config: ShardConfig
    provenance: Mapping[str, object]
    provenance_digest: str
    shard_index: int
    shard_count: int
    lineages: tuple[GeneratedLineage, ...]
    record_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "lineages": [
                {
                    "exact_reference": asdict(row.exact_reference),
                    "group": row.group.to_dict(),
                    "instance": row.instance.to_dict(),
                    "lineage_fact": asdict(row.lineage_fact),
                    "origin_metadata": dict(row.prospect.origin_metadata),
                    "prospective_seed": row.prospect.prospective_seed,
                    "prospective_slot": row.prospect.prospective_slot,
                    "realized_host_artifact": dict(row.prospect.realized_host_artifact),
                    "reference_authority_sha256": row.exact_reference.authority_sha256,
                    "reference_problem_digest": row.exact_reference.problem_digest,
                    "reference_source": dict(row.reference_source.row),
                    "reference_source_sha256": row.reference_source.record_sha256,
                    "request": asdict(row.request),
                    "task_fact": asdict(row.task_fact),
                }
                for row in self.lineages
            ],
            "record_digest": self.record_digest,
            "provenance": dict(self.provenance),
            "provenance_digest": self.provenance_digest,
            "schema": "embedbench.isingfold-corpus-shard",
            "schema_version": 2,
            "shard_count": self.shard_count,
            "shard_index": self.shard_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CorpusShard:
        """Strictly rehydrate and independently revalidate one shard receipt."""

        document = _exact_mapping(
            value,
            {
                "config",
                "lineages",
                "provenance",
                "provenance_digest",
                "record_digest",
                "schema",
                "schema_version",
                "shard_count",
                "shard_index",
            },
            "corpus shard",
        )
        if document["schema"] != "embedbench.isingfold-corpus-shard":
            raise ValueError("unsupported corpus shard schema")
        if type(document["schema_version"]) is not int or document["schema_version"] != 2:
            raise ValueError("unsupported corpus shard schema_version")
        config = _config_from_dict(document["config"])
        provenance = _generation_provenance()
        provenance_digest = content_digest(provenance)
        _sha256(document["provenance_digest"], "generation provenance digest")
        if (
            canonical_json_bytes(document["provenance"]) != canonical_json_bytes(provenance)
            or document["provenance_digest"] != provenance_digest
        ):
            raise ValueError("corpus shard generation provenance mismatch")
        if (
            type(document["shard_index"]) is not int
            or document["shard_index"] != config.shard_index
        ):
            raise ValueError("shard_index differs from generation config")
        if (
            type(document["shard_count"]) is not int
            or document["shard_count"] != config.shard_count
        ):
            raise ValueError("shard_count differs from generation config")
        _sha256(document["record_digest"], "corpus shard record digest")
        raw_lineages = document["lineages"]
        if type(raw_lineages) is not list or not raw_lineages:
            raise ValueError("corpus shard lineages must be a non-empty list")
        lineages = tuple(_lineage_from_dict(config, item) for item in raw_lineages)
        ids = tuple(row.request.lineage_id for row in lineages)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("corpus shard lineage IDs must be sorted and unique")
        expected_digest = content_digest(
            {key: item for key, item in document.items() if key != "record_digest"}
        )
        if document["record_digest"] != expected_digest:
            raise ValueError("corpus shard record digest mismatch")
        result = cls(
            config=config,
            provenance=provenance,
            provenance_digest=provenance_digest,
            shard_index=config.shard_index,
            shard_count=config.shard_count,
            lineages=lineages,
            record_digest=expected_digest,
        )
        if canonical_json_bytes(result.to_dict()) != canonical_json_bytes(document):
            raise ValueError("corpus shard does not reproduce its canonical nested records")
        return result


def _config_from_dict(value: object) -> ShardConfig:
    document = _exact_mapping(
        value,
        {
            "attempt_slots",
            "incumbent_search_slots",
            "max_split_search",
            "reads",
            "root_seed",
            "shard_count",
            "shard_index",
            "strengths",
            "sweeps",
        },
        "shard config",
    )
    strengths = document["strengths"]
    if type(strengths) not in {list, tuple}:
        raise ValueError("shard config strengths must be a list or tuple")
    return ShardConfig(
        root_seed=document["root_seed"],
        shard_index=document["shard_index"],
        shard_count=document["shard_count"],
        attempt_slots=document["attempt_slots"],
        incumbent_search_slots=document["incumbent_search_slots"],
        max_split_search=document["max_split_search"],
        strengths=tuple(strengths),
        reads=document["reads"],
        sweeps=document["sweeps"],
    )


def _request_from_dict(value: object) -> LineageRequest:
    document = _exact_mapping(
        value,
        {
            "application_family",
            "coupler_fraction",
            "distribution_regime",
            "host_size",
            "ink_chain_size",
            "ink_drop_mode",
            "lineage_id",
            "n_variables",
            "origin",
            "partition",
            "qubit_fraction",
            "topology",
        },
        "lineage request",
    )
    return LineageRequest(**document)


def lineage_shard(lineage_id: str, shard_count: int) -> int:
    checked_id = _text(lineage_id, "lineage_id")
    count = _positive_int(shard_count, "shard_count")
    return (
        int(
            content_digest({"domain": "isingfold-corpus-shard-v1", "lineage_id": checked_id})[:16],
            16,
        )
        % count
    )


def _host_seed_key(config: ShardConfig, request: LineageRequest) -> bytes:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "isingfold-corpus-realized-host-v1",
                "lineage_id": request.lineage_id,
                "root_seed": config.root_seed,
            }
        )
    ).digest()


def _application_problem(
    config: ShardConfig,
    request: LineageRequest,
    slot: int,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, float], ...],
    tuple[tuple[int, int, float], ...],
    Mapping[str, object],
]:
    seed = stable_seed(
        config.root_seed,
        "isingfold-corpus-application-v1",
        request.lineage_id,
        request.application_family,
        slot,
    )
    if request.application_family == "graphcut":
        rows = max(
            divisor
            for divisor in range(1, math.isqrt(request.n_variables) + 1)
            if request.n_variables % divisor == 0
        )
        generated = graphcut_mrf(
            rows=rows,
            cols=request.n_variables // rows,
            seed=seed,
        )
    elif request.application_family == "portfolio":
        generated = portfolio(
            n_assets=request.n_variables,
            n_factors=min(3, request.n_variables),
            seed=seed,
        )
    else:
        if request.n_variables in {12, 16}:
            n_jobs, n_machines, horizon = 2, 2, request.n_variables // 4
        elif request.n_variables in {6, 8, 10, 14}:
            n_jobs, n_machines, horizon = 2, 1, request.n_variables // 2
        else:
            n_jobs, n_machines, horizon = 1, 1, request.n_variables
        generated = jobshop(
            n_jobs=n_jobs,
            n_machines=n_machines,
            horizon=horizon,
            seed=seed,
        )
    problem = generated.problem
    nodes = tuple(sorted(problem.h))
    if nodes != tuple(range(request.n_variables)):
        raise ValueError("registered application generator did not honor n_variables")
    edges = tuple(sorted(problem.j))
    if not edges:
        raise ValueError("registered application generator produced an edgeless problem")
    return (
        nodes,
        edges,
        tuple((node, problem.h[node]) for node in nodes),
        tuple((left, right, problem.j[(left, right)]) for left, right in edges),
        {
            "derived": generated.derived,
            "family": generated.family,
            "kind": "application-derived",
            "kwargs": generated.kwargs,
            "name": generated.name,
            "seed": generated.seed,
        },
    )


def _synthetic_problem(
    config: ShardConfig,
    request: LineageRequest,
    host: nx.Graph,
    slot: int,
) -> (
    tuple[
        tuple[int, ...],
        tuple[tuple[int, int], ...],
        tuple[tuple[int, float], ...],
        tuple[tuple[int, int, float], ...],
        Mapping[str, object],
    ]
    | None
):
    try:
        generated = ink_drop(
            host,
            request.n_variables,
            request.ink_chain_size,
            mode=request.ink_drop_mode,
            seed=stable_seed(
                config.root_seed,
                "isingfold-corpus-ink-drop-v1",
                request.lineage_id,
                slot,
            ),
        )
    except (RuntimeError, ValueError):
        return None
    graph = generated.logical
    nodes = tuple(sorted(graph.nodes))
    if (
        nodes != tuple(range(request.n_variables))
        or graph.number_of_edges() == 0
        or not nx.is_connected(graph)
    ):
        return None
    edges = tuple(sorted((min(left, right), max(left, right)) for left, right in graph.edges))
    coefficient_seed = stable_seed(
        config.root_seed,
        "isingfold-corpus-coefficients-v1",
        request.lineage_id,
        slot,
    )
    field_choices = (-0.5, 0.0, 0.5)
    coupling_choices = (-1.0, -0.5, 0.5, 1.0)
    h = tuple(
        (
            node,
            field_choices[stable_seed(coefficient_seed, "linear", node) % len(field_choices)],
        )
        for node in nodes
    )
    j = tuple(
        (
            left,
            right,
            coupling_choices[
                stable_seed(coefficient_seed, "quadratic", left, right) % len(coupling_choices)
            ],
        )
        for left, right in edges
    )
    return (
        nodes,
        edges,
        h,
        j,
        {
            "chain_size": request.ink_chain_size,
            "generator": "free-growth-contact-quotient",
            "kind": "synthetic-ink-drop",
            "mode": request.ink_drop_mode,
            "seed": generated.seed,
        },
    )


def prospect_lineage(config: ShardConfig, request: LineageRequest) -> ProspectiveLineage:
    if not isinstance(config, ShardConfig) or not isinstance(request, LineageRequest):
        raise TypeError("prospecting requires ShardConfig and LineageRequest")
    seed_key = _host_seed_key(config, request)
    artifact = build_realized_host_artifact(
        topology=request.topology,
        size=request.host_size,
        qubit_fraction=request.qubit_fraction,
        coupler_fraction=request.coupler_fraction,
        seed_key=seed_key,
    )
    host = load_realized_host(
        topology=request.topology,
        size=request.host_size,
        qubit_fraction=request.qubit_fraction,
        coupler_fraction=request.coupler_fraction,
        seed_key=seed_key,
        artifact=artifact,
        expected_host_artifact_sha256=str(artifact["host_artifact_sha256"]),
        expected_host_sha256=str(artifact["host_sha256"]),
    )
    for slot in range(config.max_split_search):
        base_problem = (
            _application_problem(config, request, slot)
            if request.origin == "application-derived"
            else _synthetic_problem(config, request, host, slot)
        )
        if base_problem is None:
            continue
        nodes, edges, base_h, base_j, origin_metadata = base_problem
        prospective_seed = stable_seed(
            config.root_seed,
            "isingfold-corpus-prospective-split-v1",
            request.lineage_id,
            slot,
        )
        gauges = {node: (-1 if (prospective_seed >> node) & 1 else 1) for node in nodes}
        h = tuple((node, coefficient * gauges[node]) for node, coefficient in base_h)
        j = tuple(
            (left, right, coefficient * gauges[left] * gauges[right])
            for left, right, coefficient in base_j
        )
        split_unit_id = derive_split_unit_id(
            logical_nodes=nodes,
            logical_edges=edges,
            h=h,
            j=j,
        )
        if assign_split(split_unit_id) == request.partition:
            return ProspectiveLineage(
                request=request,
                prospective_seed=prospective_seed,
                prospective_slot=slot,
                split_unit_id=split_unit_id,
                logical_nodes=nodes,
                logical_edges=edges,
                h=h,
                j=j,
                origin_metadata=origin_metadata,
                realized_host_artifact=artifact,
            )
    if request.origin == "synthetic-ink-drop":
        raise ValueError(
            "prospective seed search found no usable connected quotient assigned to "
            f"{request.partition} in {config.max_split_search} fixed slots"
        )
    raise ValueError(
        f"prospective seed search found no {request.partition} split in "
        f"{config.max_split_search} slots"
    )


def _reference_problem(prospect: ProspectiveLineage) -> IsingProblem:
    if not isinstance(prospect, ProspectiveLineage):
        raise TypeError("prospect must be a ProspectiveLineage")
    return IsingProblem(
        variables=prospect.logical_nodes,
        linear=prospect.h,
        quadratic=prospect.j,
    )


def reference_problem_digest(prospect: ProspectiveLineage) -> str:
    return _reference_problem(prospect).problem_sha256


def derive_exact_reference(prospect: ProspectiveLineage) -> ExactReference:
    """Derive an authenticated exact reference without caller-supplied assertions."""

    problem = _reference_problem(prospect)
    if len(problem.variables) > 22:
        raise ValueError("corpus exact enumeration is limited to 22 variables")
    provenance = _generation_provenance()
    provenance_digest = content_digest(provenance)
    source_modules = cast(Mapping[str, object], provenance["source_modules"])
    source_sha256 = _sha256(source_modules["ground_certificate"], "enumerator source SHA-256")
    solver = SolverRun(
        name="embedbench-gray-code-dyadic-enumerator",
        version=f"source-sha256:{source_sha256}",
        command=("embedbench.ground_certificate.build_exact_enumeration_certificate",),
        seed=None,
        deterministic_work_limit=1 << len(problem.variables),
        safety_timeout_seconds=None,
    )
    bundle = build_exact_enumeration_certificate(
        problem,
        solver=solver,
    )
    verified = verify_ground_state_certificate(
        problem,
        bundle.certificate.to_dict(),
        bundle.proof.to_dict(),
        expected_certificate_sha256=bundle.certificate.digest,
        expected_proof_artifact_sha256=bundle.proof.digest,
    )
    energy = bundle.certificate.energy
    if verified.energy != energy or not verified.quality_eligible:
        raise RuntimeError("internally generated exact certificate did not verify")
    evaluator_protocol = InProcessExactReplay.from_verified_replay(
        problem=problem,
        proof=bundle.proof,
        verified=verified,
        source_sha256=source_sha256,
        environment_sha256=provenance_digest,
    ).to_dict()
    return ExactReference(
        energy=energy.to_float(),
        energy_integer=energy.integer,
        energy_power_of_two=energy.power_of_two,
        problem_digest=problem.problem_sha256,
        authority_sha256=content_digest(evaluator_protocol),
        certificate_sha256=bundle.certificate.digest,
        proof_artifact_sha256=bundle.proof.digest,
        evaluator_protocol=evaluator_protocol,
    )


def generate_corpus_shard(
    config: ShardConfig,
    requests: Sequence[LineageRequest],
) -> CorpusShard:
    """Generate one deterministic shard with internally certified exact references."""

    if not isinstance(config, ShardConfig):
        raise TypeError("config must be a ShardConfig")
    if not isinstance(requests, Sequence):
        raise TypeError("requests must be a sequence")
    requested = tuple(requests)
    if not requested or any(not isinstance(item, LineageRequest) for item in requested):
        raise TypeError("requests must contain LineageRequest values")
    by_id = {item.lineage_id: item for item in requested}
    if len(by_id) != len(requested):
        raise ValueError("lineage requests repeat a lineage_id")
    selected = tuple(
        by_id[lineage_id]
        for lineage_id in sorted(by_id)
        if lineage_shard(lineage_id, config.shard_count) == config.shard_index
    )
    if not selected:
        raise ValueError("selected shard contains no requested lineages")

    generated: list[GeneratedLineage] = []
    seen_split_units: set[str] = set()
    for request in selected:
        prospect = prospect_lineage(config, request)
        if prospect.split_unit_id in seen_split_units:
            raise ValueError("independent lineage requests produced one logical split unit")
        seen_split_units.add(prospect.split_unit_id)
        reference = derive_exact_reference(prospect)
        generated.append(_generate_lineage(config, prospect, reference))
    payload = _shard_payload(config, generated)
    provenance = _generation_provenance()
    return CorpusShard(
        config=config,
        provenance=provenance,
        provenance_digest=content_digest(provenance),
        shard_index=config.shard_index,
        shard_count=config.shard_count,
        lineages=tuple(generated),
        record_digest=content_digest(payload),
    )


def _load_prospect_host(config: ShardConfig, prospect: ProspectiveLineage) -> nx.Graph:
    request = prospect.request
    artifact = prospect.realized_host_artifact
    return load_realized_host(
        topology=request.topology,
        size=request.host_size,
        qubit_fraction=request.qubit_fraction,
        coupler_fraction=request.coupler_fraction,
        seed_key=_host_seed_key(config, request),
        artifact=artifact,
        expected_host_artifact_sha256=str(artifact["host_artifact_sha256"]),
        expected_host_sha256=str(artifact["host_sha256"]),
    )


def _proposal(
    *,
    host: nx.Graph,
    prospect: ProspectiveLineage,
    seed: int,
) -> tuple[tuple[int, ...], ...] | None:
    logical = nx.Graph()
    logical.add_nodes_from(prospect.logical_nodes)
    logical.add_edges_from(prospect.logical_edges)
    proposal_seed = seed & (2**31 - 1)
    if proposal_seed == 0:
        proposal_seed = 1
    try:
        raw = minorminer.find_embedding(
            logical,
            host,
            random_seed=proposal_seed,
            tries=1,
            inner_rounds=50,
            max_no_improvement=3,
            chainlength_patience=3,
            threads=1,
        )
        if set(raw) != set(prospect.logical_nodes):
            return None
        chains = tuple(tuple(sorted(raw[node])) for node in prospect.logical_nodes)
        validate_minor_embedding(
            host,
            logical_edges=[list(edge) for edge in prospect.logical_edges],
            chains={node: list(chains[index]) for index, node in enumerate(prospect.logical_nodes)},
        )
        return chains
    except (RuntimeError, TypeError, ValueError):
        return None


def _incumbent(
    config: ShardConfig,
    prospect: ProspectiveLineage,
    host: nx.Graph,
) -> tuple[tuple[int, ...], ...]:
    for slot in range(config.incumbent_search_slots):
        seed = stable_seed(
            config.root_seed,
            "isingfold-corpus-minorminer-incumbent-v1",
            prospect.request.lineage_id,
            prospect.prospective_seed,
            slot,
        )
        chains = _proposal(host=host, prospect=prospect, seed=seed)
        if chains is not None:
            return chains
    raise ValueError("Minorminer found no feasible incumbent in the fixed search slots")


def _alternative_attempts(
    config: ShardConfig,
    prospect: ProspectiveLineage,
    host: nx.Graph,
    group_id: str,
    incumbent: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[tuple[int, ...], ...], ...], tuple[RepairAttempt, ...]]:
    alternatives: list[tuple[tuple[int, ...], ...]] = []
    alternative_ids: dict[tuple[tuple[int, ...], ...], str] = {}
    attempts: list[RepairAttempt] = []
    neighborhood = prospect.logical_nodes
    for slot in range(config.attempt_slots):
        seed = stable_seed(
            config.root_seed,
            "isingfold-corpus-minorminer-alternative-v1",
            prospect.request.lineage_id,
            prospect.prospective_seed,
            slot,
        )
        chains = _proposal(host=host, prospect=prospect, seed=seed)
        candidate_id: str | None = None
        reason: str | None = None
        if chains is None:
            status = "repair_failed"
            reason = "minorminer_returned_no_valid_embedding"
        elif chains == incumbent:
            status = "no_change"
            reason = "minorminer_proposal_matches_incumbent"
        elif chains in alternative_ids:
            status = "duplicate"
            candidate_id = alternative_ids[chains]
            reason = "minorminer_proposal_duplicate"
        else:
            status = "valid"
            candidate_id = derive_candidate_id(group_id, chains)
            alternative_ids[chains] = candidate_id
            alternatives.append(chains)
        attempts.append(
            RepairAttempt(
                attempt_id=derive_attempt_id(group_id, slot),
                repair_seed=seed,
                neighborhood=neighborhood,
                status=status,
                candidate_id=candidate_id,
                slot=slot,
                transitions=0,
                reason=reason,
            )
        )
    if len(alternatives) < 2:
        raise ValueError("fixed attempt slots did not yield two unique valid alternatives")
    return tuple(alternatives), tuple(attempts)


def _evaluation_seeds(
    config: ShardConfig,
    prospect: ProspectiveLineage,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    decision = stable_seed(
        config.root_seed,
        "isingfold-corpus-decision-sa-v1",
        prospect.request.lineage_id,
    ) & (2**31 - 1)
    audit = stable_seed(
        config.root_seed,
        "isingfold-corpus-audit-sa-v1",
        prospect.request.lineage_id,
    ) & (2**31 - 1)
    if audit == decision:
        audit = (audit + 1) & (2**31 - 1)
    return (decision,), (audit,)


def _evaluation_protocol(
    config: ShardConfig, reference: ExactReference
) -> tuple[tuple[str, Any], ...]:
    return (
        ("decoder", "majority-vote-random-ties-v1"),
        ("noise_model", "none"),
        ("reference_energy", reference.energy),
        ("sampler", "dwave.samplers.SimulatedAnnealingSampler"),
        ("sampler_version", importlib.metadata.version("dwave-samplers")),
        (
            "schedule",
            {
                "beta_range": list(BETA_RANGE),
                "strengths": list(config.strengths),
                "sweeps": config.sweeps,
            },
        ),
        ("seed_derivation", "isingfold-corpus-paired-sa-v1"),
    )


def _curve(
    *,
    partition: Literal["decision", "audit"],
    seeds: tuple[int, ...],
    config: ShardConfig,
    embedding: Embedding,
    problem: LogicalProblem,
    reference: ExactReference,
) -> EvaluationCurve:
    results = [
        [
            solve_probability_at(
                embedding,
                problem,
                reference.energy,
                strength,
                num_reads=config.reads,
                num_sweeps=config.sweeps,
                seed=seed,
            )
            for seed in seeds
        ]
        for strength in config.strengths
    ]
    return EvaluationCurve(
        partition=partition,
        strengths=config.strengths,
        seeds=seeds,
        p_solve=tuple(tuple(result.p_solve for result in row) for row in results),
        residual_mean=tuple(tuple(result.mean_residual for result in row) for row in results),
        reads=config.reads,
        sweeps=config.sweeps,
        objective="solve_probability_then_residual-v1",
        evaluation_protocol=_evaluation_protocol(config, reference),
    )


def _candidate(
    *,
    group_id: str,
    chains: tuple[tuple[int, ...], ...],
    host: nx.Graph,
    problem: LogicalProblem,
    reference: ExactReference,
    config: ShardConfig,
    decision_seeds: tuple[int, ...],
    audit_seeds: tuple[int, ...],
) -> CandidateRecord:
    mapping = {node: chains[index] for index, node in enumerate(sorted(problem.h))}
    embedding = Embedding.from_chains(mapping, host, problem)
    return CandidateRecord.create(
        group_id=group_id,
        chains=chains,
        decision=_curve(
            partition="decision",
            seeds=decision_seeds,
            config=config,
            embedding=embedding,
            problem=problem,
            reference=reference,
        ),
        audit=_curve(
            partition="audit",
            seeds=audit_seeds,
            config=config,
            embedding=embedding,
            problem=problem,
            reference=reference,
        ),
    )


def _group_protocol(
    config: ShardConfig, prospect: ProspectiveLineage
) -> tuple[tuple[str, Any], ...]:
    provenance_digest = content_digest(_generation_provenance())
    return (
        ("attempt_slots", config.attempt_slots),
        ("incumbent_search_slots", config.incumbent_search_slots),
        ("minorminer_role", "feasible-proposal-only"),
        ("minorminer_version", importlib.metadata.version("minorminer")),
        (
            "proposal_limits",
            {
                "chainlength_patience": 3,
                "inner_rounds": 50,
                "max_no_improvement": 3,
                "threads": 1,
                "timeout_seconds": None,
                "tries": 1,
            },
        ),
        ("prospective_seed", prospect.prospective_seed),
        ("prospective_slot", prospect.prospective_slot),
        ("provenance_digest", provenance_digest),
        ("seed_derivation", "sha256-domain-separated-v1"),
    )


def _instance(
    config: ShardConfig,
    prospect: ProspectiveLineage,
) -> InstanceRecord:
    artifact = prospect.realized_host_artifact
    metadata: list[tuple[str, Any]] = [
        ("active_host_artifact_sha256", artifact["host_artifact_sha256"]),
        ("active_host_sha256", artifact["host_sha256"]),
        ("generator_origin", prospect.request.origin),
        ("host_seed_sha256", hashlib.sha256(_host_seed_key(config, prospect.request)).hexdigest()),
        ("lineage_id", prospect.request.lineage_id),
        ("prospective_seed", prospect.prospective_seed),
        ("prospective_slot", prospect.prospective_slot),
    ]
    if prospect.request.origin == "application-derived":
        metadata.extend(
            [
                ("application_derived", prospect.origin_metadata["derived"]),
                ("application_kwargs", prospect.origin_metadata["kwargs"]),
                ("application_name", prospect.origin_metadata["name"]),
                ("application_seed", prospect.origin_metadata["seed"]),
            ]
        )
    else:
        metadata.append(("ink_drop_generation", prospect.origin_metadata))
    return InstanceRecord.create(
        split_unit_id=prospect.split_unit_id,
        family=prospect.request.application_family,
        topology=prospect.request.topology,
        logical_nodes=prospect.logical_nodes,
        logical_edges=prospect.logical_edges,
        host_nodes=tuple(cast(Sequence[int], artifact["nodes"])),
        host_edges=tuple(tuple(edge) for edge in cast(Sequence[Sequence[int]], artifact["edges"])),
        h=prospect.h,
        j=prospect.j,
        metadata=tuple(metadata),
    )


def _design_facts(
    *,
    config: ShardConfig,
    prospect: ProspectiveLineage,
    instance: InstanceRecord,
    group: CandidateGroup,
) -> tuple[LineageFact, TaskFact]:
    problem = IsingProblem(
        variables=prospect.logical_nodes,
        linear=prospect.h,
        quadratic=prospect.j,
    )
    base_lineage_key = f"problem-sha256:{problem.problem_sha256}"
    records = (group.incumbent, *group.candidates)
    resource_counts = [record.total_qubits for record in records]
    host_size = len(instance.host_nodes)
    density_denominator = len(instance.logical_nodes) * (len(instance.logical_nodes) - 1)
    measurements = (
        (
            "candidate_resource_spread",
            float(max(resource_counts) - min(resource_counts)) / max(1, host_size),
        ),
        ("host_fill_fraction", float(group.incumbent.total_qubits) / host_size),
        (
            "logical_edge_density",
            2.0 * len(instance.logical_edges) / density_denominator,
        ),
    )
    implementation_sha256 = content_digest(_generation_provenance())
    lineage = LineageFact(
        base_lineage_key=base_lineage_key,
        application_family=prospect.request.application_family,
        generator_id=(
            "embedbench/isingfold-corpus-shard/"
            f"application-{prospect.request.application_family}-"
            f"v{2 if prospect.request.application_family in {'jobshop', 'portfolio'} else 1}"
            if prospect.request.origin == "application-derived"
            else "embedbench/isingfold-corpus-shard/synthetic-ink-drop-v1"
        ),
        generator_implementation_sha256=implementation_sha256,
        generator_kind=(
            "application" if prospect.request.origin == "application-derived" else "synthetic"
        ),
        source_instance_record_digests=(instance.record_digest,),
        measurements=measurements,
    )
    artifact = prospect.realized_host_artifact
    faulted = bool(artifact["removed_nodes"] or artifact["removed_edges"])
    gauge_active = any((prospect.prospective_seed >> node) & 1 for node in prospect.logical_nodes)
    kinds = sorted(
        [
            *(("fault",) if faulted else ()),
            *(("gauge",) if gauge_active else ()),
        ]
    )
    if not kinds:
        kinds = ["identity"]
    transform_sha256 = content_digest(
        {
            "domain": "isingfold-corpus-descendant-transform-v1",
            "host_artifact_sha256": artifact["host_artifact_sha256"],
            "kinds": kinds,
            "prospective_seed": prospect.prospective_seed,
            "split_unit_id": instance.split_unit_id,
        }
    )
    task = TaskFact(
        group_id=group.group_id,
        base_parent_lineage=base_lineage_key,
        active_topology_identity={
            "host_artifact_sha256": artifact["host_artifact_sha256"],
            "host_sha256": artifact["host_sha256"],
            "topology": prospect.request.topology,
        },
        nominal_topology_identity={
            "pristine_host_sha256": artifact["pristine_host_sha256"],
            "size": prospect.request.host_size,
            "topology": prospect.request.topology,
        },
        fault_identity={
            "fault_mask_sha256": artifact["host_artifact_sha256"],
            "status": "faulted" if faulted else "none",
        },
        calibration_identity={
            "calibration_sha256": None,
            "status": "not_applicable",
        },
        descendant_transform_identity={
            "kinds": kinds,
            "transform_sha256": transform_sha256,
        },
        distribution={
            "learning_partition": prospect.request.partition,
            "regime": prospect.request.distribution_regime,
            "source_partition": "isingfold-corpus-v4",
            "stratum": (
                f"{prospect.request.origin}/{prospect.request.topology}/"
                f"{'faulted' if faulted else 'none'}"
            ),
        },
    )
    return lineage, task


def _reference_source(
    prospect: ProspectiveLineage,
    instance: InstanceRecord,
    reference: ExactReference,
) -> AuthenticatedReferenceSource:
    row: dict[str, object] = {
        "instance_id": instance.instance_id,
        "mode": (
            "isingfold_application"
            if prospect.request.origin == "application-derived"
            else "isingfold_synthetic_ink_drop"
        ),
        "problem": {
            "J": [list(term) for term in prospect.j],
            "e0": reference.energy,
            "h": {str(node): coefficient for node, coefficient in prospect.h},
        },
    }
    raw = canonical_json_bytes(row)
    return AuthenticatedReferenceSource(
        row=row,
        canonical_bytes=raw,
        record_sha256=hashlib.sha256(raw).hexdigest(),
        authority_sha256=reference.authority_sha256,
        problem_digest=reference.problem_digest,
    )


def _generate_lineage(
    config: ShardConfig,
    prospect: ProspectiveLineage,
    reference: ExactReference,
) -> GeneratedLineage:
    host = _load_prospect_host(config, prospect)
    instance = _instance(config, prospect)
    problem = LogicalProblem.from_dicts(
        dict(prospect.h), {(left, right): coefficient for left, right, coefficient in prospect.j}
    )
    incumbent_chains = _incumbent(config, prospect, host)
    protocol = _group_protocol(config, prospect)
    group_seed = stable_seed(
        config.root_seed,
        "isingfold-corpus-candidate-group-v1",
        prospect.request.lineage_id,
        prospect.prospective_seed,
    )
    group_id = derive_group_id(instance.instance_id, incumbent_chains, protocol, group_seed)
    alternative_chains, attempts = _alternative_attempts(
        config,
        prospect,
        host,
        group_id,
        incumbent_chains,
    )
    decision_seeds, audit_seeds = _evaluation_seeds(config, prospect)
    incumbent_record = _candidate(
        group_id=group_id,
        chains=incumbent_chains,
        host=host,
        problem=problem,
        reference=reference,
        config=config,
        decision_seeds=decision_seeds,
        audit_seeds=audit_seeds,
    )
    candidates = tuple(
        _candidate(
            group_id=group_id,
            chains=chains,
            host=host,
            problem=problem,
            reference=reference,
            config=config,
            decision_seeds=decision_seeds,
            audit_seeds=audit_seeds,
        )
        for chains in alternative_chains
    )
    generation_digest = derive_generation_digest(
        group_id=group_id,
        instance_id=instance.instance_id,
        instance_record_digest=instance.record_digest,
        split_unit_id=instance.split_unit_id,
        group_seed=group_seed,
        protocol=protocol,
        incumbent_chains=incumbent_chains,
        candidate_chains=alternative_chains,
        attempts=attempts,
        rejection_reason=None,
    )
    group = CandidateGroup.create(
        group_id=group_id,
        instance_id=instance.instance_id,
        instance_record_digest=instance.record_digest,
        split_unit_id=instance.split_unit_id,
        split=assign_split(instance.split_unit_id),
        group_seed=group_seed,
        protocol=protocol,
        incumbent=incumbent_record,
        candidates=candidates,
        attempts=attempts,
        generation_digest=generation_digest,
    )
    validate_group_against_instance(group, instance)
    lineage_fact, task_fact = _design_facts(
        config=config,
        prospect=prospect,
        instance=instance,
        group=group,
    )
    return GeneratedLineage(
        request=prospect.request,
        prospect=prospect,
        exact_reference=reference,
        instance=instance,
        group=group,
        lineage_fact=lineage_fact,
        task_fact=task_fact,
        reference_source=_reference_source(prospect, instance, reference),
    )


def _shard_payload(
    config: ShardConfig,
    lineages: Sequence[GeneratedLineage],
) -> dict[str, object]:
    provenance = _generation_provenance()
    provisional = CorpusShard(
        config=config,
        provenance=provenance,
        provenance_digest=content_digest(provenance),
        shard_index=config.shard_index,
        shard_count=config.shard_count,
        lineages=tuple(lineages),
        record_digest="",
    ).to_dict()
    provisional.pop("record_digest")
    return provisional


def _lineage_fact_from_dict(value: object) -> LineageFact:
    document = _exact_mapping(
        value,
        {
            "application_family",
            "base_lineage_key",
            "generator_id",
            "generator_implementation_sha256",
            "generator_kind",
            "measurements",
            "source_instance_record_digests",
        },
        "lineage fact",
    )
    source_digests = document["source_instance_record_digests"]
    measurements = document["measurements"]
    if type(source_digests) not in {list, tuple}:
        raise ValueError("lineage fact source digests must be a list or tuple")
    if type(measurements) not in {list, tuple} or any(
        type(pair) not in {list, tuple} or len(pair) != 2 for pair in measurements
    ):
        raise ValueError("lineage fact measurements must contain two-item rows")
    return LineageFact(
        base_lineage_key=document["base_lineage_key"],
        application_family=document["application_family"],
        generator_id=document["generator_id"],
        generator_implementation_sha256=document["generator_implementation_sha256"],
        generator_kind=document["generator_kind"],
        source_instance_record_digests=tuple(source_digests),
        measurements=tuple(tuple(pair) for pair in measurements),
    )


def _task_fact_from_dict(value: object) -> TaskFact:
    document = _exact_mapping(
        value,
        {
            "active_topology_identity",
            "base_parent_lineage",
            "calibration_identity",
            "descendant_transform_identity",
            "distribution",
            "fault_identity",
            "group_id",
            "nominal_topology_identity",
        },
        "task fact",
    )
    for field in (
        "active_topology_identity",
        "calibration_identity",
        "descendant_transform_identity",
        "distribution",
        "fault_identity",
        "nominal_topology_identity",
    ):
        if not isinstance(document[field], Mapping):
            raise ValueError(f"task fact {field} must be an object")
    return TaskFact(**document)


def _frozen_protocol(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(sorted((key, _frozen_protocol(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_protocol(item) for item in value)
    return value


def _validate_persisted_protocol(
    config: ShardConfig,
    prospect: ProspectiveLineage,
    reference: ExactReference,
    group: CandidateGroup,
) -> None:
    expected_group_protocol = _group_protocol(config, prospect)
    if _frozen_protocol(group.protocol) != _frozen_protocol(expected_group_protocol):
        raise ValueError("candidate-group protocol differs from shard generation config")
    expected_group_seed = stable_seed(
        config.root_seed,
        "isingfold-corpus-candidate-group-v1",
        prospect.request.lineage_id,
        prospect.prospective_seed,
    )
    if group.group_seed != expected_group_seed:
        raise ValueError("candidate-group seed differs from shard generation config")
    expected_group_id = derive_group_id(
        group.instance_id,
        group.incumbent.chains,
        expected_group_protocol,
        expected_group_seed,
    )
    if group.group_id != expected_group_id:
        raise ValueError("candidate-group identity differs from shard generation protocol")
    decision_seeds, audit_seeds = _evaluation_seeds(config, prospect)
    expected_evaluation_protocol = _evaluation_protocol(config, reference)
    for record in (group.incumbent, *group.candidates):
        for curve, partition, seeds in (
            (record.decision, "decision", decision_seeds),
            (record.audit, "audit", audit_seeds),
        ):
            if (
                curve.partition != partition
                or curve.strengths != config.strengths
                or curve.seeds != seeds
                or curve.reads != config.reads
                or curve.sweeps != config.sweeps
                or _frozen_protocol(curve.evaluation_protocol)
                != _frozen_protocol(expected_evaluation_protocol)
            ):
                raise ValueError(
                    f"candidate evaluation protocol differs from shard config for {partition}"
                )


def _lineage_from_dict(config: ShardConfig, value: object) -> GeneratedLineage:
    document = _exact_mapping(
        value,
        {
            "exact_reference",
            "group",
            "instance",
            "lineage_fact",
            "origin_metadata",
            "prospective_seed",
            "prospective_slot",
            "realized_host_artifact",
            "reference_authority_sha256",
            "reference_problem_digest",
            "reference_source",
            "reference_source_sha256",
            "request",
            "task_fact",
        },
        "generated lineage",
    )
    request = _request_from_dict(document["request"])
    if lineage_shard(request.lineage_id, config.shard_count) != config.shard_index:
        raise ValueError("generated lineage belongs to another shard")
    prospect = prospect_lineage(config, request)
    if (
        type(document["prospective_seed"]) is not int
        or document["prospective_seed"] != prospect.prospective_seed
        or type(document["prospective_slot"]) is not int
        or document["prospective_slot"] != prospect.prospective_slot
    ):
        raise ValueError("generated lineage prospective search receipt mismatch")
    if not isinstance(document["realized_host_artifact"], Mapping) or canonical_json_bytes(
        document["realized_host_artifact"]
    ) != canonical_json_bytes(prospect.realized_host_artifact):
        raise ValueError("generated lineage realized-host artifact mismatch")
    if not isinstance(document["origin_metadata"], Mapping) or canonical_json_bytes(
        document["origin_metadata"]
    ) != canonical_json_bytes(prospect.origin_metadata):
        raise ValueError("generated lineage origin metadata mismatch")

    reference_document = _exact_mapping(
        document["exact_reference"],
        {
            "authority_sha256",
            "certificate_sha256",
            "energy",
            "energy_integer",
            "energy_power_of_two",
            "evaluator_protocol",
            "problem_digest",
            "proof_artifact_sha256",
        },
        "exact reference",
    )
    supplied_reference = ExactReference(**reference_document)
    reference = derive_exact_reference(prospect)
    if canonical_json_bytes(asdict(supplied_reference)) != canonical_json_bytes(asdict(reference)):
        raise ValueError("generated lineage differs from its internally derived exact reference")
    if (
        document["reference_authority_sha256"] != reference.authority_sha256
        or document["reference_problem_digest"] != reference.problem_digest
    ):
        raise ValueError("generated lineage exact-reference identities disagree")
    if reference.problem_digest != reference_problem_digest(prospect):
        raise ValueError("generated lineage exact reference is bound to another problem")

    if not isinstance(document["instance"], Mapping):
        raise ValueError("generated lineage instance must be an object")
    instance = InstanceRecord.from_dict(document["instance"])
    expected_instance = _instance(config, prospect)
    if canonical_json_bytes(instance.to_dict()) != canonical_json_bytes(
        expected_instance.to_dict()
    ):
        raise ValueError("generated lineage instance differs from prospective facts")
    if not isinstance(document["group"], Mapping):
        raise ValueError("generated lineage group must be an object")
    group = CandidateGroup.from_dict(document["group"])
    validate_group_against_instance(group, instance)
    _validate_persisted_protocol(config, prospect, reference, group)

    lineage_fact = _lineage_fact_from_dict(document["lineage_fact"])
    task_fact = _task_fact_from_dict(document["task_fact"])
    expected_lineage_fact, expected_task_fact = _design_facts(
        config=config,
        prospect=prospect,
        instance=instance,
        group=group,
    )
    if canonical_json_bytes(asdict(lineage_fact)) != canonical_json_bytes(
        asdict(expected_lineage_fact)
    ):
        raise ValueError("generated lineage outcome-blind lineage facts mismatch")
    if canonical_json_bytes(asdict(task_fact)) != canonical_json_bytes(asdict(expected_task_fact)):
        raise ValueError("generated lineage outcome-blind task facts mismatch")

    source_document = _exact_mapping(
        document["reference_source"],
        {"instance_id", "mode", "problem"},
        "exact-reference source row",
    )
    _exact_mapping(
        source_document["problem"],
        {"J", "e0", "h"},
        "exact-reference source problem",
    )
    reference_source = _reference_source(prospect, instance, reference)
    if canonical_json_bytes(source_document) != reference_source.canonical_bytes:
        raise ValueError("generated lineage exact-reference source row mismatch")
    if document["reference_source_sha256"] != reference_source.record_sha256:
        raise ValueError("generated lineage exact-reference source SHA-256 mismatch")

    return GeneratedLineage(
        request=request,
        prospect=prospect,
        exact_reference=reference,
        instance=instance,
        group=group,
        lineage_fact=expected_lineage_fact,
        task_fact=expected_task_fact,
        reference_source=reference_source,
    )
