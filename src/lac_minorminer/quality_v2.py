"""Self-contained quality-first candidate scoring for the deployable search path."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from ._options import integer_option
from .scoring import ResourceScorer
from .session import SearchSession


QUALITY_V2_FEATURE_NAMES = (
    "candidate_chain_length",
    "exact_total_qubits",
    "result_max_chain_length",
    "result_used_target_nodes",
    "native_max_occupancy",
    "native_total_excess_occupancy",
    "native_route_cost",
    "source_degree",
    "abs_focus_field",
    "mean_abs_incident_coupling",
    "sum_abs_incident_coupling",
    "weighted_contact_fraction",
    "chain_length_coupling_scale",
    "candidate_internal_edges",
    "candidate_cycle_surplus",
    "min_neighbor_contacts",
    "mean_neighbor_contacts",
    "total_neighbor_contacts",
    "free_boundary_edges",
)

_CHECKPOINT_FORMAT = "isingfold-quality-v2-linear"
_CHECKPOINT_VERSION = 1


def _coefficient(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"quality V2 {name} coefficients must be finite real numbers")
    return float(value)


@dataclass(frozen=True, slots=True)
class QualityV2Problem:
    """Compact Ising coefficients required to predict downstream solution quality."""

    logical_fields: tuple[float, ...]
    logical_couplings: tuple[tuple[int, int, float], ...]

    def __post_init__(self) -> None:
        fields = tuple(_coefficient(value, "linear") for value in self.logical_fields)
        couplings = []
        seen_edges = set()
        for raw in self.logical_couplings:
            if not isinstance(raw, tuple) or len(raw) != 3:
                raise ValueError("quality V2 compact quadratic entries must contain u, v, J")
            first, second, value = raw
            if (
                isinstance(first, bool)
                or isinstance(second, bool)
                or not isinstance(first, Integral)
                or not isinstance(second, Integral)
                or int(first) < 0
                or int(first) >= int(second)
            ):
                raise ValueError("quality V2 compact quadratic entries contain an invalid edge")
            edge = (int(first), int(second))
            if edge in seen_edges:
                raise ValueError("quality V2 compact quadratic entries contain duplicate edges")
            seen_edges.add(edge)
            couplings.append((edge[0], edge[1], _coefficient(value, "quadratic")))
        object.__setattr__(self, "logical_fields", fields)
        object.__setattr__(self, "logical_couplings", tuple(couplings))

    @classmethod
    def from_mappings(
        cls,
        session: SearchSession,
        *,
        linear: Mapping[Any, Real],
        quadratic: Mapping[tuple[Any, Any], Real],
    ) -> QualityV2Problem:
        if not isinstance(session, SearchSession):
            raise TypeError("session must be a SearchSession")
        if not isinstance(linear, Mapping) or set(linear) != set(session.source_labels):
            raise ValueError("quality V2 linear coefficients must cover every source variable")
        fields = tuple(_coefficient(linear[label], "linear") for label in session.source_labels)

        if not isinstance(quadratic, Mapping):
            raise ValueError("quality V2 quadratic coefficients must be an edge mapping")
        couplings: dict[tuple[int, int], float] = {}
        for raw_edge, value in quadratic.items():
            if not isinstance(raw_edge, tuple) or len(raw_edge) != 2:
                raise ValueError("quality V2 quadratic keys must be source-edge pairs")
            left_label, right_label = raw_edge
            try:
                left = session.normalized_source.index(left_label)
                right = session.normalized_source.index(right_label)
            except ValueError as error:
                raise ValueError(
                    "quality V2 quadratic coefficients contain an unknown source variable"
                ) from error
            edge = (min(left, right), max(left, right))
            if left == right or edge in couplings:
                raise ValueError("quality V2 quadratic coefficients contain an invalid edge")
            couplings[edge] = _coefficient(value, "quadratic")
        if set(couplings) != set(session.normalized_source.edges):
            raise ValueError(
                "quality V2 quadratic coefficients must cover exactly the source edges"
            )
        return cls(
            logical_fields=fields,
            logical_couplings=tuple(
                (first, second, couplings[(first, second)])
                for first, second in session.normalized_source.edges
            ),
        )


@dataclass(frozen=True, slots=True)
class QualityV2BatchInput:
    """Label-free state and candidate input passed to an inference backend."""

    generation: int
    logical_id: int
    source_size: int
    target_size: int
    logical_fields: tuple[float, ...]
    logical_couplings: tuple[tuple[int, int, float], ...]
    source_edges: tuple[tuple[int, int], ...]
    target_edges: tuple[tuple[int, int], ...]
    state_chains: tuple[tuple[int, ...], ...]
    candidate_ids: tuple[int, ...]
    candidate_chains: tuple[tuple[int, ...], ...]
    feature_names: tuple[str, ...]
    dense_features: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class QualityV2Predictions:
    """Per-candidate output contract for any V2 inference backend."""

    quality_logit: tuple[float, ...]
    quality_log_concentration: tuple[float, ...]


class QualityV2Predictor(Protocol):
    def predict(self, model_input: QualityV2BatchInput) -> QualityV2Predictions: ...


class QualityV2InferenceError(RuntimeError):
    """A recoverable inference-backend failure eligible for native fallback."""


@dataclass(frozen=True, slots=True)
class QualityV2CandidateAssessment:
    candidate_id: int
    exact_total_qubits: int
    exact_valid: bool
    eligible: bool
    quality_mean: float | None
    quality_concentration: float | None
    statistic: float | None


@dataclass(frozen=True, slots=True)
class QualityV2Decision:
    candidate_index: int | None
    candidate_id: int | None
    used_fallback: bool
    fallback_reason: str | None
    assessments: tuple[QualityV2CandidateAssessment, ...]


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"quality V2 checkpoint {name} must be finite")
    return float(value)


def _finite_tuple(name: str, values: Any, expected: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"quality V2 checkpoint {name} must contain {expected} finite values")
    result = tuple(_finite_number(name, value) for value in values)
    if len(result) != expected:
        raise ValueError(f"quality V2 checkpoint {name} must contain {expected} finite values")
    return result


@dataclass(frozen=True, slots=True)
class LinearQualityV2Checkpoint:
    """Auditable JSON checkpoint for the portable linear V2 inference baseline."""

    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    quality_weights: tuple[float, ...]
    quality_bias: float
    concentration_weights: tuple[float, ...]
    concentration_bias: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        feature_names = tuple(self.feature_names)
        if feature_names != QUALITY_V2_FEATURE_NAMES:
            raise ValueError("quality V2 checkpoint feature contract does not match this runtime")
        feature_count = len(feature_names)
        feature_mean = _finite_tuple("feature_mean", self.feature_mean, feature_count)
        feature_scale = _finite_tuple("feature_scale", self.feature_scale, feature_count)
        if any(value <= 0.0 for value in feature_scale):
            raise ValueError("quality V2 checkpoint feature_scale values must be positive")
        quality_weights = _finite_tuple("quality_weights", self.quality_weights, feature_count)
        concentration_weights = _finite_tuple(
            "concentration_weights", self.concentration_weights, feature_count
        )
        quality_bias = _finite_number("quality_bias", self.quality_bias)
        concentration_bias = _finite_number("concentration_bias", self.concentration_bias)
        if not isinstance(self.metadata, Mapping):
            raise ValueError("quality V2 checkpoint metadata must be a JSON object")
        try:
            metadata = json.loads(json.dumps(dict(self.metadata), allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("quality V2 checkpoint metadata must be JSON serializable") from error
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "feature_mean", feature_mean)
        object.__setattr__(self, "feature_scale", feature_scale)
        object.__setattr__(self, "quality_weights", quality_weights)
        object.__setattr__(self, "quality_bias", quality_bias)
        object.__setattr__(self, "concentration_weights", concentration_weights)
        object.__setattr__(self, "concentration_bias", concentration_bias)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def predict(self, model_input: QualityV2BatchInput) -> QualityV2Predictions:
        if model_input.feature_names != self.feature_names:
            raise ValueError("quality V2 model input uses an incompatible feature contract")
        quality_logits = []
        concentration_logits = []
        for row in model_input.dense_features:
            if len(row) != len(self.feature_names):
                raise ValueError("quality V2 model input has an invalid feature width")
            normalized = tuple(
                (float(value) - mean) / scale
                for value, mean, scale in zip(row, self.feature_mean, self.feature_scale)
            )
            quality_logits.append(
                self.quality_bias
                + sum(value * weight for value, weight in zip(normalized, self.quality_weights))
            )
            concentration_logits.append(
                self.concentration_bias
                + sum(
                    value * weight for value, weight in zip(normalized, self.concentration_weights)
                )
            )
        return QualityV2Predictions(tuple(quality_logits), tuple(concentration_logits))

    def save(self, path: str | Path) -> None:
        payload = {
            "format": _CHECKPOINT_FORMAT,
            "format_version": _CHECKPOINT_VERSION,
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "quality_weights": list(self.quality_weights),
            "quality_bias": self.quality_bias,
            "concentration_weights": list(self.concentration_weights),
            "concentration_bias": self.concentration_bias,
            "metadata": dict(self.metadata),
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def load_quality_v2_checkpoint(path: str | Path) -> LinearQualityV2Checkpoint:
    """Load the portable JSON format without importing a training workspace."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("quality V2 checkpoint is not readable JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("quality V2 checkpoint root must be an object")
    required = {
        "format",
        "format_version",
        "feature_names",
        "feature_mean",
        "feature_scale",
        "quality_weights",
        "quality_bias",
        "concentration_weights",
        "concentration_bias",
    }
    allowed = required | {"metadata"}
    if set(payload) - allowed or required - set(payload):
        raise ValueError("quality V2 checkpoint fields do not match the registered format")
    if payload["format"] != _CHECKPOINT_FORMAT or payload["format_version"] != _CHECKPOINT_VERSION:
        raise ValueError("quality V2 checkpoint format is unsupported")
    feature_names = payload["feature_names"]
    if isinstance(feature_names, (str, bytes)) or not isinstance(feature_names, list):
        raise ValueError("quality V2 checkpoint feature_names must be a list")
    return LinearQualityV2Checkpoint(
        feature_names=tuple(feature_names),
        feature_mean=payload["feature_mean"],
        feature_scale=payload["feature_scale"],
        quality_weights=payload["quality_weights"],
        quality_bias=payload["quality_bias"],
        concentration_weights=payload["concentration_weights"],
        concentration_bias=payload["concentration_bias"],
        metadata=payload.get("metadata", {}),
    )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


class QualityV2Scorer:
    """Rank quality predictions after exact validity and qubit-budget masking."""

    def __init__(
        self,
        *,
        session: SearchSession,
        problem: QualityV2Problem,
        predictor: QualityV2Predictor,
        total_qubit_budget: int,
        statistic: str = "mean",
        lcb_z: float = 1.0,
    ) -> None:
        if not isinstance(session, SearchSession):
            raise TypeError("session must be a SearchSession")
        if not isinstance(problem, QualityV2Problem):
            raise TypeError("problem must be a QualityV2Problem")
        if (
            len(problem.logical_fields) != len(session.source_labels)
            or tuple((first, second) for first, second, _ in problem.logical_couplings)
            != session.normalized_source.edges
        ):
            raise ValueError("problem does not match the quality V2 session")
        self._session = session
        self._problem = problem
        self._predictor = predictor
        self._budget = integer_option("total_qubit_budget", total_qubit_budget, minimum=0)
        if statistic not in {"mean", "lcb"}:
            raise ValueError("statistic must be 'mean' or 'lcb'")
        self._statistic = statistic
        if (
            isinstance(lcb_z, bool)
            or not isinstance(lcb_z, Real)
            or not math.isfinite(float(lcb_z))
            or float(lcb_z) < 0.0
        ):
            raise ValueError("lcb_z must be a finite non-negative number")
        self._lcb_z = float(lcb_z)
        self._source_adjacency = [set() for _ in session.source_labels]
        for first, second in session.normalized_source.edges:
            self._source_adjacency[first].add(second)
            self._source_adjacency[second].add(first)
        self._target_adjacency = [set() for _ in session.target_labels]
        for first, second in session.normalized_target.edges:
            self._target_adjacency[first].add(second)
            self._target_adjacency[second].add(first)
        self._couplings = {
            (first, second): value for first, second, value in problem.logical_couplings
        }
        self._last_batch: Any | None = None
        self._last_decision: QualityV2Decision | None = None

    @property
    def total_qubit_budget(self) -> int:
        return self._budget

    @property
    def last_decision(self) -> QualityV2Decision:
        if self._last_decision is None:
            raise RuntimeError("quality V2 scorer has not evaluated a candidate batch")
        return self._last_decision

    def _trial_chains(self, snapshot: Any, logical: int, candidate: Any) -> list[list[int]]:
        chains = [list(chain) for chain in snapshot.chains]
        if len(chains) != len(self._session.source_labels):
            raise ValueError("snapshot does not match the quality V2 session")
        chains[logical] = list(candidate.chain)
        return chains

    def _feature_row(
        self, logical: int, candidate: Any, chains: list[list[int]]
    ) -> tuple[float, ...]:
        candidate_chain = set(candidate.chain)
        used_nodes = {target for chain in chains for target in chain}
        exact_total_qubits = sum(len(chain) for chain in chains)
        internal_edges = (
            sum(
                neighbor in candidate_chain
                for target in candidate_chain
                for neighbor in self._target_adjacency[target]
            )
            // 2
        )
        contact_counts = []
        logical_neighbors = sorted(self._source_adjacency[logical])
        for neighbor in logical_neighbors:
            neighbor_chain = set(chains[neighbor])
            contact_counts.append(
                sum(
                    target_neighbor in neighbor_chain
                    for target in candidate_chain
                    for target_neighbor in self._target_adjacency[target]
                )
            )
        total_contacts = sum(contact_counts)
        mean_contacts = total_contacts / len(contact_counts) if contact_counts else 0.0
        min_contacts = min(contact_counts, default=0)
        incident_couplings = [
            abs(self._couplings[(min(logical, neighbor), max(logical, neighbor))])
            for neighbor in logical_neighbors
        ]
        coupling_sum = sum(incident_couplings)
        coupling_mean = coupling_sum / len(incident_couplings) if incident_couplings else 0.0
        weighted_contacts = (
            sum(
                coupling * min(contact, 3)
                for coupling, contact in zip(incident_couplings, contact_counts)
            )
            / coupling_sum
            if coupling_sum
            else 0.0
        )
        maximum_coupling = max(incident_couplings, default=0.0)
        free_boundary = sum(
            target_neighbor not in used_nodes
            for target in candidate_chain
            for target_neighbor in self._target_adjacency[target]
        )
        rank = candidate.rank
        return (
            float(len(candidate.chain)),
            float(exact_total_qubits),
            float(max(map(len, chains), default=0)),
            float(len(used_nodes)),
            float(rank.max_occupancy),
            float(rank.total_excess_occupancy),
            float(rank.route_cost),
            float(len(self._source_adjacency[logical])),
            abs(self._problem.logical_fields[logical]),
            float(coupling_mean),
            float(coupling_sum),
            float(weighted_contacts),
            float(max(0, len(candidate.chain) - 1) * maximum_coupling),
            float(internal_edges),
            float(max(0, internal_edges - len(candidate_chain) + 1)),
            float(min_contacts),
            float(mean_contacts),
            float(total_contacts),
            float(free_boundary),
        )

    def _prepare(self, snapshot: Any, candidates: Any):
        current = self._session.snapshot()
        if (
            int(candidates.session_id) != self._session.session_id
            or int(candidates.generation) != int(current.generation)
            or int(snapshot.generation) != int(current.generation)
            or tuple(tuple(chain) for chain in snapshot.chains)
            != tuple(tuple(chain) for chain in current.chains)
        ):
            raise ValueError("candidate batch or snapshot belongs to another quality V2 session")
        logical = candidates.logical
        if not 0 <= logical < len(self._session.source_labels):
            raise ValueError("candidate batch does not match the quality V2 session")
        trial_chains = [
            self._trial_chains(snapshot, logical, candidate) for candidate in candidates.candidates
        ]
        total_qubits = tuple(sum(len(chain) for chain in chains) for chains in trial_chains)
        exact_valid = tuple(
            self._session.candidate_is_valid(candidates, index)
            for index in range(len(candidates.candidates))
        )
        eligible = tuple(
            valid and qubits <= self._budget for valid, qubits in zip(exact_valid, total_qubits)
        )
        model_input = QualityV2BatchInput(
            generation=int(snapshot.generation),
            logical_id=int(logical),
            source_size=len(self._session.source_labels),
            target_size=len(self._session.target_labels),
            logical_fields=self._problem.logical_fields,
            logical_couplings=self._problem.logical_couplings,
            source_edges=self._session.normalized_source.edges,
            target_edges=self._session.normalized_target.edges,
            state_chains=tuple(tuple(chain) for chain in snapshot.chains),
            candidate_ids=tuple(int(candidate.candidate_id) for candidate in candidates.candidates),
            candidate_chains=tuple(
                tuple(int(target) for target in candidate.chain)
                for candidate in candidates.candidates
            ),
            feature_names=QUALITY_V2_FEATURE_NAMES,
            dense_features=tuple(
                self._feature_row(logical, candidate, chains)
                for candidate, chains in zip(candidates.candidates, trial_chains)
            ),
        )
        return model_input, total_qubits, exact_valid, eligible

    @staticmethod
    def _validated_predictions(
        predictions: Any, candidate_count: int
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if not isinstance(predictions, QualityV2Predictions):
            raise ValueError("predictor must return QualityV2Predictions")
        logits = _finite_tuple("quality_logit", predictions.quality_logit, candidate_count)
        concentration = _finite_tuple(
            "quality_log_concentration",
            predictions.quality_log_concentration,
            candidate_count,
        )
        return logits, concentration

    @staticmethod
    def _resource_choice(
        resource_scores: Sequence[float], eligible: tuple[bool, ...]
    ) -> int | None:
        indices = [index for index, allowed in enumerate(eligible) if allowed]
        if not indices:
            return None
        return min(indices, key=lambda index: (resource_scores[index], index))

    def score(self, snapshot: Any, candidates: Any) -> list[float]:
        model_input, total_qubits, exact_valid, eligible = self._prepare(snapshot, candidates)
        candidate_count = len(candidates.candidates)
        resource_scores = ResourceScorer().score(snapshot, candidates)
        quality_means: tuple[float | None, ...] = (None,) * candidate_count
        concentrations: tuple[float | None, ...] = (None,) * candidate_count
        statistics: tuple[float | None, ...] = (None,) * candidate_count
        used_fallback = False
        fallback_reason = None

        if not any(eligible):
            choice = None
            fallback_reason = "no_exact_feasible_candidate"
        else:
            try:
                predictions = self._predictor.predict(model_input)
                logits, concentration_logits = self._validated_predictions(
                    predictions, candidate_count
                )
                quality_means = tuple(_sigmoid(value) for value in logits)
                concentrations = tuple(
                    1.0 + 99.0 * _sigmoid(value) for value in concentration_logits
                )
                if self._statistic == "mean":
                    statistics = quality_means
                else:
                    statistics = tuple(
                        mean - self._lcb_z * math.sqrt(mean * (1.0 - mean) / (concentration + 1.0))
                        for mean, concentration in zip(quality_means, concentrations)
                    )
                choice = max(
                    (index for index, allowed in enumerate(eligible) if allowed),
                    key=lambda index: (statistics[index], -index),
                )
            except QualityV2InferenceError as error:
                choice = self._resource_choice(resource_scores, eligible)
                used_fallback = True
                fallback_reason = f"inference_error:{error}"

        assessments = tuple(
            QualityV2CandidateAssessment(
                candidate_id=int(candidate.candidate_id),
                exact_total_qubits=total_qubits[index],
                exact_valid=exact_valid[index],
                eligible=eligible[index],
                quality_mean=quality_means[index],
                quality_concentration=concentrations[index],
                statistic=statistics[index],
            )
            for index, candidate in enumerate(candidates.candidates)
        )
        decision = QualityV2Decision(
            candidate_index=choice,
            candidate_id=(
                int(candidates.candidates[choice].candidate_id) if choice is not None else None
            ),
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
            assessments=assessments,
        )
        self._last_batch = candidates
        self._last_decision = decision

        if choice is None:
            return resource_scores
        if used_fallback:
            eligible_scores = [
                resource_scores[index] for index, allowed in enumerate(eligible) if allowed
            ]
            sentinel = max(eligible_scores) + 1.0
            return [
                score if eligible[index] else sentinel + score
                for index, score in enumerate(resource_scores)
            ]
        neural_scores = [
            -float(statistic) if allowed else 0.0
            for statistic, allowed in zip(statistics, eligible)
        ]
        sentinel = (
            max(neural_scores[index] for index, allowed in enumerate(eligible) if allowed) + 1.0
        )
        return [
            neural_scores[index] if allowed else sentinel + resource_scores[index]
            for index, allowed in enumerate(eligible)
        ]

    def choice_for(self, candidates: Any) -> int | None:
        if candidates is not self._last_batch or self._last_decision is None:
            raise ValueError("quality V2 acceptance requires the scorer result for this batch")
        return self._last_decision.candidate_index

    def validate_acceptance_policy(self, policy: Any) -> None:
        if not isinstance(policy, QualityV2AcceptancePolicy) or policy.scorer is not self:
            raise ValueError(
                "QualityV2Scorer requires a QualityV2AcceptancePolicy bound to the same scorer"
            )

    def policy_audit(self) -> tuple[str, str | None]:
        decision = self.last_decision
        if decision.candidate_index is None:
            return "no_selection", decision.fallback_reason
        if decision.used_fallback:
            return "native_fallback", decision.fallback_reason
        return "learned", None


class QualityV2AcceptancePolicy:
    """Apply the scorer's exact-masked decision, including explicit no-selection."""

    def __init__(self, scorer: QualityV2Scorer) -> None:
        if not isinstance(scorer, QualityV2Scorer):
            raise TypeError("scorer must be a QualityV2Scorer")
        self._scorer = scorer

    @property
    def scorer(self) -> QualityV2Scorer:
        return self._scorer

    def choose(self, scores: Sequence[float], candidates: Any) -> int | None:
        if len(scores) != len(candidates.candidates):
            raise ValueError("quality V2 acceptance received mismatched scores")
        return self._scorer.choice_for(candidates)
