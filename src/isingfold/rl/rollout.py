"""Complete-episode rollout storage and the detached targets PPO consumes.

Spec: MODEL_SPEC sections 5.3-5.4 and 7.3. Semantic observations and the exact candidate
support are stored so the trainable trunk can be recomputed; candidates are never
regenerated during an update, because a fresh generator call would silently change the
denominator an action id refers to.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, fields
from typing import Hashable, Mapping, Sequence

import numpy as np

from isingfold.rl.contracts import (
    OPCODES,
    WORK_FIELDS,
    Candidate,
    TerminalReason,
    candidate_support_key,
    stable_digest,
)
from isingfold.rl.tensorize import Observation

ROLLOUT_REPLAY_SCHEMA = "isingfold.rollout-replay"
ROLLOUT_REPLAY_SCHEMA_VERSION = 1
STOCHASTIC_COLLECTION_RULE = "categorical-temperature-one-v1"
GREEDY_COLLECTION_RULE = "greedy-evaluation-only-v1"
COLLECTION_SCHEDULE_SCHEMA = "isingfold.ppo-collection-schedule"
COLLECTION_SCHEDULE_SCHEMA_VERSION = 1
COLLECTION_SCHEDULE = "synchronous-active-episode-waves-per-episode-pcg64-v1"
UTILITY_ADVANTAGE_TRANSFORM_SCHEMA = "isingfold.utility-advantage-transform"
UTILITY_ADVANTAGE_TRANSFORM_SCHEMA_VERSION = 1
UTILITY_ADVANTAGE_TRANSFORM = (
    "full-rollout-policy-controllable-weighted-mean-center-no-whitening-v1"
)

_ACTION_RELATION_FIELDS = (
    "actions",
    "factors",
    "routes",
    "archive",
    "edge_use_hardware",
    "edge_use_logical",
    "index_factor_action",
    "index_factor_logical",
    "index_factor_membership",
    "index_factor_archive",
    "index_route_action",
    "index_route_logical",
    "index_route_positions",
    "index_action_archive",
    "index_action_conflicts",
    "index_edge_use_factor",
    "index_edge_use_hardware",
    "index_edge_use_logical",
    "index_factor_realized_edge_use",
    "factor_roles",
    "route_roles",
    "edge_use_roles",
    "legal_mask",
    "real_action_mask",
)


@dataclass(frozen=True)
class UtilityAdvantageDiagnostics:
    """Unit-preserving signal diagnostics for one complete behavior rollout."""

    transitions: int
    positive_weight_transitions: int
    policy_controllable_transitions: int
    singleton_transitions: int
    policy_controllable_weight: float
    raw_mean: float
    raw_rms_about_mean: float
    center: float
    centered_mean: float
    policy_signal_rms: float
    policy_signal_span: float

    def is_degenerate(self, tolerance: float) -> bool:
        """Whether the actor has no resolvable utility contrast in objective units."""

        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "utility advantage degeneracy tolerance must be finite and nonnegative"
            )
        return (
            self.policy_controllable_transitions == 0
            or self.policy_controllable_weight <= 0.0
            or self.policy_signal_rms <= tolerance
        )

    def as_log_dict(self, tolerance: float) -> dict[str, float]:
        """Return finite scalar diagnostics suitable for run histories and receipts."""

        denominator = abs(self.center) + self.policy_signal_rms
        signal_fraction = self.policy_signal_rms / denominator if denominator > 0.0 else 0.0
        return {
            "utility_advantage_transform_version": float(
                UTILITY_ADVANTAGE_TRANSFORM_SCHEMA_VERSION
            ),
            "utility_advantage_raw_mean": self.raw_mean,
            "utility_advantage_raw_rms_about_mean": self.raw_rms_about_mean,
            "utility_advantage_center": self.center,
            "utility_advantage_centered_mean": self.centered_mean,
            "utility_advantage_policy_signal_rms": self.policy_signal_rms,
            "utility_advantage_policy_signal_span": self.policy_signal_span,
            "utility_advantage_signal_fraction": signal_fraction,
            "utility_advantage_policy_controllable_transitions": float(
                self.policy_controllable_transitions
            ),
            "utility_advantage_singleton_transitions": float(self.singleton_transitions),
            "utility_advantage_degeneracy_tolerance": tolerance,
            "utility_advantage_degenerate": float(self.is_degenerate(tolerance)),
        }


@dataclass(frozen=True)
class CenteredUtilityAdvantages:
    """Detached full-rollout actor advantages and their registered diagnostics."""

    values: np.ndarray
    diagnostics: UtilityAdvantageDiagnostics


def center_full_rollout_utility_advantages(
    advantages: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray,
    policy_controllable: Sequence[bool] | np.ndarray,
) -> CenteredUtilityAdvantages:
    """Subtract one actor-relevant full-rollout mean without changing utility scale.

    The center is computed once across every positive-weight transition whose exact legal
    support has at least two actions.  Singleton decisions have identically zero policy
    gradient, so allowing them to set the baseline would inject an uncontrollable offset into
    the useful rows.  Episode/stratum weights define the same empirical measure later used by
    the episode-sum PPO reduction.  There is deliberately no division by a standard deviation.
    """

    raw = np.asarray(advantages, dtype=np.float64)
    actor_weights = np.asarray(weights, dtype=np.float64)
    controllable = np.asarray(policy_controllable)
    if raw.ndim != 1 or actor_weights.ndim != 1 or controllable.ndim != 1:
        raise ValueError("utility advantages, weights and controllability must be vectors")
    if raw.shape != actor_weights.shape or raw.shape != controllable.shape:
        raise ValueError("utility advantages, weights and controllability must have equal shape")
    if controllable.dtype != np.bool_:
        raise ValueError("policy controllability must be a Boolean vector")
    if not np.isfinite(raw).all() or not np.isfinite(actor_weights).all():
        raise ValueError("utility advantages and weights must be finite")
    if np.any(actor_weights < 0.0):
        raise ValueError("utility advantage weights must be nonnegative")

    positive = actor_weights > 0.0
    actor_rows = np.logical_and(positive, controllable)
    try:
        positive_weight = math.fsum(float(value) for value in actor_weights[positive])
        actor_weight = math.fsum(float(value) for value in actor_weights[actor_rows])
    except OverflowError as exc:
        raise ValueError("utility advantage aggregate weight must be finite") from exc
    if not math.isfinite(positive_weight) or not math.isfinite(actor_weight):
        raise ValueError("utility advantage aggregate weight must be finite")

    if actor_rows.any():
        center = float(np.average(raw[actor_rows], weights=actor_weights[actor_rows]))
    elif positive.any():
        # This value affects no policy gradient because every positive-weight row is a
        # singleton.  Keeping the full-rollout mean makes diagnostics stable and finite.
        center = float(np.average(raw[positive], weights=actor_weights[positive]))
    else:
        center = 0.0
    centered = raw - center

    if positive.any():
        raw_mean = float(np.average(raw[positive], weights=actor_weights[positive]))
        raw_rms = float(
            np.sqrt(
                np.average(
                    np.square(raw[positive] - raw_mean),
                    weights=actor_weights[positive],
                )
            )
        )
    else:
        raw_mean = 0.0
        raw_rms = 0.0
    if actor_rows.any():
        centered_mean = float(np.average(centered[actor_rows], weights=actor_weights[actor_rows]))
        policy_signal_rms = float(
            np.sqrt(np.average(np.square(centered[actor_rows]), weights=actor_weights[actor_rows]))
        )
        policy_signal_span = float(np.ptp(centered[actor_rows]))
    else:
        centered_mean = 0.0
        policy_signal_rms = 0.0
        policy_signal_span = 0.0

    diagnostics = UtilityAdvantageDiagnostics(
        transitions=int(raw.size),
        positive_weight_transitions=int(np.count_nonzero(positive)),
        policy_controllable_transitions=int(np.count_nonzero(actor_rows)),
        singleton_transitions=int(np.count_nonzero(~controllable)),
        policy_controllable_weight=actor_weight,
        raw_mean=raw_mean,
        raw_rms_about_mean=raw_rms,
        center=center,
        centered_mean=centered_mean,
        policy_signal_rms=policy_signal_rms,
        policy_signal_span=policy_signal_span,
    )
    return CenteredUtilityAdvantages(values=centered, diagnostics=diagnostics)


def _typed_identity(value: Hashable) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}:{value!r}"


def _chains_payload(chains: Mapping[Hashable, Sequence[Hashable]]) -> list[dict[str, object]]:
    return [
        {
            "logical": _typed_identity(node),
            "chain": sorted(_typed_identity(qubit) for qubit in chain),
        }
        for node, chain in sorted(chains.items(), key=lambda item: _typed_identity(item[0]))
    ]


def candidate_replay_payload(candidate: Candidate) -> dict[str, object]:
    """Canonical bound-action payload retained for PPO, excluding only opaque native handles."""

    return {
        "opcode": candidate.opcode.value,
        "affected": [_typed_identity(node) for node in candidate.affected],
        "old_chains": _chains_payload(candidate.old_chains),
        "new_chains": _chains_payload(candidate.new_chains),
        "routes": [
            {
                "path": [_typed_identity(qubit) for qubit in path],
                "owner": _typed_identity(owner),
            }
            for path, owner in candidate.routes
        ],
        "application_work_bound": candidate.work.as_dict(),
        "proposal_work": candidate.proposal_work.as_dict(),
        "payload_key": candidate.payload_key,
        "archive_ref": candidate.archive_ref,
        "target_demand": (
            None
            if candidate.target_demand is None
            else [_typed_identity(node) for node in candidate.target_demand]
        ),
        "target_conflict": (
            None
            if candidate.target_conflict is None
            else _typed_identity(candidate.target_conflict)
        ),
        "restart_semantics": {
            "resets_workspace_memory": candidate.opcode.value == "RESTART",
            "restart_token_delta": -1 if candidate.opcode.value == "RESTART" else 0,
        },
        "changes_workspace": candidate.changes_workspace,
        "provenance": candidate.provenance,
    }


def snapshot_candidates(candidates: Sequence[Candidate]) -> tuple[Candidate, ...]:
    """Detach bound semantic payloads from a live environment without copying native branches."""

    return tuple(
        Candidate(
            opcode=candidate.opcode,
            affected=tuple(candidate.affected),
            old_chains={node: frozenset(chain) for node, chain in candidate.old_chains.items()},
            new_chains={node: frozenset(chain) for node, chain in candidate.new_chains.items()},
            work=candidate.work,
            payload_key=candidate.payload_key,
            proposal_work=candidate.proposal_work,
            routes=tuple((tuple(path), owner) for path, owner in candidate.routes),
            archive_ref=candidate.archive_ref,
            target_demand=candidate.target_demand,
            target_conflict=candidate.target_conflict,
            restart_cache_slot=candidate.restart_cache_slot,
            restart_cache_after_digest=candidate.restart_cache_after_digest,
            provenance=candidate.provenance,
            branch=None,
        )
        for candidate in candidates
    )


def _array_payload(array: np.ndarray) -> dict[str, object]:
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise TypeError("object arrays cannot enter a PPO replay receipt")
    contiguous = np.ascontiguousarray(value)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _observation_payload(
    observation: Observation,
    names: Sequence[str] | None = None,
) -> dict[str, object]:
    if not isinstance(observation, Observation):
        raise TypeError("PPO replay requires a normative Observation")
    selected = {item.name for item in fields(observation)} if names is None else set(names)
    payload: dict[str, object] = {}
    for item in fields(observation):
        if item.name not in selected:
            continue
        value = getattr(observation, item.name)
        if isinstance(value, np.ndarray):
            payload[item.name] = _array_payload(value)
        elif item.name in {"qubit_ids", "logical_ids"}:
            payload[item.name] = [_typed_identity(identity) for identity in value]
        else:
            raise TypeError(f"unsupported Observation replay field {item.name!r}")
    if set(payload) != selected:
        missing = sorted(selected - set(payload))
        raise TypeError(f"unknown Observation replay fields: {missing}")
    return payload


def observation_replay_digest(observation: Observation) -> str:
    """Digest every semantic tensor, incidence array, mask and typed graph ID."""

    return stable_digest(
        {
            "domain": "isingfold-rollout-observation-v1",
            "fields": _observation_payload(observation),
        }
    )


def action_relations_replay_digest(observation: Observation) -> str:
    """Bind action descriptors to all factor, route, archive and edge-use incidences."""

    return stable_digest(
        {
            "domain": "isingfold-rollout-action-relations-v1",
            "fields": _observation_payload(observation, _ACTION_RELATION_FIELDS),
        }
    )


def _work_payload(work: Mapping[str, int] | None) -> dict[str, int]:
    if not isinstance(work, Mapping) or set(work) != set(WORK_FIELDS):
        raise RuntimeError("PPO replay charged-work receipt has the wrong fields")
    result: dict[str, int] = {}
    for name in WORK_FIELDS:
        value = work[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError("PPO replay charged-work receipt must be nonnegative integers")
        result[name] = value
    return result


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class RolloutReplayReceipt:
    """Versioned integrity envelope for one already-materialized PPO decision."""

    schema: str
    schema_version: int
    collection_rule: str
    mode: str
    state_fingerprint: str
    context_version: str
    support_fingerprint: str
    support_payload_digest: str
    observation_digest: str
    action_relations_digest: str
    legal_mask: tuple[bool, ...]
    chosen_index: int
    selected_payload_key: str
    selected_candidate_payload: Mapping[str, object]
    charged_work_receipt: Mapping[str, int]
    record_digest: str

    def unsigned_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "collection_rule": self.collection_rule,
            "mode": self.mode,
            "state_fingerprint": self.state_fingerprint,
            "context_version": self.context_version,
            "support_fingerprint": self.support_fingerprint,
            "support_payload_digest": self.support_payload_digest,
            "observation_digest": self.observation_digest,
            "action_relations_digest": self.action_relations_digest,
            "legal_mask": list(self.legal_mask),
            "chosen_index": self.chosen_index,
            "selected_payload_key": self.selected_payload_key,
            "selected_candidate_payload": self.selected_candidate_payload,
            "charged_work_receipt": self.charged_work_receipt,
        }


def make_replay_receipt(
    *,
    observation: Observation,
    candidates: Sequence[Candidate],
    legal_mask: Sequence[bool],
    chosen_index: int,
    state_fingerprint: str,
    context_version: str,
    support_fingerprint: str,
    charged_work_receipt: Mapping[str, int],
    collection_rule: str,
    mode: str,
) -> RolloutReplayReceipt:
    """Create one receipt from stored payloads; proposal generation is never invoked here."""

    mask = tuple(bool(value) for value in legal_mask)
    if len(candidates) != len(mask) or not 0 <= chosen_index < len(candidates):
        raise RuntimeError("cannot receipt a mismatched candidate support or chosen index")
    if not mask[chosen_index]:
        raise RuntimeError("cannot receipt a masked chosen action")
    computed_support = candidate_support_key(candidates, mask)
    if computed_support != support_fingerprint:
        raise RuntimeError("environment candidate support differs before rollout storage")
    support_payload = [candidate_replay_payload(candidate) for candidate in candidates]
    selected_payload = support_payload[chosen_index]
    unsigned = {
        "schema": ROLLOUT_REPLAY_SCHEMA,
        "schema_version": ROLLOUT_REPLAY_SCHEMA_VERSION,
        "collection_rule": collection_rule,
        "mode": mode,
        "state_fingerprint": state_fingerprint,
        "context_version": context_version,
        "support_fingerprint": support_fingerprint,
        "support_payload_digest": stable_digest(
            {"domain": "isingfold-rollout-candidate-support-v1", "candidates": support_payload}
        ),
        "observation_digest": observation_replay_digest(observation),
        "action_relations_digest": action_relations_replay_digest(observation),
        "legal_mask": list(mask),
        "chosen_index": chosen_index,
        "selected_payload_key": candidates[chosen_index].payload_key,
        "selected_candidate_payload": selected_payload,
        "charged_work_receipt": _work_payload(charged_work_receipt),
    }
    return RolloutReplayReceipt(
        schema=ROLLOUT_REPLAY_SCHEMA,
        schema_version=ROLLOUT_REPLAY_SCHEMA_VERSION,
        collection_rule=collection_rule,
        mode=mode,
        state_fingerprint=state_fingerprint,
        context_version=context_version,
        support_fingerprint=support_fingerprint,
        support_payload_digest=str(unsigned["support_payload_digest"]),
        observation_digest=str(unsigned["observation_digest"]),
        action_relations_digest=str(unsigned["action_relations_digest"]),
        legal_mask=mask,
        chosen_index=chosen_index,
        selected_payload_key=candidates[chosen_index].payload_key,
        selected_candidate_payload=selected_payload,
        charged_work_receipt=_work_payload(charged_work_receipt),
        record_digest=stable_digest(unsigned),
    )


@dataclass
class Transition:
    observation: Observation
    legal_mask: np.ndarray
    chosen_index: int
    old_log_prob: float
    old_log_probs: np.ndarray
    old_utility: float
    old_failure: float
    reward: float
    cost: float
    terminated: bool
    episode: int
    truncated: bool = False
    payload_key: str = ""
    state_fingerprint: str = ""
    support_fingerprint: str = ""
    context_version: str = ""
    charged_work_receipt: Mapping[str, int] | None = None
    candidates: tuple[Candidate, ...] = ()
    replay_receipt: RolloutReplayReceipt | None = None


@dataclass
class Episode:
    index: int
    transitions: list[int] = field(default_factory=list)
    terminal_reward: float = 0.0
    terminal_cost: float = 0.0
    weight: float = 1.0
    reason: TerminalReason | None = None
    returned_valid: bool = False
    qubits: int = 0
    selected_strength: float | None = None


@dataclass
class RolloutBuffer:
    """The buffer of MODEL_SPEC section 7.3; every stored array is detached by construction."""

    transitions: list[Transition] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    gae_utility: np.ndarray | None = None
    gae_failure: np.ndarray | None = None
    target_utility: np.ndarray | None = None
    target_failure: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def n_transitions(self) -> int:
        return len(self.transitions)

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    def add_episode(self) -> Episode:
        episode = Episode(index=len(self.episodes))
        self.episodes.append(episode)
        return episode

    def add(self, transition: Transition) -> int:
        self.transitions.append(transition)
        index = len(self.transitions) - 1
        self.episodes[transition.episode].transitions.append(index)
        return index

    def compute_targets(self, gae_lambda: float = 0.95) -> None:
        """GAE Eq. (20) with zero terminal bootstrap, and Monte Carlo critic targets.

        The combination is deliberate: a Monte Carlo critic with a GAE actor, in raw units,
        without per-minibatch whitening.
        """

        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1]")
        for episode in self.episodes:
            if not episode.transitions:
                continue
            rows = [self.transitions[index] for index in episode.transitions]
            if any(row.episode != episode.index for row in rows):
                raise RuntimeError("rollout episode indices are inconsistent")
            if any(row.truncated for row in rows):
                raise RuntimeError(
                    "IF-Core-v1 requires complete episodes, not collector truncation"
                )
            if any(row.terminated for row in rows[:-1]) or not rows[-1].terminated:
                raise RuntimeError(
                    "rollout contains an incomplete or malformed task-terminal episode"
                )

        n = self.n_transitions
        adv_u = np.zeros(n, dtype=np.float64)
        adv_f = np.zeros(n, dtype=np.float64)
        tgt_u = np.zeros(n, dtype=np.float64)
        tgt_f = np.zeros(n, dtype=np.float64)
        for episode in self.episodes:
            running_u = running_f = 0.0
            next_transition: Transition | None = None
            for t in reversed(episode.transitions):
                tr = self.transitions[t]
                if not tr.terminated and next_transition is None:
                    raise RuntimeError("live transition has no same-episode successor")
                next_u = 0.0 if tr.terminated else next_transition.old_utility
                next_f = 0.0 if tr.terminated else next_transition.old_failure
                delta_u = tr.reward + (0.0 if tr.terminated else next_u) - tr.old_utility
                delta_f = tr.cost + (0.0 if tr.terminated else next_f) - tr.old_failure
                running_u = delta_u + gae_lambda * (0.0 if tr.terminated else running_u)
                running_f = delta_f + gae_lambda * (0.0 if tr.terminated else running_f)
                adv_u[t] = running_u
                adv_f[t] = running_f
                tgt_u[t] = episode.terminal_reward
                tgt_f[t] = episode.terminal_cost
                next_transition = tr
        self.gae_utility, self.gae_failure = adv_u, adv_f
        self.target_utility, self.target_failure = tgt_u, tgt_f
        self.metadata.update(
            {
                "utility_advantage_transform_schema": UTILITY_ADVANTAGE_TRANSFORM_SCHEMA,
                "utility_advantage_transform_schema_version": (
                    UTILITY_ADVANTAGE_TRANSFORM_SCHEMA_VERSION
                ),
                "utility_advantage_transform": UTILITY_ADVANTAGE_TRANSFORM,
            }
        )

    def utility_advantages_for_actor(self) -> CenteredUtilityAdvantages:
        """Validate and apply the registered transform to the detached full rollout."""

        if self.gae_utility is None:
            raise RuntimeError("utility GAE was not computed before actor advantage preparation")
        if (
            self.metadata.get("utility_advantage_transform_schema")
            != UTILITY_ADVANTAGE_TRANSFORM_SCHEMA
            or self.metadata.get("utility_advantage_transform_schema_version")
            != UTILITY_ADVANTAGE_TRANSFORM_SCHEMA_VERSION
            or self.metadata.get("utility_advantage_transform") != UTILITY_ADVANTAGE_TRANSFORM
        ):
            raise RuntimeError(
                "utility advantage transform schema/version is missing or incompatible"
            )
        controllable = np.asarray(
            [
                np.count_nonzero(np.asarray(transition.legal_mask, dtype=bool)) >= 2
                for transition in self.transitions
            ],
            dtype=bool,
        )
        return center_full_rollout_utility_advantages(
            self.gae_utility,
            weights=self.transition_weights(),
            policy_controllable=controllable,
        )

    def transition_weights(self) -> np.ndarray:
        """Expand fixed episode/stratum weights without normalising by episode length."""

        weights = np.empty(self.n_transitions, dtype=np.float64)
        for episode in self.episodes:
            if not np.isfinite(episode.weight) or episode.weight < 0.0:
                raise ValueError("episode importance weights must be finite and nonnegative")
            for transition_index in episode.transitions:
                weights[transition_index] = episode.weight
        return weights

    def authenticate_collection_schedule(
        self,
        *,
        expected_training_seed: int,
        expected_update_index: int,
        expected_episode_count: int,
    ) -> tuple[int, ...]:
        """Authenticate the exact registered episode schedule before replay or update."""

        if (
            self.metadata.get("collection_schedule_schema") != COLLECTION_SCHEDULE_SCHEMA
            or self.metadata.get("collection_schedule_schema_version")
            != COLLECTION_SCHEDULE_SCHEMA_VERSION
            or self.metadata.get("collection_schedule") != COLLECTION_SCHEDULE
        ):
            raise RuntimeError("PPO collection schedule schema/version/value is incompatible")
        for value, name in (
            (expected_training_seed, "expected training seed"),
            (expected_update_index, "expected update index"),
            (expected_episode_count, "expected episode count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError(f"PPO collection schedule {name} is not an integer")
        if expected_update_index < 0 or expected_episode_count <= 0:
            raise RuntimeError("PPO collection schedule expected range is invalid")

        training_seed = self.metadata.get("collection_training_seed")
        update_index = self.metadata.get("collection_update_index")
        start = self.metadata.get("episode_schedule_start")
        stop = self.metadata.get("episode_schedule_stop_exclusive")
        for value, name in (
            (training_seed, "training seed"),
            (update_index, "update index"),
            (start, "range start"),
            (stop, "range stop"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError(f"PPO collection schedule {name} is missing or non-integral")

        expected_start = expected_update_index * expected_episode_count
        expected_stop = expected_start + expected_episode_count
        if (
            training_seed != expected_training_seed
            or update_index != expected_update_index
            or start != expected_start
            or stop != expected_stop
            or stop - start != len(self.episodes)
        ):
            raise RuntimeError("PPO collection schedule identity or range is incoherent")

        raw_indices = self.metadata.get("episode_schedule_indices")
        if not isinstance(raw_indices, (list, tuple)) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in raw_indices
        ):
            raise RuntimeError("PPO collection schedule episode identities are missing or invalid")
        indices = tuple(raw_indices)
        expected_indices = tuple(range(expected_start, expected_stop))
        if indices != expected_indices:
            raise RuntimeError("PPO collection schedule episode identities are incoherent")
        if tuple(episode.index for episode in self.episodes) != tuple(range(len(self.episodes))):
            raise RuntimeError("PPO collection schedule local episode identities are incoherent")
        return indices

    def authenticate_replay(
        self,
        *,
        expected_mode: str,
        expected_training_seed: int,
        expected_update_index: int,
        expected_episode_count: int,
        training: bool = True,
    ) -> int:
        """Authenticate saved semantic supports without rerunning a proposal generator."""

        self.authenticate_collection_schedule(
            expected_training_seed=expected_training_seed,
            expected_update_index=expected_update_index,
            expected_episode_count=expected_episode_count,
        )

        if (
            self.metadata.get("replay_receipt_schema") != ROLLOUT_REPLAY_SCHEMA
            or self.metadata.get("replay_receipt_schema_version") != ROLLOUT_REPLAY_SCHEMA_VERSION
        ):
            raise RuntimeError(
                "PPO rollout replay receipt schema/version is missing or incompatible"
            )
        collection_rule = self.metadata.get("collection_rule")
        if collection_rule not in {STOCHASTIC_COLLECTION_RULE, GREEDY_COLLECTION_RULE}:
            raise RuntimeError("PPO rollout has an unknown collection rule")
        if self.metadata.get("mode") != expected_mode:
            raise RuntimeError("PPO rollout mode differs from the trainer mode")
        eligible = self.metadata.get("training_eligible")
        if training and (eligible is not True or collection_rule != STOCHASTIC_COLLECTION_RULE):
            raise RuntimeError("greedy/evaluation-only rollout cannot enter a PPO update")

        for row, transition in enumerate(self.transitions):
            self._authenticate_transition(
                transition,
                row=row,
                expected_mode=expected_mode,
                collection_rule=str(collection_rule),
            )
        return len(self.transitions)

    @staticmethod
    def _authenticate_transition(
        transition: Transition,
        *,
        row: int,
        expected_mode: str,
        collection_rule: str,
    ) -> None:
        receipt = transition.replay_receipt
        if receipt is None:
            raise RuntimeError(f"PPO replay receipt is missing at row {row}")
        if (
            receipt.schema != ROLLOUT_REPLAY_SCHEMA
            or receipt.schema_version != ROLLOUT_REPLAY_SCHEMA_VERSION
        ):
            raise RuntimeError(f"PPO replay receipt schema/version mismatch at row {row}")
        if receipt.collection_rule != collection_rule:
            raise RuntimeError(f"PPO replay receipt collection rule mismatch at row {row}")
        if receipt.mode != expected_mode:
            raise RuntimeError(f"PPO replay receipt mode mismatch at row {row}")
        if not _is_sha256(transition.state_fingerprint):
            raise RuntimeError(f"PPO replay state fingerprint is malformed at row {row}")
        if not isinstance(transition.context_version, str) or not transition.context_version:
            raise RuntimeError(f"PPO replay context version is missing at row {row}")
        if not _is_sha256(transition.support_fingerprint):
            raise RuntimeError(f"PPO replay support fingerprint is malformed at row {row}")

        legal = np.asarray(transition.legal_mask)
        if legal.dtype != np.bool_ or legal.ndim != 1:
            raise RuntimeError(f"PPO replay legal mask is not a Boolean vector at row {row}")
        exact_mask = tuple(bool(value) for value in legal)
        candidates = transition.candidates
        if len(candidates) != len(exact_mask):
            raise RuntimeError(f"PPO replay candidate support length mismatch at row {row}")
        if (
            not 0 <= transition.chosen_index < len(candidates)
            or not exact_mask[transition.chosen_index]
        ):
            raise RuntimeError(
                f"PPO replay selected action is outside the legal support at row {row}"
            )
        if len({candidate.payload_key for candidate in candidates}) != len(candidates):
            raise RuntimeError(f"PPO replay support repeats a candidate payload key at row {row}")

        observation = transition.observation
        if not isinstance(observation, Observation):
            raise RuntimeError(f"PPO replay observation has the wrong type at row {row}")
        if observation.n_actions != len(candidates):
            raise RuntimeError(f"PPO replay observation/support length mismatch at row {row}")
        if not np.array_equal(observation.legal_mask, legal):
            raise RuntimeError(f"PPO replay observation legal mask mismatch at row {row}")
        if observation.real_action_mask.shape != legal.shape or not np.all(
            observation.real_action_mask
        ):
            raise RuntimeError(f"PPO replay real-action mask mismatch at row {row}")
        opcode_rows = observation.actions[:, -len(OPCODES) :]
        expected_opcodes = np.zeros((len(candidates), len(OPCODES)), dtype=np.float32)
        for index, candidate in enumerate(candidates):
            expected_opcodes[index, candidate.opcode.index] = 1.0
        if not np.array_equal(opcode_rows, expected_opcodes):
            raise RuntimeError(f"PPO replay candidate opcode relation mismatch at row {row}")

        try:
            computed_support = candidate_support_key(candidates, exact_mask)
            work = _work_payload(transition.charged_work_receipt)
            support_payload = [candidate_replay_payload(candidate) for candidate in candidates]
            support_payload_digest = stable_digest(
                {
                    "domain": "isingfold-rollout-candidate-support-v1",
                    "candidates": support_payload,
                }
            )
            observation_digest = observation_replay_digest(observation)
            action_relations_digest = action_relations_replay_digest(observation)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"PPO replay payload cannot be authenticated at row {row}") from exc

        selected = candidates[transition.chosen_index]
        selected_payload = support_payload[transition.chosen_index]
        if transition.payload_key != selected.payload_key:
            raise RuntimeError(f"PPO replay selected payload key mismatch at row {row}")
        if computed_support != transition.support_fingerprint:
            raise RuntimeError(f"PPO replay complete support fingerprint mismatch at row {row}")
        if (
            receipt.state_fingerprint != transition.state_fingerprint
            or receipt.context_version != transition.context_version
            or receipt.support_fingerprint != transition.support_fingerprint
            or receipt.support_payload_digest != support_payload_digest
            or receipt.observation_digest != observation_digest
            or receipt.action_relations_digest != action_relations_digest
            or receipt.legal_mask != exact_mask
            or receipt.chosen_index != transition.chosen_index
            or receipt.selected_payload_key != transition.payload_key
            or receipt.selected_candidate_payload != selected_payload
            or dict(receipt.charged_work_receipt) != work
        ):
            raise RuntimeError(f"PPO replay receipt/payload mismatch at row {row}")
        if (
            not _is_sha256(receipt.record_digest)
            or stable_digest(receipt.unsigned_payload()) != receipt.record_digest
        ):
            raise RuntimeError(f"PPO replay receipt digest mismatch at row {row}")

    def summary(self) -> dict[str, float]:
        rewards = [e.terminal_reward for e in self.episodes]
        costs = [e.terminal_cost for e in self.episodes]
        lengths = [len(e.transitions) for e in self.episodes]
        valid = [1.0 if e.returned_valid else 0.0 for e in self.episodes]
        return {
            "episodes": float(len(self.episodes)),
            "transitions": float(self.n_transitions),
            "mean_return": float(np.mean(rewards)) if rewards else 0.0,
            "mean_cost": float(np.mean(costs)) if costs else 0.0,
            "valid_return_rate": float(np.mean(valid)) if valid else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        }
