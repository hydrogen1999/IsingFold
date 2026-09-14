"""Complete-episode masked PPO (MODEL_SPEC Algorithm 1).

Every decision in a batch uses one frozen behaviour snapshot; no optimizer step happens
until all episodes finish. The update recomputes the trainable trunk on the *stored*
observations, candidates and masks, so gradients reach the encoder and an action id always
means the same successor. Rewards and validity come from the environment; the learner never
rechecks workspace validity to overwrite a label.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from isingfold.rl.contracts import DecisionState, InitFailureRecord, Mode, TerminalRecord
from isingfold.rl.env import EmbeddingEnv
from isingfold.rl.model import IFCore, ModelOutput
from isingfold.rl.rollout import (
    COLLECTION_SCHEDULE,
    COLLECTION_SCHEDULE_SCHEMA,
    COLLECTION_SCHEDULE_SCHEMA_VERSION,
    GREEDY_COLLECTION_RULE,
    ROLLOUT_REPLAY_SCHEMA,
    ROLLOUT_REPLAY_SCHEMA_VERSION,
    STOCHASTIC_COLLECTION_RULE,
    UTILITY_ADVANTAGE_TRANSFORM,
    Episode,
    RolloutBuffer,
    Transition,
    make_replay_receipt,
    snapshot_candidates,
)

EnvFactory = Callable[[int, int], EmbeddingEnv]
WARM_START_RANK_COEFFICIENT = 1.0
WARM_START_ACTION_VALUE_COEFFICIENT = 1.0
WARM_START_COMMIT_DELTA_COEFFICIENT = 0.5
WARM_START_UTILITY_COEFFICIENT = 0.5
WARM_START_FULL_PROFILE_ID = "full-qmu-v4"
WARM_START_RANK_VALUE_CONTROL_PROFILE_ID = "rank-value-only-control-v1"
WARM_START_ACTOR_CRITIC_LOSS = (
    "all-action-qmu-plus-commit-delta-plus-resolved-rank-plus-state-value-v4"
)
WARM_START_REDUCTION = "exact-full-corpus-gradient-accumulation-v1"
WARM_START_ACTION_VALUE_TARGET = "bounded-frozen-continuation-qmu-v1"
WARM_START_ACTION_VALUE_WEIGHTING = "equal-state-count-over-propensity-v1"
WARM_START_COMMIT_DELTA_TARGET = "protected-commit-relative-qmu-v1"
WARM_START_UTILITY_TARGET = "uniform-legal-action-then-frozen-continuation-v1"
WARM_START_UTILITY_WEIGHTING = "bounded-variance-effective-count-v1"
PPO_UPDATE_CONTROL_SCHEMA = "isingfold.ppo-update-control"
PPO_UPDATE_CONTROL = "epoch-rollback-geometric-backtracking-v1"
PPO_UPDATE_CONTROL_SCHEMA_VERSION = 1
ENTROPY_NORMALIZATION = "legal-categorical-entropy-divided-by-log-support-v1"
ENTROPY_NORMALIZATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class WarmStartLossWeights:
    """Immutable coefficients for one registered supervised objective."""

    rank: float
    utility: float
    action_value: float
    commit_delta: float

    def __post_init__(self) -> None:
        for name in ("rank", "utility", "action_value", "commit_delta"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"warm-start loss weight {name} must be finite and nonnegative"
                )
            object.__setattr__(self, name, float(value))

    def contract(self) -> dict[str, float]:
        return {
            "rank": self.rank,
            "utility": self.utility,
            "action_value": self.action_value,
            "commit_delta": self.commit_delta,
        }


@dataclass(frozen=True, slots=True)
class WarmStartLossProfile:
    """Closed loss-profile identity used by warm training and transfer checks."""

    profile_id: str
    weights: WarmStartLossWeights
    q_mu_label_supervision: bool
    production_transfer_eligible: bool

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("warm-start loss profile ID must be nonempty text")
        if not isinstance(self.weights, WarmStartLossWeights):
            raise TypeError("warm-start loss profile weights have an unsupported type")
        if type(self.q_mu_label_supervision) is not bool:
            raise ValueError("Q-mu supervision flag must be Boolean")
        if type(self.production_transfer_eligible) is not bool:
            raise ValueError("production-transfer eligibility must be Boolean")

    def contract(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "weights": self.weights.contract(),
            "q_mu_label_supervision": self.q_mu_label_supervision,
            "production_transfer_eligible": self.production_transfer_eligible,
        }


WARM_START_FULL_LOSS_PROFILE = WarmStartLossProfile(
    profile_id=WARM_START_FULL_PROFILE_ID,
    weights=WarmStartLossWeights(
        rank=WARM_START_RANK_COEFFICIENT,
        utility=WARM_START_UTILITY_COEFFICIENT,
        action_value=WARM_START_ACTION_VALUE_COEFFICIENT,
        commit_delta=WARM_START_COMMIT_DELTA_COEFFICIENT,
    ),
    q_mu_label_supervision=True,
    production_transfer_eligible=True,
)
WARM_START_RANK_VALUE_CONTROL_PROFILE = WarmStartLossProfile(
    profile_id=WARM_START_RANK_VALUE_CONTROL_PROFILE_ID,
    weights=WarmStartLossWeights(
        rank=WARM_START_RANK_COEFFICIENT,
        utility=WARM_START_UTILITY_COEFFICIENT,
        action_value=0.0,
        commit_delta=0.0,
    ),
    q_mu_label_supervision=False,
    production_transfer_eligible=False,
)
WARM_START_LOSS_PROFILES: tuple[WarmStartLossProfile, ...] = (
    WARM_START_FULL_LOSS_PROFILE,
    WARM_START_RANK_VALUE_CONTROL_PROFILE,
)
WARM_START_LOSS_PROFILE_IDS: tuple[str, ...] = tuple(
    profile.profile_id for profile in WARM_START_LOSS_PROFILES
)


def warm_start_loss_profile(
    profile: str | WarmStartLossProfile,
) -> WarmStartLossProfile:
    """Resolve only one of the two preregistered warm-start loss profiles."""

    profile_id = profile.profile_id if isinstance(profile, WarmStartLossProfile) else profile
    if not isinstance(profile_id, str):
        raise TypeError("warm-start loss profile has an unsupported type")
    registered = {
        candidate.profile_id: candidate for candidate in WARM_START_LOSS_PROFILES
    }
    canonical = registered.get(profile_id)
    if canonical is None or (
        isinstance(profile, WarmStartLossProfile) and profile != canonical
    ):
        raise ValueError(f"unknown registered warm-start loss profile {profile_id!r}")
    return canonical


@dataclass(frozen=True)
class WarmStartUtilityTarget:
    """State-value supervision induced by the registered counterfactual draw."""

    value: float
    effective_count: float
    evaluated_actions: int
    continuation_trajectories: int
    inclusion_probability: float
    action_value: WarmStartActionValueTarget | None = None
    inclusion_probabilities: tuple[float, ...] = ()
    legal_action_count: int = 0


@dataclass(frozen=True)
class WarmStartActionValueTarget:
    """Authenticated sampled ``Q^mu(s,a)`` labels for one exact action support."""

    action_indices: tuple[int, ...]
    q_mu: tuple[float, ...]
    continuation_counts: tuple[int, ...]
    inclusion_probabilities: tuple[float, ...]
    legal_action_count: int
    commit_index: int | None = None
    support_size: int | None = None

    def __post_init__(self) -> None:
        lengths = {
            len(self.action_indices),
            len(self.q_mu),
            len(self.continuation_counts),
            len(self.inclusion_probabilities),
        }
        if lengths != {len(self.action_indices)} or not self.action_indices:
            raise ValueError("action-value target arrays must have the same nonzero length")
        if any(
            isinstance(index, bool) or not isinstance(index, (int, np.integer))
            for index in self.action_indices
        ):
            raise ValueError("action-value indices must be integers")
        if tuple(sorted(set(int(index) for index in self.action_indices))) != tuple(
            int(index) for index in self.action_indices
        ):
            raise ValueError("action-value indices must be strictly increasing")
        if (
            isinstance(self.legal_action_count, bool)
            or not isinstance(self.legal_action_count, int)
            or self.legal_action_count <= 0
            or len(self.action_indices) > self.legal_action_count
        ):
            raise ValueError("action-value legal support count must be positive")
        support_size = (
            self.legal_action_count if self.support_size is None else self.support_size
        )
        if (
            isinstance(support_size, bool)
            or not isinstance(support_size, int)
            or support_size < self.legal_action_count
            or any(index < 0 or index >= support_size for index in self.action_indices)
        ):
            raise ValueError("action-value indices must lie inside the exact legal support")
        object.__setattr__(self, "support_size", support_size)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= float(value) <= 1.0
            for value in self.q_mu
        ):
            raise ValueError("action-value Q-mu targets must be finite and bounded in [0, 1]")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in self.continuation_counts
        ):
            raise ValueError("action-value continuation counts must be positive integers")
        if any(
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0.0 < float(probability) <= 1.0
            for probability in self.inclusion_probabilities
        ):
            raise ValueError("action-value inclusion probabilities must lie in (0, 1]")
        if self.commit_index is not None and (
            isinstance(self.commit_index, bool)
            or not isinstance(self.commit_index, int)
            or self.commit_index not in self.action_indices
        ):
            raise ValueError("action-value commit index must identify an evaluated action")

    @property
    def regression_weights(self) -> tuple[float, ...]:
        """Count-aware inverse-propensity weights, normalized only inside each state."""

        return tuple(
            float(count) / float(probability)
            for count, probability in zip(
                self.continuation_counts,
                self.inclusion_probabilities,
                strict=True,
            )
        )


@dataclass(frozen=True)
class WarmStartCorpusDenominators:
    """Frozen global reductions for one authenticated warm-start corpus.

    A training minibatch is only a memory partition.  Each loss call contributes an
    additive piece of the same corpus objective, so none of these denominators may be
    recomputed from the minibatch itself.
    """

    actor_ranking_records: int
    utility_effective_count: float
    action_value_records: int
    commit_delta_records: int

    def __post_init__(self) -> None:
        for name in (
            "actor_ranking_records",
            "action_value_records",
            "commit_delta_records",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"warm-start {name} must be a nonnegative integer")
        if (
            isinstance(self.utility_effective_count, bool)
            or not isinstance(self.utility_effective_count, (int, float))
            or not math.isfinite(self.utility_effective_count)
            or float(self.utility_effective_count) <= 0.0
        ):
            raise ValueError(
                "warm-start utility_effective_count must be finite and positive"
            )
        if self.commit_delta_records > self.action_value_records:
            raise ValueError(
                "warm-start commit-delta records cannot exceed action-value records"
            )


@dataclass(frozen=True)
class WarmStartActorCriticLoss:
    """Auditable decomposition of supervised actor-critic initialization."""

    total: Tensor
    rank: Tensor
    utility: Tensor
    effective_count: float
    action_value: Tensor
    commit_delta: Tensor
    action_effective_count: float
    commit_delta_rows: int


def warm_start_utility_target(
    *,
    q_mu: Sequence[float],
    continuation_counts: Sequence[int],
    inclusion_probabilities: Sequence[float],
    action_value_target: WarmStartActionValueTarget | None = None,
    legal_action_count: int | None = None,
) -> WarmStartUtilityTarget:
    """Construct an unbiased uniform-action critic target and confidence weight.

    The production quality protocol samples actions uniformly without replacement, so
    their ``Q^mu`` means retain equal weight even when continuation counts differ.  The
    confidence rule combines bounded within-action Monte Carlo and partial-support
    sampling variance proxies.  It grows with evidence but cannot become unbounded while
    any legal actions remain unevaluated.
    """

    values = tuple(q_mu)
    counts = tuple(continuation_counts)
    probabilities = tuple(inclusion_probabilities)
    if not values or len(values) != len(counts) or len(values) != len(probabilities):
        raise ValueError("utility target arrays must have the same nonzero length")
    if action_value_target is not None and not isinstance(
        action_value_target, WarmStartActionValueTarget
    ):
        raise TypeError("utility action-value target has an unsupported type")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
        for value in values
    ):
        raise ValueError("utility Q-mu values must be finite and bounded in [0, 1]")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
        for count in counts
    ):
        raise ValueError("utility continuation counts must be positive integers")
    if any(
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0.0 < float(probability) <= 1.0
        for probability in probabilities
    ):
        raise ValueError("utility inclusion probabilities must lie in (0, 1]")
    action_count = len(values)
    common_probability = float(probabilities[0])
    common_design = all(
        math.isclose(float(probability), common_probability, rel_tol=0.0, abs_tol=1e-15)
        for probability in probabilities[1:]
    )
    if legal_action_count is None:
        if not common_design:
            raise ValueError(
                "utility labels with mixed inclusion probabilities require legal_action_count"
            )
        inferred = action_count / common_probability
        rounded = int(round(inferred))
        if rounded < action_count or not math.isclose(
            inferred, float(rounded), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("utility common inclusion probability cannot identify legal support")
        support_count = rounded
    else:
        if (
            isinstance(legal_action_count, bool)
            or not isinstance(legal_action_count, int)
            or legal_action_count < action_count
        ):
            raise ValueError("utility legal_action_count must cover every evaluated action")
        support_count = legal_action_count
    ht_weights = tuple(
        1.0 / (support_count * float(probability)) for probability in probabilities
    )
    if not math.isclose(math.fsum(ht_weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("utility inclusion probabilities do not define the registered fixed draw")
    value = math.fsum(
        weight * float(q_value) for weight, q_value in zip(ht_weights, values, strict=True)
    )
    if not -1e-12 <= value <= 1.0 + 1e-12:
        raise ValueError("utility Horvitz-Thompson target is outside [0, 1]")
    value = min(1.0, max(0.0, value))
    within_action_proxy = math.fsum(
        weight * weight / count
        for weight, count in zip(ht_weights, counts, strict=True)
    )
    partial_support_proxy = math.fsum(
        (1.0 - float(probability)) * weight * weight
        for weight, probability in zip(ht_weights, probabilities, strict=True)
    )
    effective_count = 1.0 / (within_action_proxy + partial_support_proxy)
    if not math.isfinite(effective_count) or effective_count <= 0.0:
        raise RuntimeError("utility effective continuation count is not finite and positive")
    return WarmStartUtilityTarget(
        value=value,
        effective_count=effective_count,
        evaluated_actions=action_count,
        continuation_trajectories=sum(counts),
        inclusion_probability=min(float(probability) for probability in probabilities),
        action_value=action_value_target,
        inclusion_probabilities=tuple(float(probability) for probability in probabilities),
        legal_action_count=support_count,
    )


def normalized_legal_entropy(
    masked_log_probs: Tensor,
    legal_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return entropy divided by maximum entropy for each exact legal support.

    A singleton support has both raw and normalized entropy zero.  Illegal or padded
    actions never enter a multiplication with negative infinity.
    """

    if masked_log_probs.ndim != 2 or legal_mask.shape != masked_log_probs.shape:
        raise ValueError("entropy log probabilities and legal mask must be matching matrices")
    if legal_mask.dtype is not torch.bool:
        raise ValueError("entropy legal mask must be Boolean")
    legal_counts = legal_mask.sum(dim=1)
    if bool(torch.any(legal_counts <= 0)):
        raise ValueError("entropy requires a nonempty exact legal support in every row")
    if not bool(torch.all(torch.isfinite(masked_log_probs[legal_mask]))):
        raise ValueError("entropy legal log probabilities must be finite")
    if bool(torch.any(~torch.isneginf(masked_log_probs[~legal_mask]))):
        raise ValueError("entropy masked actions must have negative-infinite log probability")
    safe_log_probs = torch.where(
        legal_mask, masked_log_probs, torch.zeros_like(masked_log_probs)
    )
    probabilities = torch.where(
        legal_mask, torch.exp(masked_log_probs), torch.zeros_like(masked_log_probs)
    )
    raw = -(probabilities * safe_log_probs).sum(dim=1)
    maximum = torch.log(legal_counts.to(dtype=raw.dtype).clamp(min=2))
    normalized = raw / maximum
    if not bool(torch.all(torch.isfinite(raw))) or not bool(
        torch.all(torch.isfinite(normalized))
    ):
        raise RuntimeError("normalized legal entropy became non-finite")
    return normalized, raw


def _episode_rng(
    *,
    seed: int,
    update_index: int,
    episode_schedule_index: int,
    domain: str,
) -> np.random.Generator:
    """Create a stable domain-separated stream for one registered episode.

    Environment retries and policy samples deliberately use separate streams.  A change in
    another episode's horizon, retry count or terminal state therefore cannot move this
    episode's random stream merely because active episodes are collected in waves.
    """

    payload = "\x00".join(
        (
            COLLECTION_SCHEDULE,
            str(seed),
            str(update_index),
            str(episode_schedule_index),
            domain,
        )
    ).encode("utf-8")
    stream_seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
    return np.random.Generator(np.random.PCG64(stream_seed))


@dataclass
class _ActiveEpisode:
    env: object
    decision: DecisionState
    episode_index: int
    policy_rng: np.random.Generator


def _finish_episode(episode: Episode, terminal: TerminalRecord) -> None:
    if terminal.training_reward is None or terminal.training_cost is None:
        raise RuntimeError("training collection received a deployment-only terminal record")
    episode.terminal_reward = float(terminal.training_reward)
    episode.terminal_cost = float(terminal.training_cost)
    episode.reason = terminal.terminal_reason
    episode.returned_valid = terminal.returned_valid
    episode.selected_strength = terminal.selected_strength
    if terminal.embedding is not None:
        episode.qubits = sum(len(chain) for chain in terminal.embedding.values())


def _forward_decision_wave(
    behaviour: IFCore,
    decisions: Sequence[DecisionState],
    device: torch.device,
) -> list[tuple[np.ndarray, float, float]]:
    """Evaluate one active wave, batching every production IF-Core observation."""

    if not decisions:
        return []
    for decision in decisions:
        observation = decision.observation
        support_size = len(decision.candidates)
        decision_mask = np.asarray(decision.legal_mask, dtype=bool)
        if observation.actions.shape[0] != support_size or decision_mask.shape != (support_size,):
            raise RuntimeError("decision observation differs from its exact action support")
        if not np.array_equal(observation.legal_mask, decision_mask):
            raise RuntimeError("decision and observation legal masks differ")
        if observation.real_action_mask.shape != decision_mask.shape or not np.all(
            observation.real_action_mask
        ):
            raise RuntimeError("collected decision contains padded or malformed action rows")
    if isinstance(behaviour, IFCore):
        output = behaviour(
            [decision.observation for decision in decisions],
            [decision.observation.actions for decision in decisions],
            [np.asarray(decision.legal_mask, dtype=bool) for decision in decisions],
            device=device,
        )
        log_prob_matrix = output.masked_log_probs.detach().cpu().numpy()
        values_and_counts = (
            torch.stack(
                (
                    output.utility_value,
                    output.failure_value,
                    output.action_count.to(dtype=output.utility_value.dtype),
                ),
                dim=-1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        if log_prob_matrix.ndim != 2 or log_prob_matrix.shape[0] != len(decisions):
            raise RuntimeError("batched behavior policy returned the wrong log-probability shape")
        if values_and_counts.shape != (len(decisions), 3):
            raise RuntimeError("batched behavior policy returned the wrong value/count shape")
        if not np.isfinite(values_and_counts).all():
            raise RuntimeError("batched behavior policy returned a non-finite value or count")
        rows: list[tuple[np.ndarray, float, float]] = []
        for row, decision in enumerate(decisions):
            support_size = len(decision.candidates)
            if support_size > log_prob_matrix.shape[1]:
                raise RuntimeError("batched behavior policy truncated an action support")
            if not np.isneginf(log_prob_matrix[row, support_size:]).all():
                raise RuntimeError("batched behavior policy assigned mass beyond an action support")
            if values_and_counts[row, 2] != support_size:
                raise RuntimeError("batched behavior policy reported the wrong action count")
            rows.append(
                (
                    log_prob_matrix[row, :support_size].copy(),
                    float(values_and_counts[row, 0]),
                    float(values_and_counts[row, 1]),
                )
            )
        return rows

    # Compatibility path for small protocol doubles and external clients that expose only
    # the historical scalar interface.  Production IF-Core families can never enter it.
    forward_single = getattr(behaviour, "forward_single", None)
    if not callable(forward_single):
        raise TypeError("behavior policy must implement the IF-Core batch or scalar protocol")
    rows = []
    for decision in decisions:
        output = forward_single(decision.observation, device)
        utility = float(output.utility_value.detach().cpu().item())
        failure = float(output.failure_value.detach().cpu().item())
        if not math.isfinite(utility) or not math.isfinite(failure):
            raise RuntimeError("scalar behavior policy returned a non-finite critic value")
        rows.append(
            (
                output.masked_log_probs.detach().cpu().numpy().copy(),
                utility,
                failure,
            )
        )
    return rows


def _sample_action(
    log_probs: np.ndarray,
    legal_mask: Sequence[bool],
    *,
    policy_rng: np.random.Generator,
    greedy: bool,
) -> int:
    legal = np.asarray(legal_mask, dtype=bool)
    if log_probs.ndim != 1 or log_probs.shape != legal.shape:
        raise RuntimeError("behavior policy output shape differs from its exact action support")
    if (
        not legal.any()
        or not np.isfinite(log_probs[legal]).all()
        or not np.isneginf(log_probs[~legal]).all()
    ):
        raise RuntimeError("behavior policy finite support differs from the exact legal mask")
    probabilities = np.zeros_like(log_probs, dtype=np.float64)
    probabilities[legal] = np.exp(log_probs[legal])
    total = float(probabilities.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("behavior policy produced a non-finite or empty action distribution")
    if not np.isclose(total, 1.0, atol=1e-4, rtol=0.0):
        raise RuntimeError("behavior policy action distribution is not normalized")
    probabilities /= total
    return (
        int(np.argmax(probabilities))
        if greedy
        else int(policy_rng.choice(len(probabilities), p=probabilities))
    )


def _model_mode(model: nn.Module) -> Mode:
    improvement_mode = getattr(model, "improvement_mode", None)
    if type(improvement_mode) is not bool:
        raise ValueError("PPO model must declare a Boolean improvement_mode")
    return Mode.IMPROVEMENT if improvement_mode else Mode.CONSTRUCTION


def _require_environment_mode(env: object, behaviour_mode: Mode) -> None:
    """Reject a real environment/model mode mismatch before reset or collection."""

    if not hasattr(env, "mode"):
        # Minimal failure doubles may omit mode because they never yield a decision. Real
        # EmbeddingEnv instances always expose it and are checked before reset.
        return
    environment_mode = getattr(env, "mode")
    if not isinstance(environment_mode, Mode) or environment_mode is not behaviour_mode:
        raise ValueError(
            "PPO collection mode mismatch: "
            f"model={behaviour_mode.value}, environment={environment_mode!r}"
        )


@dataclass(frozen=True)
class PPOConfig:
    """Reference hyperparameters of MODEL_SPEC section 7.5; all frozen before a run."""

    episodes_per_batch: int = 64
    epochs: int = 4
    minibatch: int = 256
    clip: float = 0.2
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    entropy_initial: float = 0.01
    entropy_floor: float = 0.001
    entropy_floor_at: float = 0.8
    grad_norm: float = 0.5
    kl_target: float = 0.01
    kl_stop: float = 0.02
    kl_backtrack_factor: float = 0.5
    kl_max_backtracks: int = 3
    value_weight: float = 0.5
    reference_length: int = 32
    failure_constraint_enabled: bool = False
    failure_target: float | None = None
    dual_step: float = 0.05
    init_attempt_cap: int = 8
    seed: int = 0
    utility_advantage_transform: str = UTILITY_ADVANTAGE_TRANSFORM
    utility_advantage_degeneracy_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        for name in (
            "episodes_per_batch",
            "epochs",
            "minibatch",
            "reference_length",
            "init_attempt_cap",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.kl_max_backtracks, bool)
            or not isinstance(self.kl_max_backtracks, int)
            or self.kl_max_backtracks < 0
        ):
            raise ValueError("kl_max_backtracks must be a nonnegative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.gamma != 1.0:
            raise ValueError("IF-Core-v1 registers undiscounted gamma=1")
        if not math.isfinite(self.gae_lambda) or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1]")
        if not math.isfinite(self.clip) or not 0.0 < self.clip < 1.0:
            raise ValueError("PPO clip must lie in (0, 1)")
        positive = (
            "learning_rate",
            "grad_norm",
            "kl_target",
            "kl_stop",
            "entropy_initial",
            "entropy_floor",
        )
        if any(
            not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0
            for name in positive
        ):
            raise ValueError(
                "learning rate, gradient cap, KL thresholds and entropy coefficients "
                "must be positive"
            )
        if self.kl_target >= self.kl_stop:
            raise ValueError("kl_target must be strictly below the hard kl_stop limit")
        if (
            not math.isfinite(self.kl_backtrack_factor)
            or not 0.0 < self.kl_backtrack_factor < 1.0
        ):
            raise ValueError("kl_backtrack_factor must lie strictly between zero and one")
        if self.entropy_initial < self.entropy_floor:
            raise ValueError("entropy_initial must be at least the positive entropy_floor")
        nonnegative = ("value_weight", "dual_step")
        if any(
            not math.isfinite(getattr(self, name)) or getattr(self, name) < 0.0
            for name in nonnegative
        ):
            raise ValueError("value weight and dual step must be nonnegative")
        if not math.isfinite(self.entropy_floor_at) or not 0.0 < self.entropy_floor_at <= 1.0:
            raise ValueError("entropy_floor_at must lie in (0, 1]")
        if self.utility_advantage_transform != UTILITY_ADVANTAGE_TRANSFORM:
            raise ValueError(
                "utility advantage transform must use the registered full-rollout version"
            )
        if (
            not math.isfinite(self.utility_advantage_degeneracy_tolerance)
            or self.utility_advantage_degeneracy_tolerance < 0.0
        ):
            raise ValueError(
                "utility advantage degeneracy tolerance must be finite and nonnegative"
            )
        if self.failure_constraint_enabled and self.failure_target is None:
            raise ValueError(
                "failure_target is required when the construction failure constraint is enabled"
            )
        if self.failure_target is not None and (
            not math.isfinite(self.failure_target) or not 0.0 <= self.failure_target <= 1.0
        ):
            raise ValueError("failure_target must lie in [0, 1]")


def episode_weighted_reduction(values: np.ndarray, weights: np.ndarray) -> float:
    """Implement ``M^-1 sum_i w_i value_i`` from the registered CMDP objective."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape or not len(values):
        raise ValueError("episode values and weights must be equal nonempty vectors")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("episode values and weights must be finite with nonnegative weights")
    return float(np.sum(weights * values) / len(values))


def collect(
    env_factory: EnvFactory,
    behaviour: IFCore,
    config: PPOConfig,
    *,
    update_index: int = 0,
    device: torch.device | None = None,
    greedy: bool = False,
) -> tuple[RolloutBuffer, dict[str, float]]:
    """Run ``episodes_per_batch`` complete episodes under one frozen snapshot."""

    if isinstance(update_index, bool) or not isinstance(update_index, int) or update_index < 0:
        raise ValueError("update_index must be a nonnegative integer")
    device = device or next(behaviour.parameters()).device
    buffer = RolloutBuffer()
    init_failures = 0
    behaviour_mode = _model_mode(behaviour)
    collection_rule = GREEDY_COLLECTION_RULE if greedy else STOCHASTIC_COLLECTION_RULE
    buffer.metadata.update(
        {
            "replay_receipt_schema": ROLLOUT_REPLAY_SCHEMA,
            "replay_receipt_schema_version": ROLLOUT_REPLAY_SCHEMA_VERSION,
            "collection_rule": collection_rule,
            "training_eligible": not greedy,
            "mode": behaviour_mode.value,
            "collection_schedule_schema": COLLECTION_SCHEDULE_SCHEMA,
            "collection_schedule_schema_version": COLLECTION_SCHEDULE_SCHEMA_VERSION,
            "collection_schedule": COLLECTION_SCHEDULE,
            "collection_training_seed": config.seed,
            "collection_update_index": update_index,
        }
    )
    behaviour.eval()

    schedule_start = update_index * config.episodes_per_batch
    buffer.metadata.update(
        {
            "episode_schedule_start": schedule_start,
            "episode_schedule_stop_exclusive": schedule_start + config.episodes_per_batch,
            "episode_schedule_indices": tuple(
                range(schedule_start, schedule_start + config.episodes_per_batch)
            ),
        }
    )

    active: list[_ActiveEpisode] = []
    for local_episode_index in range(config.episodes_per_batch):
        # The immutable episode index is separate from every environment/initializer RNG
        # seed.  Scientific callers use it to schedule base lineages cyclically, so a
        # lineage with more generated conditions or variants cannot receive more PPO
        # weight.  Initializer retries keep the same episode index and therefore the same
        # task; only their environment seed changes.
        episode_schedule_index = schedule_start + local_episode_index
        environment_rng = _episode_rng(
            seed=config.seed,
            update_index=update_index,
            episode_schedule_index=episode_schedule_index,
            domain="environment-initialization",
        )
        policy_rng = _episode_rng(
            seed=config.seed,
            update_index=update_index,
            episode_schedule_index=episode_schedule_index,
            domain="categorical-policy",
        )
        result: DecisionState | TerminalRecord | InitFailureRecord | None = None
        env: object | None = None
        for _ in range(config.init_attempt_cap):
            env = env_factory(int(environment_rng.integers(2**31)), episode_schedule_index)
            _require_environment_mode(env, behaviour_mode)
            result = env.reset()
            if not isinstance(result, InitFailureRecord):
                break
            init_failures += 1
        if isinstance(result, InitFailureRecord) or result is None:
            raise RuntimeError(
                "unable to fill the complete behavior batch within the registered "
                f"initializer-attempt cap ({config.init_attempt_cap})"
            )

        episode = buffer.add_episode()
        if isinstance(result, DecisionState):
            if env is None:  # pragma: no cover - the attempt loop always assigns it
                raise RuntimeError("collector lost the initialized environment")
            active.append(
                _ActiveEpisode(
                    env=env,
                    decision=result,
                    episode_index=episode.index,
                    policy_rng=policy_rng,
                )
            )
        elif isinstance(result, TerminalRecord):
            _finish_episode(episode, result)
        else:
            raise RuntimeError("collector stopped while the registered task was still live")

    wave_count = 0
    active_rows_total = 0
    while active:
        wave_count += 1
        active_rows_total += len(active)
        decisions = [item.decision for item in active]
        with torch.no_grad():
            policy_rows = _forward_decision_wave(behaviour, decisions, device)
        if len(policy_rows) != len(active):
            raise RuntimeError("behavior policy returned the wrong active-wave batch size")

        next_active: list[_ActiveEpisode] = []
        for item, (log_probs, old_utility, old_failure) in zip(active, policy_rows, strict=True):
            result = item.decision
            index = _sample_action(
                log_probs,
                result.legal_mask,
                policy_rng=item.policy_rng,
                greedy=greedy,
            )
            stored_observation = copy.deepcopy(result.observation)
            stored_candidates = snapshot_candidates(result.candidates)
            stored_mask = np.asarray(result.legal_mask, dtype=bool).copy()
            work_receipt = result.charged_work_receipt.as_dict()
            replay_receipt = make_replay_receipt(
                observation=stored_observation,
                candidates=stored_candidates,
                legal_mask=stored_mask,
                chosen_index=index,
                state_fingerprint=result.state_fingerprint,
                context_version=result.context_version,
                support_fingerprint=result.support_fingerprint,
                charged_work_receipt=work_receipt,
                collection_rule=collection_rule,
                mode=behaviour_mode.value,
            )
            step = item.env.step(result, index)
            buffer.add(
                Transition(
                    observation=stored_observation,
                    legal_mask=stored_mask,
                    chosen_index=index,
                    old_log_prob=float(log_probs[index]),
                    old_log_probs=log_probs.copy(),
                    old_utility=old_utility,
                    old_failure=old_failure,
                    reward=float(step.reward),
                    cost=float(step.failure_cost),
                    terminated=bool(step.terminated),
                    truncated=bool(step.truncated),
                    episode=item.episode_index,
                    payload_key=result.candidates[index].payload_key,
                    state_fingerprint=result.state_fingerprint,
                    support_fingerprint=result.support_fingerprint,
                    context_version=result.context_version,
                    charged_work_receipt=work_receipt,
                    candidates=stored_candidates,
                    replay_receipt=replay_receipt,
                )
            )
            successor = step.next_decision_or_terminal
            if isinstance(successor, DecisionState):
                item.decision = successor
                next_active.append(item)
            elif isinstance(successor, TerminalRecord):
                _finish_episode(buffer.episodes[item.episode_index], successor)
            else:
                raise RuntimeError("collector stopped while the registered task was still live")
        active = next_active

    if buffer.n_episodes != config.episodes_per_batch:
        raise RuntimeError("collector failed to produce the registered complete behavior batch")

    buffer.metadata.update(
        {
            "collection_wave_count": wave_count,
            "collection_active_rows_total": active_rows_total,
        }
    )

    buffer.compute_targets(config.gae_lambda)
    stats = buffer.summary()
    stats.update(
        buffer.utility_advantages_for_actor().diagnostics.as_log_dict(
            config.utility_advantage_degeneracy_tolerance
        )
    )
    stats["init_failures"] = float(init_failures)
    return buffer, stats


class PPOTrainer:
    """One optimizer owning the shared trunk, actor and active critics."""

    def __init__(
        self,
        model: IFCore,
        config: PPOConfig,
        *,
        mode: Mode = Mode.IMPROVEMENT,
        total_updates: int = 100,
        device: torch.device | None = None,
    ) -> None:
        if not isinstance(mode, Mode):
            raise ValueError("PPO trainer mode must be a Mode value")
        declared_mode = _model_mode(model)
        if declared_mode is not mode:
            raise ValueError(
                "PPO trainer/model mode mismatch: "
                f"trainer={mode.value}, model={declared_mode.value}"
            )
        failure_trainable = [parameter.requires_grad for parameter in model.failure.parameters()]
        if mode is Mode.CONSTRUCTION and (not failure_trainable or not all(failure_trainable)):
            raise ValueError("construction mode requires a trainable failure head")
        if mode is Mode.IMPROVEMENT and any(failure_trainable):
            raise ValueError("improvement mode requires a frozen failure head")
        self.model = model
        self.config = config
        self.mode = mode
        if (
            isinstance(total_updates, bool)
            or not isinstance(total_updates, int)
            or total_updates <= 0
        ):
            raise ValueError("total_updates must be a positive integer fixed before training")
        self.total_updates = total_updates
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-5,
            weight_decay=0.0,
        )
        self.lambda_f = 0.0
        self.updates_done = 0

    def _require_training_mode(self) -> None:
        declared_mode = _model_mode(self.model)
        if declared_mode is not self.mode:
            raise RuntimeError(
                "PPO trainer/model mode changed after construction: "
                f"trainer={self.mode.value}, model={declared_mode.value}"
            )
        failure_trainable = [
            parameter.requires_grad for parameter in self.model.failure.parameters()
        ]
        if self.mode is Mode.CONSTRUCTION and (not failure_trainable or not all(failure_trainable)):
            raise RuntimeError("construction PPO failure head is no longer trainable")
        if self.mode is Mode.IMPROVEMENT and any(failure_trainable):
            raise RuntimeError("improvement PPO failure head is no longer frozen")

    def entropy_coefficient(self) -> float:
        span = self.config.entropy_floor_at * self.total_updates
        progress = min(1.0, self.updates_done / max(1e-9, span))
        return self.config.entropy_initial + progress * (
            self.config.entropy_floor - self.config.entropy_initial
        )

    def _optimizer_learning_rate(self) -> float:
        rates = {float(group["lr"]) for group in self.optimizer.param_groups}
        if len(rates) != 1:
            raise RuntimeError("PPO KL control requires one common optimizer learning rate")
        rate = next(iter(rates))
        if not math.isfinite(rate) or rate <= 0.0:
            raise RuntimeError("PPO optimizer learning rate is not finite and positive")
        return rate

    def _set_optimizer_learning_rate(self, learning_rate: float) -> None:
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise RuntimeError("PPO backtracking produced an invalid learning rate")
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def behaviour_snapshot(self) -> IFCore:
        self._require_training_mode()
        snapshot = copy.deepcopy(self.model)
        for parameter in snapshot.parameters():
            parameter.requires_grad_(False)
        snapshot.eval()
        return snapshot

    def _forward(self, transition: Transition) -> ModelOutput:
        return self.model.forward_single(transition.observation, self.device)

    def _forward_many(self, transitions: Sequence[Transition]) -> ModelOutput:
        """Recompute a stored support as one segmented model batch when available."""

        rows = list(transitions)
        if not rows:
            raise ValueError("batched PPO forward requires at least one transition")
        if isinstance(self.model, IFCore):
            return self.model(
                [transition.observation for transition in rows],
                [transition.observation.actions for transition in rows],
                [transition.legal_mask for transition in rows],
                device=self.device,
            )

        # Small protocol doubles used by downstream clients may only implement the original
        # single-row interface.  Preserve that compatibility without putting production
        # IF-Core training back on the scalar path.
        outputs = [self._forward(transition) for transition in rows]
        width = max(int(output.masked_log_probs.shape[0]) for output in outputs)
        padded = outputs[0].masked_log_probs.new_full((len(outputs), width), float("-inf"))
        for row, output in enumerate(outputs):
            padded[row, : output.masked_log_probs.shape[0]] = output.masked_log_probs
        return ModelOutput(
            masked_log_probs=padded,
            utility_value=torch.stack([output.utility_value for output in outputs]),
            failure_logit=torch.stack([output.failure_logit for output in outputs]),
            failure_value=torch.stack([output.failure_value for output in outputs]),
            action_count=torch.stack([output.action_count for output in outputs]),
        )

    def replay_check(self, buffer: RolloutBuffer, tolerance: float = 1e-4) -> float:
        """Unchanged-policy replay must reproduce every complete categorical distribution."""

        self._require_training_mode()
        buffer.authenticate_collection_schedule(
            expected_training_seed=self.config.seed,
            expected_update_index=self.updates_done,
            expected_episode_count=self.config.episodes_per_batch,
        )
        worst = 0.0
        self.model.eval()
        with torch.no_grad():
            for start in range(0, buffer.n_transitions, self.config.minibatch):
                transitions = buffer.transitions[start : start + self.config.minibatch]
                outputs = self._forward_many(transitions)
                for local_row, transition in enumerate(transitions):
                    row = start + local_row
                    old = np.asarray(transition.old_log_probs, dtype=np.float64)
                    legal = np.asarray(transition.legal_mask, dtype=bool)
                    now = outputs.masked_log_probs[local_row, : old.shape[0]].detach().cpu().numpy()
                    if now.shape != old.shape or old.shape != legal.shape:
                        raise RuntimeError(f"likelihood replay support shape mismatch at row {row}")
                    if (
                        not 0 <= transition.chosen_index < old.shape[0]
                        or not legal[transition.chosen_index]
                    ):
                        raise RuntimeError(
                            f"likelihood replay has an invalid chosen action at row {row}"
                        )
                    if not np.array_equal(np.isfinite(now), legal) or not np.array_equal(
                        np.isfinite(old), legal
                    ):
                        raise RuntimeError(f"likelihood replay legal-mask mismatch at row {row}")
                    if legal.any():
                        if not np.isclose(np.exp(old[legal]).sum(), 1.0, atol=tolerance, rtol=0.0):
                            raise RuntimeError(
                                f"stored behavior distribution is not normalized at row {row}"
                            )
                        if not np.isclose(np.exp(now[legal]).sum(), 1.0, atol=tolerance, rtol=0.0):
                            raise RuntimeError(
                                f"replayed behavior distribution is not normalized at row {row}"
                            )
                        worst = max(worst, float(np.max(np.abs(now[legal] - old[legal]))))
                    if abs(old[transition.chosen_index] - transition.old_log_prob) > tolerance:
                        raise RuntimeError(
                            f"stored chosen likelihood disagrees with its distribution at row {row}"
                        )
        if worst > tolerance:
            raise RuntimeError(
                f"likelihood replay mismatch {worst:.2e}: a contract failure, not a reason to widen clipping"
            )
        return worst

    def update(self, buffer: RolloutBuffer) -> dict[str, float | str]:
        cfg = self.config
        self._require_training_mode()
        buffer.authenticate_replay(
            expected_mode=self.mode.value,
            expected_training_seed=cfg.seed,
            expected_update_index=self.updates_done,
            expected_episode_count=cfg.episodes_per_batch,
            training=True,
        )
        if buffer.n_episodes != cfg.episodes_per_batch:
            raise RuntimeError("PPO update requires the registered complete episode count")
        n_tr = buffer.n_transitions
        if n_tr == 0:
            # A construction episode may terminate before exposing any decision.  It still
            # belongs to the registered episode denominator and therefore must update the
            # failure dual.  There is no actor/critic gradient in this case.
            beta = self.entropy_coefficient()
            self._update_failure_dual(buffer)
            self.updates_done += 1
            logs = {
                "entropy_coefficient": beta,
                "entropy_floor": cfg.entropy_floor,
                "entropy_raw_optimization_mean": 0.0,
                "entropy_normalized_optimization_mean": 0.0,
                "lambda_f": self.lambda_f,
                "optimizer_steps": 0.0,
                "optimizer_steps_attempted": 0.0,
                "optimizer_steps_rolled_back": 0.0,
                "epochs_attempted": 0.0,
                "epochs_accepted": 0.0,
                "epochs_run": 0.0,
                "kl": 0.0,
                "kl_target": cfg.kl_target,
                "kl_hard_limit": cfg.kl_stop,
                "kl_backtracks": 0.0,
                "kl_rejected_epochs": 0.0,
                "kl_nonfinite_rejections": 0.0,
                "kl_early_stop": 0.0,
                "kl_early_stop_reason": "no-transitions",
                "kl_hard_limit_satisfied": 1.0,
                "effective_learning_rate": self._optimizer_learning_rate(),
            }
            logs.update(buffer.summary())
            return logs
        self.replay_check(buffer)
        if any(
            value is None
            for value in (
                buffer.gae_utility,
                buffer.gae_failure,
                buffer.target_utility,
                buffer.target_failure,
            )
        ):
            raise RuntimeError("PPO rollout targets were not computed before update")

        actor_advantages = buffer.utility_advantages_for_actor()
        advantage_diagnostics = actor_advantages.diagnostics
        if self.mode is Mode.IMPROVEMENT and advantage_diagnostics.is_degenerate(
            cfg.utility_advantage_degeneracy_tolerance
        ):
            raise RuntimeError(
                "degenerate PPO utility advantage signal: "
                f"policy_controllable_transitions="
                f"{advantage_diagnostics.policy_controllable_transitions}, "
                f"policy_signal_rms={advantage_diagnostics.policy_signal_rms:.9g}, "
                f"tolerance={cfg.utility_advantage_degeneracy_tolerance:.9g}"
            )

        adv_u = torch.as_tensor(actor_advantages.values, dtype=torch.float32, device=self.device)
        adv_f = torch.as_tensor(buffer.gae_failure, dtype=torch.float32, device=self.device)
        tgt_u = torch.as_tensor(buffer.target_utility, dtype=torch.float32, device=self.device)
        tgt_f = torch.as_tensor(buffer.target_failure, dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(
            np.asarray([t.old_log_prob for t in buffer.transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        weights = torch.as_tensor(
            buffer.transition_weights(), dtype=torch.float32, device=self.device
        )
        beta = self.entropy_coefficient()
        scale = n_tr / (cfg.episodes_per_batch * cfg.reference_length)

        logs: dict[str, float | str] = advantage_diagnostics.as_log_dict(
            cfg.utility_advantage_degeneracy_tolerance
        )
        accepted_kl = 0.0
        max_tentative_kl = 0.0
        epochs_attempted = 0
        epochs_accepted = 0
        rejected_epochs = 0
        backtracks_performed = 0
        nonfinite_kl_rejections = 0
        optimizer_steps_attempted = 0
        optimizer_steps_accepted = 0
        optimizer_steps_rolled_back = 0
        accepted_entropy_raw_sum = 0.0
        accepted_entropy_normalized_sum = 0.0
        accepted_entropy_rows = 0
        last_accepted_loss = 0.0
        early_stop_reason = "epoch-budget"
        self.model.train()

        def run_tentative_epoch(order: np.ndarray) -> dict[str, float | int]:
            entropy_raw_sum = 0.0
            entropy_normalized_sum = 0.0
            entropy_rows = 0
            optimizer_steps = 0
            last_loss = 0.0
            for start in range(0, n_tr, cfg.minibatch):
                batch = order[start : start + cfg.minibatch]
                if not len(batch):
                    continue
                batch_rows = [buffer.transitions[int(t)] for t in batch]
                out = self._forward_many(batch_rows)
                selected = torch.as_tensor(
                    [transition.chosen_index for transition in batch_rows],
                    dtype=torch.long,
                    device=self.device,
                )
                row_index = torch.arange(len(batch_rows), dtype=torch.long, device=self.device)
                batch_index = torch.as_tensor(batch, dtype=torch.long, device=self.device)
                logp = out.masked_log_probs[row_index, selected]
                ratio = torch.exp(logp - old_logp[batch_index])
                clipped = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip)
                batch_weights = weights[batch_index]
                surrogate = (
                    torch.min(ratio * adv_u[batch_index], clipped * adv_u[batch_index])
                    * batch_weights
                ).sum()
                cost_surrogate = (
                    torch.max(ratio * adv_f[batch_index], clipped * adv_f[batch_index])
                    * batch_weights
                ).sum()
                legal = torch.zeros_like(out.masked_log_probs, dtype=torch.bool)
                for row, transition in enumerate(batch_rows):
                    count = int(transition.legal_mask.shape[0])
                    legal[row, :count] = torch.as_tensor(
                        transition.legal_mask, dtype=torch.bool, device=self.device
                    )
                normalized_entropy, raw_entropy = normalized_legal_entropy(
                    out.masked_log_probs, legal
                )
                entropy_term = (normalized_entropy * batch_weights).sum()
                value_loss = ((out.utility_value - tgt_u[batch_index]) ** 2).sum()
                reduce = scale / len(batch)
                loss = -reduce * surrogate - beta * reduce * entropy_term
                loss = loss + cfg.value_weight * value_loss / len(batch)
                if self.mode is Mode.CONSTRUCTION:
                    failure_loss = nn.functional.binary_cross_entropy_with_logits(
                        out.failure_logit, tgt_f[batch_index], reduction="sum"
                    )
                    loss = loss + cfg.value_weight * failure_loss / len(batch)
                    if cfg.failure_constraint_enabled:
                        loss = loss + self.lambda_f * reduce * cost_surrogate
                self.optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    raise RuntimeError("PPO loss became non-finite")
                loss.backward()
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in self.model.parameters()
                ):
                    raise RuntimeError("PPO gradient became non-finite")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_norm)
                self.optimizer.step()
                optimizer_steps += 1
                last_loss = float(loss.detach().item())
                entropy_raw_sum += float(raw_entropy.detach().sum().item())
                entropy_normalized_sum += float(
                    normalized_entropy.detach().sum().item()
                )
                entropy_rows += len(batch_rows)
            return {
                "last_loss": last_loss,
                "optimizer_steps": optimizer_steps,
                "entropy_raw_sum": entropy_raw_sum,
                "entropy_normalized_sum": entropy_normalized_sum,
                "entropy_rows": entropy_rows,
            }

        for epoch in range(cfg.epochs):
            order = np.random.default_rng(cfg.seed + epoch + self.updates_done).permutation(n_tr)
            boundary_model = copy.deepcopy(self.model.state_dict())
            boundary_optimizer = copy.deepcopy(self.optimizer.state_dict())
            boundary_cpu_rng = torch.get_rng_state().clone()
            boundary_cuda_rng = (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            )
            epoch_learning_rate = self._optimizer_learning_rate()

            def restore_boundary() -> None:
                self.model.load_state_dict(boundary_model)
                self.optimizer.load_state_dict(boundary_optimizer)
                torch.set_rng_state(boundary_cpu_rng)
                if boundary_cuda_rng is not None:
                    torch.cuda.set_rng_state_all(boundary_cuda_rng)

            accepted = False
            for backtrack in range(cfg.kl_max_backtracks + 1):
                restore_boundary()
                if backtrack > 0:
                    backtracks_performed += 1
                attempt_learning_rate = epoch_learning_rate * (
                    cfg.kl_backtrack_factor**backtrack
                )
                self._set_optimizer_learning_rate(attempt_learning_rate)
                epochs_attempted += 1
                try:
                    attempt = run_tentative_epoch(order)
                    tentative_kl = self.full_buffer_kl(buffer)
                except BaseException:
                    restore_boundary()
                    raise
                attempt_steps = int(attempt["optimizer_steps"])
                optimizer_steps_attempted += attempt_steps
                if math.isfinite(tentative_kl):
                    max_tentative_kl = max(max_tentative_kl, tentative_kl)
                if not math.isfinite(tentative_kl) or tentative_kl > cfg.kl_stop:
                    rejected_epochs += 1
                    optimizer_steps_rolled_back += attempt_steps
                    if not math.isfinite(tentative_kl):
                        nonfinite_kl_rejections += 1
                    continue

                accepted = True
                accepted_kl = tentative_kl
                epochs_accepted += 1
                optimizer_steps_accepted += attempt_steps
                last_accepted_loss = float(attempt["last_loss"])
                accepted_entropy_raw_sum += float(attempt["entropy_raw_sum"])
                accepted_entropy_normalized_sum += float(
                    attempt["entropy_normalized_sum"]
                )
                accepted_entropy_rows += int(attempt["entropy_rows"])
                break

            if not accepted:
                restore_boundary()
                self._set_optimizer_learning_rate(
                    epoch_learning_rate
                    * cfg.kl_backtrack_factor ** (cfg.kl_max_backtracks + 1)
                )
                early_stop_reason = "hard-limit-no-safe-step"
                break
            if accepted_kl >= cfg.kl_target:
                early_stop_reason = "soft-target"
                break

        self._update_failure_dual(buffer)
        self.updates_done += 1
        if accepted_kl > cfg.kl_stop or not math.isfinite(accepted_kl):
            raise RuntimeError("PPO KL rollback failed to restore a hard-limit-safe boundary")
        logs.update(
            {
                "loss": last_accepted_loss,
                "kl": accepted_kl,
                "max_tentative_kl": max_tentative_kl,
                "kl_target": cfg.kl_target,
                "kl_hard_limit": cfg.kl_stop,
                "kl_backtracks": float(backtracks_performed),
                "kl_rejected_epochs": float(rejected_epochs),
                "kl_nonfinite_rejections": float(nonfinite_kl_rejections),
                "kl_early_stop": float(early_stop_reason != "epoch-budget"),
                "kl_early_stop_reason": early_stop_reason,
                "kl_hard_limit_satisfied": 1.0,
                "epochs_attempted": float(epochs_attempted),
                "epochs_accepted": float(epochs_accepted),
                "epochs_run": float(epochs_accepted),
                "optimizer_steps": float(optimizer_steps_accepted),
                "optimizer_steps_attempted": float(optimizer_steps_attempted),
                "optimizer_steps_rolled_back": float(optimizer_steps_rolled_back),
                "effective_learning_rate": self._optimizer_learning_rate(),
                "entropy_raw_optimization_mean": (
                    accepted_entropy_raw_sum / max(1, accepted_entropy_rows)
                ),
                "entropy_normalized_optimization_mean": (
                    accepted_entropy_normalized_sum / max(1, accepted_entropy_rows)
                ),
            }
        )
        logs["entropy_coefficient"] = beta
        logs["entropy_floor"] = cfg.entropy_floor
        logs["lambda_f"] = self.lambda_f
        logs.update(buffer.summary())
        return logs

    def _update_failure_dual(self, buffer: RolloutBuffer) -> None:
        """Apply the episode-level CMDP dual update, including zero-decision failures."""

        cfg = self.config
        if self.mode is not Mode.CONSTRUCTION or not cfg.failure_constraint_enabled:
            return
        episode_weights = np.asarray([episode.weight for episode in buffer.episodes])
        episode_costs = np.asarray([episode.terminal_cost for episode in buffer.episodes])
        mean_cost = episode_weighted_reduction(episode_costs, episode_weights)
        if cfg.failure_target is None:  # Defensive; PPOConfig rejects this state.
            raise RuntimeError("enabled failure constraint has no registered target")
        self.lambda_f = max(0.0, self.lambda_f + cfg.dual_step * (mean_cost - cfg.failure_target))

    def full_buffer_kl(self, buffer: RolloutBuffer) -> float:
        """Support-wise categorical KL of Eq. (26), a per-decision diagnostic mean."""

        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, buffer.n_transitions, self.config.minibatch):
                transitions = buffer.transitions[start : start + self.config.minibatch]
                outputs = self._forward_many(transitions)
                for row, transition in enumerate(transitions):
                    old = transition.old_log_probs
                    legal = np.isfinite(old)
                    if not legal.any():
                        continue
                    new = outputs.masked_log_probs[row, : old.shape[0]].detach().cpu().numpy()
                    p_old = np.exp(old[legal])
                    total += float(np.sum(p_old * (old[legal] - new[legal])))
                    count += 1
        return total / max(1, count)


def _validated_warm_start_ranking_rows(
    observations: Sequence,
    best_sets: Sequence[Sequence[int]],
    evaluated_sets: Sequence[Sequence[int]],
) -> list[tuple[int, object, Sequence[int], Sequence[int]]]:
    if not (
        len(observations) == len(best_sets) == len(evaluated_sets)
    ):
        raise ValueError("warm-start observation and ranking arrays must have the same length")
    valid: list[tuple[int, object, Sequence[int], Sequence[int]]] = []
    for row, (obs, best, evaluated) in enumerate(
        zip(observations, best_sets, evaluated_sets, strict=True)
    ):
        if not best or not evaluated:
            continue
        action_count = int(obs.actions.shape[0])
        if any(
            isinstance(index, bool) or not isinstance(index, (int, np.integer))
            for index in (*best, *evaluated)
        ):
            raise ValueError("ranking action indices must be integers")
        if any(index < 0 or index >= action_count for index in (*best, *evaluated)):
            raise ValueError("ranking action index is outside its stored support")
        best_indices = {int(index) for index in best}
        evaluated_indices = {int(index) for index in evaluated}
        if len(best_indices) != len(best) or len(evaluated_indices) != len(evaluated):
            raise ValueError("ranking best/evaluated sets must not contain duplicate indices")
        if not best_indices <= evaluated_indices:
            raise ValueError("ranking best set must be a subset of the evaluated set")
        legal_indices = set(np.flatnonzero(np.logical_and(obs.legal_mask, obs.real_action_mask)))
        if not evaluated_indices <= legal_indices:
            raise ValueError("ranking evaluated set must be inside the exact legal support")
        # A plausible-best set equal to the evaluated set contains no actor preference.
        # Keep the source row in the actor-critic batch so that its authenticated utility
        # target still supervises V(s), but exclude it from the mean ranking denominator.
        if best_indices == evaluated_indices:
            continue
        valid.append((row, obs, best, evaluated))
    return valid


def _warm_start_rank_from_outputs(
    outputs: ModelOutput,
    rows: Sequence[tuple[int, object, Sequence[int], Sequence[int]]],
    *,
    corpus_records: int | None = None,
) -> Tensor:
    if not rows:
        return outputs.utility_value.new_zeros(())
    best_mask = torch.zeros_like(outputs.masked_log_probs, dtype=torch.bool)
    evaluated_mask = torch.zeros_like(outputs.masked_log_probs, dtype=torch.bool)
    device = outputs.masked_log_probs.device
    for row, _obs, best, evaluated in rows:
        best_mask[row, torch.as_tensor(best, dtype=torch.long, device=device)] = True
        evaluated_mask[row, torch.as_tensor(evaluated, dtype=torch.long, device=device)] = True
    negative_infinity = torch.full_like(outputs.masked_log_probs, float("-inf"))
    denominator = torch.logsumexp(
        torch.where(evaluated_mask, outputs.masked_log_probs, negative_infinity), dim=1
    )
    numerator = torch.logsumexp(
        torch.where(best_mask, outputs.masked_log_probs, negative_infinity), dim=1
    )
    row_indices = torch.as_tensor(
        [row[0] for row in rows], dtype=torch.long, device=device
    )
    row_losses = -(numerator[row_indices] - denominator[row_indices])
    loss = (
        row_losses.mean()
        if corpus_records is None
        else row_losses.sum() / float(corpus_records)
    )
    if not torch.isfinite(loss):
        raise RuntimeError("supervised ranking support produced a non-finite loss")
    return loss


def warm_start_rank_loss(
    model: IFCore,
    observations: Sequence,
    best_sets: Sequence[Sequence[int]],
    evaluated_sets: Sequence[Sequence[int]],
    device: torch.device | None = None,
) -> Tensor:
    """Eq. (19): subset-normalised ranking over *evaluated* actions only.

    Actions that were never evaluated are not labelled inferior; they still provide the
    model's action context.
    """

    device = device or next(model.parameters()).device
    rows = _validated_warm_start_ranking_rows(observations, best_sets, evaluated_sets)
    if not rows:
        return torch.zeros((), device=device)
    compact_rows = [
        (row, obs, best, evaluated)
        for row, (_source_row, obs, best, evaluated) in enumerate(rows)
    ]
    outputs = model([row[1] for row in rows], device=device)
    return _warm_start_rank_from_outputs(outputs, compact_rows)


def warm_start_actor_critic_loss(
    model: IFCore,
    observations: Sequence,
    best_sets: Sequence[Sequence[int]],
    evaluated_sets: Sequence[Sequence[int]],
    utility_targets: Sequence[float],
    utility_effective_counts: Sequence[float],
    action_value_targets: Sequence[WarmStartActionValueTarget | None] | None = None,
    device: torch.device | None = None,
    *,
    corpus_denominators: WarmStartCorpusDenominators,
    loss_profile: WarmStartLossProfile = WARM_START_FULL_LOSS_PROFILE,
) -> WarmStartActorCriticLoss:
    """Return one additive memory-minibatch contribution to the corpus loss.

    Counterfactual outcomes are stopped-gradient scalar targets.  They are deliberately
    constructed only after the model forward and can never enter observation, candidate,
    mask, actor-logit, or critic-input tensors.  ``corpus_denominators`` must be computed
    once from the authenticated corpus; recomputing it per call changes the registered
    objective and overweights sparse or short-tail minibatches.
    """

    if not isinstance(corpus_denominators, WarmStartCorpusDenominators):
        raise TypeError("warm-start corpus denominators have an unsupported type")
    profile = warm_start_loss_profile(loss_profile)

    n_rows = len(observations)
    if not (
        n_rows
        == len(best_sets)
        == len(evaluated_sets)
        == len(utility_targets)
        == len(utility_effective_counts)
    ):
        raise ValueError("warm-start actor and critic arrays must have the same length")
    if n_rows == 0:
        raise ValueError("warm-start actor-critic supervision must not be empty")
    if any(
        isinstance(target, bool)
        or not isinstance(target, (int, float))
        or not math.isfinite(target)
        or not 0.0 <= float(target) <= 1.0
        for target in utility_targets
    ):
        raise ValueError("warm-start utility targets must be finite and bounded in [0, 1]")
    if any(
        isinstance(count, bool)
        or not isinstance(count, (int, float))
        or not math.isfinite(count)
        or float(count) <= 0.0
        for count in utility_effective_counts
    ):
        raise ValueError("warm-start utility effective counts must be finite and positive")
    if action_value_targets is not None and len(action_value_targets) != n_rows:
        raise ValueError("warm-start action-value targets must have the same length")
    quality_targets = (
        [None] * n_rows if action_value_targets is None else list(action_value_targets)
    )
    for observation, evaluated, target in zip(
        observations, evaluated_sets, quality_targets, strict=True
    ):
        if target is None:
            continue
        if not isinstance(target, WarmStartActionValueTarget):
            raise TypeError("warm-start action-value target has an unsupported type")
        legal = np.logical_and(observation.legal_mask, observation.real_action_mask)
        if target.legal_action_count != int(np.count_nonzero(legal)):
            raise ValueError("action-value legal support differs from its observation")
        support_size = (
            target.legal_action_count
            if target.support_size is None
            else target.support_size
        )
        if support_size != int(observation.actions.shape[0]):
            raise ValueError("action-value tensor support differs from its observation")
        if tuple(int(index) for index in evaluated) != target.action_indices:
            raise ValueError("action-value target indices differ from the evaluated action set")
        if any(not legal[index] for index in target.action_indices):
            raise ValueError("action-value target includes an action outside the exact legal mask")
    rows = _validated_warm_start_ranking_rows(observations, best_sets, evaluated_sets)
    local_effective_count = float(
        math.fsum(float(value) for value in utility_effective_counts)
    )
    local_action_value_records = sum(target is not None for target in quality_targets)
    local_commit_delta_records = sum(
        target is not None
        and target.commit_index is not None
        and len(target.action_indices) > 1
        for target in quality_targets
    )
    local_counts = (
        ("actor-ranking", len(rows), corpus_denominators.actor_ranking_records),
        (
            "action-value",
            local_action_value_records,
            corpus_denominators.action_value_records,
        ),
        (
            "commit-delta",
            local_commit_delta_records,
            corpus_denominators.commit_delta_records,
        ),
    )
    for name, local, corpus in local_counts:
        if local > corpus:
            raise ValueError(
                f"warm-start minibatch {name} rows exceed the corpus denominator"
            )
    effective_count_tolerance = max(
        1e-9,
        1e-12 * float(corpus_denominators.utility_effective_count),
    )
    if (
        local_effective_count
        > float(corpus_denominators.utility_effective_count) + effective_count_tolerance
    ):
        raise ValueError(
            "warm-start minibatch utility mass exceeds the corpus denominator"
        )

    device = device or next(model.parameters()).device
    outputs = model(observations, device=device)
    rank_loss = profile.weights.rank * _warm_start_rank_from_outputs(
        outputs, rows, corpus_records=corpus_denominators.actor_ranking_records
    )
    # Targets are intentionally materialized after the policy/critic forward.  They are
    # labels for the loss only, never features consumed by the shared representation.
    targets = torch.as_tensor(
        utility_targets, dtype=outputs.utility_value.dtype, device=device
    ).detach()
    counts = torch.as_tensor(
        utility_effective_counts, dtype=outputs.utility_value.dtype, device=device
    ).detach()
    utility_loss = profile.weights.utility * (
        counts * (outputs.utility_value - targets).square()
    ).sum() / float(corpus_denominators.utility_effective_count)
    if outputs.action_quality_value is None or outputs.action_quality_logit is None:
        if any(target is not None for target in quality_targets):
            raise RuntimeError("registered model omitted action-quality predictions")
        action_value_loss = outputs.utility_value.new_zeros(())
        commit_delta_loss = outputs.utility_value.new_zeros(())
        action_effective_count = 0.0
        commit_delta_rows = 0
    else:
        action_rows: list[Tensor] = []
        delta_rows: list[Tensor] = []
        action_effective_count = 0.0
        for row, target in enumerate(quality_targets):
            if target is None:
                continue
            indices = torch.as_tensor(target.action_indices, dtype=torch.long, device=device)
            q_mu = torch.as_tensor(
                target.q_mu,
                dtype=outputs.action_quality_value.dtype,
                device=device,
            ).detach()
            weights = torch.as_tensor(
                target.regression_weights,
                dtype=outputs.action_quality_value.dtype,
                device=device,
            ).detach()
            prediction = outputs.action_quality_value[row, indices]
            quality_logit = outputs.action_quality_logit[row, indices]
            action_rows.append(
                (
                    weights
                    * torch.nn.functional.binary_cross_entropy_with_logits(
                        quality_logit,
                        q_mu,
                        reduction="none",
                    )
                ).sum()
                / weights.sum()
            )
            action_effective_count += math.fsum(target.regression_weights)
            if target.commit_index is None or len(target.action_indices) == 1:
                continue
            commit_position = target.action_indices.index(target.commit_index)
            other_positions = tuple(
                position
                for position in range(len(target.action_indices))
                if position != commit_position
            )
            other = torch.as_tensor(other_positions, dtype=torch.long, device=device)
            commit_prediction = prediction[commit_position]
            commit_target = q_mu[commit_position]
            delta_prediction = prediction[other] - commit_prediction
            delta_target = q_mu[other] - commit_target
            delta_weights = weights[other]
            delta_rows.append(
                (delta_weights * (delta_prediction - delta_target).square()).sum()
                / delta_weights.sum()
            )
        action_value_loss = (
            profile.weights.action_value
            * torch.stack(action_rows).sum()
            / float(corpus_denominators.action_value_records)
            if action_rows
            else outputs.utility_value.new_zeros(())
        )
        commit_delta_loss = (
            profile.weights.commit_delta
            * torch.stack(delta_rows).sum()
            / float(corpus_denominators.commit_delta_records)
            if delta_rows
            else outputs.utility_value.new_zeros(())
        )
        commit_delta_rows = len(delta_rows)
    total = rank_loss + utility_loss + action_value_loss + commit_delta_loss
    if not all(
        bool(torch.isfinite(value))
        for value in (utility_loss, action_value_loss, commit_delta_loss, total)
    ):
        raise RuntimeError("supervised actor-critic warm start produced a non-finite loss")
    return WarmStartActorCriticLoss(
        total=total,
        rank=rank_loss,
        utility=utility_loss,
        effective_count=local_effective_count,
        action_value=action_value_loss,
        commit_delta=commit_delta_loss,
        action_effective_count=float(action_effective_count),
        commit_delta_rows=commit_delta_rows,
    )
