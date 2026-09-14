"""Quality counterfactuals that name their continuation policy.

Spec: Rev2 section "Quality counterfactuals must name their continuation policy". A partial
state has no quality without a continuation, so ``Q^mu(s, a)`` is defined against a frozen
executable policy, its budget, its archive rule and the frozen strength selector. Changing any
of them changes the label target and requires a new label version.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Callable, Hashable, Mapping, Sequence

import numpy as np

from isingfold.rl.contracts import (
    Context,
    DecisionState,
    InitFailureRecord,
    Opcode,
    TerminalRecord,
    stable_digest,
)
from isingfold.rl.data.action_certificate import (
    StateActionEnvelopeV1,
    apply_envelope_action,
)
from isingfold.rl.data.prepared import PreparedTask
from isingfold.rl.env import EmbeddingEnv, EmbeddingTask, StrengthSelector
from isingfold.rl.evaluator import sample_program
from isingfold.rl.program import Program, compile_registry, program_features, strength_registry
from isingfold.rl.initializer_bank import (
    InitializerSnapshotBank,
    public_task_digest,
)
from isingfold.rl.proposal import Initializer, LEGACY_ONLINE_INITIALIZER_RESTARTS_V1

Node = Hashable
Qubit = Hashable

ContinuationPolicy = Callable[[DecisionState, random.Random], int]

CONTINUATION_POLICY_ID = "uniform-exact-legal-support-v1"
COMMIT_FIRST_POLICY_ID = "commit-first-then-uniform-v1"
CONTINUATION_MAX_STEPS = 32
BEST_ACTION_RULE_ID = "simultaneous-hoeffding-overlap-v1"
BEST_ACTION_ALPHA = 0.05
ACTION_SUBSET_POLICY_ID = "protected-incumbent-commit-plus-uniform-legal-remainder-v1"
QUALITY_LABEL_VERSION = "if-q3-s0-qmu-7"
CONTINUATION_RECEIPT_SCHEMA = "isingfold.quality-continuation"
CONTINUATION_RECEIPT_VERSION = 2
QUALITY_INITIALIZER_BINDING_SCHEMA = "isingfold.quality-initializer-binding"
QUALITY_INITIALIZER_BINDING_VERSION = 1
QUALITY_INITIALIZER_BANK_PROTOCOL = "authenticated-persistent-k2-initializer-bank-v1"
QUALITY_INITIALIZER_LEGACY_PROTOCOL = "legacy-online-initializer-restarts-diagnostic-v1"
QUALITY_INITIALIZER_RESTART_CACHE_WIDTH = 2


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def legacy_quality_initializer_binding() -> dict[str, object]:
    """Identify the retained non-publication legacy replay path explicitly."""

    body: dict[str, object] = {
        "schema": QUALITY_INITIALIZER_BINDING_SCHEMA,
        "schema_version": QUALITY_INITIALIZER_BINDING_VERSION,
        "protocol": QUALITY_INITIALIZER_LEGACY_PROTOCOL,
        "publication_eligible": False,
    }
    return {**body, "record_digest": stable_digest(body)}


def quality_initializer_bank_contract(
    bank: InitializerSnapshotBank,
    *,
    expected_manifest_sha256: str,
    allow_test_bank: bool = False,
) -> dict[str, object]:
    """Authenticate the target-free global bank identity used by quality labels.

    Loading the bank verifies its files.  This second boundary binds that verified object to
    the caller's out-of-band manifest pin and to the exact persistent K=2 deployment contract.
    Test-only initializer backends remain available only through an explicit diagnostic opt-in.
    """

    if not isinstance(bank, InitializerSnapshotBank):
        raise TypeError("quality labeling requires an InitializerSnapshotBank")
    expected = _digest(expected_manifest_sha256, "initializer-bank manifest pin")
    access = bank.access_receipt.as_dict()
    plan = bank.plan.as_dict()
    if (
        access.get("manifest_sha256") != expected
        or access.get("manifest_record_digest") != bank.access_receipt.manifest_record_digest
        or access.get("plan_record_digest") != bank.plan.record_digest
        or access.get("prepared_manifest_sha256") != bank.plan.prepared_manifest_sha256
        or access.get("partition") != "train"
    ):
        raise ValueError("initializer-bank manifest pin or access identity differs")
    if (
        access.get("opened_evaluator_targets") is not False
        or plan.get("opened_evaluator_targets", False) is not False
    ):
        raise ValueError("quality initializer bank must not open evaluator targets")
    if (
        bank.plan.restart_cache_slots_per_episode != QUALITY_INITIALIZER_RESTART_CACHE_WIDTH
        or plan.get("restart_cache_slots_per_episode") != QUALITY_INITIALIZER_RESTART_CACHE_WIDTH
    ):
        raise ValueError("quality initializer bank must seal exactly two restart-cache slots")
    if bank.plan.partition != "train" or plan.get("partition") != "train":
        raise ValueError("quality initializer bank must be train-only")
    if bank.plan.publication_eligible is not True and not allow_test_bank:
        raise ValueError("quality initializer bank is not publication-eligible")
    body: dict[str, object] = {
        "schema": "isingfold.quality-initializer-bank-contract",
        "schema_version": 1,
        "protocol": QUALITY_INITIALIZER_BANK_PROTOCOL,
        "manifest_sha256": expected,
        "manifest_record_digest": bank.access_receipt.manifest_record_digest,
        "access_receipt_record_digest": bank.access_receipt.record_digest,
        "plan_record_digest": bank.plan.record_digest,
        "prepared_manifest_sha256": bank.plan.prepared_manifest_sha256,
        "partition": bank.plan.partition,
        "config_digest": bank.plan.config_digest,
        "context_digest": bank.plan.context_digest,
        "training_seed": bank.plan.training_seed,
        "episode_schedule_start": bank.plan.episode_schedule_start,
        "episode_schedule_stop_exclusive": bank.plan.episode_schedule_stop_exclusive,
        "conditional_episode_count": len(bank),
        "restart_cache_slots_per_episode": bank.plan.restart_cache_slots_per_episode,
        "opened_evaluator_data": False,
        "publication_eligible": bank.plan.publication_eligible,
    }
    contract = {**body, "record_digest": stable_digest(body)}
    validate_quality_initializer_bank_contract(
        contract,
        require_publication=not allow_test_bank,
    )
    return contract


def validate_quality_initializer_bank_contract(
    contract: Mapping[str, object],
    *,
    require_publication: bool = True,
) -> None:
    """Validate a serialized bank contract without silently accepting an older shape."""

    fields = {
        "access_receipt_record_digest",
        "conditional_episode_count",
        "config_digest",
        "context_digest",
        "episode_schedule_start",
        "episode_schedule_stop_exclusive",
        "manifest_record_digest",
        "manifest_sha256",
        "opened_evaluator_data",
        "partition",
        "plan_record_digest",
        "prepared_manifest_sha256",
        "protocol",
        "publication_eligible",
        "record_digest",
        "restart_cache_slots_per_episode",
        "schema",
        "schema_version",
        "training_seed",
    }
    if not isinstance(contract, Mapping) or set(contract) != fields:
        raise ValueError("quality initializer-bank contract schema differs")
    recorded = _digest(contract.get("record_digest"), "quality initializer-bank contract digest")
    unsigned = {key: contract[key] for key in contract if key != "record_digest"}
    if not hmac.compare_digest(recorded, stable_digest(unsigned)):
        raise ValueError("quality initializer-bank contract digest mismatch")
    if (
        contract.get("schema") != "isingfold.quality-initializer-bank-contract"
        or contract.get("schema_version") != 1
        or contract.get("protocol") != QUALITY_INITIALIZER_BANK_PROTOCOL
        or contract.get("partition") != "train"
        or contract.get("opened_evaluator_data") is not False
        or contract.get("restart_cache_slots_per_episode")
        != QUALITY_INITIALIZER_RESTART_CACHE_WIDTH
    ):
        raise ValueError("quality initializer-bank contract differs from persistent K=2")
    if require_publication and contract.get("publication_eligible") is not True:
        raise ValueError("quality initializer bank is not publication-eligible")
    for field in (
        "access_receipt_record_digest",
        "config_digest",
        "context_digest",
        "manifest_record_digest",
        "manifest_sha256",
        "plan_record_digest",
        "prepared_manifest_sha256",
    ):
        _digest(contract.get(field), f"quality initializer-bank contract {field}")
    start = contract.get("episode_schedule_start")
    stop = contract.get("episode_schedule_stop_exclusive")
    count = contract.get("conditional_episode_count")
    training_seed = contract.get("training_seed")
    if (
        type(start) is not int
        or type(stop) is not int
        or type(count) is not int
        or type(training_seed) is not int
        or start < 0
        or stop <= start
        or count != stop - start
        or training_seed < 0
    ):
        raise ValueError("quality initializer-bank schedule is malformed")


def quality_initializer_bank_episode_binding(
    bank: InitializerSnapshotBank,
    *,
    expected_manifest_sha256: str,
    episode_schedule_index: int,
    task: EmbeddingTask,
    ctx: Context,
    allow_test_bank: bool = False,
) -> dict[str, object]:
    """Bind one quality state to one immutable initializer/cache episode."""

    if (
        isinstance(episode_schedule_index, bool)
        or not isinstance(episode_schedule_index, int)
        or episode_schedule_index < 0
    ):
        raise ValueError("quality initializer-bank episode index must be nonnegative")
    contract = quality_initializer_bank_contract(
        bank,
        expected_manifest_sha256=expected_manifest_sha256,
        allow_test_bank=allow_test_bank,
    )
    try:
        bootstrap = bank.bootstrap_outcome(episode_schedule_index)
    except KeyError as error:
        raise ValueError("quality initializer-bank episode is outside the sealed bank") from error
    if (
        bootstrap.context_digest != context_digest(ctx)
        or bootstrap.plan_record_digest != contract["plan_record_digest"]
        or bootstrap.manifest_record_digest != contract["manifest_record_digest"]
        or bootstrap.config_digest != contract["config_digest"]
        or bootstrap.restart_cache_slot_count != QUALITY_INITIALIZER_RESTART_CACHE_WIDTH
        or len(bootstrap.restart_cache_snapshots) != QUALITY_INITIALIZER_RESTART_CACHE_WIDTH
    ):
        raise ValueError("quality initializer bootstrap differs from the bank contract")
    if (
        task.lineage != bootstrap.base_lineage
        or public_task_digest(task, instance_id=bootstrap.instance_id)
        != bootstrap.public_task_digest
    ):
        raise ValueError("quality task differs from the initializer-bank episode")
    body: dict[str, object] = {
        "schema": QUALITY_INITIALIZER_BINDING_SCHEMA,
        "schema_version": QUALITY_INITIALIZER_BINDING_VERSION,
        "protocol": QUALITY_INITIALIZER_BANK_PROTOCOL,
        "publication_eligible": contract["publication_eligible"],
        "bank_contract_record_digest": contract["record_digest"],
        "manifest_sha256": contract["manifest_sha256"],
        "manifest_record_digest": contract["manifest_record_digest"],
        "access_receipt_record_digest": contract["access_receipt_record_digest"],
        "plan_record_digest": contract["plan_record_digest"],
        "prepared_manifest_sha256": contract["prepared_manifest_sha256"],
        "config_digest": contract["config_digest"],
        "context_digest": contract["context_digest"],
        "episode_schedule_index": episode_schedule_index,
        "instance_id": bootstrap.instance_id,
        "base_lineage": bootstrap.base_lineage,
        "public_task_digest": bootstrap.public_task_digest,
        "bootstrap_record_digest": bootstrap.record_digest,
        "initial_snapshot_record_digest": bootstrap.initial_snapshot.record_digest,
        "restart_snapshot_record_digests": [
            snapshot.record_digest for snapshot in bootstrap.restart_cache_snapshots
        ],
        "restart_cache_slots": QUALITY_INITIALIZER_RESTART_CACHE_WIDTH,
        "opened_evaluator_data": False,
    }
    return {**body, "record_digest": stable_digest(body)}


def validate_quality_initializer_binding(binding: Mapping[str, object]) -> None:
    """Validate the exact v1 initializer identity embedded in a continuation receipt."""

    if not isinstance(binding, Mapping):
        raise ValueError("quality continuation initializer binding must be an object")
    digest = _digest(binding.get("record_digest"), "quality initializer binding digest")
    unsigned = {key: binding[key] for key in binding if key != "record_digest"}
    if not hmac.compare_digest(digest, stable_digest(unsigned)):
        raise ValueError("quality initializer binding digest mismatch")
    if (
        binding.get("schema") != QUALITY_INITIALIZER_BINDING_SCHEMA
        or binding.get("schema_version") != QUALITY_INITIALIZER_BINDING_VERSION
    ):
        raise ValueError("unsupported quality initializer binding")
    protocol = binding.get("protocol")
    if protocol == QUALITY_INITIALIZER_LEGACY_PROTOCOL:
        if (
            set(binding)
            != {
                "schema",
                "schema_version",
                "protocol",
                "publication_eligible",
                "record_digest",
            }
            or binding.get("publication_eligible") is not False
        ):
            raise ValueError("legacy quality initializer binding is malformed")
        return
    bank_fields = {
        "access_receipt_record_digest",
        "bank_contract_record_digest",
        "base_lineage",
        "bootstrap_record_digest",
        "config_digest",
        "context_digest",
        "episode_schedule_index",
        "initial_snapshot_record_digest",
        "instance_id",
        "manifest_record_digest",
        "manifest_sha256",
        "opened_evaluator_data",
        "plan_record_digest",
        "prepared_manifest_sha256",
        "protocol",
        "public_task_digest",
        "publication_eligible",
        "record_digest",
        "restart_cache_slots",
        "restart_snapshot_record_digests",
        "schema",
        "schema_version",
    }
    if protocol != QUALITY_INITIALIZER_BANK_PROTOCOL or set(binding) != bank_fields:
        raise ValueError("quality initializer-bank binding schema differs")
    if (
        binding.get("opened_evaluator_data") is not False
        or binding.get("restart_cache_slots") != QUALITY_INITIALIZER_RESTART_CACHE_WIDTH
        or type(binding.get("episode_schedule_index")) is not int
        or int(binding["episode_schedule_index"]) < 0
        or not isinstance(binding.get("instance_id"), str)
        or not binding["instance_id"]
        or not isinstance(binding.get("base_lineage"), str)
        or not binding["base_lineage"]
    ):
        raise ValueError("quality initializer-bank binding scalar fields are malformed")
    for field in (
        "access_receipt_record_digest",
        "bank_contract_record_digest",
        "bootstrap_record_digest",
        "config_digest",
        "context_digest",
        "initial_snapshot_record_digest",
        "manifest_record_digest",
        "manifest_sha256",
        "plan_record_digest",
        "prepared_manifest_sha256",
        "public_task_digest",
    ):
        _digest(binding.get(field), f"quality initializer binding {field}")
    restart_digests = binding.get("restart_snapshot_record_digests")
    if (
        not isinstance(restart_digests, list)
        or len(restart_digests) != QUALITY_INITIALIZER_RESTART_CACHE_WIDTH
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in restart_digests
        )
    ):
        raise ValueError("quality initializer binding restart snapshots are malformed")


def random_masked_policy(decision: DecisionState, rng: random.Random) -> int:
    """The frozen reference continuation ``mu``: uniform over the exact legal support."""

    legal = [k for k, m in enumerate(decision.legal_mask) if m]
    return rng.choice(legal)


def commit_first_policy(decision: DecisionState, rng: random.Random) -> int:
    """A second registered continuation: stop at the first admissible archive entry."""

    for k, cand in enumerate(decision.candidates):
        if cand.opcode.value == "COMMIT" and decision.legal_mask[k]:
            return k
    return random_masked_policy(decision, rng)


def continuation_policy_id(policy: ContinuationPolicy) -> str:
    """Return the immutable semantic identity of a registered executable continuation."""

    registered = {
        random_masked_policy: CONTINUATION_POLICY_ID,
        commit_first_policy: COMMIT_FIRST_POLICY_ID,
    }
    try:
        return registered[policy]
    except KeyError as error:
        raise ValueError("quality labels require a registered continuation policy") from error


@dataclass(frozen=True)
class DeterministicActionSample:
    """One outcome-blind action draw and its exact first-order propensities."""

    action_indices: tuple[int, ...]
    inclusion_probabilities: tuple[float, ...]
    protected_commit_index: int | None

    def __post_init__(self) -> None:
        if (
            not self.action_indices
            or tuple(sorted(set(self.action_indices))) != self.action_indices
        ):
            raise ValueError("quality action sample indices must be nonempty, unique, and sorted")
        if len(self.inclusion_probabilities) != len(self.action_indices):
            raise ValueError("quality action sample probabilities differ from its indices")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 < value <= 1.0
            for value in self.inclusion_probabilities
        ):
            raise ValueError("quality action sample probabilities must lie in (0, 1]")
        if self.protected_commit_index is not None:
            if (
                isinstance(self.protected_commit_index, bool)
                or not isinstance(self.protected_commit_index, int)
                or self.protected_commit_index not in self.action_indices
            ):
                raise ValueError("protected COMMIT must be present in the quality action sample")
            if self.inclusion_probability(self.protected_commit_index) != 1.0:
                raise ValueError("protected COMMIT must have unit inclusion probability")

    def inclusion_probability(self, action_index: int) -> float:
        """Return the registered propensity for one selected absolute action index."""

        try:
            position = self.action_indices.index(action_index)
        except ValueError as error:
            raise KeyError(f"action {action_index} is absent from the quality sample") from error
        return float(self.inclusion_probabilities[position])


def deterministic_action_sample(
    decision: DecisionState,
    *,
    evaluated_actions: int,
    seed: int,
) -> DeterministicActionSample:
    """Reproduce the registered, outcome-blind action draw for one decision.

    The protected incumbent COMMIT is always retained when present.  Every remaining legal
    action, including other COMMIT and STOP actions, belongs to one uniform without-replacement
    pool.  Generation and authentication share this function, including its propensities.
    """

    if not isinstance(decision, DecisionState):
        raise TypeError("quality action sampling requires an exact DecisionState")
    if (
        isinstance(evaluated_actions, bool)
        or not isinstance(evaluated_actions, int)
        or evaluated_actions <= 0
    ):
        raise ValueError("evaluated_actions must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("quality subset seed must be a nonnegative 63-bit integer")
    if not isinstance(decision.state_fingerprint, str) or not decision.state_fingerprint:
        raise ValueError("quality subset requires a state fingerprint")
    if len(decision.candidates) != len(decision.legal_mask):
        raise ValueError("quality subset candidate support and legal mask differ")
    if any(type(value) is not bool for value in decision.legal_mask):
        raise ValueError("quality subset legal mask must be Boolean")
    legal = [index for index, is_legal in enumerate(decision.legal_mask) if is_legal]
    if not legal:
        raise ValueError("quality subset requires at least one legal action")
    protected = [
        index
        for index in legal
        if decision.candidates[index].opcode is Opcode.COMMIT
        and decision.candidates[index].archive_ref == 0
    ]
    if len(protected) > 1:
        raise ValueError("quality subset found multiple legal protected COMMIT actions")

    sample_size = min(evaluated_actions, len(legal))
    anchor = protected[0] if protected else None
    if anchor is not None and len(legal) > 1 and sample_size < 2:
        raise ValueError(
            "protected-COMMIT sampling requires at least two evaluated actions when other "
            "legal actions exist"
        )
    rng = random.Random(
        _stable_seed(
            seed,
            "quality-action-subset-protected-commit-v1",
            decision.state_fingerprint,
        )
    )
    if sample_size == len(legal):
        chosen = legal
        probability_by_index = {index: 1.0 for index in legal}
    elif anchor is None:
        chosen = rng.sample(legal, sample_size)
        probability = sample_size / len(legal)
        probability_by_index = {index: probability for index in chosen}
    else:
        remaining = [index for index in legal if index != anchor]
        remaining_size = sample_size - 1
        chosen = [anchor, *rng.sample(remaining, remaining_size)]
        remaining_probability = remaining_size / len(remaining)
        probability_by_index = {
            anchor: 1.0,
            **{index: remaining_probability for index in chosen if index != anchor},
        }
    indices = tuple(sorted(chosen))
    return DeterministicActionSample(
        action_indices=indices,
        inclusion_probabilities=tuple(probability_by_index[index] for index in indices),
        protected_commit_index=anchor,
    )


@dataclass(frozen=True)
class ContinuationResult:
    returned_valid: bool
    reward: float
    embedding: Mapping[Node, frozenset[Qubit]] | None
    selected_index: int | None
    steps: int
    receipt: Mapping[str, object] | None = None


def _quality_environment(
    task: EmbeddingTask,
    ctx: Context,
    *,
    initializer: Initializer | None,
    initializer_bank: InitializerSnapshotBank | None,
    expected_initializer_bank_manifest_sha256: str | None,
    initializer_bank_episode_index: int | None,
    prepared_task: PreparedTask | None,
    target_access: Mapping[str, object] | None,
    ground_partition_receipt: Mapping[str, object] | None,
    allow_test_initializer_bank: bool,
    selector: StrengthSelector,
    reward_reads: int | None,
    seed: int,
) -> tuple[EmbeddingEnv, Mapping[str, object]]:
    """Create exactly one legacy diagnostic or authenticated bank-backed environment."""

    bank_inputs_present = any(
        value is not None
        for value in (
            initializer_bank,
            expected_initializer_bank_manifest_sha256,
            initializer_bank_episode_index,
            prepared_task,
            target_access,
            ground_partition_receipt,
        )
    )
    if (initializer is not None) == bank_inputs_present:
        raise ValueError(
            "quality replay requires exactly one initializer mode: legacy diagnostic or "
            "authenticated initializer bank"
        )
    if initializer is not None:
        return (
            EmbeddingEnv(
                task,
                ctx,
                initializer=initializer,
                selector=selector,
                reward_reads=reward_reads,
                improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
                seed=seed,
            ),
            legacy_quality_initializer_binding(),
        )
    if (
        initializer_bank is None
        or expected_initializer_bank_manifest_sha256 is None
        or initializer_bank_episode_index is None
    ):
        raise ValueError("banked quality replay requires the bank, manifest pin, and episode index")
    binding = quality_initializer_bank_episode_binding(
        initializer_bank,
        expected_manifest_sha256=expected_initializer_bank_manifest_sha256,
        episode_schedule_index=initializer_bank_episode_index,
        task=task,
        ctx=ctx,
        allow_test_bank=allow_test_initializer_bank,
    )
    bootstrap = initializer_bank.bootstrap_outcome(initializer_bank_episode_index)
    if seed != bootstrap.initial_snapshot.system_seed:
        raise ValueError("quality environment seed differs from its initializer-bank episode")
    if prepared_task is None:
        if target_access is not None or ground_partition_receipt is not None:
            raise ValueError("target authority requires an explicit train PreparedTask")
        environment = initializer_bank.environment(
            initializer_bank_episode_index,
            context=ctx,
            selector=selector,
            reward_reads=reward_reads,
        )
    else:
        if task is not prepared_task.task:
            raise ValueError("quality replay task differs from its PreparedTask")
        normalized_target_access = _jsonable(target_access)
        normalized_ground_receipt = _jsonable(ground_partition_receipt)
        if not isinstance(normalized_target_access, dict) or not isinstance(
            normalized_ground_receipt, dict
        ):
            raise ValueError("quality target authorities must be JSON objects")
        environment = initializer_bank.environment(
            initializer_bank_episode_index,
            context=ctx,
            selector=selector,
            reward_reads=reward_reads,
            training_task=prepared_task,
            target_access=normalized_target_access,
            ground_partition_receipt=normalized_ground_receipt,
        )
    return environment, binding


def run_continuation(
    task: EmbeddingTask,
    ctx: Context,
    *,
    initializer: Initializer | None,
    initializer_bank: InitializerSnapshotBank | None = None,
    expected_initializer_bank_manifest_sha256: str | None = None,
    initializer_bank_episode_index: int | None = None,
    prepared_task: PreparedTask | None = None,
    target_access: Mapping[str, object] | None = None,
    ground_partition_receipt: Mapping[str, object] | None = None,
    allow_test_initializer_bank: bool = False,
    selector: StrengthSelector,
    prefix: Sequence[int],
    policy: ContinuationPolicy = random_masked_policy,
    seed: int = 0,
    continuation_seed: int | None = None,
    reward_reads: int | None = None,
    max_steps: int = 64,
) -> ContinuationResult | None:
    """Replay a recorded prefix deterministically, then continue under ``policy``.

    The environment is deterministic given its seed and the selected indices, so a prefix is
    replayed exactly rather than snapshotted.
    """

    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    env, initializer_binding = _quality_environment(
        task,
        ctx,
        initializer=initializer,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=(expected_initializer_bank_manifest_sha256),
        initializer_bank_episode_index=initializer_bank_episode_index,
        prepared_task=prepared_task,
        target_access=target_access,
        ground_partition_receipt=ground_partition_receipt,
        allow_test_initializer_bank=allow_test_initializer_bank,
        selector=selector,
        reward_reads=reward_reads,
        seed=seed,
    )
    result = env.reset(seed)
    if isinstance(result, (InitFailureRecord, TerminalRecord)):
        return None
    action_trace: list[dict[str, object]] = []

    def trace(decision: DecisionState, index: int, *, source: str) -> None:
        candidate = decision.candidates[index]
        action_trace.append(
            {
                "position": len(action_trace),
                "source": source,
                "action_index": index,
                "opcode": candidate.opcode.value,
                "payload_key": candidate.payload_key,
                "state_fingerprint": decision.state_fingerprint,
                "support_fingerprint": decision.support_fingerprint,
            }
        )

    fork_position = len(prefix) - 1 if prefix and continuation_seed is not None else len(prefix)
    for position, index in enumerate(prefix):
        if not isinstance(result, DecisionState):
            return None
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(result.candidates)
            or not result.legal_mask[index]
        ):
            return None
        # Replay the recorded state/support with the common prefix RNG.  The environment
        # verifies that support first, then forks transition randomness before preparing
        # the successor.  This keeps the chosen payload identical without staling it.
        trace(result, index, source="forced-prefix")
        branch_seed = continuation_seed if position == fork_position else None
        step = env.step(result, index, transition_seed=branch_seed)
        result = step.next_decision_or_terminal
    policy_root = seed if continuation_seed is None else continuation_seed
    policy_identity = (
        f"{getattr(policy, '__module__', '')}.{getattr(policy, '__qualname__', repr(policy))}"
    )
    rng = random.Random(
        _stable_seed(policy_root, "quality-continuation-policy-rng", policy_identity)
    )
    steps = 0
    while isinstance(result, DecisionState) and steps < max_steps:
        branch_seed = continuation_seed if not prefix and steps == 0 else None
        chosen = policy(result, rng)
        trace(result, chosen, source="continuation-policy")
        step = env.step(result, chosen, transition_seed=branch_seed)
        result = step.next_decision_or_terminal
        steps += 1
    if not isinstance(result, TerminalRecord):
        return None
    if result.returned_valid and result.training_reward is None:
        raise RuntimeError("quality continuation ended without its registered reward block")
    receipt = continuation_receipt(
        result,
        action_trace=action_trace,
        continuation_seed=policy_root,
        continuation_policy=continuation_policy_id(policy),
        continuation_steps=steps,
        reward_reads=env.reward_reads,
        num_sweeps=ctx.num_sweeps,
        initializer_binding=initializer_binding,
    )
    return ContinuationResult(
        returned_valid=result.returned_valid,
        reward=float(result.training_reward if result.training_reward is not None else 0.0),
        embedding=result.embedding,
        selected_index=result.selected_index,
        steps=steps,
        receipt=receipt,
    )


@dataclass(frozen=True)
class ActionQuality:
    action_index: int
    payload_key: str
    opcode: str
    q_mu: float
    continuations: int
    valid_returns: int
    inclusion_probability: float
    selected_payload_digest: str
    applied_action_record_digest: str | None
    continuation_seeds: tuple[int, ...] = ()
    continuation_rewards: tuple[float, ...] = ()
    continuation_valid: tuple[bool, ...] = ()
    continuation_receipts: tuple[Mapping[str, object], ...] = ()

    def validate(self) -> None:
        """Recompute every scalar denominator from the retained continuation outcomes."""

        if (
            isinstance(self.action_index, bool)
            or not isinstance(self.action_index, int)
            or self.action_index < 0
        ):
            raise ValueError("quality action index must be a nonnegative integer")
        if not isinstance(self.payload_key, str) or not self.payload_key:
            raise ValueError("quality action has no payload identity")
        if not isinstance(self.opcode, str) or not self.opcode:
            raise ValueError("quality action has no opcode identity")
        try:
            opcode = Opcode(self.opcode)
        except ValueError as error:
            raise ValueError("quality action opcode is not registered") from error
        if (
            not isinstance(self.selected_payload_digest, str)
            or len(self.selected_payload_digest) != 64
            or any(
                character not in "0123456789abcdef" for character in self.selected_payload_digest
            )
        ):
            raise ValueError("quality action has no valid selected-payload digest")
        applied_digest = self.applied_action_record_digest
        if opcode.is_terminal:
            if applied_digest is not None:
                raise ValueError(
                    "a terminal quality action opcode cannot carry a workspace successor"
                )
        elif (
            not isinstance(applied_digest, str)
            or len(applied_digest) != 64
            or any(character not in "0123456789abcdef" for character in applied_digest)
        ):
            raise ValueError("a workspace quality action requires an applied-action digest")
        if (
            isinstance(self.continuations, bool)
            or not isinstance(self.continuations, int)
            or self.continuations <= 0
        ):
            raise ValueError("quality continuation denominator must be positive")
        if (
            isinstance(self.valid_returns, bool)
            or not isinstance(self.valid_returns, int)
            or not 0 <= self.valid_returns <= self.continuations
        ):
            raise ValueError("quality valid-return count is outside its denominator")
        if (
            isinstance(self.inclusion_probability, bool)
            or not isinstance(self.inclusion_probability, (int, float))
            or not math.isfinite(self.inclusion_probability)
            or not 0.0 < self.inclusion_probability <= 1.0
        ):
            raise ValueError("quality inclusion probability must lie in (0, 1]")
        if (
            len(self.continuation_seeds) != self.continuations
            or len(self.continuation_rewards) != self.continuations
            or len(self.continuation_valid) != self.continuations
            or len(self.continuation_receipts) != self.continuations
        ):
            raise ValueError("quality continuation arrays differ from their denominator")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
            for seed in self.continuation_seeds
        ):
            raise ValueError("quality continuation seed is not a nonnegative 63-bit integer")
        if len(set(self.continuation_seeds)) != self.continuations:
            raise ValueError("quality continuation seeds are not unique")
        if any(type(flag) is not bool for flag in self.continuation_valid):
            raise ValueError("quality continuation-valid array is not Boolean")
        if sum(self.continuation_valid) != self.valid_returns:
            raise ValueError("quality valid-return count differs from its Boolean outcomes")
        for position, receipt in enumerate(self.continuation_receipts):
            validate_continuation_receipt(receipt)
            if (
                receipt["continuation_seed"] != self.continuation_seeds[position]
                or receipt["returned_valid"] is not self.continuation_valid[position]
                or not math.isclose(
                    float(receipt["reward"]),
                    float(self.continuation_rewards[position]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError("quality continuation receipt disagrees with its action arrays")
        if any(
            isinstance(reward, bool) or not isinstance(reward, (int, float))
            for reward in self.continuation_rewards
        ):
            raise ValueError("quality continuation rewards must be numeric, not Boolean")
        rewards = np.asarray(self.continuation_rewards, dtype=np.float64)
        if (
            rewards.shape != (self.continuations,)
            or not np.isfinite(rewards).all()
            or np.any(rewards < 0.0)
            or np.any(rewards > 1.0)
        ):
            raise ValueError("quality continuation rewards must be finite and bounded")
        if any(
            not flag and reward != 0.0 for flag, reward in zip(self.continuation_valid, rewards)
        ):
            raise ValueError("an invalid continuation must contribute exactly zero utility")
        mean = float(rewards.mean())
        if (
            isinstance(self.q_mu, bool)
            or not isinstance(self.q_mu, (int, float))
            or not math.isfinite(self.q_mu)
            or not 0.0 <= self.q_mu <= 1.0
            or not math.isclose(mean, self.q_mu, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError("quality mean differs from its recorded continuation outcomes")


@dataclass(frozen=True)
class QualityRecord:
    """One decision with its full stored support and labels for an evaluated subset."""

    instance: str
    lineage: str
    prefix: tuple[int, ...]
    support: tuple[Mapping[str, object], ...]
    evaluated: tuple[ActionQuality, ...]
    continuation_policy: str
    action_envelope_record_digest: str
    state_fingerprint: str = ""
    support_fingerprint: str = ""
    context_version: str = ""
    charged_work_receipt: Mapping[str, int] | None = None
    exact_state: Mapping[str, object] | None = None
    observation: Mapping[str, object] | None = None
    environment_seed: int = 0
    context_digest: str = ""
    label_version: str = QUALITY_LABEL_VERSION

    def best_actions(self, alpha: float = BEST_ACTION_ALPHA) -> tuple[int, ...]:
        """Actions not ruled out as best by simultaneous bounded-mean intervals.

        The returned set is intentionally conservative.  A row for which every evaluated
        action remains plausible contributes zero subset-ranking signal and should be
        skipped by the warm-start loader, rather than converted into a noisy winner.
        """

        if not self.evaluated:
            return ()
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("best-action alpha must lie strictly between zero and one")
        arm_count = len(self.evaluated)
        intervals: list[tuple[ActionQuality, float, float]] = []
        for action in self.evaluated:
            action.validate()
            rewards = np.asarray(action.continuation_rewards, dtype=np.float64)
            mean = float(rewards.mean())
            radius = math.sqrt(math.log(2.0 * arm_count / alpha) / (2.0 * action.continuations))
            intervals.append((action, max(0.0, mean - radius), min(1.0, mean + radius)))
        best_lower = max(lower for _action, lower, _upper in intervals)
        return tuple(
            action.action_index for action, _lower, upper in intervals if upper >= best_lower
        )


def label_decision(
    task: EmbeddingTask,
    ctx: Context,
    *,
    initializer: Initializer | None,
    initializer_bank: InitializerSnapshotBank | None = None,
    expected_initializer_bank_manifest_sha256: str | None = None,
    initializer_bank_episode_index: int | None = None,
    prepared_task: PreparedTask | None = None,
    target_access: Mapping[str, object] | None = None,
    ground_partition_receipt: Mapping[str, object] | None = None,
    allow_test_initializer_bank: bool = False,
    selector: StrengthSelector,
    prefix: Sequence[int],
    seed: int,
    provenance_fingerprint: str,
    evaluated_actions: int = 8,
    continuations: int = 2,
    reward_reads: int = 128,
    policy: ContinuationPolicy = random_masked_policy,
) -> QualityRecord | None:
    """Store the full candidate support, then label a recorded random subset of it."""

    for name, value in (
        ("evaluated_actions", evaluated_actions),
        ("continuations", continuations),
        ("reward_reads", reward_reads),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    replayed = replay_decision_with_action_envelope(
        task,
        ctx,
        initializer=initializer,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=(expected_initializer_bank_manifest_sha256),
        initializer_bank_episode_index=initializer_bank_episode_index,
        prepared_task=prepared_task,
        target_access=target_access,
        ground_partition_receipt=ground_partition_receipt,
        allow_test_initializer_bank=allow_test_initializer_bank,
        selector=selector,
        prefix=prefix,
        seed=seed,
        reward_reads=reward_reads,
        provenance_fingerprint=provenance_fingerprint,
    )
    if replayed is None:
        return None
    result, action_envelope = replayed
    support, exact_state, observation = decision_snapshot(result)
    action_sample = deterministic_action_sample(
        result,
        evaluated_actions=evaluated_actions,
        seed=seed,
    )

    labels: list[ActionQuality] = []
    for k in action_sample.action_indices:
        bound_action = action_envelope.candidates[k]
        applied_action_digest = None
        if not bound_action.opcode.is_terminal:
            applied_action = apply_envelope_action(action_envelope, k)
            applied_action.verify_against(action_envelope)
            applied_action_digest = applied_action.record_digest
        values: list[float] = []
        valid_flags: list[bool] = []
        continuation_seeds: list[int] = []
        continuation_receipts: list[Mapping[str, object]] = []
        valid = 0
        for m in range(continuations):
            seed_for_continuation = continuation_seed(
                seed, task.name, result.state_fingerprint, k, m
            )
            outcome = run_continuation(
                task,
                ctx,
                initializer=initializer,
                initializer_bank=initializer_bank,
                expected_initializer_bank_manifest_sha256=(
                    expected_initializer_bank_manifest_sha256
                ),
                initializer_bank_episode_index=initializer_bank_episode_index,
                prepared_task=prepared_task,
                target_access=target_access,
                ground_partition_receipt=ground_partition_receipt,
                allow_test_initializer_bank=allow_test_initializer_bank,
                selector=selector,
                prefix=(*prefix, k),
                policy=policy,
                seed=seed,
                continuation_seed=seed_for_continuation,
                reward_reads=reward_reads,
                max_steps=CONTINUATION_MAX_STEPS,
            )
            if outcome is None:
                raise RuntimeError(
                    "a materialized legal counterfactual failed exact prefix replay; "
                    "the label job must abort instead of shrinking its denominator"
                )
            continuation_seeds.append(seed_for_continuation)
            if outcome.receipt is None:
                raise RuntimeError("quality continuation omitted its exact receipt")
            validate_continuation_receipt(outcome.receipt)
            continuation_receipts.append(outcome.receipt)
            valid += 1 if outcome.returned_valid else 0
            values.append(outcome.reward if outcome.returned_valid else 0.0)
            valid_flags.append(outcome.returned_valid)
        if len(values) != continuations:
            raise RuntimeError("counterfactual continuation count differs from its registry")
        labels.append(
            ActionQuality(
                action_index=k,
                payload_key=result.candidates[k].payload_key,
                opcode=result.candidates[k].opcode.value,
                q_mu=float(np.mean(values)),
                continuations=len(values),
                valid_returns=valid,
                inclusion_probability=action_sample.inclusion_probability(k),
                selected_payload_digest=bound_action.payload_digest,
                applied_action_record_digest=applied_action_digest,
                continuation_seeds=tuple(continuation_seeds),
                continuation_rewards=tuple(values),
                continuation_valid=tuple(valid_flags),
                continuation_receipts=tuple(continuation_receipts),
            )
        )

    return QualityRecord(
        instance=task.name,
        lineage=task.lineage,
        prefix=tuple(prefix),
        support=support,
        evaluated=tuple(labels),
        continuation_policy=continuation_policy_id(policy),
        action_envelope_record_digest=action_envelope.record_digest,
        state_fingerprint=result.state_fingerprint,
        support_fingerprint=result.support_fingerprint,
        context_version=result.context_version,
        charged_work_receipt=result.charged_work_receipt.as_dict(),
        exact_state=exact_state,
        observation=observation,
        environment_seed=seed,
        context_digest=context_digest(ctx),
        label_version=QUALITY_LABEL_VERSION,
    )


def replay_decision(
    task: EmbeddingTask,
    ctx: Context,
    *,
    initializer: Initializer | None,
    initializer_bank: InitializerSnapshotBank | None = None,
    expected_initializer_bank_manifest_sha256: str | None = None,
    initializer_bank_episode_index: int | None = None,
    prepared_task: PreparedTask | None = None,
    target_access: Mapping[str, object] | None = None,
    ground_partition_receipt: Mapping[str, object] | None = None,
    allow_test_initializer_bank: bool = False,
    selector: StrengthSelector,
    prefix: Sequence[int],
    seed: int,
    reward_reads: int,
) -> DecisionState | None:
    """Materialize one recorded state without evaluating terminal quality."""

    replayed = _replay_live_decision(
        task,
        ctx,
        initializer=initializer,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=(expected_initializer_bank_manifest_sha256),
        initializer_bank_episode_index=initializer_bank_episode_index,
        prepared_task=prepared_task,
        target_access=target_access,
        ground_partition_receipt=ground_partition_receipt,
        allow_test_initializer_bank=allow_test_initializer_bank,
        selector=selector,
        prefix=prefix,
        seed=seed,
        reward_reads=reward_reads,
    )
    return None if replayed is None else replayed[1]


def replay_decision_with_action_envelope(
    task: EmbeddingTask,
    ctx: Context,
    *,
    initializer: Initializer | None,
    initializer_bank: InitializerSnapshotBank | None = None,
    expected_initializer_bank_manifest_sha256: str | None = None,
    initializer_bank_episode_index: int | None = None,
    prepared_task: PreparedTask | None = None,
    target_access: Mapping[str, object] | None = None,
    ground_partition_receipt: Mapping[str, object] | None = None,
    allow_test_initializer_bank: bool = False,
    selector: StrengthSelector,
    prefix: Sequence[int],
    seed: int,
    reward_reads: int,
    provenance_fingerprint: str,
) -> tuple[DecisionState, StateActionEnvelopeV1] | None:
    """Replay a live state and bind its complete support before quality is observed."""

    replayed = _replay_live_decision(
        task,
        ctx,
        initializer=initializer,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=(expected_initializer_bank_manifest_sha256),
        initializer_bank_episode_index=initializer_bank_episode_index,
        prepared_task=prepared_task,
        target_access=target_access,
        ground_partition_receipt=ground_partition_receipt,
        allow_test_initializer_bank=allow_test_initializer_bank,
        selector=selector,
        prefix=prefix,
        seed=seed,
        reward_reads=reward_reads,
    )
    if replayed is None:
        return None
    env, decision = replayed
    envelope = StateActionEnvelopeV1.capture(
        env=env,
        decision=decision,
        provenance_fingerprint=provenance_fingerprint,
    )
    return decision, envelope


def _replay_live_decision(
    task: EmbeddingTask,
    ctx: Context,
    *,
    initializer: Initializer | None,
    initializer_bank: InitializerSnapshotBank | None = None,
    expected_initializer_bank_manifest_sha256: str | None = None,
    initializer_bank_episode_index: int | None = None,
    prepared_task: PreparedTask | None = None,
    target_access: Mapping[str, object] | None = None,
    ground_partition_receipt: Mapping[str, object] | None = None,
    allow_test_initializer_bank: bool = False,
    selector: StrengthSelector,
    prefix: Sequence[int],
    seed: int,
    reward_reads: int,
) -> tuple[EmbeddingEnv, DecisionState] | None:
    """Materialize one recorded state while retaining the live environment binding."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("quality environment seed must be a nonnegative 63-bit integer")
    env, _initializer_binding = _quality_environment(
        task,
        ctx,
        initializer=initializer,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=(expected_initializer_bank_manifest_sha256),
        initializer_bank_episode_index=initializer_bank_episode_index,
        prepared_task=prepared_task,
        target_access=target_access,
        ground_partition_receipt=ground_partition_receipt,
        allow_test_initializer_bank=allow_test_initializer_bank,
        selector=selector,
        reward_reads=reward_reads,
        seed=seed,
    )
    result = env.reset(seed)
    if isinstance(result, (InitFailureRecord, TerminalRecord)):
        return None
    for index in prefix:
        if not isinstance(result, DecisionState):
            return None
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(result.candidates)
            or not result.legal_mask[index]
        ):
            return None
        result = env.step(result, index, evaluate_training_reward=False).next_decision_or_terminal
    return (env, result) if isinstance(result, DecisionState) else None


def _typed_identity(value: Hashable) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}:{value!r}"


def _encoded_chains(
    chains: Mapping[Node, frozenset[Qubit]],
) -> list[dict[str, object]]:
    return [
        {
            "logical": _typed_identity(node),
            "chain": sorted(_typed_identity(qubit) for qubit in chain),
        }
        for node, chain in sorted(chains.items(), key=lambda item: _typed_identity(item[0]))
    ]


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"cannot encode {type(value).__name__} in a quality identity")


def _canonical_program(program: Program) -> dict[str, object]:
    def pair(left: object, right: object) -> tuple[str, str]:
        a, b = _typed_identity(left), _typed_identity(right)
        return (a, b) if a <= b else (b, a)

    return {
        "strength": float(program.strength),
        "strength_index": int(program.strength_index),
        "scale": float(program.scale),
        "offset": float(program.offset),
        "h_phys": sorted(
            (_typed_identity(qubit), float(value)) for qubit, value in program.h_phys.items()
        ),
        "j_phys": sorted(
            (*pair(left, right), float(value)) for (left, right), value in program.j_phys.items()
        ),
        "chain_edges": sorted(
            (
                _typed_identity(owner),
                sorted(pair(left, right) for left, right in edges),
            )
            for owner, edges in program.chain_edges.items()
        ),
        "contact_counts": sorted(
            (*pair(left, right), int(count))
            for (left, right), count in program.contact_counts.items()
        ),
    }


def continuation_receipt(
    terminal: TerminalRecord,
    *,
    action_trace: Sequence[Mapping[str, object]],
    continuation_seed: int,
    continuation_policy: str,
    continuation_steps: int,
    reward_reads: int,
    num_sweeps: int,
    initializer_binding: Mapping[str, object],
) -> dict[str, object]:
    """Materialize every terminal fact needed to replay one Q^mu sample."""

    reward = float(terminal.training_reward if terminal.training_reward is not None else 0.0)
    terminal_evidence: dict[str, object] | None = None
    if terminal.returned_valid:
        if (
            terminal.embedding is None
            or terminal.selected_index is None
            or terminal.selected_strength is None
            or terminal.evaluator_counts is None
            or terminal.evaluator_seed is None
            or not isinstance(terminal.compiled_programs, tuple)
            or len(terminal.compiled_programs) != 4
            or any(not isinstance(program, Program) for program in terminal.compiled_programs)
        ):
            raise RuntimeError("valid quality continuation omitted terminal evidence")
        hits, reads = terminal.evaluator_counts
        programs = [_canonical_program(program) for program in terminal.compiled_programs]
        embedding = _encoded_chains(terminal.embedding)
        terminal_evidence = {
            "embedding": embedding,
            "programs": programs,
            "selected_index": terminal.selected_index,
            "selected_strength": float(terminal.selected_strength),
            "selected_program_digest": stable_digest(
                {
                    "program": programs[terminal.selected_index],
                    "embedding": embedding,
                }
            ),
            "evaluator_count_block": {
                "seed": terminal.evaluator_seed,
                "hits": hits,
                "reads": reads,
                "num_sweeps": num_sweeps,
            },
        }
    elif reward != 0.0:
        raise RuntimeError("invalid quality continuation must have zero utility")

    validate_quality_initializer_binding(initializer_binding)
    payload: dict[str, object] = {
        "schema": CONTINUATION_RECEIPT_SCHEMA,
        "schema_version": CONTINUATION_RECEIPT_VERSION,
        "continuation_seed": continuation_seed,
        "continuation_policy": continuation_policy,
        "continuation_steps": continuation_steps,
        "action_trace": [_jsonable(row) for row in action_trace],
        "returned_valid": terminal.returned_valid,
        "terminal_reason": terminal.terminal_reason.value,
        "reward": reward,
        "requested_reward_reads": reward_reads,
        "validation_receipt": _jsonable(terminal.validation_receipt),
        "cumulative_work": terminal.cumulative_work.as_dict(),
        "terminal_evidence": terminal_evidence,
        "initializer_binding": _jsonable(initializer_binding),
    }
    payload["record_digest"] = stable_digest(payload)
    return payload


def validate_continuation_receipt(receipt: Mapping[str, object]) -> None:
    expected = {
        "schema",
        "schema_version",
        "continuation_seed",
        "continuation_policy",
        "continuation_steps",
        "action_trace",
        "returned_valid",
        "terminal_reason",
        "reward",
        "requested_reward_reads",
        "validation_receipt",
        "cumulative_work",
        "terminal_evidence",
        "initializer_binding",
        "record_digest",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected:
        raise ValueError("quality continuation receipt fields differ from its schema")
    if (
        receipt["schema"] != CONTINUATION_RECEIPT_SCHEMA
        or receipt["schema_version"] != CONTINUATION_RECEIPT_VERSION
    ):
        raise ValueError("unsupported quality continuation receipt")
    digest = receipt["record_digest"]
    unsigned = {key: receipt[key] for key in receipt if key != "record_digest"}
    if not isinstance(digest, str) or digest != stable_digest(unsigned):
        raise ValueError("quality continuation receipt digest mismatch")
    if (
        type(receipt["continuation_seed"]) is not int
        or not 0 <= receipt["continuation_seed"] < 2**63
        or not isinstance(receipt["continuation_policy"], str)
        or not receipt["continuation_policy"]
        or type(receipt["continuation_steps"]) is not int
        or receipt["continuation_steps"] < 0
        or type(receipt["returned_valid"]) is not bool
        or not isinstance(receipt["terminal_reason"], str)
        or type(receipt["requested_reward_reads"]) is not int
        or receipt["requested_reward_reads"] <= 0
    ):
        raise ValueError("quality continuation receipt has invalid scalar identity fields")
    validate_quality_initializer_binding(receipt["initializer_binding"])
    reward = receipt["reward"]
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(float(reward))
        or not 0.0 <= float(reward) <= 1.0
    ):
        raise ValueError("quality continuation receipt has invalid reward")
    trace = receipt["action_trace"]
    if not isinstance(trace, list) or any(
        not isinstance(row, Mapping)
        or set(row)
        != {
            "position",
            "source",
            "action_index",
            "opcode",
            "payload_key",
            "state_fingerprint",
            "support_fingerprint",
        }
        or row["position"] != position
        for position, row in enumerate(trace)
    ):
        raise ValueError("quality continuation action trace is malformed")
    evidence = receipt["terminal_evidence"]
    if receipt["returned_valid"]:
        if not isinstance(evidence, Mapping):
            raise ValueError("valid quality continuation omitted terminal evidence")
        if set(evidence) != {
            "embedding",
            "programs",
            "selected_index",
            "selected_strength",
            "selected_program_digest",
            "evaluator_count_block",
        }:
            raise ValueError("quality continuation terminal evidence is malformed")
        programs = evidence["programs"]
        selected = evidence["selected_index"]
        block = evidence["evaluator_count_block"]
        if (
            not isinstance(programs, list)
            or len(programs) != 4
            or type(selected) is not int
            or not 0 <= selected < 4
            or not isinstance(block, Mapping)
            or set(block) != {"seed", "hits", "reads", "num_sweeps"}
            or type(block["seed"]) is not int
            or not 0 <= block["seed"] < 2**31
            or type(block["hits"]) is not int
            or type(block["reads"]) is not int
            or not 0 <= block["hits"] <= block["reads"]
            or block["reads"] != receipt["requested_reward_reads"]
            or type(block["num_sweeps"]) is not int
            or block["num_sweeps"] <= 0
            or not math.isclose(
                float(receipt["reward"]),
                block["hits"] / block["reads"],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("quality continuation evaluator evidence is inconsistent")
        expected_program_digest = stable_digest(
            {"program": programs[selected], "embedding": evidence["embedding"]}
        )
        if evidence["selected_program_digest"] != expected_program_digest:
            raise ValueError("quality continuation selected program digest is inconsistent")
    elif evidence is not None or float(receipt["reward"]) != 0.0:
        raise ValueError("invalid quality continuation carries terminal evidence or utility")


def validate_publication_quality_record_initializer_binding(
    record: QualityRecord,
    *,
    expected_initializer_bank_contract: Mapping[str, object] | None = None,
    expected_initializer_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Require one publication-grade initializer identity across a quality record.

    ``ActionQuality.validate`` is deliberately the first boundary: a caller cannot use this
    helper to bless a receipt whose signed contents disagree with the action's retained seed,
    validity, or reward arrays.  Publication then requires every evaluated continuation to
    name the same complete persistent-K=2 episode binding.  The optional expected values let
    an artifact loader additionally pin that binding to its authenticated bank authority.

    Returns a copy of the common episode binding after successful validation.
    """

    if not isinstance(record, QualityRecord):
        raise TypeError("publication quality initializer validation requires a QualityRecord")

    receipts: list[Mapping[str, object]] = []
    for action in record.evaluated:
        action.validate()
        receipts.extend(action.continuation_receipts)
    if not receipts:
        raise ValueError("publication quality record has no evaluated continuation receipts")

    common_binding: Mapping[str, object] | None = None
    for receipt in receipts:
        # ActionQuality.validate() already checks the receipt.  Keep the call explicit here so
        # this publication boundary remains correct if ActionQuality's implementation changes.
        validate_continuation_receipt(receipt)
        binding = receipt["initializer_binding"]
        validate_quality_initializer_binding(binding)
        if binding.get("protocol") == QUALITY_INITIALIZER_LEGACY_PROTOCOL:
            raise ValueError(
                "legacy quality initializer protocol is diagnostic-only, not publication"
            )
        if binding.get("protocol") != QUALITY_INITIALIZER_BANK_PROTOCOL:
            raise ValueError("publication quality record has an unsupported initializer protocol")
        if binding.get("publication_eligible") is not True:
            raise ValueError("quality initializer-bank binding is not publication-eligible")
        if common_binding is None:
            common_binding = binding
        elif dict(binding) != dict(common_binding):
            raise ValueError(
                "publication quality record does not use exactly one initializer-bank binding"
            )

    if common_binding is None:  # Defensive: receipts is nonempty above.
        raise ValueError("publication quality record has no evaluated continuation receipts")

    if expected_initializer_bank_contract is not None:
        validate_quality_initializer_bank_contract(
            expected_initializer_bank_contract,
            require_publication=True,
        )
        contract_fields = {
            "bank_contract_record_digest": "record_digest",
            "manifest_sha256": "manifest_sha256",
            "manifest_record_digest": "manifest_record_digest",
            "access_receipt_record_digest": "access_receipt_record_digest",
            "plan_record_digest": "plan_record_digest",
            "prepared_manifest_sha256": "prepared_manifest_sha256",
            "config_digest": "config_digest",
            "context_digest": "context_digest",
            "publication_eligible": "publication_eligible",
        }
        if any(
            common_binding[binding_field] != expected_initializer_bank_contract[contract_field]
            for binding_field, contract_field in contract_fields.items()
        ):
            raise ValueError(
                "quality record initializer binding differs from the expected bank contract"
            )
        if (
            common_binding["restart_cache_slots"]
            != expected_initializer_bank_contract["restart_cache_slots_per_episode"]
        ):
            raise ValueError(
                "quality record initializer binding differs from the expected bank contract"
            )

    if expected_initializer_binding is not None:
        validate_quality_initializer_binding(expected_initializer_binding)
        if expected_initializer_binding.get("protocol") != QUALITY_INITIALIZER_BANK_PROTOCOL:
            raise ValueError("expected publication initializer binding is not a bank binding")
        if expected_initializer_binding.get("publication_eligible") is not True:
            raise ValueError("expected initializer-bank binding is not publication-eligible")
        if dict(common_binding) != dict(expected_initializer_binding):
            raise ValueError(
                "quality record initializer binding differs from the expected episode binding"
            )

    return dict(common_binding)


def context_digest(ctx: Context) -> str:
    """Bind one row to the complete immutable environment registry."""

    return stable_digest(_jsonable(ctx))


def decision_snapshot(
    result: DecisionState,
) -> tuple[
    tuple[Mapping[str, object], ...],
    Mapping[str, object],
    Mapping[str, object],
]:
    """Canonical audit snapshot used both by generation and exact loader replay."""

    support = tuple(
        {
            "index": index,
            "opcode": candidate.opcode.value,
            "payload_key": candidate.payload_key,
            "affected": [_typed_identity(node) for node in candidate.affected],
            "old_chains": _encoded_chains(candidate.old_chains),
            "new_chains": _encoded_chains(candidate.new_chains),
            "routes": [
                {
                    "path": [_typed_identity(qubit) for qubit in path],
                    "owner": _typed_identity(owner),
                }
                for path, owner in candidate.routes
            ],
            "work": candidate.work.as_dict(),
            "proposal_work": candidate.proposal_work.as_dict(),
            "archive_ref": candidate.archive_ref,
            "restart_cache_slot": candidate.restart_cache_slot,
            "restart_cache_after_digest": candidate.restart_cache_after_digest,
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
            "legal": bool(result.legal_mask[index]),
        }
        for index, candidate in enumerate(result.candidates)
    )
    exact = result.exact_state

    def keyed(values: Mapping[object, int]) -> list[list[object]]:
        return [
            [_typed_identity(key), value]
            for key, value in sorted(values.items(), key=lambda pair: _typed_identity(pair[0]))
        ]

    exact_state = {
        "chains": _encoded_chains(exact.chains),
        "archive": [
            {
                "chains": _encoded_chains(entry.chains),
                "protected": entry.protected,
                "age": entry.age,
                "admissible": entry.admissible,
                "qubits": entry.qubits,
                "max_chain": entry.max_chain,
                "key": entry.key,
            }
            for entry in exact.archive
        ],
        "ages": {
            "chain": keyed(exact.ages.chain),
            "claim": keyed(exact.ages.claim),
            "conflict": keyed(exact.ages.conflict),
            "demand": keyed(exact.ages.demand),
            "occupancy": keyed(exact.ages.occupancy),
        },
        "remaining": exact.remaining.as_dict(),
        "spent": exact.spent.as_dict(),
        "restarts_left": exact.restarts_left,
        "decisions_used": exact.decisions_used,
        "workspace_valid": exact.workspace_valid,
        "random_state": result.random_state,
    }
    observation = {
        name: (
            value.tolist()
            if isinstance(value, np.ndarray)
            else [_typed_identity(item) for item in value]
        )
        for name, value in vars(result.observation).items()
    }
    return support, exact_state, observation


def _stable_seed(root: int, domain: str, *parts: object) -> int:
    """Domain-separated 63-bit seed; Python's randomized ``hash`` is never used."""

    raw = json.dumps(
        {"domain": domain, "parts": [root, *parts], "version": 1},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & (2**63 - 1)


def continuation_seed(
    environment_seed: int,
    task_id: str,
    state_fingerprint: str,
    action_index: int,
    continuation_index: int,
) -> int:
    """Public seed registry used by generation and independent loader verification."""

    return _stable_seed(
        environment_seed,
        "quality-continuation",
        task_id,
        state_fingerprint,
        action_index,
        continuation_index,
    )


def strength_counts(
    task: EmbeddingTask,
    chains: Mapping[Node, frozenset[Qubit]],
    ctx: Context,
    *,
    reads: int,
    seed: int,
) -> tuple[tuple[Mapping[str, float], ...], tuple[int, ...], tuple[int, ...]]:
    """Count-complete labels for all four registered strengths of one valid embedding."""

    strengths = strength_registry(task.problem, ctx.strength_ratios, ctx.epsilon_strength)
    programs = compile_registry(
        chains, task.host, task.problem, strengths, ctx.field_limit, ctx.coupler_limit
    )
    features = tuple(program_features(p, chains, task.problem) for p in programs)
    hits: list[int] = []
    totals: list[int] = []
    for j, program in enumerate(programs):
        block = sample_program(
            program,
            chains,
            task.problem,
            task.ground_energy,
            num_reads=reads,
            seed=seed + 101 * j,
            num_sweeps=ctx.num_sweeps,
        )
        hits.append(block.hits)
        totals.append(block.reads)
    return features, tuple(hits), tuple(totals)
