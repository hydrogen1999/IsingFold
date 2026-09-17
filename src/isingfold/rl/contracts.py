"""Immutable schemas: context, work ledger, actions, decision states and terminals.

Spec: MODEL_SPEC sections 2.1-2.5 and 7.2; Rev2 sections 2-3. Nothing here depends on a
learned score, an evaluator outcome or a planted witness. A candidate is a *fully
materialised* successor: the actor picks a row, it never emits raw sets.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Hashable, Mapping, Sequence

Node = Hashable
Qubit = Hashable

#: Opcode order is fixed by the spec; the one-hot uses exactly this order.
OPCODES: tuple[str, ...] = (
    "PLACE",
    "ROUTE",
    "REWRITE_ONE",
    "REWRITE_GROUP",
    "REPAIR_GROUP",
    "RESTART",
    "COMMIT",
    "STOP",
)
CANDIDATE_SUPPORT_VERSION = "if-action-support-v2-restart-cache"


class Opcode(str, Enum):
    PLACE = "PLACE"
    ROUTE = "ROUTE"
    REWRITE_ONE = "REWRITE_ONE"
    REWRITE_GROUP = "REWRITE_GROUP"
    REPAIR_GROUP = "REPAIR_GROUP"
    RESTART = "RESTART"
    COMMIT = "COMMIT"
    STOP = "STOP"

    @property
    def index(self) -> int:
        return OPCODES.index(self.value)

    @property
    def is_terminal(self) -> bool:
        return self in (Opcode.COMMIT, Opcode.STOP)


class Mode(str, Enum):
    IMPROVEMENT = "improvement"
    """Profile I: start from a validated initializer held in a protected archive slot."""
    CONSTRUCTION = "construction"
    """Profile C: start from empty chains and an empty archive; STOP becomes reachable."""


class TerminalReason(str, Enum):
    COMMIT = "COMMIT"
    STOP_NO_VALID = "STOP_NO_VALID"
    BUDGET_NO_VALID = "BUDGET_NO_VALID"
    NO_ACTION_NO_VALID = "NO_ACTION_NO_VALID"


WORK_FIELDS: tuple[str, ...] = (
    "decisions",
    "route_expansions",
    "materializations",
    "compiler_calls",
    "validator_calls",
    "cut_edge_visits",
    "restart_work",
    "evaluator_reads",
    "feature_work",
)


@dataclass(frozen=True)
class WorkVector:
    """The work ledger of Rev2 section 3.6. Every coordinate is nonnegative and monotone."""

    decisions: int = 0
    route_expansions: int = 0
    materializations: int = 0
    compiler_calls: int = 0
    validator_calls: int = 0
    cut_edge_visits: int = 0
    restart_work: int = 0
    evaluator_reads: int = 0
    feature_work: int = 0

    def __add__(self, other: WorkVector) -> WorkVector:
        return WorkVector(**{f: getattr(self, f) + getattr(other, f) for f in WORK_FIELDS})

    def __sub__(self, other: WorkVector) -> WorkVector:
        return WorkVector(**{f: getattr(self, f) - getattr(other, f) for f in WORK_FIELDS})

    def fits_in(self, budget: WorkVector) -> bool:
        """Coordinatewise ``self <= budget``: the ``preceq`` of MODEL_SPEC Eq. (3f)."""

        return all(getattr(self, f) <= getattr(budget, f) for f in WORK_FIELDS)

    @property
    def is_nonnegative(self) -> bool:
        return all(getattr(self, f) >= 0 for f in WORK_FIELDS)

    def fractions_of(self, caps: WorkVector) -> dict[str, float | None]:
        """Remaining fraction per coordinate; ``None`` marks a non-binding coordinate."""

        out: dict[str, float | None] = {}
        for f in WORK_FIELDS:
            cap = getattr(caps, f)
            out[f] = None if cap <= 0 else max(0.0, min(1.0, getattr(self, f) / cap))
        return out

    def as_dict(self) -> dict[str, int]:
        return {f: getattr(self, f) for f in WORK_FIELDS}


@dataclass(frozen=True)
class OverlapProfile:
    """Envelope of MODEL_SPEC Eq. (3): O0 is exclusive, O1 the pilot, O2 the wide variant."""

    name: str = "O1"
    max_occupancy: int = 2
    excess_fraction_of_qubit_cap: float = 0.10

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("overlap profile name must be non-empty")
        if (
            isinstance(self.max_occupancy, bool)
            or not isinstance(self.max_occupancy, int)
            or self.max_occupancy < 1
        ):
            raise ValueError("maximum occupancy must be a positive integer")
        if (
            not math.isfinite(self.excess_fraction_of_qubit_cap)
            or not 0.0 <= self.excess_fraction_of_qubit_cap <= 1.0
        ):
            raise ValueError("overlap excess fraction must lie in [0, 1]")

    def excess_cap(self, qubit_cap: int) -> int:
        import math

        return int(math.ceil(self.excess_fraction_of_qubit_cap * qubit_cap))


# The wide construction support: a decision may see every frontier placement of a large
# instance instead of a 64-candidate shortlist. Registered by its own context version so
# no measurement under the 64-candidate registration is ever mistaken for one under this.
WIDE_STATE_CHANGING = 512
WIDE_SUFFIX = "-wide512"

DEFAULT_CAPS = WorkVector(
    decisions=32,
    route_expansions=200_000,
    materializations=4_096,
    compiler_calls=2_048,
    validator_calls=2_048,
    cut_edge_visits=200_000,
    restart_work=64,
    evaluator_reads=8_192,
    feature_work=200_000,
)

RESERVE = WorkVector(
    decisions=1,
    route_expansions=0,
    materializations=8,
    compiler_calls=12,
    validator_calls=4,
    cut_edge_visits=0,
    restart_work=0,
    evaluator_reads=256,
    feature_work=8_192,
)
"""Terminal reserve: one COMMIT decision, its four program compilations, an independent
revalidation and one reward-read block (MODEL_SPEC section 2.5)."""


@dataclass(frozen=True)
class Context:
    """The immutable registry Omega. Changing any field changes the task, not a checkpoint."""

    qubit_cap: int
    overlap: OverlapProfile = OverlapProfile()
    caps: WorkVector = DEFAULT_CAPS
    reserve: WorkVector = RESERVE
    strength_ratios: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    epsilon_strength: float = 1e-6
    beta_range: tuple[float, float] | None = None
    """Inverse-temperature schedule for the surrogate annealer, or None to let it choose.

    Left at None the sampler derives the range from the programmed h and J, so multiplying a
    whole Hamiltonian by c rescales the schedule by 1/c and the product of beta and energy is
    preserved: the common autoscale that the fixed-temperature theory relies on is undone
    before it can have an effect. A two-qubit control makes this exact, and an external audit
    reproduced it. Registering a range here pins the schedule in program units instead, which
    is what a claim about energy-scale compression needs. It changes every measured utility,
    so it is a registry field and not a call-site argument."""

    n_est_reads: int = 256
    audit_reads: int = 4_096
    num_sweeps: int = 200
    decode: str = "majority"
    field_limit: float = 4.0
    coupler_limit: float = 2.0
    group_sizes: tuple[int, ...] = (2, 3, 4, 8)
    max_state_changing: int = 64
    max_commit: int = 8
    padded_actions: int = 73
    archive_protected: int = 1
    archive_fifo: int = 7
    restart_allowance: int = 2
    tabu_tenure: int = 0
    quotas: Mapping[str, int] = field(
        default_factory=lambda: {
            "single": 16,
            "group2": 8,
            "group3": 8,
            "group4": 8,
            "group8": 8,
            "repair": 12,
            "restart": 4,
        }
    )
    construction_quotas: Mapping[str, int] = field(
        default_factory=lambda: {
            "place": 24,
            "route": 24,
            "rewrite": 8,
            "repair": 4,
            "restart": 4,
        }
    )
    endpoint: str = "IF-Q3-S0"
    context_version: str = "rev2-pilot-3-restart-cache"

    def __post_init__(self) -> None:
        if isinstance(self.qubit_cap, bool) or not isinstance(self.qubit_cap, int) or self.qubit_cap <= 0:
            raise ValueError("qubit_cap must be a positive integer")
        for label, work in (("work caps", self.caps), ("terminal reserve", self.reserve)):
            if any(
                isinstance(getattr(work, name), bool)
                or not isinstance(getattr(work, name), int)
                for name in WORK_FIELDS
            ):
                raise ValueError(f"{label} must contain integer coordinates")
            if not work.is_nonnegative:
                raise ValueError(f"{label} must be nonnegative")
        if not self.reserve.fits_in(self.caps):
            raise ValueError("terminal reserve must be nonnegative and fit within every work cap")
        integer_fields = (
            "max_state_changing",
            "max_commit",
            "padded_actions",
            "archive_protected",
            "archive_fifo",
            "restart_allowance",
            "tabu_tenure",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.max_commit != 8:
            raise ValueError("IF-Core-v1 fixes the COMMIT capacity at 8")
        if self.max_state_changing not in (64, WIDE_STATE_CHANGING):
            raise ValueError(
                "IF-Core-v1 fixes the state-changing capacity at 64; the wide construction "
                f"registration allows {WIDE_STATE_CHANGING}"
            )
        if self.max_state_changing == WIDE_STATE_CHANGING and WIDE_SUFFIX not in str(self.context_version):
            raise ValueError("a wide support must be registered by its context version")
        if self.max_state_changing == 64 and self.padded_actions != 73:
            raise ValueError("IF-Core-v1 fixes the action capacities at 64 + 8 + 1 = 73")
        if self.archive_protected != 1 or self.archive_fifo != 7:
            raise ValueError("IF-Core-v1 fixes the improvement archive at 1 protected + 7 FIFO")
        if self.restart_allowance != 2:
            raise ValueError("IF-Core-v1 fixes the selected-restart allowance at two")
        if self.tabu_tenure != 0:
            raise ValueError("IF-Core-v1 keeps tabu disabled")
        if self.archive_protected + self.archive_fifo > self.max_commit:
            raise ValueError("archive slots exceed the COMMIT capacity")
        if self.max_state_changing + self.max_commit + 1 != self.padded_actions:
            raise ValueError("padded action capacity must be state-changing + commit + STOP")
        if len(self.strength_ratios) != 4:
            raise ValueError("IF-Q3-S0 registers exactly four strengths")
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in self.strength_ratios)
            or len(set(self.strength_ratios)) != len(self.strength_ratios)
        ):
            raise ValueError("strength ratios must be four distinct positive finite values")
        if not math.isfinite(self.epsilon_strength) or self.epsilon_strength <= 0.0:
            raise ValueError("epsilon_strength must be positive and finite")
        for name in ("n_est_reads", "audit_reads", "num_sweeps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_est_reads > self.reserve.evaluator_reads:
            raise ValueError("terminal reward reads must fit the registered terminal reserve")
        if self.audit_reads > self.caps.evaluator_reads:
            raise ValueError("audit reads must fit the registered evaluator cap")
        if self.reserve.compiler_calls < self.max_commit + 4:
            raise ValueError(
                "terminal reserve must cover every archive overlay and four COMMIT programs"
            )
        minimum_terminal_features = 32 * (self.max_commit + self.qubit_cap)
        if self.reserve.feature_work < minimum_terminal_features:
            raise ValueError(
                "terminal reserve feature work cannot encode a full-cap COMMIT-only support"
            )
        for name in ("field_limit", "coupler_limit"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.beta_range is not None:
            if (
                not isinstance(self.beta_range, tuple)
                or len(self.beta_range) != 2
                or any(not math.isfinite(v) or v <= 0.0 for v in self.beta_range)
                or self.beta_range[0] >= self.beta_range[1]
            ):
                raise ValueError("beta_range must be an increasing pair of positive finites")
        if self.decode != "majority":
            raise ValueError("the rev2 pilot registers only majority decoding")
        if self.endpoint != "IF-Q3-S0":
            raise ValueError("the rev2 pilot registers only endpoint IF-Q3-S0")
        if not isinstance(self.context_version, str) or not self.context_version:
            raise ValueError("context_version must be a non-empty string")
        if (
            not isinstance(self.group_sizes, tuple)
            or not self.group_sizes
            or len(set(self.group_sizes)) != len(self.group_sizes)
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size not in {2, 3, 4, 8}
                for size in self.group_sizes
            )
        ):
            raise ValueError("group sizes must be distinct members of {2, 3, 4, 8}")
        allowed_quotas = {
            "quotas": {"single", "group2", "group3", "group4", "group8", "repair", "restart"},
            "construction_quotas": {"place", "route", "grow", "shrink", "rewrite", "repair", "restart"},
        }
        for attribute in ("quotas", "construction_quotas"):
            quota_copy = dict(getattr(self, attribute))
            if any(
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for name, value in quota_copy.items()
            ):
                raise ValueError(
                    "proposal quotas must map non-empty names to nonnegative integers"
                )
            unknown = set(quota_copy) - allowed_quotas[attribute]
            if unknown:
                raise ValueError(f"unknown {attribute} families: {sorted(unknown)}")
            object.__setattr__(self, attribute, MappingProxyType(quota_copy))

    @property
    def archive_slots(self) -> int:
        return self.archive_protected + self.archive_fifo

    def with_caps(self, **kwargs: int) -> Context:
        return replace(self, caps=replace(self.caps, **kwargs))


@dataclass(frozen=True)
class RestartCacheSlot:
    """One ordered, authenticated, episode-local persistent restart exemplar."""

    slot_index: int
    status: str
    chains: Mapping[Node, frozenset[Qubit]] | None
    snapshot_record_digest: str
    attempt_receipt_root: str
    consumed: bool = False

    def __post_init__(self) -> None:
        if type(self.slot_index) is not int or self.slot_index < 0:
            raise ValueError("restart-cache slot index must be nonnegative")
        if self.status not in {
            "SUCCESS",
            "FAILED",
            "BUDGET_NOT_INVOKED",
            "TIME_NOT_INVOKED",
        }:
            raise ValueError("restart-cache slot has an unknown status")
        for digest in (self.snapshot_record_digest, self.attempt_receipt_root):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("restart-cache evidence must use SHA-256 identities")
        if type(self.consumed) is not bool:
            raise TypeError("restart-cache consumed flag must be Boolean")
        if self.status == "SUCCESS":
            if not isinstance(self.chains, Mapping) or not self.chains:
                raise ValueError("successful restart-cache slot needs an embedding")
            if any(type(chain) is not frozenset for chain in self.chains.values()):
                raise ValueError("restart-cache chains must be exact frozen sets")
        elif self.chains is not None or self.consumed:
            raise ValueError("failed restart-cache slot cannot carry or consume an embedding")

    def consume(self) -> RestartCacheSlot:
        if self.status != "SUCCESS" or self.consumed:
            raise ValueError("restart-cache slot is unavailable")
        return replace(self, consumed=True)


@dataclass(frozen=True)
class Candidate:
    """A bound, already materialised successor (MODEL_SPEC section 2.4).

    ``new_chains`` holds the complete replacement for every affected logical id; unaffected
    chains are untouched. ``work`` is the audited upper bound on applying it. ``payload_key``
    identifies the successor for likelihood replay and deduplication.
    """

    opcode: Opcode
    affected: tuple[Node, ...]
    old_chains: Mapping[Node, frozenset[Qubit]]
    new_chains: Mapping[Node, frozenset[Qubit]]
    work: WorkVector
    payload_key: str
    proposal_work: WorkVector = WorkVector()
    """Exact work charged to materialise this retained proposal during preparation.

    Rejected and duplicate attempts remain represented only in the batch-wide preparation
    receipt.  This per-row receipt is an observed proposal-cost feature, never work charged
    again when the action is selected.
    """
    routes: tuple[tuple[tuple[Qubit, ...], Node], ...] = ()
    """Ordered owner-consistent route segments, each with its logical owner."""
    archive_ref: int | None = None
    """Archive index for COMMIT or a bound REWRITE_GROUP restore; ``None`` otherwise."""
    target_demand: tuple[Node, Node] | None = None
    """The construction/repair demand this fully bound action addresses, when applicable."""
    target_conflict: Qubit | None = None
    """The contested qubit whose complete claimant set defines a repair group."""
    restart_cache_slot: int | None = None
    """Persistent cache slot consumed by a selected cached RESTART."""
    restart_cache_after_digest: str | None = None
    """Exact post-selection cache inventory identity for bound transition replay."""
    provenance: str = ""
    branch: Any = field(default=None, repr=False, compare=False)
    """Opaque native search branch already holding the successor, when available."""

    @property
    def changes_workspace(self) -> bool:
        return self.opcode not in (Opcode.COMMIT, Opcode.STOP)


@dataclass(frozen=True)
class ArchiveEntry:
    """An outcome-blind record of an independently validated embedding."""

    chains: Mapping[Node, frozenset[Qubit]]
    protected: bool
    age: int
    admissible: bool
    qubits: int
    max_chain: int
    key: str
    branch: Any = field(default=None, repr=False, compare=False)

    def aged(self) -> ArchiveEntry:
        return replace(self, age=self.age + 1)


@dataclass
class DecisionState:
    """What the actor sees: a charged state, its candidate batch and the exact mask."""

    observation: Any
    candidates: tuple[Candidate, ...]
    legal_mask: tuple[bool, ...]
    state_fingerprint: str
    context_version: str
    charged_work_receipt: WorkVector
    support_fingerprint: str
    exact_state: Any = field(default=None, repr=False)
    random_state: Any = field(default=None, repr=False)
    """Exact exogenous proposal RNG state; provenance/replay only, never a model feature."""

    @property
    def n_legal(self) -> int:
        return sum(1 for m in self.legal_mask if m)


@dataclass(frozen=True)
class TerminalRecord:
    returned_valid: bool
    terminal_reason: TerminalReason
    embedding: Mapping[Node, frozenset[Qubit]] | None
    selected_strength: float | None
    selected_index: int | None
    validation_receipt: Mapping[str, Any]
    cumulative_work: WorkVector
    selected_program: Any | None = field(default=None, repr=False)
    compiled_programs: tuple[Any, ...] | None = field(default=None, repr=False)
    """Exact compiler outputs already charged at terminal materialisation."""
    training_reward: float | None = None
    training_cost: float | None = None
    evaluator_counts: tuple[int, int] | None = None
    evaluator_seed: int | None = None


@dataclass(frozen=True)
class InitFailureRecord:
    """A failed improvement initializer: a system outcome, never a PPO episode."""

    reason: str
    work: WorkVector


@dataclass(frozen=True)
class StepResult:
    next_decision_or_terminal: DecisionState | TerminalRecord
    reward: float
    failure_cost: float
    terminated: bool
    truncated: bool = False
    terminal_reason: TerminalReason | None = None
    work_receipt: WorkVector = WorkVector()


def chain_key(chains: Mapping[Node, Sequence[Qubit] | frozenset[Qubit]]) -> str:
    """A stable, order-free identity of a complete assignment, for dedup and archives."""

    payload = [
        [
            _typed_identity(node),
            sorted((_typed_identity(qubit) for qubit in chains[node]), key=lambda item: item),
        ]
        for node in sorted(chains, key=_typed_identity)
    ]
    return stable_digest(payload)


def candidate_support_key(candidates: Sequence[Candidate], legal_mask: Sequence[bool]) -> str:
    """Bind the complete materialised support, not just its row positions.

    Provenance and native branch handles are deliberately excluded.  They cannot change
    policy mass when the successor and charged semantics are identical.
    """

    if len(candidates) != len(legal_mask):
        raise ValueError("candidate support and legal mask have different lengths")
    rows = []
    for candidate, legal in zip(candidates, legal_mask, strict=True):
        rows.append(
            {
                "opcode": candidate.opcode.value,
                "affected": [_typed_identity(node) for node in candidate.affected],
                "old": chain_key(candidate.old_chains),
                "new": chain_key(candidate.new_chains),
                "work": candidate.work.as_dict(),
                "proposal_work": candidate.proposal_work.as_dict(),
                "payload_key": candidate.payload_key,
                "routes": [
                    {
                        "owner": _typed_identity(owner),
                        "path": [_typed_identity(qubit) for qubit in path],
                    }
                    for path, owner in candidate.routes
                ],
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
                "restart_cache_slot": candidate.restart_cache_slot,
                "restart_cache_after_digest": candidate.restart_cache_after_digest,
                "legal": bool(legal),
            }
        )
    return stable_digest(
        {
            "schema": CANDIDATE_SUPPORT_VERSION,
            "rows": rows,
        }
    )


def stable_digest(payload: object) -> str:
    """SHA-256 over canonical JSON for replay and integrity receipts."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _typed_identity(value: Hashable) -> str:
    """Keep values such as integer ``1`` and string ``"1"`` distinct in digests."""

    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    return f"{kind}:{value!r}"
