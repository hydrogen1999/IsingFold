"""The exact environment: prepare, decide, apply, prepare once.

Spec: MODEL_SPEC sections 2.2-2.5 and 7.2. The environment owns graph facts, candidate
materialisation, resource accounting and legality; the policy only reorders legal rows. A
returned embedding always passes an independent validator, whatever the parameters say.
"""

from __future__ import annotations

import copy
import os
import math
import operator
import random
from dataclasses import dataclass, field
from typing import Callable, Hashable, Mapping, Sequence

import networkx as nx

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import (
    ArchiveEntry,
    Candidate,
    Context,
    DecisionState,
    InitFailureRecord,
    Mode,
    Opcode,
    RestartCacheSlot,
    StepResult,
    TerminalReason,
    TerminalRecord,
    WORK_FIELDS,
    WorkVector,
    candidate_support_key,
    chain_key,
    stable_digest,
)
from isingfold.rl.evaluator import ReadBlock, sample_program
from isingfold.rl.program import Program, instance_scale, program_features, strength_registry
from isingfold.rl.proposal import (
    AUTHENTICATED_RESTART_CACHE_V1,
    Initializer,
    LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
    ProposalGenerator,
    bound_successor_key,
    restart_cache_digest,
)
from isingfold.rl.tensorize import Ages, build_observation
from isingfold.rl.validate import (
    ValidationReceipt,
    occupancy,
    p_embed,
    p_return,
    p_search,
    unrealized_demands,
)

Node = Hashable
Qubit = Hashable

StrengthSelector = Callable[[Sequence[Program], Sequence[Mapping[str, float]]], int]


def _identity_key(value: Hashable) -> tuple[str, str, str]:
    """A total ordering which never conflates heterogeneous graph identifiers."""

    kind = type(value)
    return kind.__module__, kind.__qualname__, repr(value)


def _identity_payload(value: Hashable) -> dict[str, str]:
    module, qualname, representation = _identity_key(value)
    return {"type": f"{module}.{qualname}", "repr": representation}


def _canonical_pair(left: Node, right: Node) -> tuple[Node, Node]:
    return (left, right) if _identity_key(left) <= _identity_key(right) else (right, left)


def fixed_strength_selector(index: int = 1) -> StrengthSelector:
    """The no-learning control of Rev2 section 2.5: always the registered ``F_2``."""

    def _select(programs: Sequence[Program], features: Sequence[Mapping[str, float]]) -> int:
        return min(index, len(programs) - 1)

    return _select


class IntegrityError(RuntimeError):
    """A contract violation: never a low reward, never a training label."""



_SKIP_INTERNAL_ASSERTS = os.environ.get("ISINGFOLD_FAST_INTERNAL_ASSERTS", "") == "1"
"""Skip the defensive invariant re-checks. Set only for timing runs, never for a result."""

@dataclass(frozen=True)
class EmbeddingTask:
    """One registered instance: public inputs plus evaluator-only certificates."""

    name: str
    logical: nx.Graph
    host: nx.Graph
    problem: LogicalProblem
    ground_energy: float | None
    lineage: str = ""
    witness: Mapping[Node, frozenset[Qubit]] | None = field(default=None, repr=False)
    """Evaluator-only feasibility witness. Never tensorised, never a quality label."""
    initial_embedding: Mapping[Node, frozenset[Qubit]] | None = field(
        default=None,
        repr=False,
    )
    """Registered Profile-I incumbent. It is visible only through the current state."""


def task_initializer(task: EmbeddingTask, fallback: Initializer | None = None) -> Initializer:
    """Return the task's authenticated initializer, or an explicitly supplied fallback."""

    if task.initial_embedding is None:
        if fallback is None:
            raise IntegrityError(f"task {task.name!r} has no registered Profile-I initializer")
        return fallback
    frozen = {node: frozenset(chain) for node, chain in task.initial_embedding.items()}

    def _initializer(logical: nx.Graph, host: nx.Graph, seed: int):
        del seed
        if set(frozen) != set(logical.nodes()):
            raise IntegrityError("registered initializer logical domain mismatch")
        if any(not set(chain) <= set(host.nodes()) for chain in frozen.values()):
            raise IntegrityError("registered initializer active-host mismatch")
        return dict(frozen)

    return _initializer


@dataclass
class EnvState:
    chains: dict[Node, frozenset[Qubit]]
    archive: list[ArchiveEntry]
    ages: Ages
    remaining: WorkVector
    restarts_left: int
    restart_cache: list[RestartCacheSlot]
    decisions_used: int = 0
    spent: WorkVector = WorkVector()
    budget_debit: WorkVector = WorkVector()
    """Exact work spent before this environment took ownership of execution.

    The debit reduces the registered budget and is part of the transition state, but it is
    not environment work and therefore is not added to ``spent`` or terminal cumulative
    work.  Complete-system receipts account for it in the initializer ledger instead.
    """
    workspace_valid: bool = False
    reference_program: Program | None = None


class EmbeddingEnv:
    """Finite-horizon constrained MDP at macro-decision epochs."""

    def __init__(
        self,
        task: EmbeddingTask,
        ctx: Context,
        *,
        mode: Mode = Mode.IMPROVEMENT,
        initializer: Initializer | None = None,
        selector: StrengthSelector | None = None,
        reward_reads: int | None = None,
        coefficient_scale: float | None = None,
        budget_debit: WorkVector = WorkVector(),
        initializer_precomputed: bool = False,
        restart_cache: Sequence[RestartCacheSlot] | None = None,
        restart_cache_manifest_digest: str | None = None,
        improvement_restart_protocol: str = AUTHENTICATED_RESTART_CACHE_V1,
        seed: int = 0,
    ) -> None:
        self.task = task
        self.ctx = ctx
        self.mode = mode
        self.initializer = initializer
        self.selector = selector or fixed_strength_selector()
        self.reward_reads = ctx.n_est_reads if reward_reads is None else reward_reads
        if not isinstance(budget_debit, WorkVector):
            raise TypeError("budget_debit must be an exact WorkVector")
        if not budget_debit.is_nonnegative:
            raise ValueError("budget_debit must be nonnegative")
        if type(initializer_precomputed) is not bool:
            raise TypeError("initializer_precomputed must be Boolean")
        if initializer_precomputed and mode is not Mode.IMPROVEMENT:
            raise ValueError("only an improvement initializer can be precomputed")
        if initializer_precomputed and budget_debit == WorkVector():
            raise ValueError("a precomputed initializer requires a nonzero exact budget debit")
        self.budget_debit = budget_debit
        self.initializer_precomputed = initializer_precomputed
        if improvement_restart_protocol not in {
            AUTHENTICATED_RESTART_CACHE_V1,
            LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        }:
            raise ValueError("unknown improvement restart protocol")
        if (
            mode is Mode.IMPROVEMENT
            and ctx.restart_allowance > 0
            and int(ctx.quotas.get("restart", 0)) > 0
            and restart_cache is None
            and improvement_restart_protocol == AUTHENTICATED_RESTART_CACHE_V1
        ):
            raise ValueError(
                "registered Profile-I improvement requires an authenticated restart cache"
            )
        if (
            mode is Mode.IMPROVEMENT
            and restart_cache is not None
            and improvement_restart_protocol == LEGACY_ONLINE_INITIALIZER_RESTARTS_V1
        ):
            raise ValueError("legacy online-restart mode cannot consume a restart cache")
        self.improvement_restart_protocol = improvement_restart_protocol
        if restart_cache is None:
            if restart_cache_manifest_digest is not None:
                raise ValueError("restart-cache manifest supplied without cache slots")
            self._restart_cache_template: tuple[RestartCacheSlot, ...] | None = None
        else:
            slots = tuple(restart_cache)
            if any(not isinstance(slot, RestartCacheSlot) for slot in slots):
                raise TypeError("restart cache must contain typed slots")
            if tuple(slot.slot_index for slot in slots) != tuple(range(len(slots))):
                raise ValueError("restart-cache slots must be contiguous and ordered")
            if len(slots) != ctx.restart_allowance:
                raise ValueError("restart cache must bind every registered cache slot")
            if (
                not isinstance(restart_cache_manifest_digest, str)
                or len(restart_cache_manifest_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in restart_cache_manifest_digest
                )
            ):
                raise ValueError("restart cache requires an authenticated manifest digest")
            self._restart_cache_template = slots
        self.restart_cache_manifest_digest = restart_cache_manifest_digest
        if coefficient_scale is None:
            selector_scale = getattr(self.selector, "coefficient_transform_scale", 1.0)
            coefficient_scale = (
                float(selector_scale.item())
                if hasattr(selector_scale, "item")
                else float(selector_scale)
            )
        if not math.isfinite(coefficient_scale) or coefficient_scale <= 0.0:
            raise ValueError("coefficient_scale must be positive and finite")
        self.coefficient_scale = coefficient_scale
        self.rng = random.Random(seed)
        self.generator = ProposalGenerator(
            ctx,
            task.logical,
            task.host,
            task.problem,
            initializer=initializer,
            mode=mode,
            improvement_restart_protocol=improvement_restart_protocol,
        )
        self.strengths = strength_registry(task.problem, ctx.strength_ratios, ctx.epsilon_strength)
        self.scale = instance_scale(task.problem, ctx.epsilon_strength)
        self.state: EnvState | None = None
        self._seed = seed
        self._prepared_state_fingerprint: str | None = None
        self._prepared_support_fingerprint: str | None = None
        self._integrity_seal: str | None = None
        # Return validation is deterministic for a fixed task and context.  Keep an
        # environment-private copy so invariant checks do not silently add unmetered
        # compiler calls at every state boundary.  Public state never aliases this cache.
        self._return_validation_cache: dict[
            str, tuple[ValidationReceipt, tuple[Program, ...]]
        ] = {}
        if isinstance(self.reward_reads, bool) or not isinstance(self.reward_reads, int):
            raise ValueError("reward_reads must be a positive integer")
        if self.reward_reads <= 0 or self.reward_reads > self.ctx.reserve.evaluator_reads:
            raise ValueError("reward_reads must fit the registered terminal evaluator reserve")
        if self.mode is Mode.IMPROVEMENT:
            initialization_work = WorkVector(
                compiler_calls=4,
                validator_calls=1,
                restart_work=0 if self.initializer_precomputed else 1,
            )
            if not (
                self.budget_debit + initialization_work + self.ctx.reserve
            ).fits_in(self.ctx.caps):
                raise ValueError(
                    "improvement work caps must fit the budget debit, initialization, "
                    "and terminal reserve"
                )
        elif not (self.budget_debit + self.ctx.reserve).fits_in(self.ctx.caps):
            raise ValueError("construction work caps must fit the budget debit and reserve")

    # -- lifecycle ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> DecisionState | TerminalRecord | InitFailureRecord:
        # A failed reset must never leave a previously prepared state reusable.
        self.state = None
        self._prepared_state_fingerprint = None
        self._prepared_support_fingerprint = None
        self._integrity_seal = None
        self._return_validation_cache.clear()
        if seed is not None:
            self._seed = seed
            self.rng = random.Random(seed)
        spent = WorkVector()
        if self.mode is Mode.IMPROVEMENT:
            if self.initializer is None:
                raise IntegrityError("improvement mode needs a registered initializer")
            chains = self.initializer(self.task.logical, self.task.host, self._seed)
            spent = WorkVector(
                restart_work=0 if self.initializer_precomputed else 1,
                validator_calls=1,
                compiler_calls=4,
            )
            if chains is None:
                return InitFailureRecord("initializer returned no embedding", spent)
            receipt, programs = self._validated_return(chains, fresh=True)
            if not receipt.valid:
                return InitFailureRecord(f"initializer output rejected: {receipt.reasons}", spent)
            entry = self._entry(chains, protected=True)
            archive = [entry]
            reference_program = copy.deepcopy(programs[0])
            workspace_valid = True
        else:
            chains = {v: frozenset() for v in self.task.logical.nodes()}
            archive = []
            reference_program = None
            workspace_valid = False
        self.state = EnvState(
            chains=dict(chains),
            archive=archive,
            ages=self._zero_workspace_ages(chains),
            remaining=self.ctx.caps - self.budget_debit - spent,
            restarts_left=self.ctx.restart_allowance,
            restart_cache=(
                []
                if self._restart_cache_template is None
                else copy.deepcopy(list(self._restart_cache_template))
            ),
            spent=spent,
            budget_debit=self.budget_debit,
            workspace_valid=workspace_valid,
            reference_program=reference_program,
        )
        self._refresh_integrity_seal()
        return self._prepare()

    # -- preparation ----------------------------------------------------------------

    def _entry(self, chains: Mapping[Node, frozenset[Qubit]], *, protected: bool) -> ArchiveEntry:
        lengths = [len(c) for c in chains.values()] or [0]
        return ArchiveEntry(
            chains={i: frozenset(c) for i, c in chains.items()},
            protected=protected,
            age=0,
            admissible=True,
            qubits=len(occupancy(chains)),
            max_chain=max(lengths),
            key=chain_key(chains),
        )

    def _free_budget(self) -> WorkVector:
        """Search proposals may spend only what is left after the terminal reserve."""

        st = self._require_state()
        return st.remaining - self.ctx.reserve

    def _prepare(self, *, training_labels: bool = True) -> DecisionState | TerminalRecord:
        st = self._require_state()
        self._assert_search_state()
        free = self._free_budget()
        decisions_left = st.remaining.decisions

        raw_candidates: list[Candidate] = []
        charged = WorkVector()
        # Reserve a deterministic upper bound for all phase-specific feature overlays before
        # proposal generation.  The actual charge below can be smaller, but every surviving
        # application must still fit after preparation has completed.
        preparation_bound = WorkVector(
            compiler_calls=(
                self.ctx.max_state_changing
                + self.ctx.max_commit
                + 4 * self.ctx.quotas.get("restart", 0)
            ),
            validator_calls=self.ctx.max_state_changing,
            feature_work=32 * (self.ctx.padded_actions + len(st.chains)),
        )
        proposal_allowance = free - preparation_bound
        # State-changing generation needs room for its own decision and the reserved final one.
        if decisions_left >= 2 and proposal_allowance.is_nonnegative:
            batch = self.generator.generate(
                st.chains,
                self.rng,
                restarts_left=st.restarts_left,
                archive=st.archive,
                restart_cache=(
                    None
                    if self._restart_cache_template is None
                    else st.restart_cache
                ),
                allowance=proposal_allowance,
            )
            charged = batch.work
            if not charged.fits_in(proposal_allowance):
                overrun = {
                    field: (getattr(charged, field), getattr(proposal_allowance, field))
                    for field in charged.as_dict()
                    if getattr(charged, field) > getattr(proposal_allowance, field)
                }
                raise IntegrityError(
                    f"proposal generator exceeded the registered free-work bound: {overrun}"
                )
            retained_proposal_work = WorkVector()
            for candidate in batch.candidates:
                retained_proposal_work = retained_proposal_work + candidate.proposal_work
            if not retained_proposal_work.fits_in(charged):
                raise IntegrityError(
                    "retained candidates contain proposal work absent from the batch receipt"
                )
            self._charge(charged)
            raw_candidates.extend(batch.candidates)

        if self.ctx.reserve.fits_in(st.remaining):
            for k, entry in enumerate(st.archive[: self.ctx.max_commit]):
                if not entry.admissible:
                    continue
                raw_candidates.append(
                    Candidate(
                        opcode=Opcode.COMMIT,
                        affected=(),
                        old_chains={},
                        new_chains={},
                        work=self._commit_application_work(),
                        payload_key=f"COMMIT:{entry.key}",
                        archive_ref=k,
                        provenance="archive",
                    )
                )
        if self.mode is Mode.CONSTRUCTION and not st.archive and decisions_left >= 1:
            raw_candidates.append(
                Candidate(
                    opcode=Opcode.STOP,
                    affected=(),
                    old_chains={},
                    new_chains={},
                    work=WorkVector(decisions=1),
                    payload_key="STOP",
                    provenance="voluntary",
                )
            )

        if not raw_candidates:
            return self._terminal_failure(
                TerminalReason.BUDGET_NO_VALID
                if decisions_left <= 0
                else TerminalReason.NO_ACTION_NO_VALID,
                training_labels=training_labels,
            )

        raw_candidates = raw_candidates[: self.ctx.padded_actions]
        phase_compiles = self._phase_compiler_calls(raw_candidates)
        state_changing = [candidate for candidate in raw_candidates if candidate.changes_workspace]
        restart_validations = sum(
            1
            for candidate in state_changing
            if candidate.opcode is Opcode.RESTART and self.mode is Mode.IMPROVEMENT
        )
        feature_work = WorkVector(
            compiler_calls=phase_compiles + 4 * restart_validations,
            validator_calls=len(state_changing),
            feature_work=32 * (len(raw_candidates) + len(st.chains)),
        )
        if not feature_work.fits_in(st.remaining):
            raise IntegrityError("registered preparation/terminal reserve cannot fit feature work")
        self._charge(feature_work)

        candidates = raw_candidates
        mask = [
            (
                self._is_legal(candidate)
                if candidate.changes_workspace
                else self._is_legal_terminal(candidate)
            )
            for candidate in candidates
        ]
        if not any(mask):
            return self._terminal_failure(
                TerminalReason.BUDGET_NO_VALID
                if decisions_left <= 0
                else TerminalReason.NO_ACTION_NO_VALID,
                training_labels=training_labels,
            )

        program = st.reference_program if st.workspace_valid else None

        observation = build_observation(
            ctx=self.ctx,
            logical=self.task.logical,
            host=self.task.host,
            problem=self.task.problem,
            chains=st.chains,
            candidates=candidates,
            legal_mask=mask,
            archive=st.archive,
            remaining=st.remaining,
            ages=st.ages,
            strengths=self.strengths,
            program=program,
            mode_is_improvement=self.mode is Mode.IMPROVEMENT,
            restarts_left=st.restarts_left,
            workspace_valid=st.workspace_valid,
            protected_available=any(e.protected for e in st.archive),
            instance_scale=self.scale,
            coef_scale=self.coefficient_scale,
        )
        self._assert_search_state()
        state_fingerprint = self._state_fingerprint()
        support_fingerprint = candidate_support_key(candidates, mask)
        self._prepared_state_fingerprint = state_fingerprint
        self._prepared_support_fingerprint = support_fingerprint
        return DecisionState(
            observation=observation,
            candidates=tuple(candidates),
            legal_mask=tuple(mask),
            state_fingerprint=state_fingerprint,
            context_version=self.ctx.context_version,
            charged_work_receipt=charged + feature_work,
            support_fingerprint=support_fingerprint,
            exact_state=copy.deepcopy(st),
            random_state=copy.deepcopy(self.rng.getstate()),
        )

    def _disjoint(self) -> bool:
        st = self._require_state()
        occ = occupancy(st.chains)
        return bool(occ) and max(occ.values()) <= 1 and all(st.chains.values())

    def _is_legal(self, cand: Candidate) -> bool:
        """MODEL_SPEC Eq. (3f): bound, preconditions, dry ``p_search``, and work fits."""

        st = self._require_state()
        if not cand.changes_workspace or cand.opcode in (Opcode.COMMIT, Opcode.STOP):
            return False
        affected = tuple(cand.affected)
        affected_set = set(affected)
        is_archive_restore = (
            cand.opcode is Opcode.REWRITE_GROUP and cand.archive_ref is not None
        )
        if (
            not affected
            or len(affected_set) != len(affected)
            or affected_set != set(cand.old_chains)
            or affected_set != set(cand.new_chains)
            or not affected_set <= set(self.task.logical.nodes())
            or not cand.work.is_nonnegative
            or not cand.proposal_work.is_nonnegative
            or cand.proposal_work.materializations != 1
            or cand.work.decisions != 1
        ):
            return False
        if cand.archive_ref is not None and not is_archive_restore:
            return False
        if cand.opcode is not Opcode.RESTART and (
            cand.restart_cache_slot is not None
            or cand.restart_cache_after_digest is not None
        ):
            return False
        if any(st.chains.get(i, frozenset()) != cand.old_chains[i] for i in affected):
            return False
        if cand.opcode is Opcode.RESTART and st.restarts_left <= 0:
            return False

        successor = dict(st.chains)
        successor.update({i: frozenset(cand.new_chains[i]) for i in affected})
        if cand.payload_key != bound_successor_key(
            successor,
            cand.work,
            restart=cand.opcode is Opcode.RESTART,
            restart_cache_after_digest=cand.restart_cache_after_digest,
        ):
            return False
        if not self._routes_are_bound(cand, successor):
            return False

        old = cand.old_chains
        new = cand.new_chains
        changed = any(old[i] != new[i] for i in affected)
        if cand.opcode is Opcode.PLACE:
            if (
                self.mode is not Mode.CONSTRUCTION
                or len(affected) != 1
                or bool(old[affected[0]])
                or not bool(new[affected[0]])
                or cand.target_demand is not None
                or cand.target_conflict is not None
            ):
                return False
        elif cand.opcode is Opcode.ROUTE:
            target = cand.target_demand
            if (
                self.mode is not Mode.CONSTRUCTION
                or target is None
                or cand.target_conflict is not None
                or len(target) != 2
                or frozenset(target) not in {
                    frozenset(edge) for edge in self.task.logical.edges()
                }
                or not set(affected) <= set(target)
                or not cand.routes
                or any(not old[i] or not new[i] or not old[i] <= new[i] for i in affected)
            ):
                return False
            left, right = target
            if not st.chains.get(left) or not st.chains.get(right):
                return False
            if self._has_contact(st.chains[left], st.chains[right]):
                return False
            if not self._has_contact(successor[left], successor[right]):
                return False
        elif cand.opcode is Opcode.REWRITE_ONE:
            if (
                len(affected) != 1
                or not old[affected[0]]
                or not new[affected[0]]
                or not changed
                or cand.target_demand is not None
                or cand.target_conflict is not None
            ):
                return False
        elif cand.opcode is Opcode.REWRITE_GROUP:
            if (
                len(affected) not in self.ctx.group_sizes
                or any(not old[i] or not new[i] for i in affected)
                or not changed
                or cand.target_demand is not None
                or cand.target_conflict is not None
            ):
                return False
            if is_archive_restore:
                assert cand.archive_ref is not None
                if not 0 <= cand.archive_ref < len(st.archive):
                    return False
                entry = st.archive[cand.archive_ref]
                if (
                    not entry.admissible
                    or set(entry.chains) != set(self.task.logical.nodes())
                    or any(
                        frozenset(new[node]) != frozenset(entry.chains[node])
                        for node in affected
                    )
                ):
                    return False
        elif cand.opcode is Opcode.REPAIR_GROUP:
            if (
                len(affected) not in self.ctx.group_sizes
                or any(not old[i] or not new[i] for i in affected)
                or not changed
                or (cand.target_demand is None) == (cand.target_conflict is None)
            ):
                return False
            if cand.target_conflict is not None:
                claimants = {
                    i for i, chain in st.chains.items() if cand.target_conflict in chain
                }
                if len(claimants) < 2 or not claimants <= affected_set:
                    return False
            if cand.target_demand is not None:
                left, right = cand.target_demand
                if (
                    frozenset((left, right))
                    not in {frozenset(edge) for edge in self.task.logical.edges()}
                    or not {left, right} <= affected_set
                    or not st.chains.get(left)
                    or not st.chains.get(right)
                    or self._has_contact(st.chains[left], st.chains[right])
                ):
                    return False
        elif cand.opcode is Opcode.RESTART:
            if (
                affected_set != set(self.task.logical.nodes())
                or cand.routes
                or cand.target_demand is not None
                or cand.target_conflict is not None
            ):
                return False
            if self._restart_cache_template is None:
                if (
                    cand.restart_cache_slot is not None
                    or cand.restart_cache_after_digest is not None
                ):
                    return False
            else:
                if (
                    type(cand.restart_cache_slot) is not int
                    or not 0 <= cand.restart_cache_slot < len(st.restart_cache)
                    or cand.restart_cache_after_digest is None
                ):
                    return False
                slot = st.restart_cache[cand.restart_cache_slot]
                if (
                    slot.status != "SUCCESS"
                    or slot.consumed
                    or slot.chains is None
                    or dict(slot.chains) != successor
                ):
                    return False
                after = tuple(
                    item.consume()
                    if item.slot_index == cand.restart_cache_slot
                    else item
                    for item in st.restart_cache
                )
                if cand.restart_cache_after_digest != restart_cache_digest(after):
                    return False
            if self.mode is Mode.IMPROVEMENT:
                if self._restart_cache_template is None:
                    if cand.proposal_work.restart_work < 1:
                        return False
                elif cand.proposal_work.restart_work != 0:
                    return False
            if self.mode is Mode.CONSTRUCTION:
                if any(successor.values()):
                    return False
            else:
                receipt, _ = p_return(
                    successor,
                    self.task.logical,
                    self.task.host,
                    self.task.problem,
                    self.ctx,
                )
                if not receipt.valid:
                    return False
        else:
            return False

        receipt = p_search(
            successor,
            self.task.logical,
            self.task.host,
            self.ctx.qubit_cap,
            self.ctx.overlap,
            allow_empty=self.mode is Mode.CONSTRUCTION,
        )
        if not receipt.valid:
            return False
        return cand.work.fits_in(self._free_budget())

    def _is_legal_terminal(self, cand: Candidate) -> bool:
        st = self._require_state()
        if cand.opcode is Opcode.COMMIT:
            return (
                not cand.affected
                and not cand.old_chains
                and not cand.new_chains
                and cand.archive_ref is not None
                and 0 <= cand.archive_ref < len(st.archive)
                and st.archive[cand.archive_ref].admissible
                and cand.work == self._commit_application_work()
                and cand.work.fits_in(st.remaining)
            )
        if cand.opcode is Opcode.STOP:
            return (
                self.mode is Mode.CONSTRUCTION
                and not st.archive
                and not cand.affected
                and not cand.old_chains
                and not cand.new_chains
                and cand.archive_ref is None
                and WorkVector(decisions=1).fits_in(st.remaining)
            )
        return False

    def _commit_application_work(self) -> WorkVector:
        return WorkVector(
            decisions=1,
            compiler_calls=4,
            validator_calls=1,
            evaluator_reads=self.reward_reads,
            feature_work=4,
        )

    def _routes_are_bound(
        self,
        cand: Candidate,
        successor: Mapping[Node, frozenset[Qubit]],
    ) -> bool:
        route_owners: set[Node] = set()
        for path, owner in cand.routes:
            if owner not in cand.affected or not path:
                return False
            if len(path) != len(set(path)):
                return False
            if any(q not in self.task.host or q not in successor[owner] for q in path):
                return False
            if any(not self.task.host.has_edge(left, right) for left, right in zip(path, path[1:])):
                return False
            route_owners.add(owner)

        if cand.opcode is not Opcode.ROUTE:
            return True

        if cand.target_demand is None or route_owners != set(cand.affected):
            return False
        left, right = cand.target_demand
        for owner in cand.affected:
            if owner not in (left, right):
                return False
            old_chain = cand.old_chains[owner]
            new_chain = cand.new_chains[owner]
            additions = set(new_chain) - set(old_chain)
            if not additions:
                return False
            owner_paths = [path for path, route_owner in cand.routes if route_owner == owner]
            if any(path[0] not in old_chain for path in owner_paths):
                return False
            declared = {qubit for path in owner_paths for qubit in path}
            if declared - set(old_chain) != additions:
                return False
            other = right if owner == left else left
            if any(
                not any(
                    endpoint != target
                    and self.task.host.has_edge(endpoint, target)
                    for target in successor[other]
                )
                for endpoint in (path[-1] for path in owner_paths)
            ):
                return False
        return True

    def _has_contact(
        self,
        left: frozenset[Qubit],
        right: frozenset[Qubit],
    ) -> bool:
        return any(
            q != r and self.task.host.has_edge(q, r)
            for q in left
            for r in right
        )

    def _phase_compiler_calls(self, candidates: Sequence[Candidate]) -> int:
        """Count the exact complete phase overlays built by ``build_observation``."""

        st = self._require_state()
        calls = sum(1 for entry in st.archive if entry.admissible)
        for candidate in candidates:
            if not candidate.changes_workspace:
                continue
            successor = dict(st.chains)
            successor.update(candidate.new_chains)
            receipt = p_embed(
                successor,
                self.task.logical,
                self.task.host,
                self.ctx.qubit_cap,
            )
            if receipt.valid:
                calls += 1
        return calls

    # -- transition -----------------------------------------------------------------

    def step(
        self,
        decision: DecisionState,
        chosen_local_index: int,
        *,
        evaluate_training_reward: bool = True,
        transition_seed: int | None = None,
    ) -> StepResult:
        """Apply one bound action and prepare its successor exactly once.

        ``transition_seed`` is an evaluator/counterfactual-only hook for choosing fresh
        exogenous randomness *after* the actor has selected from the already bound support.
        It is never part of an observation or action.  Ordinary rollouts leave it unset and
        advance the environment's registered random tape.
        """

        st = self._require_state()
        self._assert_search_state()
        current_fingerprint = self._state_fingerprint()
        if (
            decision.state_fingerprint != current_fingerprint
            or decision.state_fingerprint != self._prepared_state_fingerprint
        ):
            raise IntegrityError("stale decision state: fingerprint does not match the workspace")
        supplied_support = candidate_support_key(decision.candidates, decision.legal_mask)
        if (
            decision.support_fingerprint != supplied_support
            or decision.support_fingerprint != self._prepared_support_fingerprint
        ):
            raise IntegrityError("candidate payload/support fingerprint mismatch")
        if decision.context_version != self.ctx.context_version:
            raise IntegrityError("decision state uses a different context version")
        if not (0 <= chosen_local_index < len(decision.candidates)):
            raise IntegrityError("action index outside the materialised support")
        if not decision.legal_mask[chosen_local_index]:
            raise IntegrityError("a masked action was selected")
        cand = decision.candidates[chosen_local_index]
        if transition_seed is not None:
            if (
                isinstance(transition_seed, bool)
                or not isinstance(transition_seed, int)
                or not 0 <= transition_seed < 2**63
            ):
                raise ValueError("transition_seed must be a nonnegative 63-bit integer")
            # The current action/support was verified above.  Fork only the transition's
            # future proposal/evaluator randomness, then bind the resulting next decision
            # to that new tape in its own state fingerprint.
            self.rng = random.Random(transition_seed)
        self._prepared_state_fingerprint = None
        self._prepared_support_fingerprint = None

        if cand.opcode is Opcode.COMMIT:
            return self._commit(cand, evaluate_training_reward)
        if cand.opcode is Opcode.STOP:
            self._charge(WorkVector(decisions=1))
            record = self._terminal_failure(
                TerminalReason.STOP_NO_VALID,
                training_labels=evaluate_training_reward,
            )
            return StepResult(record, 0.0, 1.0, True, terminal_reason=TerminalReason.STOP_NO_VALID)

        # Eq. (3i): the decision coordinate of application work is one, charged exactly once.
        self._charge(cand.work)
        previous = dict(st.chains)
        st.chains.update({i: frozenset(c) for i, c in cand.new_chains.items()})
        if cand.opcode is Opcode.RESTART:
            if cand.restart_cache_slot is not None:
                st.restart_cache[cand.restart_cache_slot] = st.restart_cache[
                    cand.restart_cache_slot
                ].consume()
            st.restarts_left -= 1
            self._reset_workspace_ages()
            st.archive[:] = [entry.aged() for entry in st.archive]
        else:
            self._advance_ages(previous)

        receipt, programs = self._validated_return(st.chains, fresh=True)
        st.workspace_valid = receipt.valid
        st.reference_program = copy.deepcopy(programs[0]) if receipt.valid else None
        if receipt.valid:
            self._admit(st.chains)

        self._refresh_integrity_seal()

        nxt = self._prepare(training_labels=evaluate_training_reward)
        if isinstance(nxt, TerminalRecord):
            reward = 0.0
            cost = 0.0 if nxt.returned_valid else 1.0
            return StepResult(nxt, reward, cost, True, terminal_reason=nxt.terminal_reason)
        return StepResult(nxt, 0.0, 0.0, False, work_receipt=cand.work)

    def _charge(self, work: WorkVector) -> None:
        st = self._require_state()
        self._assert_search_state()
        if not work.is_nonnegative:
            raise IntegrityError("attempted to charge negative work")
        if not work.fits_in(st.remaining):
            raise IntegrityError("operation exceeded a registered binding work cap")
        st.remaining = st.remaining - work
        st.spent = st.spent + work
        st.decisions_used = st.spent.decisions
        if st.remaining + st.spent + st.budget_debit != self.ctx.caps:
            raise IntegrityError("work ledger no longer partitions the registered fixed caps")
        self._refresh_integrity_seal()

    def _validated_return(
        self,
        chains: Mapping[Node, frozenset[Qubit]],
        *,
        fresh: bool = False,
    ) -> tuple[ValidationReceipt, tuple[Program, ...]]:
        """Return an independently derived return predicate without aliasing public state.

        A fresh check is used at initializer acceptance, after every selected successor and
        again at COMMIT.  Invariant-only checks reuse an environment-private copy keyed by
        the exact assignment.  This preserves fail-closed checking without inventing
        unmetered compiler work at each internal assertion.
        """

        key = chain_key(chains)
        cached = self._return_validation_cache.get(key)
        if cached is None or fresh:
            checked = p_return(
                chains,
                self.task.logical,
                self.task.host,
                self.task.problem,
                self.ctx,
            )
            self._return_validation_cache[key] = copy.deepcopy(checked)
        if _SKIP_INTERNAL_ASSERTS:
            # The cached entry is already an environment-private deep copy that nothing else
            # holds a reference to, so the second copy on every read exists only to stop a
            # caller mutating what it was handed. Under the fast path it is the caller's
            # contract not to mutate; the checked path keeps the copy.
            return self._return_validation_cache[key]
        return copy.deepcopy(self._return_validation_cache[key])

    def _zero_workspace_ages(
        self, chains: Mapping[Node, frozenset[Qubit]]
    ) -> Ages:
        owners = self._owners(chains)
        return Ages(
            chain={node: 0 for node in self.task.logical.nodes()},
            claim={
                (node, qubit): 0
                for node, chain in chains.items()
                for qubit in chain
            },
            conflict={qubit: 0 for qubit, claimants in owners.items() if len(claimants) > 1},
            demand={
                _canonical_pair(left, right): 0
                for left, right in self.task.logical.edges()
            },
            occupancy={qubit: 0 for qubit in owners},
        )

    @staticmethod
    def _owners(
        chains: Mapping[Node, frozenset[Qubit]],
    ) -> dict[Qubit, frozenset[Node]]:
        owners: dict[Qubit, set[Node]] = {}
        for node, chain in chains.items():
            for qubit in chain:
                owners.setdefault(qubit, set()).add(node)
        return {qubit: frozenset(nodes) for qubit, nodes in owners.items()}

    @staticmethod
    def _work_is_exact(work: object) -> bool:
        return isinstance(work, WorkVector) and all(
            type(getattr(work, field)) is int for field in WORK_FIELDS
        )

    @staticmethod
    def _age_map_is_exact(
        values: object,
        expected_keys: set[object],
        decisions_used: int,
    ) -> bool:
        return (
            isinstance(values, dict)
            and set(values) == expected_keys
            and all(
                type(age) is int and 0 <= age <= decisions_used
                for age in values.values()
            )
        )

    def _assert_search_state(self, *, verify_seal: bool = True) -> None:
        if _SKIP_INTERNAL_ASSERTS:
            # Opt-in fast path for timing measurements, off unless the environment variable is
            # set. These are defensive re-checks of invariants the transitions already maintain,
            # not the fail-closed return validation, which is never skipped. Any timing reported
            # under this flag has to be shown to produce identical episodes to the checked path,
            # which `probes/check_fast_path.py` does.
            return
        """Enforce the whole exact-state form of ``P_search`` and fail closed.

        This validates derived ownership through exact age-map domains, the archive and
        return-program context, every memory/counter domain and the complete work identity.
        A separate private seal catches otherwise plausible out-of-transition mutations,
        such as decrementing a restart counter while keeping it in range.
        """

        st = self._require_state()
        logical_nodes = set(self.task.logical.nodes())
        if not isinstance(st.chains, dict) or set(st.chains) != logical_nodes:
            raise IntegrityError("search-state chains do not exactly cover the logical domain")
        if any(type(chain) is not frozenset for chain in st.chains.values()):
            raise IntegrityError("search-state chains must be stored as exact frozen sets")

        for label, work in (
            ("remaining", st.remaining),
            ("spent", st.spent),
            ("budget debit", st.budget_debit),
        ):
            if not self._work_is_exact(work):
                raise IntegrityError(f"{label} work is not an exact integer WorkVector")
            if not work.is_nonnegative:
                raise IntegrityError(f"{label} work contains a negative coordinate")
        if st.budget_debit != self.budget_debit:
            raise IntegrityError("state budget debit differs from the registered fixed debit")
        if st.remaining + st.spent + st.budget_debit != self.ctx.caps:
            raise IntegrityError("work ledger does not partition the registered fixed caps")
        if type(st.decisions_used) is not int or st.decisions_used < 0:
            raise IntegrityError("decision count must be a nonnegative exact integer")
        if st.decisions_used != st.spent.decisions:
            raise IntegrityError("decision count differs from the charged decision ledger")
        if (
            type(st.restarts_left) is not int
            or not 0 <= st.restarts_left <= self.ctx.restart_allowance
        ):
            raise IntegrityError("remaining selected-restart count is outside its registry")
        if self._restart_cache_template is None:
            if st.restart_cache or self.restart_cache_manifest_digest is not None:
                raise IntegrityError("unregistered restart-cache state is present")
        else:
            if (
                len(st.restart_cache) != len(self._restart_cache_template)
                or any(
                    not isinstance(slot, RestartCacheSlot)
                    for slot in st.restart_cache
                )
                or tuple(slot.slot_index for slot in st.restart_cache)
                != tuple(range(len(st.restart_cache)))
            ):
                raise IntegrityError("restart-cache inventory is malformed")
            for slot, registered in zip(
                st.restart_cache,
                self._restart_cache_template,
                strict=True,
            ):
                if (
                    slot.slot_index != registered.slot_index
                    or slot.status != registered.status
                    or slot.chains != registered.chains
                    or slot.snapshot_record_digest
                    != registered.snapshot_record_digest
                    or slot.attempt_receipt_root != registered.attempt_receipt_root
                ):
                    raise IntegrityError("restart-cache evidence changed after reset")
                if slot.chains is not None and (
                    set(slot.chains) != logical_nodes
                    or any(
                        type(chain) is not frozenset
                        or not set(chain) <= set(self.task.host.nodes())
                        for chain in slot.chains.values()
                    )
                ):
                    raise IntegrityError("restart-cache embedding changed graph domain")
            consumed = sum(slot.consumed for slot in st.restart_cache)
            if consumed != self.ctx.restart_allowance - st.restarts_left:
                raise IntegrityError(
                    "restart-cache consumption differs from the selected-restart ledger"
                )

        search_receipt = p_search(
            st.chains,
            self.task.logical,
            self.task.host,
            self.ctx.qubit_cap,
            self.ctx.overlap,
            remaining=st.remaining,
            allow_empty=self.mode is Mode.CONSTRUCTION,
        )
        if not search_receipt.valid:
            raise IntegrityError(
                f"workspace violates P_search: {search_receipt.reasons}"
            )

        owners = self._owners(st.chains)
        if not isinstance(st.ages, Ages):
            raise IntegrityError("workspace ages do not use the registered Ages schema")
        age_domains: tuple[tuple[str, object, set[object]], ...] = (
            ("chain", st.ages.chain, set(logical_nodes)),
            (
                "claim",
                st.ages.claim,
                {
                    (node, qubit)
                    for node, chain in st.chains.items()
                    for qubit in chain
                },
            ),
            (
                "conflict",
                st.ages.conflict,
                {qubit for qubit, claimants in owners.items() if len(claimants) > 1},
            ),
            (
                "demand",
                st.ages.demand,
                {
                    _canonical_pair(left, right)
                    for left, right in self.task.logical.edges()
                },
            ),
            ("occupancy", st.ages.occupancy, set(owners)),
        )
        for label, values, expected_keys in age_domains:
            if not self._age_map_is_exact(values, expected_keys, st.decisions_used):
                raise IntegrityError(
                    f"{label} ages have a wrong key domain or invalid decision age"
                )

        if not isinstance(st.archive, list) or any(
            not isinstance(entry, ArchiveEntry) for entry in st.archive
        ):
            raise IntegrityError("archive does not contain only ArchiveEntry records")
        protected = [index for index, entry in enumerate(st.archive) if entry.protected]
        if self.mode is Mode.IMPROVEMENT:
            if not st.archive or protected != [0]:
                raise IntegrityError(
                    "improvement archive must retain exactly its protected first slot"
                )
            if len(st.archive) > self.ctx.archive_slots:
                raise IntegrityError("improvement archive exceeds its registered capacity")
        else:
            if protected:
                raise IntegrityError("construction archive cannot contain a protected slot")
            if len(st.archive) > self.ctx.max_commit:
                raise IntegrityError("construction archive exceeds its registered capacity")

        archive_keys: set[str] = set()
        for entry in st.archive:
            if (
                type(entry.protected) is not bool
                or entry.admissible is not True
                or type(entry.age) is not int
                or not 0 <= entry.age <= st.decisions_used
                or type(entry.qubits) is not int
                or type(entry.max_chain) is not int
            ):
                raise IntegrityError("archive metadata has an invalid type, flag or age")
            if not isinstance(entry.chains, Mapping) or set(entry.chains) != logical_nodes:
                raise IntegrityError("archive entry does not cover the logical domain")
            if any(type(chain) is not frozenset for chain in entry.chains.values()):
                raise IntegrityError("archive chains must be stored as exact frozen sets")
            computed_key = chain_key(entry.chains)
            lengths = [len(chain) for chain in entry.chains.values()] or [0]
            if (
                entry.key != computed_key
                or entry.qubits != len(occupancy(entry.chains))
                or entry.max_chain != max(lengths)
            ):
                raise IntegrityError("archive metadata does not match its exact assignment")
            if entry.key in archive_keys:
                raise IntegrityError("archive contains duplicate exact assignments")
            archive_keys.add(entry.key)
            try:
                archive_receipt, _ = self._validated_return(entry.chains)
            except Exception as exc:
                raise IntegrityError("archive return validation raised an exception") from exc
            if not archive_receipt.valid:
                raise IntegrityError(
                    f"archive entry violates P_return: {archive_receipt.reasons}"
                )

        if type(st.workspace_valid) is not bool:
            raise IntegrityError("workspace-valid flag must be an exact Boolean")
        try:
            workspace_receipt, workspace_programs = self._validated_return(st.chains)
        except Exception as exc:
            raise IntegrityError("workspace return validation raised an exception") from exc
        if st.workspace_valid is not workspace_receipt.valid:
            raise IntegrityError("workspace-valid flag differs from exact P_return")
        if workspace_receipt.valid:
            if (
                not workspace_programs
                or not isinstance(st.reference_program, Program)
                or st.reference_program != workspace_programs[0]
            ):
                raise IntegrityError(
                    "valid workspace reference program differs from its exact F_1 program"
                )
        elif st.reference_program is not None:
            raise IntegrityError("nonreturnable workspace retained a reference program")

        if (
            verify_seal
            and self._integrity_seal is not None
            and self._integrity_seal != self._search_state_digest()
        ):
            raise IntegrityError("exact search state changed outside a registered transition")

    def _refresh_integrity_seal(self) -> None:
        self._assert_search_state(verify_seal=False)
        self._integrity_seal = self._search_state_digest()

    @staticmethod
    def _mapping_payload(values: Mapping[object, int]) -> list[list[object]]:
        def key_payload(key: object) -> object:
            if isinstance(key, tuple):
                return {"tuple": [key_payload(value) for value in key]}
            if not isinstance(key, Hashable):
                raise IntegrityError("exact-state mapping contains an unhashable key")
            return _identity_payload(key)

        rows = [[key_payload(key), value] for key, value in values.items()]
        return sorted(rows, key=lambda row: repr(row[0]))

    @staticmethod
    def _program_payload(program: Program | None) -> object:
        if program is None:
            return None

        def scalar_rows(values: Mapping[Hashable, float]) -> list[list[object]]:
            return sorted(
                [[_identity_payload(key), value] for key, value in values.items()],
                key=lambda row: repr(row[0]),
            )

        def pair_rows(values: Mapping[tuple[Hashable, Hashable], object]) -> list[list[object]]:
            return sorted(
                [
                    [_identity_payload(left), _identity_payload(right), value]
                    for (left, right), value in values.items()
                ],
                key=lambda row: repr(row[:2]),
            )

        return {
            "strength": program.strength,
            "strength_index": program.strength_index,
            "scale": program.scale,
            "h_phys": scalar_rows(program.h_phys),
            "j_phys": pair_rows(program.j_phys),
            "chain_edges": sorted(
                [
                    [
                        _identity_payload(node),
                        sorted(
                            [
                                [_identity_payload(left), _identity_payload(right)]
                                for left, right in edges
                            ],
                            key=repr,
                        ),
                    ]
                    for node, edges in program.chain_edges.items()
                ],
                key=lambda row: repr(row[0]),
            ),
            "contact_counts": pair_rows(program.contact_counts),
            "offset": program.offset,
        }

    def _exact_state_payload(self) -> dict[str, object]:
        st = self._require_state()
        return {
            "chains": chain_key(st.chains),
            "archive": [
                {
                    "chains": chain_key(entry.chains),
                    "protected": entry.protected,
                    "age": entry.age,
                    "admissible": entry.admissible,
                    "qubits": entry.qubits,
                    "max_chain": entry.max_chain,
                    "key": entry.key,
                }
                for entry in st.archive
            ],
            "ages": {
                "chain": self._mapping_payload(st.ages.chain),
                "claim": self._mapping_payload(st.ages.claim),
                "conflict": self._mapping_payload(st.ages.conflict),
                "demand": self._mapping_payload(st.ages.demand),
                "occupancy": self._mapping_payload(st.ages.occupancy),
            },
            "remaining": st.remaining.as_dict(),
            "spent": st.spent.as_dict(),
            "budget_debit": st.budget_debit.as_dict(),
            "restarts_left": st.restarts_left,
            "restart_cache_manifest_digest": self.restart_cache_manifest_digest,
            "restart_cache": [
                {
                    "slot_index": slot.slot_index,
                    "status": slot.status,
                    "chains": None if slot.chains is None else chain_key(slot.chains),
                    "snapshot_record_digest": slot.snapshot_record_digest,
                    "attempt_receipt_root": slot.attempt_receipt_root,
                    "consumed": slot.consumed,
                }
                for slot in st.restart_cache
            ],
            "decisions_used": st.decisions_used,
            "workspace_valid": st.workspace_valid,
            "reference_program": self._program_payload(st.reference_program),
            "mode": self.mode.value,
            "context_version": self.ctx.context_version,
        }

    def _search_state_digest(self) -> str:
        return stable_digest(self._exact_state_payload())

    def _state_fingerprint(self) -> str:
        """Digest every transition-relevant exact-state field after preparation charges."""

        payload = self._exact_state_payload()
        # Exogenous randomness is transition-relevant because it determines the next
        # proposal support.  It is bound here and retained for replay, but deliberately
        # absent from the neural observation and the mutation-only integrity seal.
        payload["random_state"] = self.rng.getstate()
        return stable_digest(payload)

    def _advance_ages(self, previous: Mapping[Node, frozenset[Qubit]]) -> None:
        st = self._require_state()
        ages = st.ages
        for i, chain in st.chains.items():
            if previous.get(i) != chain:
                ages.chain[i] = 0
            else:
                ages.chain[i] = ages.chain.get(i, 0) + 1
        owners = self._owners(st.chains)
        before_owners = self._owners(previous)
        for q, claimants in owners.items():
            same_claimants = before_owners.get(q) == claimants
            if len(claimants) > 1:
                ages.conflict[q] = (
                    ages.conflict.get(q, 0) + 1
                    if same_claimants and len(before_owners.get(q, frozenset())) > 1
                    else 0
                )
            else:
                ages.conflict.pop(q, None)
            ages.occupancy[q] = ages.occupancy.get(q, 0) + 1 if same_claimants else 0
        for q in set(before_owners) - set(owners):
            ages.occupancy.pop(q, None)
            ages.conflict.pop(q, None)
        live_claims = {(i, q) for i, chain in st.chains.items() for q in chain}
        for stale in set(ages.claim) - live_claims:
            ages.claim.pop(stale, None)
        for i, chain in st.chains.items():
            for q in chain:
                key = (i, q)
                if q in previous.get(i, frozenset()):
                    ages.claim[key] = ages.claim.get(key, 0) + 1
                else:
                    ages.claim[key] = 0
        for u, v in self.task.logical.edges():
            pair = _canonical_pair(u, v)
            was_realized = any(
                a != b and self.task.host.has_edge(a, b)
                for a in previous.get(u, frozenset())
                for b in previous.get(v, frozenset())
            )
            realized = any(
                a != b and self.task.host.has_edge(a, b)
                for a in st.chains.get(u, frozenset())
                for b in st.chains.get(v, frozenset())
            )
            if realized or was_realized or pair not in ages.demand:
                ages.demand[pair] = 0
            else:
                ages.demand[pair] += 1
        st.archive[:] = [e.aged() for e in st.archive]

    def _reset_workspace_ages(self) -> None:
        """Reset every workspace-specific clock after a selected RESTART.

        Archive entries survive the restart and are aged separately by the selected decision.
        A membership that happens to equal its predecessor is still new restart state and must
        not inherit or increment the old chain/claim/occupancy clock.
        """

        st = self._require_state()
        st.ages = self._zero_workspace_ages(st.chains)

    def _admit(self, chains: Mapping[Node, frozenset[Qubit]]) -> None:
        """Only the selected, realised and independently validated workspace is archived."""

        st = self._require_state()
        key = chain_key(chains)
        for entry in st.archive:
            if entry.key == key:
                return
        entry = self._entry(chains, protected=False)
        mutable = [e for e in st.archive if not e.protected]
        protected = [e for e in st.archive if e.protected]
        mutable.append(entry)
        mutable_capacity = (
            self.ctx.max_commit
            if self.mode is Mode.CONSTRUCTION
            else self.ctx.archive_fifo
        )
        while len(mutable) > mutable_capacity:
            mutable.pop(0)
        st.archive[:] = protected + mutable

    # -- terminals ------------------------------------------------------------------

    def _commit(self, cand: Candidate, evaluate: bool) -> StepResult:
        st = self._require_state()
        if cand.archive_ref is None or cand.archive_ref >= len(st.archive):
            raise IntegrityError("COMMIT referenced a missing archive entry")
        entry = st.archive[cand.archive_ref]
        receipt, programs = self._validated_return(entry.chains, fresh=True)
        self._charge(
            WorkVector(
                decisions=1,
                validator_calls=1,
                compiler_calls=4,
                feature_work=4,
            )
        )
        if not receipt.valid:
            raise IntegrityError(f"archived entry failed revalidation: {receipt.reasons}")

        features = [program_features(p, entry.chains, self.task.problem) for p in programs]
        if all(
            hasattr(self.selector, name)
            for name in (
                "select_embedding",
                "deployment_ready",
                "normalizer_digest",
                "coefficient_transform_scale",
            )
        ):
            graph_selector = self.selector
            if not graph_selector.deployment_ready:
                raise IntegrityError("graph strength selector is not frozen/deployment-ready")
            from isingfold.rl.strength_tensorize import build_strength_inputs

            graph_inputs = build_strength_inputs(
                ctx=self.ctx,
                logical=self.task.logical,
                host=self.task.host,
                problem=self.task.problem,
                chains=entry.chains,
                programs=programs,
                coef_scale=float(graph_selector.coefficient_transform_scale),
                normalizer_digest=graph_selector.normalizer_digest,
            )
            raw_index = graph_selector.select_embedding(graph_inputs)
        else:
            raw_index = self.selector(programs, features)
        try:
            index = operator.index(raw_index)
        except TypeError as exc:
            raise IntegrityError("strength selector returned a non-integral index") from exc
        if not 0 <= index < len(programs):
            raise IntegrityError("strength selector returned an out-of-range index")
        chosen = programs[index]

        counts: tuple[int, int] | None = None
        evaluator_seed: int | None = None
        reward = 0.0
        if evaluate:
            if self.task.ground_energy is None:
                raise IntegrityError("training evaluation requires a certified reference energy")
            evaluator_seed = self.rng.randrange(2**31)
            block: ReadBlock = sample_program(
                chosen,
                entry.chains,
                self.task.problem,
                self.task.ground_energy,
                num_reads=self.reward_reads,
                seed=evaluator_seed,
                num_sweeps=self.ctx.num_sweeps,
            )
            counts = (block.hits, block.reads)
            reward = block.rate
            self._charge(WorkVector(evaluator_reads=block.reads))

        record = TerminalRecord(
            returned_valid=True,
            terminal_reason=TerminalReason.COMMIT,
            embedding=entry.chains,
            selected_strength=chosen.strength,
            selected_index=index,
            validation_receipt=receipt.as_dict(),
            cumulative_work=st.spent,
            selected_program=chosen,
            compiled_programs=programs,
            training_reward=reward if evaluate else None,
            training_cost=0.0 if evaluate else None,
            evaluator_counts=counts,
            evaluator_seed=evaluator_seed,
        )
        return StepResult(record, reward, 0.0, True, terminal_reason=TerminalReason.COMMIT)

    def _terminal_failure(
        self,
        reason: TerminalReason,
        *,
        training_labels: bool = True,
    ) -> TerminalRecord:
        st = self._require_state()
        self._assert_search_state()
        return TerminalRecord(
            returned_valid=False,
            terminal_reason=reason,
            embedding=None,
            selected_strength=None,
            selected_index=None,
            validation_receipt={"valid": False, "reasons": [reason.value]},
            cumulative_work=st.spent,
            training_reward=0.0 if training_labels else None,
            training_cost=1.0 if training_labels else None,
        )

    def _require_state(self) -> EnvState:
        if self.state is None:
            raise IntegrityError("environment used before reset")
        return self.state

    # -- diagnostics ----------------------------------------------------------------

    def workspace_report(self) -> dict[str, object]:
        st = self._require_state()
        self._assert_search_state()
        return {
            "qubits": len(occupancy(st.chains)),
            "unrealized": unrealized_demands(st.chains, self.task.logical, self.task.host),
            "archive": len(st.archive),
            "decisions_used": st.decisions_used,
            "remaining": st.remaining.as_dict(),
        }
