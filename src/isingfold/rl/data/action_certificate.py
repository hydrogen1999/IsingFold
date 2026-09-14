"""Environment-aligned state/action envelopes and exact structural certificates.

The bridge in this module starts from a live :class:`~isingfold.rl.env.EmbeddingEnv`
decision.  It binds the exact public task, prepared search state, complete materialised
candidate support and every legal-mask bit before any structural label is produced.

Version 1 deliberately certifies only a finite *persistent-core, disjoint-terminal*
continuation domain.  Applying an action is atomic, and every nonempty post-action chain is
a mandatory core of that variable's eventual chain.  In particular, choosing ``PLACE i at
{q}`` can never be labelled by a completion that silently places ``i`` somewhere else.

Overlap-enabled repair needs a richer continuation grammar in which later bound rewrites
can move claims.  That oracle is not implemented here.  A post-action overlap therefore
produces an explicit unsupported result with ``feasible=None``; it is never presented as an
infeasibility certificate.  Likewise, hitting the deterministic DFS-node limit is unknown,
not false.  No evaluator target, planted witness, solve count, or Minorminer outcome enters
the envelope or the oracle.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Hashable, Iterable, Mapping, Sequence

import networkx as nx

from isingfold.rl.contracts import (
    WORK_FIELDS,
    Candidate,
    DecisionState,
    Opcode,
    WorkVector,
    candidate_support_key,
    stable_digest,
)
from isingfold.rl.env import EmbeddingEnv, EmbeddingTask, EnvState
from isingfold.rl.proposal import bound_successor_key
from isingfold.rl.validate import occupancy, p_embed

Node = Hashable
Qubit = Hashable

ENVELOPE_SCHEMA = "isingfold.state-action-envelope"
BOUND_CANDIDATE_SCHEMA = "isingfold.bound-action-candidate"
APPLIED_ACTION_SCHEMA = "isingfold.applied-state-action"
CONTINUATION_SCHEMA = "isingfold.persistent-continuation-domain"
CERTIFICATE_SCHEMA = "isingfold.structural-action-certificate"
PUBLIC_TASK_SCHEMA_VERSION = 1
BOUND_CANDIDATE_SCHEMA_VERSION = 2
ENVELOPE_SCHEMA_VERSION = 2
APPLIED_ACTION_SCHEMA_VERSION = 2
CONTINUATION_SCHEMA_VERSION = 1
CERTIFICATE_SCHEMA_VERSION = 1

# The implementation registry historically imports this name. It identifies the current
# envelope contract, whose v2 rows carry authenticated restart-cache evidence.
SCHEMA_VERSION = ENVELOPE_SCHEMA_VERSION

_PERSISTENCE_SEMANTICS = "post-action-nonempty-chains-are-connected-superset-cores-v1"
_TERMINAL_SEMANTICS = "complete-connected-disjoint-contact-realising-within-qubit-cap-v1"
_OVERLAP_SUPPORT = "unsupported-fail-closed-v1"
_HEX = frozenset("0123456789abcdef")


class ActionCertificateError(ValueError):
    """The envelope or registered continuation domain violates its contract."""


class UnsupportedActionCertificateError(ActionCertificateError):
    """The selected action has no structural-continuation semantics in this version."""


class CertificateStatus(str, Enum):
    """Exhaustive status; only the first two values are exact Boolean labels."""

    CERTIFIED_FEASIBLE = "certified_feasible"
    CERTIFIED_INFEASIBLE = "certified_infeasible"
    UNKNOWN_NODE_LIMIT = "unknown_node_limit"
    UNSUPPORTED_OVERLAP_REPAIR = "unsupported_overlap_repair"


def _identity_key(value: Hashable) -> tuple[str, str]:
    if type(value) is int:
        return "integer", str(value)
    if type(value) is str:
        return "text", value
    raise ActionCertificateError("persistent identities must be a canonical integer or text scalar")


def _identity_payload(value: Hashable) -> dict[str, object]:
    kind, _canonical_value = _identity_key(value)
    return {"kind": kind, "value": value}


def _canonical_pair(left: Hashable, right: Hashable) -> tuple[Hashable, Hashable]:
    return (left, right) if _identity_key(left) <= _identity_key(right) else (right, left)


def _pair_payload(left: Hashable, right: Hashable) -> list[dict[str, object]]:
    first, second = _canonical_pair(left, right)
    return [_identity_payload(first), _identity_payload(second)]


def _chain_rows(
    chains: Mapping[Node, Iterable[Qubit]],
) -> list[dict[str, object]]:
    return [
        {
            "node": _identity_payload(node),
            "qubits": [
                _identity_payload(qubit) for qubit in sorted(chains[node], key=_identity_key)
            ],
        }
        for node in sorted(chains, key=_identity_key)
    ]


def _snapshot_chains(
    chains: Mapping[Node, Iterable[Qubit]], *, label: str
) -> Mapping[Node, frozenset[Qubit]]:
    if not isinstance(chains, Mapping):
        raise ActionCertificateError(f"{label} must be a mapping")
    snapshot: dict[Node, frozenset[Qubit]] = {}
    for node, chain in chains.items():
        try:
            hash(node)
            _identity_key(node)
            frozen = frozenset(chain)
            for qubit in frozen:
                hash(qubit)
                _identity_key(qubit)
        except (TypeError, ValueError) as exc:
            raise ActionCertificateError(f"{label} contains an unhashable identity") from exc
        snapshot[node] = frozen
    return MappingProxyType(snapshot)


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ActionCertificateError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ActionCertificateError(f"{label} must be a positive exact integer")
    return value


def _validate_restart_cache_binding(
    *,
    opcode: Opcode,
    slot: int | None,
    after_digest: str | None,
    label: str,
) -> None:
    has_slot = slot is not None
    has_digest = after_digest is not None
    if has_slot != has_digest:
        raise ActionCertificateError(
            f"{label} restart-cache slot and after digest must be present together"
        )
    if opcode is not Opcode.RESTART:
        if has_slot:
            raise ActionCertificateError(
                f"{label} restart-cache evidence is valid only for RESTART"
            )
        return
    if not has_slot:
        # Legacy online restarts remain a distinct diagnostic transition. They carry no
        # authenticated persistent-cache evidence and cannot be confused with the K=2 path.
        return
    if type(slot) is not int or slot < 0:
        raise ActionCertificateError(
            f"{label} restart_cache_slot must be a nonnegative exact integer"
        )
    _require_sha256(after_digest, label=f"{label} restart_cache_after_digest")


def _work_payload(work: WorkVector) -> dict[str, int]:
    if not isinstance(work, WorkVector) or any(
        type(getattr(work, field)) is not int for field in WORK_FIELDS
    ):
        raise ActionCertificateError("candidate work must be an exact integer WorkVector")
    return work.as_dict()


def _public_task_payload(task: EmbeddingTask) -> dict[str, object]:
    """Public task identity, with all evaluator-only and initializer fields excluded."""

    if not isinstance(task, EmbeddingTask):
        raise TypeError("task must be an EmbeddingTask")
    logical_nodes = set(task.logical.nodes)
    if set(task.problem.graph.nodes) != logical_nodes:
        raise ActionCertificateError("logical graph and problem node domains differ")
    logical_edges = {_canonical_pair(left, right) for left, right in task.logical.edges}
    problem_edges = {_canonical_pair(left, right) for left, right in task.problem.graph.edges}
    if logical_edges != problem_edges:
        raise ActionCertificateError("logical graph and problem coupling supports differ")
    return {
        "base_lineage": task.lineage or task.name,
        "host_edges": sorted(
            (_pair_payload(left, right) for left, right in task.host.edges),
            key=repr,
        ),
        "host_nodes": [
            _identity_payload(node) for node in sorted(task.host.nodes, key=_identity_key)
        ],
        "instance_id": task.name,
        "logical_edges": sorted(
            (_pair_payload(left, right) for left, right in task.logical.edges),
            key=repr,
        ),
        "logical_nodes": [
            _identity_payload(node) for node in sorted(task.logical.nodes, key=_identity_key)
        ],
        "problem_h": [
            [_identity_payload(node), float(value)]
            for node, value in sorted(
                task.problem.h.items(), key=lambda item: _identity_key(item[0])
            )
        ],
        "problem_j": sorted(
            (
                [*_pair_payload(left, right), float(value)]
                for (left, right), value in task.problem.j.items()
            ),
            key=repr,
        ),
        "schema": "isingfold.public-embedding-task",
        "schema_version": PUBLIC_TASK_SCHEMA_VERSION,
    }


def public_task_fingerprint(task: EmbeddingTask) -> str:
    """Digest only deployment-visible task content.

    ``ground_energy``, planted/witness material, and ``initial_embedding`` are intentionally
    absent.  The current workspace is bound independently by the state fingerprint.
    """

    return stable_digest(_public_task_payload(task))


@dataclass(frozen=True, slots=True)
class BoundCandidateV1:
    """One complete v2 candidate row retained in its original support position.

    The Python class name is retained for source compatibility. Its serialized payload is
    exclusively v2 and therefore cannot validate a pre-cache v1 row.
    """

    index: int
    legal: bool
    opcode: Opcode
    affected: tuple[Node, ...]
    old_chains: Mapping[Node, frozenset[Qubit]]
    new_chains: Mapping[Node, frozenset[Qubit]]
    work: WorkVector
    proposal_work: WorkVector
    payload_key: str
    routes: tuple[tuple[tuple[Qubit, ...], Node], ...]
    archive_ref: int | None
    target_demand: tuple[Node, Node] | None
    target_conflict: Qubit | None
    restart_cache_slot: int | None
    restart_cache_after_digest: str | None
    proposal_provenance: str

    def __post_init__(self) -> None:
        _validate_restart_cache_binding(
            opcode=self.opcode,
            slot=self.restart_cache_slot,
            after_digest=self.restart_cache_after_digest,
            label="bound candidate",
        )

    @classmethod
    def capture(cls, *, index: int, candidate: Candidate, legal: bool) -> BoundCandidateV1:
        if type(index) is not int or index < 0:
            raise ActionCertificateError("candidate index must be a nonnegative exact integer")
        if not isinstance(candidate, Candidate):
            raise TypeError("candidate support must contain Candidate values")
        if type(legal) is not bool:
            raise ActionCertificateError("every legal-mask bit must be an exact Boolean")
        if not isinstance(candidate.opcode, Opcode):
            raise ActionCertificateError("candidate opcode must be a registered Opcode")
        if type(candidate.payload_key) is not str or not candidate.payload_key:
            raise ActionCertificateError("candidate payload_key must be nonempty")
        if type(candidate.provenance) is not str:
            raise ActionCertificateError("candidate proposal provenance must be a string")
        return cls(
            index=index,
            legal=legal,
            opcode=candidate.opcode,
            affected=tuple(candidate.affected),
            old_chains=_snapshot_chains(candidate.old_chains, label="candidate old_chains"),
            new_chains=_snapshot_chains(candidate.new_chains, label="candidate new_chains"),
            work=candidate.work,
            proposal_work=candidate.proposal_work,
            payload_key=candidate.payload_key,
            routes=tuple((tuple(path), owner) for path, owner in candidate.routes),
            archive_ref=candidate.archive_ref,
            target_demand=(
                None if candidate.target_demand is None else tuple(candidate.target_demand)
            ),
            target_conflict=candidate.target_conflict,
            restart_cache_slot=candidate.restart_cache_slot,
            restart_cache_after_digest=candidate.restart_cache_after_digest,
            proposal_provenance=candidate.provenance,
        )

    @property
    def changes_workspace(self) -> bool:
        return self.opcode not in (Opcode.COMMIT, Opcode.STOP)

    def payload_dict(self) -> dict[str, object]:
        return {
            "affected": [_identity_payload(node) for node in self.affected],
            "archive_ref": self.archive_ref,
            "new_chains": _chain_rows(self.new_chains),
            "old_chains": _chain_rows(self.old_chains),
            "opcode": self.opcode.value,
            "payload_key": self.payload_key,
            "proposal_provenance": self.proposal_provenance,
            "proposal_work": _work_payload(self.proposal_work),
            "restart_cache_after_digest": self.restart_cache_after_digest,
            "restart_cache_slot": self.restart_cache_slot,
            "routes": [
                {
                    "owner": _identity_payload(owner),
                    "path": [_identity_payload(qubit) for qubit in path],
                }
                for path, owner in self.routes
            ],
            "schema": BOUND_CANDIDATE_SCHEMA,
            "schema_version": BOUND_CANDIDATE_SCHEMA_VERSION,
            "target_conflict": (
                None if self.target_conflict is None else _identity_payload(self.target_conflict)
            ),
            "target_demand": (
                None
                if self.target_demand is None
                else [_identity_payload(node) for node in self.target_demand]
            ),
            "work": _work_payload(self.work),
        }

    @property
    def payload_digest(self) -> str:
        return stable_digest(self.payload_dict())

    def as_dict(self) -> dict[str, object]:
        payload = self.payload_dict()
        return {
            "index": self.index,
            "legal": self.legal,
            "payload": payload,
            "payload_digest": stable_digest(payload),
        }


@dataclass(frozen=True, slots=True)
class StateActionEnvelopeV1:
    """Authenticated v2 public snapshot of one real environment decision epoch."""

    task_id: str
    base_lineage: str
    task_fingerprint: str
    provenance_fingerprint: str
    state_fingerprint: str
    support_fingerprint: str
    context_version: str
    mode: str
    state_chains: Mapping[Node, frozenset[Qubit]]
    charged_work_receipt: WorkVector
    candidates: tuple[BoundCandidateV1, ...]
    record_digest: str

    @classmethod
    def capture(
        cls,
        *,
        env: EmbeddingEnv,
        decision: DecisionState,
        provenance_fingerprint: str,
    ) -> StateActionEnvelopeV1:
        """Capture a decision only while it is the live prepared support of ``env``."""

        if not isinstance(env, EmbeddingEnv):
            raise TypeError("env must be an EmbeddingEnv")
        if not isinstance(decision, DecisionState):
            raise TypeError("decision must be a DecisionState")
        provenance_fingerprint = _require_sha256(
            provenance_fingerprint, label="provenance_fingerprint"
        )
        if len(decision.candidates) != len(decision.legal_mask):
            raise ActionCertificateError("candidate support and legal mask have different lengths")
        if any(type(bit) is not bool for bit in decision.legal_mask):
            raise ActionCertificateError("every legal-mask bit must be an exact Boolean")
        if not decision.candidates:
            raise ActionCertificateError("a structural envelope cannot bind an empty support")

        try:
            env._assert_search_state()
            current_state_fingerprint = env._state_fingerprint()
        except Exception as exc:
            raise ActionCertificateError(
                "environment search state failed exact validation"
            ) from exc
        supplied_support = candidate_support_key(decision.candidates, decision.legal_mask)
        if (
            decision.state_fingerprint != current_state_fingerprint
            or decision.state_fingerprint != env._prepared_state_fingerprint
        ):
            raise ActionCertificateError("decision is not the environment's live prepared state")
        if (
            decision.support_fingerprint != supplied_support
            or decision.support_fingerprint != env._prepared_support_fingerprint
        ):
            raise ActionCertificateError("decision support is not bound by the environment")
        if decision.context_version != env.ctx.context_version:
            raise ActionCertificateError("decision and environment context versions differ")
        if not isinstance(decision.exact_state, EnvState) or env.state is None:
            raise ActionCertificateError("decision has no exact environment-state snapshot")
        if decision.exact_state.chains != env.state.chains:
            raise ActionCertificateError("decision exact chains differ from the live environment")
        if decision.random_state != env.rng.getstate():
            raise ActionCertificateError("decision random tape differs from the live environment")
        _require_sha256(decision.state_fingerprint, label="state_fingerprint")
        _require_sha256(decision.support_fingerprint, label="support_fingerprint")

        rows = tuple(
            BoundCandidateV1.capture(index=index, candidate=candidate, legal=legal)
            for index, (candidate, legal) in enumerate(
                zip(decision.candidates, decision.legal_mask, strict=True)
            )
        )
        values: dict[str, Any] = {
            "task_id": env.task.name,
            "base_lineage": env.task.lineage or env.task.name,
            "task_fingerprint": public_task_fingerprint(env.task),
            "provenance_fingerprint": provenance_fingerprint,
            "state_fingerprint": decision.state_fingerprint,
            "support_fingerprint": decision.support_fingerprint,
            "context_version": decision.context_version,
            "mode": env.mode.value,
            "state_chains": _snapshot_chains(env.state.chains, label="environment chains"),
            "charged_work_receipt": decision.charged_work_receipt,
            "candidates": rows,
        }
        unsigned = cls._payload(**values)
        return cls(**values, record_digest=stable_digest(unsigned))

    @staticmethod
    def _payload(
        *,
        task_id: str,
        base_lineage: str,
        task_fingerprint: str,
        provenance_fingerprint: str,
        state_fingerprint: str,
        support_fingerprint: str,
        context_version: str,
        mode: str,
        state_chains: Mapping[Node, frozenset[Qubit]],
        charged_work_receipt: WorkVector,
        candidates: Sequence[BoundCandidateV1],
    ) -> dict[str, object]:
        return {
            "base_lineage": base_lineage,
            "candidates": [candidate.as_dict() for candidate in candidates],
            "charged_work_receipt": _work_payload(charged_work_receipt),
            "context_version": context_version,
            "mode": mode,
            "provenance_fingerprint": provenance_fingerprint,
            "schema": ENVELOPE_SCHEMA,
            "schema_version": ENVELOPE_SCHEMA_VERSION,
            "state_chains": _chain_rows(state_chains),
            "state_fingerprint": state_fingerprint,
            "support_fingerprint": support_fingerprint,
            "task_fingerprint": task_fingerprint,
            "task_id": task_id,
        }

    def unsigned_dict(self) -> dict[str, object]:
        return self._payload(
            task_id=self.task_id,
            base_lineage=self.base_lineage,
            task_fingerprint=self.task_fingerprint,
            provenance_fingerprint=self.provenance_fingerprint,
            state_fingerprint=self.state_fingerprint,
            support_fingerprint=self.support_fingerprint,
            context_version=self.context_version,
            mode=self.mode,
            state_chains=self.state_chains,
            charged_work_receipt=self.charged_work_receipt,
            candidates=self.candidates,
        )

    def verify_digest(self) -> None:
        _require_sha256(self.record_digest, label="envelope record_digest")
        if stable_digest(self.unsigned_dict()) != self.record_digest:
            raise ActionCertificateError("state/action envelope digest mismatch")

    def as_dict(self) -> dict[str, object]:
        self.verify_digest()
        return {**self.unsigned_dict(), "record_digest": self.record_digest}


@dataclass(frozen=True, slots=True)
class AppliedStateActionV1:
    """Atomic v2 structural successor with the selected replacement retained as cores."""

    task_fingerprint: str
    provenance_fingerprint: str
    envelope_record_digest: str
    state_fingerprint: str
    support_fingerprint: str
    selected_index: int
    selected_opcode: Opcode
    selected_payload_key: str
    selected_payload_digest: str
    selected_restart_cache_slot: int | None
    selected_restart_cache_after_digest: str | None
    chains_before: Mapping[Node, frozenset[Qubit]]
    chains_after: Mapping[Node, frozenset[Qubit]]
    persistent_cores: Mapping[Node, frozenset[Qubit]]
    successor_fingerprint: str
    record_digest: str

    def __post_init__(self) -> None:
        _validate_restart_cache_binding(
            opcode=self.selected_opcode,
            slot=self.selected_restart_cache_slot,
            after_digest=self.selected_restart_cache_after_digest,
            label="applied action",
        )

    @staticmethod
    def _payload(
        *,
        task_fingerprint: str,
        provenance_fingerprint: str,
        envelope_record_digest: str,
        state_fingerprint: str,
        support_fingerprint: str,
        selected_index: int,
        selected_opcode: Opcode,
        selected_payload_key: str,
        selected_payload_digest: str,
        selected_restart_cache_slot: int | None,
        selected_restart_cache_after_digest: str | None,
        chains_before: Mapping[Node, frozenset[Qubit]],
        chains_after: Mapping[Node, frozenset[Qubit]],
        persistent_cores: Mapping[Node, frozenset[Qubit]],
        successor_fingerprint: str,
    ) -> dict[str, object]:
        return {
            "chains_after": _chain_rows(chains_after),
            "chains_before": _chain_rows(chains_before),
            "envelope_record_digest": envelope_record_digest,
            "persistence_semantics": _PERSISTENCE_SEMANTICS,
            "persistent_cores": _chain_rows(persistent_cores),
            "provenance_fingerprint": provenance_fingerprint,
            "schema": APPLIED_ACTION_SCHEMA,
            "schema_version": APPLIED_ACTION_SCHEMA_VERSION,
            "selected_index": selected_index,
            "selected_opcode": selected_opcode.value,
            "selected_payload_digest": selected_payload_digest,
            "selected_payload_key": selected_payload_key,
            "selected_restart_cache_after_digest": selected_restart_cache_after_digest,
            "selected_restart_cache_slot": selected_restart_cache_slot,
            "state_fingerprint": state_fingerprint,
            "successor_fingerprint": successor_fingerprint,
            "support_fingerprint": support_fingerprint,
            "task_fingerprint": task_fingerprint,
        }

    def unsigned_dict(self) -> dict[str, object]:
        return self._payload(
            task_fingerprint=self.task_fingerprint,
            provenance_fingerprint=self.provenance_fingerprint,
            envelope_record_digest=self.envelope_record_digest,
            state_fingerprint=self.state_fingerprint,
            support_fingerprint=self.support_fingerprint,
            selected_index=self.selected_index,
            selected_opcode=self.selected_opcode,
            selected_payload_key=self.selected_payload_key,
            selected_payload_digest=self.selected_payload_digest,
            selected_restart_cache_slot=self.selected_restart_cache_slot,
            selected_restart_cache_after_digest=self.selected_restart_cache_after_digest,
            chains_before=self.chains_before,
            chains_after=self.chains_after,
            persistent_cores=self.persistent_cores,
            successor_fingerprint=self.successor_fingerprint,
        )

    def verify_digest(self) -> None:
        _require_sha256(self.record_digest, label="applied-action record_digest")
        if stable_digest(self.unsigned_dict()) != self.record_digest:
            raise ActionCertificateError("applied-action digest mismatch")

    def verify_against(self, envelope: StateActionEnvelopeV1) -> None:
        """Replay the selected support row and require this exact successor record."""

        if not isinstance(envelope, StateActionEnvelopeV1):
            raise TypeError("envelope must be a StateActionEnvelopeV1")
        self.verify_digest()
        expected = apply_envelope_action(envelope, self.selected_index)
        if (
            self.record_digest != expected.record_digest
            or self.unsigned_dict() != expected.unsigned_dict()
        ):
            raise ActionCertificateError(
                "applied action does not exactly replay from its bound envelope"
            )

    def as_dict(self) -> dict[str, object]:
        self.verify_digest()
        return {**self.unsigned_dict(), "record_digest": self.record_digest}


def apply_envelope_action(
    envelope: StateActionEnvelopeV1, selected_index: int
) -> AppliedStateActionV1:
    """Apply one bound legal workspace action atomically, without preparing a new support."""

    if not isinstance(envelope, StateActionEnvelopeV1):
        raise TypeError("envelope must be a StateActionEnvelopeV1")
    envelope.verify_digest()
    if type(selected_index) is not int or not 0 <= selected_index < len(envelope.candidates):
        raise ActionCertificateError("selected index leaves the materialised support")
    selected = envelope.candidates[selected_index]
    if selected.index != selected_index:
        raise ActionCertificateError("candidate support positions are not canonical")
    if not selected.legal:
        raise ActionCertificateError("a masked action cannot enter a structural certificate")
    if not selected.changes_workspace:
        raise UnsupportedActionCertificateError(
            f"{selected.opcode.value} has no structural continuation in schema v1"
        )

    affected = set(selected.affected)
    if (
        not affected
        or len(affected) != len(selected.affected)
        or affected != set(selected.old_chains)
        or affected != set(selected.new_chains)
        or not affected <= set(envelope.state_chains)
    ):
        raise ActionCertificateError("selected candidate has a malformed affected domain")
    if any(envelope.state_chains[node] != selected.old_chains[node] for node in affected):
        raise ActionCertificateError("selected candidate old_chains differ from the bound state")

    successor = dict(envelope.state_chains)
    successor.update({node: frozenset(selected.new_chains[node]) for node in selected.affected})
    expected_payload_key = bound_successor_key(
        successor,
        selected.work,
        restart=selected.opcode is Opcode.RESTART,
        restart_cache_after_digest=selected.restart_cache_after_digest,
    )
    if selected.payload_key != expected_payload_key:
        raise ActionCertificateError("selected candidate payload_key does not bind its successor")

    before = _snapshot_chains(envelope.state_chains, label="pre-action chains")
    after = _snapshot_chains(successor, label="post-action chains")
    persistent = _snapshot_chains(
        {node: selected.new_chains[node] for node in selected.affected},
        label="selected persistent cores",
    )
    successor_fingerprint = stable_digest(
        {
            "chains_after": _chain_rows(after),
            "domain": "isingfold.structural-successor",
            "selected_payload_digest": selected.payload_digest,
            "state_fingerprint": envelope.state_fingerprint,
            "support_fingerprint": envelope.support_fingerprint,
            "schema_version": APPLIED_ACTION_SCHEMA_VERSION,
        }
    )
    values: dict[str, Any] = {
        "task_fingerprint": envelope.task_fingerprint,
        "provenance_fingerprint": envelope.provenance_fingerprint,
        "envelope_record_digest": envelope.record_digest,
        "state_fingerprint": envelope.state_fingerprint,
        "support_fingerprint": envelope.support_fingerprint,
        "selected_index": selected_index,
        "selected_opcode": selected.opcode,
        "selected_payload_key": selected.payload_key,
        "selected_payload_digest": selected.payload_digest,
        "selected_restart_cache_slot": selected.restart_cache_slot,
        "selected_restart_cache_after_digest": selected.restart_cache_after_digest,
        "chains_before": before,
        "chains_after": after,
        "persistent_cores": persistent,
        "successor_fingerprint": successor_fingerprint,
    }
    unsigned = AppliedStateActionV1._payload(**values)
    return AppliedStateActionV1(**values, record_digest=stable_digest(unsigned))


@dataclass(frozen=True, slots=True)
class PersistentContinuationDomainV1:
    """Finite deterministic completion domain anchored to one selected action."""

    task_fingerprint: str
    applied_action_record_digest: str
    successor_fingerprint: str
    movable: tuple[Node, ...]
    window: frozenset[Qubit]
    max_chain: int
    qubit_cap: int
    max_nodes: int
    record_digest: str

    def __post_init__(self) -> None:
        _require_sha256(self.task_fingerprint, label="continuation task_fingerprint")
        _require_sha256(
            self.applied_action_record_digest,
            label="continuation applied_action_record_digest",
        )
        _require_sha256(self.successor_fingerprint, label="continuation successor_fingerprint")
        _require_positive_int(self.max_chain, label="max_chain")
        _require_positive_int(self.qubit_cap, label="qubit_cap")
        _require_positive_int(self.max_nodes, label="max_nodes")
        if len(self.movable) != len(set(self.movable)):
            raise ActionCertificateError("movable variables must be unique")

    @classmethod
    def build(
        cls,
        *,
        task: EmbeddingTask,
        envelope: StateActionEnvelopeV1,
        applied: AppliedStateActionV1,
        movable: Iterable[Node],
        window: frozenset[Qubit],
        max_chain: int,
        qubit_cap: int,
        max_nodes: int,
    ) -> PersistentContinuationDomainV1:
        if not isinstance(applied, AppliedStateActionV1):
            raise TypeError("applied must be an AppliedStateActionV1")
        applied.verify_against(envelope)
        fingerprint = public_task_fingerprint(task)
        if fingerprint != applied.task_fingerprint:
            raise ActionCertificateError("task differs from the applied action's public task")
        try:
            movable_tuple = tuple(movable)
            window_frozen = frozenset(window)
        except TypeError as exc:
            raise ActionCertificateError(
                "movable and window must contain hashable identities"
            ) from exc
        if len(movable_tuple) != len(set(movable_tuple)):
            raise ActionCertificateError("movable variables must be unique")
        logical_nodes = set(task.logical.nodes)
        if set(applied.chains_after) != logical_nodes:
            raise ActionCertificateError("post-action chains do not cover the logical domain")
        if not set(applied.persistent_cores) <= logical_nodes:
            raise ActionCertificateError("selected persistent cores leave the logical domain")
        if not set(movable_tuple) <= logical_nodes:
            raise ActionCertificateError("movable variables leave the logical domain")
        if any(not applied.chains_after[node] for node in logical_nodes - set(movable_tuple)):
            raise ActionCertificateError(
                "an empty chain cannot be frozen outside the movable domain"
            )
        if not window_frozen <= set(task.host.nodes):
            raise ActionCertificateError("continuation window leaves the active host")
        max_chain = _require_positive_int(max_chain, label="max_chain")
        qubit_cap = _require_positive_int(qubit_cap, label="qubit_cap")
        max_nodes = _require_positive_int(max_nodes, label="max_nodes")
        canonical_movable = tuple(sorted(movable_tuple, key=_identity_key))
        values: dict[str, Any] = {
            "task_fingerprint": fingerprint,
            "applied_action_record_digest": applied.record_digest,
            "successor_fingerprint": applied.successor_fingerprint,
            "movable": canonical_movable,
            "window": window_frozen,
            "max_chain": max_chain,
            "qubit_cap": qubit_cap,
            "max_nodes": max_nodes,
        }
        unsigned = cls._payload(**values)
        return cls(**values, record_digest=stable_digest(unsigned))

    @staticmethod
    def _payload(
        *,
        task_fingerprint: str,
        applied_action_record_digest: str,
        successor_fingerprint: str,
        movable: Sequence[Node],
        window: frozenset[Qubit],
        max_chain: int,
        qubit_cap: int,
        max_nodes: int,
    ) -> dict[str, object]:
        return {
            "applied_action_record_digest": applied_action_record_digest,
            "max_chain": max_chain,
            "max_nodes": max_nodes,
            "movable": [_identity_payload(node) for node in movable],
            "overlap_repair_support": _OVERLAP_SUPPORT,
            "persistence_semantics": _PERSISTENCE_SEMANTICS,
            "qubit_cap": qubit_cap,
            "schema": CONTINUATION_SCHEMA,
            "schema_version": CONTINUATION_SCHEMA_VERSION,
            "successor_fingerprint": successor_fingerprint,
            "task_fingerprint": task_fingerprint,
            "terminal_semantics": _TERMINAL_SEMANTICS,
            "window": [_identity_payload(qubit) for qubit in sorted(window, key=_identity_key)],
        }

    def unsigned_dict(self) -> dict[str, object]:
        return self._payload(
            task_fingerprint=self.task_fingerprint,
            applied_action_record_digest=self.applied_action_record_digest,
            successor_fingerprint=self.successor_fingerprint,
            movable=self.movable,
            window=self.window,
            max_chain=self.max_chain,
            qubit_cap=self.qubit_cap,
            max_nodes=self.max_nodes,
        )

    def verify_digest(self) -> None:
        _require_sha256(self.record_digest, label="continuation-domain record_digest")
        if stable_digest(self.unsigned_dict()) != self.record_digest:
            raise ActionCertificateError("continuation-domain digest mismatch")

    def as_dict(self) -> dict[str, object]:
        self.verify_digest()
        return {**self.unsigned_dict(), "record_digest": self.record_digest}


@dataclass(frozen=True, slots=True)
class StructuralActionCertificateV1:
    """Boolean proof, unknown receipt, or explicit unsupported-scope receipt."""

    task_fingerprint: str
    provenance_fingerprint: str
    applied_action_record_digest: str
    continuation_domain_record_digest: str
    status: CertificateStatus
    feasible: bool | None
    exhausted: bool
    nodes_explored: int
    witness: Mapping[Node, frozenset[Qubit]] | None
    reason: str
    record_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, CertificateStatus):
            raise ActionCertificateError("certificate status is not registered")
        if type(self.exhausted) is not bool:
            raise ActionCertificateError("certificate exhausted flag must be Boolean")
        if type(self.nodes_explored) is not int or self.nodes_explored < 0:
            raise ActionCertificateError("nodes_explored must be a nonnegative exact integer")
        if type(self.reason) is not str or not self.reason:
            raise ActionCertificateError("certificate reason must be nonempty")
        if self.status is CertificateStatus.CERTIFIED_FEASIBLE:
            if self.feasible is not True or self.witness is None:
                raise ActionCertificateError("a feasible certificate requires a concrete witness")
        elif self.status is CertificateStatus.CERTIFIED_INFEASIBLE:
            if self.feasible is not False or not self.exhausted or self.witness is not None:
                raise ActionCertificateError(
                    "an infeasible certificate requires exhausted enumeration"
                )
        elif self.feasible is not None or self.exhausted or self.witness is not None:
            raise ActionCertificateError(
                "unknown/unsupported receipts cannot carry a Boolean label"
            )

    @property
    def is_exact(self) -> bool:
        return self.status in {
            CertificateStatus.CERTIFIED_FEASIBLE,
            CertificateStatus.CERTIFIED_INFEASIBLE,
        }

    @staticmethod
    def _payload(
        *,
        task_fingerprint: str,
        provenance_fingerprint: str,
        applied_action_record_digest: str,
        continuation_domain_record_digest: str,
        status: CertificateStatus,
        feasible: bool | None,
        exhausted: bool,
        nodes_explored: int,
        witness: Mapping[Node, frozenset[Qubit]] | None,
        reason: str,
    ) -> dict[str, object]:
        return {
            "applied_action_record_digest": applied_action_record_digest,
            "continuation_domain_record_digest": continuation_domain_record_digest,
            "exhausted": exhausted,
            "feasible": feasible,
            "nodes_explored": nodes_explored,
            "provenance_fingerprint": provenance_fingerprint,
            "reason": reason,
            "schema": CERTIFICATE_SCHEMA,
            "schema_version": CERTIFICATE_SCHEMA_VERSION,
            "status": status.value,
            "task_fingerprint": task_fingerprint,
            "witness": None if witness is None else _chain_rows(witness),
        }

    def unsigned_dict(self) -> dict[str, object]:
        return self._payload(
            task_fingerprint=self.task_fingerprint,
            provenance_fingerprint=self.provenance_fingerprint,
            applied_action_record_digest=self.applied_action_record_digest,
            continuation_domain_record_digest=self.continuation_domain_record_digest,
            status=self.status,
            feasible=self.feasible,
            exhausted=self.exhausted,
            nodes_explored=self.nodes_explored,
            witness=self.witness,
            reason=self.reason,
        )

    def verify_digest(self) -> None:
        _require_sha256(self.record_digest, label="structural certificate record_digest")
        if stable_digest(self.unsigned_dict()) != self.record_digest:
            raise ActionCertificateError("structural action certificate digest mismatch")

    def as_dict(self) -> dict[str, object]:
        self.verify_digest()
        return {**self.unsigned_dict(), "record_digest": self.record_digest}


def _certificate(
    *,
    applied: AppliedStateActionV1,
    domain: PersistentContinuationDomainV1,
    status: CertificateStatus,
    feasible: bool | None,
    exhausted: bool,
    nodes_explored: int,
    witness: Mapping[Node, frozenset[Qubit]] | None,
    reason: str,
) -> StructuralActionCertificateV1:
    if status is CertificateStatus.CERTIFIED_FEASIBLE:
        if feasible is not True or witness is None:
            raise ActionCertificateError("a feasible certificate requires a concrete witness")
    elif status is CertificateStatus.CERTIFIED_INFEASIBLE:
        if feasible is not False or not exhausted or witness is not None:
            raise ActionCertificateError("an infeasible certificate requires exhausted enumeration")
    elif feasible is not None or exhausted or witness is not None:
        raise ActionCertificateError("unknown/unsupported receipts cannot carry a Boolean label")
    if type(nodes_explored) is not int or nodes_explored < 0:
        raise ActionCertificateError("nodes_explored must be a nonnegative exact integer")
    values: dict[str, Any] = {
        "task_fingerprint": applied.task_fingerprint,
        "provenance_fingerprint": applied.provenance_fingerprint,
        "applied_action_record_digest": applied.record_digest,
        "continuation_domain_record_digest": domain.record_digest,
        "status": status,
        "feasible": feasible,
        "exhausted": exhausted,
        "nodes_explored": nodes_explored,
        "witness": None if witness is None else _snapshot_chains(witness, label="witness"),
        "reason": reason,
    }
    unsigned = StructuralActionCertificateV1._payload(**values)
    return StructuralActionCertificateV1(**values, record_digest=stable_digest(unsigned))


class _NodeLimitReached(RuntimeError):
    pass


@dataclass(slots=True)
class _NodeBudget:
    limit: int
    explored: int = 0

    def enter(self) -> None:
        if self.explored >= self.limit:
            raise _NodeLimitReached
        self.explored += 1


def _connected_supersets(
    *,
    host: nx.Graph,
    core: frozenset[Qubit],
    available_extras: frozenset[Qubit],
    max_chain: int,
) -> Iterable[frozenset[Qubit]]:
    minimum_extras = 0 if core else 1
    maximum_extras = max_chain - len(core)
    if maximum_extras < minimum_extras:
        return
    ordered = tuple(sorted(available_extras, key=_identity_key))
    for count in range(minimum_extras, maximum_extras + 1):
        for additions in itertools.combinations(ordered, count):
            candidate = core | frozenset(additions)
            if candidate and (len(candidate) == 1 or nx.is_connected(host.subgraph(candidate))):
                yield candidate


def _has_contact(
    host: nx.Graph,
    left: frozenset[Qubit],
    right: frozenset[Qubit],
) -> bool:
    return any(q != r and host.has_edge(q, r) for q in left for r in right)


def certify_persistent_completion(
    *,
    task: EmbeddingTask,
    envelope: StateActionEnvelopeV1,
    applied: AppliedStateActionV1,
    domain: PersistentContinuationDomainV1,
) -> StructuralActionCertificateV1:
    """Decide the registered finite domain, or return a truthful non-label receipt.

    A feasible result is exact as soon as its concrete witness passes ``p_embed``.  A false
    result is emitted only when the deterministic search domain is completely exhausted.
    If entering another DFS state would exceed ``max_nodes``, the result is unknown.
    """

    if not isinstance(applied, AppliedStateActionV1):
        raise TypeError("applied must be an AppliedStateActionV1")
    if not isinstance(domain, PersistentContinuationDomainV1):
        raise TypeError("domain must be a PersistentContinuationDomainV1")
    applied.verify_against(envelope)
    domain.verify_digest()
    task_fingerprint = public_task_fingerprint(task)
    if (
        task_fingerprint != applied.task_fingerprint
        or task_fingerprint != domain.task_fingerprint
        or domain.applied_action_record_digest != applied.record_digest
        or domain.successor_fingerprint != applied.successor_fingerprint
    ):
        raise ActionCertificateError("task, applied action, and continuation identities differ")

    logical_nodes = set(task.logical.nodes)
    if set(applied.chains_after) != logical_nodes:
        raise ActionCertificateError("post-action chains do not cover the logical domain")
    if nx.number_of_selfloops(task.logical):
        raise ActionCertificateError("schema v1 does not define logical self-loop realization")

    post = {node: frozenset(chain) for node, chain in applied.chains_after.items()}
    host_nodes = set(task.host.nodes)
    for node, chain in post.items():
        if not set(chain) <= host_nodes:
            raise ActionCertificateError(f"post-action chain {node!r} leaves the active host")
        if len(chain) > 1 and not nx.is_connected(task.host.subgraph(chain)):
            raise ActionCertificateError(f"post-action chain {node!r} is disconnected")
    if any(not applied.persistent_cores[node] <= post[node] for node in applied.persistent_cores):
        raise ActionCertificateError("selected candidate cores were not retained post-action")

    if any(count > 1 for count in occupancy(post).values()):
        return _certificate(
            applied=applied,
            domain=domain,
            status=CertificateStatus.UNSUPPORTED_OVERLAP_REPAIR,
            feasible=None,
            exhausted=False,
            nodes_explored=0,
            witness=None,
            reason=(
                "schema v1 cannot certify a continuation that must later move overlapping "
                "claims; no Boolean label was produced"
            ),
        )

    movable = set(domain.movable)
    frozen = {node: post[node] for node in logical_nodes - movable}
    if any(not chain for chain in frozen.values()):
        raise ActionCertificateError("continuation domain freezes an empty exterior chain")
    all_cores = {node: post[node] for node in movable}
    persistent_qubits = frozenset(qubit for chain in post.values() for qubit in chain)

    if any(len(chain) > domain.max_chain for chain in post.values()):
        return _certificate(
            applied=applied,
            domain=domain,
            status=CertificateStatus.CERTIFIED_INFEASIBLE,
            feasible=False,
            exhausted=True,
            nodes_explored=0,
            witness=None,
            reason="the persistent post-action core exceeds the registered chain cap",
        )
    minimum_qubits = len(persistent_qubits) + sum(not all_cores[node] for node in movable)
    if minimum_qubits > domain.qubit_cap:
        return _certificate(
            applied=applied,
            domain=domain,
            status=CertificateStatus.CERTIFIED_INFEASIBLE,
            feasible=False,
            exhausted=True,
            nodes_explored=0,
            witness=None,
            reason="the exact persistent-core lower bound exceeds the qubit cap",
        )
    for left, right in task.logical.edges:
        if (
            left in frozen
            and right in frozen
            and not _has_contact(task.host, frozen[left], frozen[right])
        ):
            return _certificate(
                applied=applied,
                domain=domain,
                status=CertificateStatus.CERTIFIED_INFEASIBLE,
                feasible=False,
                exhausted=True,
                nodes_explored=0,
                witness=None,
                reason="a frozen-exterior logical demand has no physical contact",
            )

    order = tuple(
        sorted(
            domain.movable,
            key=lambda node: (
                0 if all_cores[node] else 1,
                -task.logical.degree[node],
                _identity_key(node),
            ),
        )
    )
    extras = frozenset(domain.window - persistent_qubits)
    assignment: dict[Node, frozenset[Qubit]] = dict(frozen)
    budget = _NodeBudget(domain.max_nodes)

    def recurse(
        index: int,
        used_extras: frozenset[Qubit],
    ) -> dict[Node, frozenset[Qubit]] | None:
        budget.enter()
        remaining = order[index:]
        lower_bound = (
            len(persistent_qubits)
            + len(used_extras)
            + sum(not all_cores[node] for node in remaining)
        )
        if lower_bound > domain.qubit_cap:
            return None
        if index == len(order):
            receipt = p_embed(assignment, task.logical, task.host, domain.qubit_cap)
            return dict(assignment) if receipt.valid else None

        node = order[index]
        core = all_cores[node]
        available = extras - used_extras
        for chain in _connected_supersets(
            host=task.host,
            core=core,
            available_extras=available,
            max_chain=domain.max_chain,
        ):
            additions = chain - core
            if len(persistent_qubits) + len(used_extras | additions) > domain.qubit_cap:
                continue
            if any(
                neighbour in assignment
                and not _has_contact(task.host, chain, assignment[neighbour])
                for neighbour in task.logical.neighbors(node)
            ):
                continue
            assignment[node] = chain
            witness = recurse(index + 1, used_extras | additions)
            del assignment[node]
            if witness is not None:
                return witness
        return None

    try:
        witness = recurse(0, frozenset())
    except _NodeLimitReached:
        return _certificate(
            applied=applied,
            domain=domain,
            status=CertificateStatus.UNKNOWN_NODE_LIMIT,
            feasible=None,
            exhausted=False,
            nodes_explored=budget.explored,
            witness=None,
            reason="deterministic max_nodes reached before the continuation domain was exhausted",
        )

    if witness is None:
        return _certificate(
            applied=applied,
            domain=domain,
            status=CertificateStatus.CERTIFIED_INFEASIBLE,
            feasible=False,
            exhausted=True,
            nodes_explored=budget.explored,
            witness=None,
            reason="deterministic continuation domain exhausted without a valid completion",
        )
    for node, core in post.items():
        if core and not core <= witness[node]:
            raise RuntimeError("exact search violated a persistent post-action core")
    receipt = p_embed(witness, task.logical, task.host, domain.qubit_cap)
    if not receipt.valid:
        raise RuntimeError("exact search returned a witness rejected by p_embed")
    return _certificate(
        applied=applied,
        domain=domain,
        status=CertificateStatus.CERTIFIED_FEASIBLE,
        feasible=True,
        exhausted=False,
        nodes_explored=budget.explored,
        witness=witness,
        reason="concrete persistent-core completion independently passes p_embed",
    )


__all__ = [
    "APPLIED_ACTION_SCHEMA",
    "APPLIED_ACTION_SCHEMA_VERSION",
    "BOUND_CANDIDATE_SCHEMA",
    "BOUND_CANDIDATE_SCHEMA_VERSION",
    "CERTIFICATE_SCHEMA",
    "CERTIFICATE_SCHEMA_VERSION",
    "CONTINUATION_SCHEMA",
    "CONTINUATION_SCHEMA_VERSION",
    "ENVELOPE_SCHEMA",
    "ENVELOPE_SCHEMA_VERSION",
    "PUBLIC_TASK_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "ActionCertificateError",
    "AppliedStateActionV1",
    "BoundCandidateV1",
    "CertificateStatus",
    "PersistentContinuationDomainV1",
    "StateActionEnvelopeV1",
    "StructuralActionCertificateV1",
    "UnsupportedActionCertificateError",
    "apply_envelope_action",
    "certify_persistent_completion",
    "public_task_fingerprint",
]
