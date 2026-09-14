"""Tensorization boundary for the frozen terminal strength selector.

This module deliberately accepts only public problem, hardware, embedding and compiled
program objects.  It has no parameter for a reference energy, planted witness, evaluator
receipt, success count or random key.  The search tensorizer is reused for the common
Appendix A feature definitions, after which every search-only value and knownness bit is
removed explicitly.
"""

from __future__ import annotations

import math
from typing import Hashable, Mapping, Sequence

import networkx as nx
import numpy as np

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import Context
from isingfold.rl.program import (
    Program,
    check_faithfulness,
    instance_scale,
    strength_registry,
)
from isingfold.rl.strength import (
    ALLOWED_GLOBAL_SLOTS,
    MASKED_CLAIM_SLOTS,
    MASKED_HARDWARE_SLOTS,
    MASKED_LOGICAL_EDGE_SLOTS,
    MASKED_LOGICAL_SLOTS,
    N_CLAIM,
    N_GLOBAL,
    N_HARDWARE,
    N_LOGICAL,
    N_LOGICAL_EDGE,
    N_STRENGTHS,
    StrengthGraphInput,
)
from isingfold.rl.tensorize import Ages, build_observation

Node = Hashable
Qubit = Hashable


def _edge_set(graph: nx.Graph) -> set[frozenset[Hashable]]:
    return {frozenset((left, right)) for left, right in graph.edges()}


def _mask_packed(
    array: np.ndarray,
    scalar_count: int,
    masked_slots: Sequence[int],
) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32).copy()
    columns = tuple(masked_slots) + tuple(scalar_count + slot for slot in masked_slots)
    if columns:
        result[..., list(columns)] = 0.0
    return result


def _validate_embedding(
    logical: nx.Graph,
    host: nx.Graph,
    problem: LogicalProblem,
    chains: Mapping[Node, frozenset[Qubit]],
    qubit_cap: int,
) -> None:
    if set(logical.nodes()) != set(problem.graph.nodes()):
        raise ValueError("logical graph and Ising problem node supports differ")
    if _edge_set(logical) != _edge_set(problem.graph):
        raise ValueError("logical graph and Ising problem edge supports differ")
    if set(chains) != set(logical.nodes()):
        raise ValueError("embedding does not cover the logical node support exactly")
    owner: dict[Qubit, Node] = {}
    for node, chain in chains.items():
        if not chain:
            raise ValueError(f"selector input chain {node!r} is empty")
        if not set(chain) <= set(host.nodes()):
            raise ValueError(f"selector input chain {node!r} leaves active hardware")
        if len(chain) > 1 and not nx.is_connected(host.subgraph(chain)):
            raise ValueError(f"selector input chain {node!r} is disconnected")
        for qubit in chain:
            if qubit in owner:
                raise ValueError("selector input embedding contains overlapping chains")
            owner[qubit] = node
    if len(owner) > qubit_cap:
        raise ValueError("selector input embedding exceeds the registered qubit cap")
    for left, right in logical.edges():
        if not any(
            q != r and host.has_edge(q, r)
            for q in chains[left]
            for r in chains[right]
        ):
            raise ValueError(f"logical demand {(left, right)!r} has no active contact")


def build_strength_inputs(
    *,
    ctx: Context,
    logical: nx.Graph,
    host: nx.Graph,
    problem: LogicalProblem,
    chains: Mapping[Node, frozenset[Qubit]],
    programs: Sequence[Program],
    coef_scale: float = 1.0,
    normalizer_digest: str = "selector-normalizer-unit-v1",
) -> tuple[StrengthGraphInput, ...]:
    """Build the four program-conditioned graph inputs required by IF-Q3-S0.

    ``coef_scale`` is a positive training-fitted physical-unit transform scale.  The caller
    must use the identical recorded value at fitting and deployment.
    """

    if not math.isfinite(coef_scale) or coef_scale <= 0.0:
        raise ValueError("coef_scale must be positive and finite")
    if not isinstance(normalizer_digest, str) or not normalizer_digest:
        raise ValueError("normalizer_digest must be a non-empty registered identifier")
    if len(programs) != N_STRENGTHS:
        raise ValueError("IF-Q3-S0 requires exactly four compiled programs")
    _validate_embedding(logical, host, problem, chains, ctx.qubit_cap)
    expected_strengths = strength_registry(
        problem,
        ctx.strength_ratios,
        ctx.epsilon_strength,
    )
    if tuple(program.strength_index for program in programs) != tuple(range(N_STRENGTHS)):
        raise ValueError("compiled programs must follow strength indices 0, 1, 2, 3")
    for index, (program, expected_strength) in enumerate(
        zip(programs, expected_strengths, strict=True)
    ):
        tolerance = 1e-12 * max(1.0, abs(expected_strength))
        if abs(program.strength - expected_strength) > tolerance:
            raise ValueError(f"program {index} does not use its registered strength")
        report = check_faithfulness(
            program,
            chains,
            host,
            problem,
            ctx.field_limit,
            ctx.coupler_limit,
        )
        if not report.ok:
            raise ValueError(f"program {index} is not faithfully programmable: {report.as_dict()}")

    selector_inputs: list[StrengthGraphInput] = []
    scale = instance_scale(problem, ctx.epsilon_strength)
    for program in programs:
        # Search fields are populated only to reuse one semantic implementation.  They are
        # then masked below and masked again by StrengthSelectorModel at its trust boundary.
        observation = build_observation(
            ctx=ctx,
            logical=logical,
            host=host,
            problem=problem,
            chains=chains,
            candidates=(),
            legal_mask=(),
            archive=(),
            remaining=ctx.caps,
            ages=Ages(),
            strengths=expected_strengths,
            program=program,
            mode_is_improvement=False,
            restarts_left=0,
            workspace_valid=True,
            protected_available=False,
            instance_scale=scale,
            coef_scale=coef_scale,
        )
        masked_global_slots = tuple(
            slot for slot in range(N_GLOBAL) if slot not in ALLOWED_GLOBAL_SLOTS
        )
        selector_input = StrengthGraphInput(
            logical=_mask_packed(
                observation.logical,
                N_LOGICAL,
                MASKED_LOGICAL_SLOTS,
            ),
            hardware=_mask_packed(
                observation.hardware,
                N_HARDWARE,
                MASKED_HARDWARE_SLOTS,
            ),
            logical_edges=_mask_packed(
                observation.logical_edges,
                N_LOGICAL_EDGE,
                MASKED_LOGICAL_EDGE_SLOTS,
            ),
            hardware_edges=np.asarray(observation.hardware_edges, dtype=np.float32).copy(),
            claims=_mask_packed(
                observation.claims,
                N_CLAIM,
                MASKED_CLAIM_SLOTS,
            ),
            globals_=_mask_packed(
                observation.globals_[0],
                N_GLOBAL,
                masked_global_slots,
            ),
            index_logical_edges=np.asarray(
                observation.index_logical_edges,
                dtype=np.int64,
            ).copy(),
            index_hardware_edges=np.asarray(
                observation.index_hardware_edges,
                dtype=np.int64,
            ).copy(),
            index_claims=np.asarray(observation.index_claims, dtype=np.int64).copy(),
            strength_index=program.strength_index,
            strength=program.strength,
            scale=program.scale,
            normalizer_digest=normalizer_digest,
        )
        selector_inputs.append(selector_input.validate())
    return tuple(selector_inputs)
