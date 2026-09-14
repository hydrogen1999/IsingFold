"""Chain-level scorer: f(state, candidate chain) trained pairwise on chain-seam samples.

Node features on the window (per qubit): free, adjacent to the chain of neighbour k (one
indicator per required neighbour, summed), number of frozen qubits adjacent, degree share,
free-neighbour share, articulation point of the free window subgraph. Candidate features:
size, internal couplers, contacts per required neighbour (min and mean), mean degree of its
qubits. The network runs message passing on the window, pools the candidate's qubit states
(mean and max) with the window pool and the candidate features, and outputs a scalar.

Evaluation is by regret: p_solve(best) - p_solve(picked), using each candidate's stored
estimate (stage-2 where available), against the resource-best pick, the original chain when
the state came from minorminer, and a random pick.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from embedbench.hamiltonian_context import (
    HAMILTONIAN_CONTEXT_DIMENSION,
    encode_hamiltonian_context,
)

NODE_F = 10
CAND_F = 10
USE_NEIGHBOUR_FEATS = False
"""When True, two candidate features are appended (mean neighbour degree / 6, mean neighbour
chain size / 4) and CAND_F is 12 for models built after the flag is set (codex audit: the
fields were computed and never used)."""

AUDIT_SCHEMA = "embedbench.independent-quality-audit"
AUDIT_SCHEMA_VERSION = 2
AUDIT_GENERATOR = "EmbedBench/scripts/rescore_quality.py"
AUDIT_SEED_SCHEDULE = "(base_seed + 7919*candidate_index + 31*strength_index) % 2**31"
REGISTERED_STRENGTH_COUNT = 4
AUDIT_STRENGTH_SCHEDULE = f"default_strength_grid(problem, {REGISTERED_STRENGTH_COUNT})"
AUDIT_AGGREGATION = "max_p_solve_over_strength_schedule"
MIN_HIGH_READS_PER_STRENGTH = 1000


@dataclass
class ChainEncoded:
    x: np.ndarray  # [n, NODE_F]
    adj: np.ndarray  # [n, n]
    cand_masks: np.ndarray  # [m, n]
    cand_feats: np.ndarray  # [m, CAND_F]
    p: list[float]
    stage: list[int]
    best_index: int
    resource_index: int
    original_index: int
    source: str
    instance_id: str
    topology: str
    difficulty: str
    hamiltonian_context: np.ndarray = field(
        default_factory=lambda: np.zeros(HAMILTONIAN_CONTEXT_DIMENSION, dtype=np.float32)
    )
    lengths: list | None = None  # candidate chain lengths (length-balanced training)


@dataclass(frozen=True)
class IndependentAuditEvidence:
    """Untrusted high-read row loaded from an audit JSONL artifact."""

    key: tuple[str, str, int]
    scores: tuple[float, ...]
    audit_schema: str | None
    audit_schema_version: int | None
    corpus_sha256: str | None
    candidate_signature: str | None
    reads: int | None
    provenance: Mapping[str, object] | None
    source: str
    line_number: int


@dataclass(frozen=True)
class VerifiedIndependentAuditScores:
    """Candidate-aligned scores bound to an exact corpus row and audit protocol."""

    key: tuple[str, str, int]
    scores: tuple[float | None, ...]
    corpus_sha256: str
    candidate_signature: str
    provenance: Mapping[str, object]

    @property
    def context(self) -> tuple[tuple[str, str, int], str, str]:
        return self.key, self.corpus_sha256, self.candidate_signature


def deployment_view(rec: dict, host, max_free: int = 8) -> dict:
    """Rebuild a quality record's window the way the deployment seam builds it (seam._record):
    window = candidate qubits and up to `max_free` free qubits within two hops, in
    deterministic order; frozen chains adjacent to that window; frozen adjacency on it.
    Candidate-policy indices are deliberately not read. Records carrying ``all_chains``
    (v1.1) use every other chain; older records use their stored ``frozen`` (chains adjacent
    to the corpus window), an approximation for the two-hop extras.
    """
    chains = rec.get("all_chains") or rec["frozen"]
    frozen = {int(v): sorted(int(q) for q in c) for v, c in chains.items()}
    cands = [set(c) for c in rec["candidates"]]
    seed_set = set().union(*cands)
    blocked = {q for c in frozen.values() for q in c}
    first = sorted(
        {n for q in seed_set for n in host.neighbors(q) if n not in blocked and n not in seed_set}
    )
    second = sorted(
        {
            n
            for q in first
            for n in host.neighbors(q)
            if n not in blocked and n not in seed_set and n not in first
        }
    )
    window = sorted(seed_set | set((first + second)[:max_free]))
    wset = set(window)
    fq = blocked
    out = dict(rec)
    out["window_nodes"] = [int(q) for q in window]
    out["window_edges"] = sorted([min(a, b), max(a, b)] for a, b in host.subgraph(wset).edges())
    out["frozen"] = {
        int(v): c for v, c in frozen.items() if any(n in wset for q in c for n in host.neighbors(q))
    }
    out["frozen_adjacency"] = {
        int(q): sorted(int(n) for n in host.neighbors(q) if n in fq)
        for q in window
        if any(n in fq for n in host.neighbors(q))
    }
    out["_deployment_view"] = True
    return out


DROP_LENGTH_FEATS = False
"""When True, the two explicit length features of a candidate (length / l_cap and
(length - 1) * max |J|) are zeroed at encoding time (L-115: the scorer's length prior)."""


def encode_chain(
    rec: dict,
    *,
    use_neighbour_feats: bool | None = None,
    drop_length_feats: bool | None = None,
    require_hamiltonian_context: bool = False,
) -> ChainEncoded:
    if use_neighbour_feats is None:
        use_neighbour_feats = USE_NEIGHBOUR_FEATS
    if drop_length_feats is None:
        drop_length_feats = DROP_LENGTH_FEATS
    if require_hamiltonian_context or "problem" in rec:
        hamiltonian_context = encode_hamiltonian_context(rec).values
    else:
        hamiltonian_context = np.zeros(HAMILTONIAN_CONTEXT_DIMENSION, dtype=np.float32)
    nodes = list(rec["window_nodes"])
    idx = {q: i for i, q in enumerate(nodes)}
    n = len(nodes)
    adjl = {q: [] for q in nodes}
    for a, b in rec["window_edges"]:
        adjl[a].append(b)
        adjl[b].append(a)
    frozen = {int(v): set(c) for v, c in rec["frozen"].items()}
    fa = {int(q): set(t) for q, t in rec.get("frozen_adjacency", {}).items()}
    req = [frozen.get(u, set()) for u in rec["neighbours"]]
    J = rec.get("edge_J", [0.0] * len(req))
    absJ = [abs(j) for j in J]
    ndeg = rec.get("neighbour_degree", [1] * len(req))
    ncs = rec.get("neighbour_chain_size", [1] * len(req))
    fq_all = set().union(*frozen.values()) if frozen else set()
    free = set(nodes)  # window qubits are all free in the state (focus chain removed)
    fg = nx.Graph()
    fg.add_nodes_from(nodes)
    fg.add_edges_from(map(tuple, rec["window_edges"]))
    art = set(nx.articulation_points(fg)) if n else set()
    maxdeg = max(1, max(len(adjl[q]) + len(fa.get(q, ())) for q in nodes))
    x = np.zeros((n, NODE_F), dtype=np.float32)
    for i, q in enumerate(nodes):
        touch = fa.get(q, set())
        x[i, 0] = 1.0
        x[i, 1] = sum(1 for r in req if touch & r) / max(1, len(req))
        x[i, 2] = len(touch) / maxdeg
        x[i, 3] = (len(adjl[q]) + len(touch)) / maxdeg
        x[i, 4] = len(adjl[q]) / max(1, len(adjl[q]) + len(touch))
        x[i, 5] = q in art
        x[i, 6] = len(touch & fq_all) > 0
        x[i, 7] = sum(1 for r in req if touch & r) == len(req) and len(req) > 0
        x[i, 8] = max((absJ[k] for k, r in enumerate(req) if touch & r), default=0.0)
        x[i, 9] = sum(absJ[k] for k, r in enumerate(req) if touch & r) / max(1e-6, sum(absJ))
    A = np.zeros((n, n), dtype=np.float32)
    for a, b in rec["window_edges"]:
        A[idx[a], idx[b]] = 1.0
        A[idx[b], idx[a]] = 1.0
    A = (A + np.eye(n, dtype=np.float32)) / (A.sum(1, keepdims=True) + 1.0)
    cands = rec["candidates"]
    masks = np.zeros((len(cands), n), dtype=np.float32)
    feats = np.zeros(
        (len(cands), CAND_F + (2 if use_neighbour_feats else 0)),
        dtype=np.float32,
    )
    for k, c in enumerate(cands):
        cs = set(c)
        for q in c:
            masks[k, idx[q]] = 1.0
        internal = sum(1 for a, b in rec["window_edges"] if a in cs and b in cs)
        contacts = [sum(1 for q in c for t in fa.get(q, ()) if t in r) for r in req]
        wcont = sum(absJ[j] * min(contacts[j], 3) for j in range(len(req))) / max(1e-6, sum(absJ))
        feats[k, :CAND_F] = [
            len(c) / rec["l_cap"],
            internal / max(1, len(c)),
            min(contacts) / 3.0 if contacts else 0.0,
            (sum(contacts) / max(1, len(contacts))) / 3.0,
            np.mean([x[idx[q], 3] for q in c]),
            (internal - (len(c) - 1)) / max(1, len(c)),
            wcont,
            sum(absJ) / max(1, len(absJ)),
            abs(rec.get("focus_h", 0.0)),
            (len(c) - 1) * max(absJ, default=0.0),
        ]
        if use_neighbour_feats:
            feats[k, CAND_F] = float(np.mean(ndeg)) / 6.0 if len(ndeg) else 0.0
            feats[k, CAND_F + 1] = float(np.mean(ncs)) / 4.0 if len(ncs) else 0.0
    if drop_length_feats:
        feats[:, 0] = 0.0
        feats[:, 9] = 0.0
    enc = ChainEncoded(
        x=x,
        adj=A,
        cand_masks=masks,
        cand_feats=feats,
        p=list(rec["p_solve"]),
        stage=list(rec.get("stage", [1] * len(cands))),
        best_index=rec["best_index"],
        resource_index=rec["resource_index"],
        original_index=rec["original_index"],
        source=rec["source"],
        instance_id=rec["instance_id"],
        topology=rec["topology"],
        difficulty=rec.get("difficulty", "base"),
        hamiltonian_context=hamiltonian_context,
    )
    enc.lengths = [len(c) for c in cands]
    return enc


def load_chain_records(paths):
    out = []
    for p in paths:
        with open(p) as fh:
            out += [json.loads(l) for l in fh]
    return out


def quality_audit_key(record: Mapping[str, object]) -> tuple[str, str, int]:
    """Return the corpus-qualified identity used by independent high-read labels."""
    raw_file = record.get("_file", record.get("file"))
    if not isinstance(raw_file, str) or not raw_file:
        raise ValueError("quality audit record requires a corpus file")
    corpus_file = Path(raw_file).name
    if not corpus_file:
        raise ValueError("quality audit record requires a corpus file basename")
    instance_id = record.get("instance_id")
    focus = record.get("focus")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("quality audit record requires a non-empty instance_id")
    if isinstance(focus, bool) or not isinstance(focus, numbers.Integral):
        raise ValueError("quality audit record requires an integer focus")
    return corpus_file, instance_id, int(focus)


def quality_candidate_signature(record: Mapping[str, object]) -> str:
    """Hash the ordered candidate support while treating each chain as a set."""
    candidates = record.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("quality record requires an ordered candidates sequence")
    canonical = []
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
            raise ValueError(f"candidate {candidate_index} must be a qubit sequence")
        chain = []
        for qubit in candidate:
            if isinstance(qubit, bool) or not isinstance(qubit, numbers.Integral):
                raise ValueError(f"candidate {candidate_index} contains a non-integer qubit")
            chain.append(int(qubit))
        canonical.append(sorted(chain))
    if not canonical:
        raise ValueError("quality record requires at least one candidate")
    payload = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _optional_audit_string(row, field: str, location: str) -> str | None:
    value = row.get(field)
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{location}: audit field {field} must be a non-empty string")
    return value


def _optional_audit_integer(row, field: str, location: str) -> int | None:
    value = row.get(field)
    if value is not None and (isinstance(value, bool) or not isinstance(value, numbers.Integral)):
        raise ValueError(f"{location}: audit field {field} must be an integer")
    return int(value) if value is not None else None


def load_independent_audit_scores(
    paths: Sequence[str | Path],
) -> dict[tuple[str, str, int], IndependentAuditEvidence]:
    """Load candidate-aligned independent scores produced by ``rescore_quality.py``."""
    labels = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError(f"{path}:{line_number}: audit row must be an object")
                key = quality_audit_key(row)
                if key in labels:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate independent audit labels for {key!r}"
                    )
                raw_scores = row.get("high_read_scores")
                if not isinstance(raw_scores, list) or not raw_scores:
                    raise ValueError(f"{path}:{line_number}: audit row requires high_read_scores")
                location = f"{path}:{line_number}"
                raw_provenance = row.get("provenance")
                if raw_provenance is not None and not isinstance(raw_provenance, Mapping):
                    raise ValueError(f"{location}: audit field provenance must be an object")
                labels[key] = IndependentAuditEvidence(
                    key=key,
                    scores=tuple(
                        _probability_vector(
                            raw_scores,
                            len(raw_scores),
                            "independent audit labels",
                        )
                    ),
                    audit_schema=_optional_audit_string(row, "audit_schema", location),
                    audit_schema_version=_optional_audit_integer(
                        row, "audit_schema_version", location
                    ),
                    corpus_sha256=_optional_audit_string(row, "corpus_sha256", location),
                    candidate_signature=_optional_audit_string(
                        row, "candidate_signature", location
                    ),
                    reads=_optional_audit_integer(row, "reads", location),
                    provenance=dict(raw_provenance) if raw_provenance is not None else None,
                    source=str(path),
                    line_number=line_number,
                )
    return labels


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def quality_audit_context(
    record: Mapping[str, object],
    corpus_sha256: str,
) -> tuple[tuple[str, str, int], str, str]:
    """Build the immutable identity required to consume a verified audit vector."""
    if not _valid_sha256(corpus_sha256):
        raise ValueError("quality corpus SHA-256 must be a lowercase hexadecimal digest")
    return (
        quality_audit_key(record),
        corpus_sha256,
        quality_candidate_signature(record),
    )


def _audit_protocol(evidence: IndependentAuditEvidence) -> dict[str, object]:
    location = f"{evidence.source}:{evidence.line_number}"
    if (
        evidence.audit_schema != AUDIT_SCHEMA
        or evidence.audit_schema_version != AUDIT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{location}: verified high-read provenance requires {AUDIT_SCHEMA} "
            f"schema version {AUDIT_SCHEMA_VERSION}"
        )
    provenance = evidence.provenance
    if provenance is None:
        raise ValueError(f"{location}: verified high-read provenance is missing")
    required = {
        "generator": AUDIT_GENERATOR,
        "registered_strength_count": REGISTERED_STRENGTH_COUNT,
        "seed_schedule": AUDIT_SEED_SCHEDULE,
        "strength_schedule": AUDIT_STRENGTH_SCHEDULE,
        "aggregation": AUDIT_AGGREGATION,
    }
    for field, expected in required.items():
        if field not in provenance:
            raise ValueError(f"{location}: verified high-read provenance requires {field}")
        if provenance[field] != expected:
            raise ValueError(f"{location}: unsupported high-read provenance {field}")
    reads = provenance.get("reads_per_strength")
    if (
        isinstance(reads, bool)
        or not isinstance(reads, numbers.Integral)
        or reads < MIN_HIGH_READS_PER_STRENGTH
    ):
        raise ValueError(
            f"{location}: high-read provenance requires at least "
            f"{MIN_HIGH_READS_PER_STRENGTH} reads_per_strength"
        )
    if evidence.reads != reads:
        raise ValueError(f"{location}: top-level reads must match provenance reads_per_strength")
    base_seed = provenance.get("base_seed")
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, numbers.Integral)
        or not 0 <= base_seed < 2**31
    ):
        raise ValueError(f"{location}: high-read provenance requires a valid base_seed")
    sweeps = provenance.get("num_sweeps")
    if isinstance(sweeps, bool) or not isinstance(sweeps, numbers.Integral) or sweeps <= 0:
        raise ValueError(f"{location}: high-read provenance requires positive num_sweeps")
    return dict(provenance)


def verify_independent_audit_scores(
    record: Mapping[str, object],
    evidence: IndependentAuditEvidence,
    corpus_sha256: str,
) -> VerifiedIndependentAuditScores:
    """Bind untrusted audit evidence to the exact corpus bytes and candidate order."""
    context = quality_audit_context(record, corpus_sha256)
    key, bound_corpus_sha256, signature = context
    if key != evidence.key:
        raise ValueError("independent audit identity does not match the quality record")
    if evidence.corpus_sha256 != bound_corpus_sha256:
        raise ValueError("independent audit corpus SHA-256 does not match the input corpus")
    if evidence.candidate_signature != signature:
        raise ValueError("independent audit candidate signature does not match candidate order")
    if len(evidence.scores) != len(record["candidates"]):
        raise ValueError("independent audit scores do not align with the signed candidate support")
    provenance = _audit_protocol(evidence)
    return VerifiedIndependentAuditScores(
        key=evidence.key,
        scores=evidence.scores,
        corpus_sha256=bound_corpus_sha256,
        candidate_signature=signature,
        provenance=provenance,
    )


def split_by_instance(encs, frac, seed):
    ids = sorted({e.instance_id.rsplit("-", 1)[0] for e in encs})
    rng = random.Random(seed)
    rng.shuffle(ids)
    test = set(ids[: max(1, int(len(ids) * frac))])
    key = lambda e: e.instance_id.rsplit("-", 1)[0]
    return [e for e in encs if key(e) not in test], [e for e in encs if key(e) in test]


def build_chain_model(hidden=64, layers=3, neighbour_feats: bool | None = None):
    import torch, torch.nn as nn

    if neighbour_feats is None:
        neighbour_feats = USE_NEIGHBOUR_FEATS
    candidate_features = CAND_F + (2 if neighbour_feats else 0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(NODE_F, hidden), nn.ReLU())
            self.msg = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
                    )
                    for _ in range(layers)
                ]
            )
            self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.read = nn.Sequential(
                nn.Linear(3 * hidden + candidate_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x, adj, masks, feats):
            h = self.inp(x)
            for m, ln in zip(self.msg, self.norm):
                h = ln(h + m(torch.cat([h, adj @ h], -1)))
            pooled = h.mean(0, keepdim=True).expand(masks.shape[0], -1)
            cmean = (masks @ h) / masks.sum(1, keepdim=True).clamp(min=1.0)
            cmax = torch.stack([h[masks[k] > 0].max(0).values for k in range(masks.shape[0])])
            return self.read(torch.cat([cmean, cmax, pooled, feats], -1)).squeeze(-1)

    return Net()


def build_chain_linear(neighbour_feats: bool | None = None):
    import torch, torch.nn as nn

    if neighbour_feats is None:
        neighbour_feats = USE_NEIGHBOUR_FEATS
    candidate_features = CAND_F + (2 if neighbour_feats else 0)

    class Lin(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(candidate_features, 1)

        def forward(self, x, adj, masks, feats):
            return self.w(feats).squeeze(-1)

    return Lin()


def _t(e, device):
    import torch

    return (
        torch.from_numpy(e.x).to(device),
        torch.from_numpy(e.adj).to(device),
        torch.from_numpy(e.cand_masks).to(device),
        torch.from_numpy(e.cand_feats).to(device),
    )


def _pairs(p, stage, tol, tol1):
    """Ordered pairs (i better than j) worth training on: both stage-2 and apart by tol, or
    at least one stage-1 and apart by tol1 (stage-1 estimates are noisier)."""
    out = []
    for i in range(len(p)):
        for j in range(len(p)):
            t = tol if (stage[i] == 2 and stage[j] == 2) else tol1
            if p[i] - p[j] >= t:
                out.append((i, j))
    return out


def pair_loss(
    scores, p, tol, stage=None, tol1=None, lengths=None, length_balance=False, stage2_only=False
):
    """Pairwise logistic loss over (better, worse) pairs. `length_balance`: the pairs where the
    better candidate is longer and those where it is shorter get equal total weight (pairs of
    equal length keep weight 1), so the loss cannot be lowered by a length prior alone.
    `stage2_only`: only pairs among stage-2 (reliable) candidates."""
    import torch

    stage = stage or [1] * len(p)
    tol1 = tol1 if tol1 is not None else 2 * tol
    li, lj = [], []
    for i, j in _pairs(p, stage, tol, tol1):
        if stage2_only and not (stage[i] == 2 and stage[j] == 2):
            continue
        li.append(i)
        lj.append(j)
    if not li:
        return None
    losses = torch.nn.functional.softplus(-(scores[li] - scores[lj]))
    if length_balance and lengths is not None:
        sign = [
            (1 if lengths[i] > lengths[j] else -1 if lengths[i] < lengths[j] else 0)
            for i, j in zip(li, lj)
        ]
        n_long = sum(1 for x in sign if x == 1)
        n_short = sum(1 for x in sign if x == -1)
        if n_long and n_short:
            w = torch.tensor(
                [(n_short / n_long) if x == 1 else 1.0 for x in sign],
                dtype=losses.dtype,
                device=losses.device,
            )
            w = w * (len(sign) / w.sum())
            return (losses * w).mean()
    return losses.mean()


def _candidate_count(example: object) -> int:
    """Read support size from deployable candidate data, never from a label vector."""
    explicit = getattr(example, "candidate_count", None)
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, numbers.Integral):
            raise ValueError("candidate_count must be an integer")
        count = int(explicit)
    else:
        candidates = getattr(example, "cand_masks", None)
        if candidates is None:
            candidates = getattr(example, "candidates", None)
        if candidates is None:
            raise ValueError(
                "quality example must expose candidate_count, cand_masks, or candidates"
            )
        count = len(candidates)
    if count <= 0:
        raise ValueError("quality example must contain at least one candidate")
    return count


def _probability_vector(values: Sequence[object], count: int, name: str) -> list[float]:
    if len(values) != count:
        raise ValueError(f"{name} must contain one value per candidate")
    out = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} values must be finite probabilities")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} values must be finite probabilities in [0, 1]")
        out.append(number)
    return out


def _audit_vector(
    values: (
        Sequence[object | None] | IndependentAuditEvidence | VerifiedIndependentAuditScores | None
    ),
    count: int,
) -> list[float | None]:
    if values is None:
        return [None] * count
    raw_values = (
        values.scores
        if isinstance(
            values,
            (IndependentAuditEvidence, VerifiedIndependentAuditScores),
        )
        else values
    )
    if len(raw_values) != count:
        raise ValueError("independent audit labels must contain one value per candidate")
    out = []
    for value in raw_values:
        if value is None:
            out.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError("independent audit labels must be finite probabilities")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError("independent audit labels must be finite probabilities in [0, 1]")
        out.append(number)
    return out


def _score_vector(score_fn: Callable[[object], Sequence[float]], example, count: int):
    raw = score_fn(example)
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    scores = np.asarray(raw, dtype=float)
    if scores.ndim != 1 or len(scores) != count:
        raise ValueError("quality scorer must return one score per candidate")
    if not np.isfinite(scores).all():
        raise ValueError("quality scorer returned a non-finite score")
    return scores


def _support_indices(example, count: int, support: str) -> list[int]:
    if support == "full":
        return list(range(count))
    if support != "legacy-reliable":
        raise ValueError("evaluation support must be 'full' or 'legacy-reliable'")
    stage = getattr(example, "stage", None)
    if stage is None or len(stage) != count:
        raise ValueError("legacy reliable evaluation requires one stage value per candidate")
    reliable = [index for index, value in enumerate(stage) if value == 2]
    return reliable or list(range(count))


def _selection_index(example, name: str, count: int, support: Sequence[int]) -> int | None:
    value = getattr(example, f"{name}_index", -1)
    if name == "original" and value == -1:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name}_index must be an integer")
    index = int(value)
    if not 0 <= index < count:
        raise ValueError(f"{name}_index is outside the candidate support")
    if index not in support:
        raise ValueError(
            f"{name}_index is excluded by {support!r}; controls must use identical support"
        )
    return index


def _quality_metrics(rows, *, tol: float, top_tolerance: float) -> dict[str, object]:
    regrets = {name: [] for name in ("model", "resource", "random", "original")}
    tops = {name: 0 for name in regrets}
    pair_ok = pair_count = 0
    original_count = 0
    for row in rows:
        labels = row["labels"]
        support = row["support"]
        best = max(labels[index] for index in support)
        for name in ("model", "resource", "random"):
            index = row[f"{name}_index"]
            regrets[name].append(best - labels[index])
            tops[name] += labels[index] >= best - top_tolerance
        original = row["original_index"]
        if original is not None:
            original_count += 1
            regrets["original"].append(best - labels[original])
            tops["original"] += labels[original] >= best - top_tolerance
        scores = row["scores"]
        for first in support:
            for second in support:
                if labels[first] - labels[second] >= tol:
                    pair_count += 1
                    pair_ok += scores[first] > scores[second]

    count = len(rows)
    denominator = max(1, count)
    out = {
        "n": count,
        "pair_acc": pair_ok / max(1, pair_count),
        "pairs": pair_count,
    }
    for name in ("model", "resource", "random"):
        out[f"{name}_regret"] = float(np.mean(regrets[name])) if regrets[name] else None
        out[f"{name}_top"] = float(tops[name] / denominator)
    out["original_regret"] = float(np.mean(regrets["original"])) if regrets["original"] else None
    out["original_top"] = float(tops["original"] / original_count) if original_count else None
    out["n_original"] = original_count
    return out


def evaluate_quality_support(
    score_fn: Callable[[object], Sequence[float]],
    examples: Sequence[object],
    *,
    support: str = "full",
    mode: str = "development",
    audit_scores: Sequence[object | None] | None = None,
    tol: float = 0.05,
    top_tolerance: float = 0.02,
    random_seed: int = 0,
) -> dict[str, object]:
    """Evaluate quality selectors on one identical candidate support.

    ``full`` support is derived only from deployable candidate tensors.  The named
    ``legacy-reliable`` diagnostic intentionally reproduces the V1 stage-2 shortlist.
    Development metrics use the release labels and are marked provisional; ``audit`` and
    ``paper`` require an independent label for every candidate in every evaluated support.
    """
    if mode not in {"development", "audit", "paper"}:
        raise ValueError("evaluation mode must be 'development', 'audit', or 'paper'")
    if mode in {"audit", "paper"} and support != "full":
        raise ValueError(
            f"{mode} evaluation requires full candidate support; "
            "legacy-reliable is a label-derived development diagnostic"
        )
    if tol < 0 or top_tolerance < 0:
        raise ValueError("quality evaluation tolerances must be non-negative")
    items = list(examples)
    if mode in {"audit", "paper"} and not items:
        raise ValueError(f"{mode} evaluation requires at least one example")
    audits = [None] * len(items) if audit_scores is None else list(audit_scores)
    if len(audits) != len(items):
        raise ValueError("audit_scores must align one-to-one with examples")

    rng = random.Random(random_seed)
    development_rows = []
    complete_audit_rows = []
    selections = []
    support_sizes = []
    audit_labeled = 0
    audit_records_any = 0
    audit_records_verified = 0
    fidelity_stages = set()
    for record_index, (example, raw_audit) in enumerate(zip(items, audits, strict=True)):
        provenance_verified = isinstance(
            raw_audit,
            VerifiedIndependentAuditScores,
        )
        if mode in {"audit", "paper"} and raw_audit is not None and not provenance_verified:
            raise ValueError(f"{mode} evaluation requires verified high-read provenance")
        if provenance_verified and getattr(example, "audit_context", None) != raw_audit.context:
            raise ValueError(f"{mode} evaluation audit context does not match the encoded example")
        count = _candidate_count(example)
        scores = _score_vector(score_fn, example, count)
        indices = _support_indices(example, count, support)
        if not indices:
            raise ValueError("quality evaluation support cannot be empty")
        labels = _probability_vector(example.p, count, "development labels")
        audit = _audit_vector(raw_audit, count)
        resource = _selection_index(example, "resource", count, indices)
        original = _selection_index(example, "original", count, indices)
        model = max(indices, key=lambda index: scores[index])
        random_index = rng.choice(indices)
        support_sizes.append(len(indices))
        stage = getattr(example, "stage", None)
        if stage is not None and len(stage) == count:
            fidelity_stages.update(stage[index] for index in indices)

        labeled_here = sum(audit[index] is not None for index in indices)
        complete = labeled_here == len(indices)
        audit_labeled += labeled_here
        audit_records_any += labeled_here > 0
        audit_records_verified += provenance_verified and complete
        row = {
            "scores": scores,
            "support": indices,
            "model_index": model,
            "resource_index": resource,
            "random_index": random_index,
            "original_index": original,
        }
        development_rows.append({**row, "labels": labels})
        if complete:
            complete_audit_rows.append({**row, "labels": audit})
        selections.append(
            {
                "record_index": record_index,
                "instance_id": getattr(example, "instance_id", None),
                "support_indices": list(indices),
                "support_size": len(indices),
                "model_index": model,
                "model_has_audit_label": audit[model] is not None,
                "audit_provenance_verified": provenance_verified,
                "resource_index": resource,
                "resource_has_audit_label": audit[resource] is not None,
                "original_index": original,
                "original_has_audit_label": (
                    audit[original] is not None if original is not None else None
                ),
                "random_index": random_index,
                "random_has_audit_label": audit[random_index] is not None,
            }
        )

    total_supported = sum(support_sizes)
    audit_coverage = {
        "records_with_any_label": audit_records_any,
        "records_with_complete_support": len(complete_audit_rows),
        "record_fraction_complete": (len(complete_audit_rows) / len(items) if items else 0.0),
        "labeled_candidates": audit_labeled,
        "supported_candidates": total_supported,
        "candidate_fraction": audit_labeled / total_supported if total_supported else 0.0,
    }
    if mode in {"audit", "paper"} and len(complete_audit_rows) != len(items):
        raise ValueError(
            f"{mode} evaluation requires independent audit labels for the complete support"
        )

    development = _quality_metrics(
        development_rows,
        tol=tol,
        top_tolerance=top_tolerance,
    )
    audit = (
        _quality_metrics(
            complete_audit_rows,
            tol=tol,
            top_tolerance=top_tolerance,
        )
        if complete_audit_rows
        else None
    )
    active = audit if mode in {"audit", "paper"} else development
    if mode in {"audit", "paper"}:
        label_fidelity = "independent-audit"
    elif fidelity_stages == {2}:
        label_fidelity = "stage2-release"
    elif fidelity_stages:
        label_fidelity = "mixed-stage-release"
    else:
        label_fidelity = "release-unspecified"

    return {
        **active,
        "evaluation_support": support,
        "evaluation_mode": mode,
        "support_uses_labels": support == "legacy-reliable",
        "label_fidelity": label_fidelity,
        "provisional": mode == "development",
        "candidate_support": {
            "records": len(items),
            "total_candidates": sum(_candidate_count(example) for example in items),
            "total_supported_candidates": total_supported,
            "minimum_support_size": min(support_sizes) if support_sizes else 0,
            "maximum_support_size": max(support_sizes) if support_sizes else 0,
            "mean_support_size": (float(np.mean(support_sizes)) if support_sizes else 0.0),
        },
        "audit_coverage": audit_coverage,
        "audit_provenance": {
            "verified_records": audit_records_verified,
            "all_complete_support_verified": (
                bool(complete_audit_rows) and audit_records_verified == len(complete_audit_rows)
            ),
        },
        "development_metrics": development,
        "audit_metrics": audit,
        "selections": selections,
    }


def evaluate(
    model,
    encs,
    device="cpu",
    tol=0.05,
    *,
    support="full",
    mode="development",
    audit_scores=None,
    top_tolerance=0.02,
    random_seed=0,
):
    """Model-oriented wrapper around :func:`evaluate_quality_support`.

    Full, label-independent support is the default.  Call
    :func:`evaluate_reliable_shortlist` to reproduce the V1 diagnostic.
    """
    import torch

    model.eval()

    def score(example):
        return model(*_t(example, device)).detach().cpu().numpy()

    with torch.no_grad():
        return evaluate_quality_support(
            score,
            encs,
            support=support,
            mode=mode,
            audit_scores=audit_scores,
            tol=tol,
            top_tolerance=top_tolerance,
            random_seed=random_seed,
        )


def evaluate_reliable_shortlist(model, encs, device="cpu", tol=0.05):
    """Named V1 diagnostic that restricts every selector to stage-2 candidates."""
    return evaluate(
        model,
        encs,
        device=device,
        tol=tol,
        support="legacy-reliable",
    )


evaluate_legacy_reliable = evaluate_reliable_shortlist


def train(model, tr, va, *, epochs=30, lr=1e-3, seed=0, device="cpu", tol=0.05, log=print):
    import torch

    torch.manual_seed(seed)
    rng = random.Random(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = None
    for ep in range(epochs):
        model.train()
        order = list(range(len(tr)))
        rng.shuffle(order)
        for b0 in range(0, len(order), 16):
            opt.zero_grad()
            loss = 0.0
            m = 0
            for k in order[b0 : b0 + 16]:
                e = tr[k]
                s = model(*_t(e, device))
                l = pair_loss(s, e.p, tol, e.stage)
                if l is not None:
                    loss = loss + l
                    m += 1
            if m:
                (loss / m).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        ev = evaluate_reliable_shortlist(model, va, device, tol)
        log(
            f"epoch {ep + 1:3d} val regret model {ev['model_regret']:.4f} resource {ev['resource_regret']:.4f} random {ev['random_regret']:.4f} pair {ev['pair_acc']:.3f}"
        )
        if best is None or ev["model_regret"] < best[0]:
            best = (
                ev["model_regret"],
                ep + 1,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
    model.load_state_dict(best[2])
    return best[1]


def save_chain_model(model, path, meta=None):
    from pathlib import Path

    import torch

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save({"state": state, "meta": meta or {}}, output)


def checkpoint_preprocessing(meta: Mapping[str, object]) -> dict[str, bool]:
    """Read the explicit preprocessing contract, including legacy flat metadata."""
    nested = meta.get("preprocessing", {})
    if not isinstance(nested, Mapping):
        raise ValueError("checkpoint preprocessing metadata must be an object")
    flags = {}
    for key in ("neighbour_feats", "no_length_feats", "deploy_view"):
        value = nested.get(key, meta.get(key, False))
        if not isinstance(value, bool):
            raise ValueError(f"checkpoint preprocessing field {key!r} must be boolean")
        flags[key] = value
    return flags


def _prepare_scorer_record(rec: dict, deploy_view_required: bool) -> dict:
    already_deployment_view = bool(rec.get("_deployment_view")) or rec.get("source") == "seam"
    if already_deployment_view:
        if not deploy_view_required:
            raise ValueError(
                "chain scorer was trained without deploy-view preprocessing and cannot "
                "faithfully score deployment seam records"
            )
        return rec
    if not deploy_view_required:
        return rec
    topology = rec.get("topology")
    size = rec.get("size")
    if not isinstance(topology, str) or not isinstance(size, int):
        raise ValueError(
            "deploy-view checkpoint requires either a seam record or a corpus record "
            "with topology and integer size"
        )
    from embedbench.structural import host_graph

    return deployment_view(rec, host_graph(topology, size))


def _attach_preprocessing(model, flags: Mapping[str, bool]):
    model._neighbour_feats = flags["neighbour_feats"]
    model._no_length_feats = flags["no_length_feats"]
    model._deploy_view = flags["deploy_view"]
    return model


def load_chain_model(path):
    import torch

    ck = torch.load(path, map_location="cpu")
    meta = ck.get("meta", {})
    if not isinstance(meta, Mapping):
        raise ValueError("checkpoint metadata must be an object")
    arch = meta.get("arch", "mpnn")
    if arch == "hetero":
        raise ValueError("hetero checkpoint requires embedbench.models_hetero.load_scorer")
    flags = checkpoint_preprocessing(meta)
    model = build_chain_arch(
        arch,
        meta.get("hidden", 64),
        meta.get("layers", 3),
        meta.get("heads", 4),
        neighbour_feats=flags["neighbour_feats"],
    )
    model.load_state_dict(ck["state"])
    model.eval()
    return _attach_preprocessing(model, flags)


class ChainScorer:
    def __init__(
        self,
        model,
        *,
        neighbour_feats: bool | None = None,
        no_length_feats: bool | None = None,
        deploy_view: bool | None = None,
    ):
        self.model = model
        self.neighbour_feats = (
            bool(getattr(model, "_neighbour_feats", False))
            if neighbour_feats is None
            else neighbour_feats
        )
        self.no_length_feats = (
            bool(getattr(model, "_no_length_feats", False))
            if no_length_feats is None
            else no_length_feats
        )
        self.deploy_view = (
            bool(getattr(model, "_deploy_view", False)) if deploy_view is None else deploy_view
        )

    def __call__(self, rec: dict) -> list[float]:
        import torch

        prepared = _prepare_scorer_record(rec, self.deploy_view)
        e = encode_chain(
            prepared,
            use_neighbour_feats=self.neighbour_feats,
            drop_length_feats=self.no_length_feats,
        )
        with torch.no_grad():
            return [float(v) for v in self.model(*_t(e, "cpu"))]


# ---------------------------------------------------------------- architecture variants (2026-09-09)


def build_chain_arch(
    arch: str = "mpnn",
    hidden: int = 64,
    layers: int = 3,
    heads: int = 4,
    *,
    neighbour_feats: bool | None = None,
):
    """Same bodies as `models_structural.build_arch` (mpnn, gin, gatv2, gps) with the chain
    readout: pooled candidate-qubit states (mean and max), the window mean, and the candidate
    features."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from embedbench.models_structural import hop_matrix

    if neighbour_feats is None:
        neighbour_feats = USE_NEIGHBOUR_FEATS
    candidate_features = CAND_F + (2 if neighbour_feats else 0)

    if arch == "mpnn":
        return build_chain_model(hidden, layers, neighbour_feats)

    class ChainRead(nn.Module):
        def __init__(self):
            super().__init__()
            self.read = nn.Sequential(
                nn.Linear(3 * hidden + candidate_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, h, masks, feats):
            pooled = h.mean(0, keepdim=True).expand(masks.shape[0], -1)
            cmean = (masks @ h) / masks.sum(1, keepdim=True).clamp(min=1.0)
            cmax = torch.stack([h[masks[k] > 0].max(0).values for k in range(masks.shape[0])])
            return self.read(torch.cat([cmean, cmax, pooled, feats], -1)).squeeze(-1)

    class GIN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(NODE_F, hidden)
            self.eps = nn.Parameter(torch.zeros(layers))
            self.mlps = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                    for _ in range(layers)
                ]
            )
            self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.out = ChainRead()

        def forward(self, x, adj, masks, feats):
            A = (adj > 0).float()
            A = A - torch.diag(torch.diag(A))
            h = F.relu(self.inp(x))
            for k in range(layers):
                h = self.norm[k](h + self.mlps[k]((1 + self.eps[k]) * h + A @ h))
            return self.out(h, masks, feats)

    class GATv2(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(NODE_F, hidden)
            self.W = nn.ModuleList([nn.Linear(2 * hidden, hidden * heads) for _ in range(layers)])
            self.a = nn.ModuleList([nn.Linear(hidden, 1, bias=False) for _ in range(layers)])
            self.proj = nn.ModuleList([nn.Linear(hidden * heads, hidden) for _ in range(layers)])
            self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.out = ChainRead()

        def forward(self, x, adj, masks, feats):
            n = x.shape[0]
            A = (adj > 0).float()
            h = F.relu(self.inp(x))
            for k in range(layers):
                hi = h.unsqueeze(1).expand(n, n, hidden)
                hj = h.unsqueeze(0).expand(n, n, hidden)
                z = F.leaky_relu(self.W[k](torch.cat([hi, hj], -1)), 0.2).view(n, n, heads, hidden)
                e = self.a[k](z).squeeze(-1).masked_fill(A.unsqueeze(-1) == 0, float("-inf"))
                att = torch.softmax(e, dim=1)
                msg = torch.einsum("ijh,ijhd->ihd", att, z).reshape(n, heads * hidden)
                h = self.norm[k](h + self.proj[k](msg))
            return self.out(h, masks, feats)

    class GPS(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(NODE_F, hidden)
            self.local = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                    for _ in range(layers)
                ]
            )
            self.attn = nn.ModuleList(
                [nn.MultiheadAttention(hidden, heads, batch_first=True) for _ in range(layers)]
            )
            self.bias = nn.Parameter(torch.zeros(layers, heads, 6))
            self.ff = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, hidden)
                    )
                    for _ in range(layers)
                ]
            )
            self.n1 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.n2 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.n3 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.out = ChainRead()

        def forward(self, x, adj, masks, feats):
            n = x.shape[0]
            A = (adj > 0).float()
            A = A - torch.diag(torch.diag(A))
            hops = torch.from_numpy(hop_matrix(adj.detach().cpu().numpy())).to(x.device)
            h = F.relu(self.inp(x))
            for k in range(layers):
                h = self.n1[k](h + self.local[k](h + A @ h))
                a, _ = self.attn[k](
                    h.unsqueeze(0),
                    h.unsqueeze(0),
                    h.unsqueeze(0),
                    attn_mask=(-self.bias[k][:, hops]).reshape(heads, n, n),
                    need_weights=False,
                )
                h = self.n2[k](h + a.squeeze(0))
                h = self.n3[k](h + self.ff[k](h))
            return self.out(h, masks, feats)

    return {"gin": GIN, "gatv2": GATv2, "gps": GPS}[arch]()
