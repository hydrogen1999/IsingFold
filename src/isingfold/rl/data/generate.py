"""Corpus driver: instances, witnesses, certificates, strength counts and decision labels.

Spec: Rev2 section "A staged corpus plan and release gates". Public inputs and evaluator-only
metadata are written to physically separate files, every record carries its lineage and
generator version, and the four release gates are reported rather than assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import networkx as nx

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import Context
from isingfold.rl.data.inkdrop import InkDropError, apply_faults, ink_drop
from isingfold.rl.data.lineage import Lineage, digest, split_by_lineage
from isingfold.rl.data.planting import PlantingError, frustrated_loops
from isingfold.rl.data.quality import strength_counts
from isingfold.rl.env import EmbeddingTask

GENERATOR_VERSION = "rev2-inkdrop-frustrated-1"


def host_graph(name: str, size: int) -> nx.Graph:
    """Named hosts. Grids need no optional dependency; the device families need dwave-networkx."""

    if name == "grid":
        return nx.convert_node_labels_to_integers(nx.grid_2d_graph(size, size))
    import dwave_networkx as dnx

    builders = {"chimera": dnx.chimera_graph, "pegasus": dnx.pegasus_graph, "zephyr": dnx.zephyr_graph}
    if name not in builders:
        raise ValueError(f"unregistered host family {name!r}")
    return builders[name](size)


@dataclass(frozen=True)
class GeneratedInstance:
    task: EmbeddingTask
    lineage: Lineage
    witness_receipt: Mapping[str, object]
    clause_report: Mapping[str, object]

    def public_record(self) -> dict[str, object]:
        problem = self.task.problem
        return {
            "instance": self.task.name,
            "lineage": self.lineage.lineage_id,
            "lineage_root": self.lineage.root,
            "family": self.lineage.family,
            "host": self.lineage.host,
            "host_size": self.lineage.size,
            "n_vars": problem.n,
            "edges": [[str(u), str(v)] for u, v in self.task.logical.edges()],
            "h": {str(k): v for k, v in problem.h.items()},
            "J": [[str(u), str(v), w] for (u, v), w in problem.j.items()],
            "active_nodes": [str(q) for q in self.task.host.nodes()],
            "active_edges": [[str(a), str(b)] for a, b in self.task.host.edges()],
            "generator_version": GENERATOR_VERSION,
            "coefficient_digest": digest({"h": {str(k): v for k, v in problem.h.items()},
                                          "J": [[str(u), str(v), w] for (u, v), w in problem.j.items()]}),
        }

    def evaluator_record(self) -> dict[str, object]:
        return {
            "instance": self.task.name,
            "lineage_root": self.lineage.root,
            "ground_energy": self.task.ground_energy,
            "witness": {str(k): sorted(map(str, v)) for k, v in (self.task.witness or {}).items()},
            "witness_receipt": dict(self.witness_receipt),
            "clause_report": dict(self.clause_report),
            "certificate": "planted frustrated-loop clause decomposition",
        }


def generate_instances(
    *,
    host_family: str = "grid",
    host_size: int = 10,
    n_instances: int = 8,
    n_variables: int = 12,
    chain_size: int = 3,
    fault_rate: float = 0.02,
    alpha: float = 0.5,
    density: float = 1.0,
    modes: Sequence[str] = ("compact", "contact_seeking", "elongated", "bottleneck"),
    weight_choices: Sequence[float] = (1.0,),
    clause_length: int = 6,
    seed: int = 0,
    max_attempts: int = 12,
) -> list[GeneratedInstance]:
    """Ink-drop a feasible support, then plant a certified logical optimum on it.

    Embedding difficulty and sampling difficulty are separate axes (Rev2 section
    "Separate embedding difficulty from sampling difficulty"): ``n_variables``, ``chain_size``
    and ``fault_rate`` press on the embedding, while ``alpha``, ``clause_length`` and the
    coefficient dynamic range in ``weight_choices`` press on the energy landscape.
    """

    base = host_graph(host_family, host_size)
    out: list[GeneratedInstance] = []
    for k in range(n_instances):
        mode = modes[k % len(modes)]
        for attempt in range(max_attempts):
            local = seed * 1000 + k * 31 + attempt
            try:
                active = apply_faults(base, fault_rate, local)
                drop = ink_drop(
                    active, n_variables, chain_size, seed=local, mode=mode, density=density
                )
                planted = frustrated_loops(
                    drop.logical,
                    alpha=alpha,
                    seed=local,
                    max_length=clause_length,
                    weight_choices=weight_choices,
                )
            except (InkDropError, PlantingError):
                continue
            name = f"{host_family}{host_size}-{mode}-{k}"
            lineage = Lineage(
                lineage_id=f"{name}-l{local}",
                family=mode,
                host=f"{host_family}{host_size}",
                size=n_variables,
                generator_version=GENERATOR_VERSION,
            )
            task = EmbeddingTask(
                name=name,
                logical=planted.problem.graph,
                host=active,
                problem=planted.problem,
                ground_energy=planted.ground_energy,
                lineage=lineage.lineage_id,
                witness=drop.witness,
            )
            out.append(
                GeneratedInstance(task, lineage, drop.receipt, planted.verify())
            )
            break
    return out


def write_corpus(
    instances: Sequence[GeneratedInstance],
    out_dir: str | Path,
    *,
    ctx: Context | None = None,
    strength_reads: int = 128,
    seed: int = 0,
    ood_predicate: Callable[[Lineage], bool] | None = None,
) -> dict[str, object]:
    """Write public inputs, evaluator-only metadata, splits and four-strength count labels."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "instances.jsonl").open("w") as fh:
        for instance in instances:
            fh.write(json.dumps(instance.public_record()) + "\n")
    with (out / "evaluator_only.jsonl").open("w") as fh:
        for instance in instances:
            fh.write(json.dumps(instance.evaluator_record()) + "\n")

    split = split_by_lineage(
        (i.lineage for i in instances), ood_predicate=ood_predicate, seed=seed
    )
    (out / "splits.json").write_text(json.dumps(split.as_dict(), indent=1))

    strength_rows = 0
    if ctx is not None:
        with (out / "strength_counts.jsonl").open("w") as fh:
            for k, instance in enumerate(instances):
                if instance.task.witness is None:
                    continue
                if split.partition_of(instance.lineage.root) not in ("train", "validation"):
                    continue
                features, hits, reads = strength_counts(
                    instance.task,
                    instance.task.witness,
                    ctx,
                    reads=strength_reads,
                    seed=seed + 17 * k,
                )
                fh.write(
                    json.dumps(
                        {
                            "instance": instance.task.name,
                            "lineage_root": instance.lineage.root,
                            "features": [dict(f) for f in features],
                            "hits": list(hits),
                            "reads": list(reads),
                            "source": "witness",
                        }
                    )
                    + "\n"
                )
                strength_rows += 1

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "instances": len(instances),
        "strength_records": strength_rows,
        "split": {k: len(v) for k, v in split.as_dict().items()},
        "files": sorted(p.name for p in out.iterdir()),
        "digest": digest([i.public_record() for i in instances]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load_instances(path: str | Path) -> list[EmbeddingTask]:
    """Rebuild tasks from the public records plus the evaluator-only certificates."""

    directory = Path(path)
    public = [json.loads(line) for line in (directory / "instances.jsonl").read_text().splitlines()]
    evaluator = {
        json.loads(line)["instance"]: json.loads(line)
        for line in (directory / "evaluator_only.jsonl").read_text().splitlines()
    }
    tasks: list[EmbeddingTask] = []
    for record in public:
        host = nx.Graph()
        host.add_nodes_from(record["active_nodes"])
        host.add_edges_from([tuple(e) for e in record["active_edges"]])
        logical = nx.Graph()
        logical.add_nodes_from(record["h"].keys())
        logical.add_edges_from([(u, v) for u, v, _ in record["J"]])
        problem = LogicalProblem.from_dicts(
            {k: float(v) for k, v in record["h"].items()},
            {(u, v): float(w) for u, v, w in record["J"]},
        )
        meta = evaluator[record["instance"]]
        witness = {k: frozenset(v) for k, v in meta["witness"].items()} or None
        tasks.append(
            EmbeddingTask(
                name=record["instance"],
                logical=logical,
                host=host,
                problem=problem,
                ground_energy=float(meta["ground_energy"]),
                lineage=record["lineage_root"],
                witness=witness,
            )
        )
    return tasks
