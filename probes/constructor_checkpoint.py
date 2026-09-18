"""Validation-only selection and reproducible identities for constructor runs."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch


def run_contract(args, train, heldout):
    """Hash public instance contents and run settings, without outcomes or witnesses."""
    def signature(task):
        def ordered(values):
            return sorted(map(repr, values))
        value = {
            "name": task.name, "lineage": task.lineage,
            "logical_nodes": ordered(task.logical.nodes()),
            "logical_edges": sorted(sorted((repr(u), repr(v))) for u, v in task.logical.edges()),
            "host_nodes": ordered(task.host.nodes()),
            "host_edges": sorted(sorted((repr(u), repr(v))) for u, v in task.host.edges()),
            "h": sorted((repr(v), float(h)) for v, h in task.problem.h.items()),
            "j": sorted((sorted((repr(u), repr(v))), float(j)) for (u, v), j in task.problem.j.items()),
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    options = vars(args).copy()
    options.pop("out", None)
    checkpoint_hash = (hashlib.sha256(Path(args.init).read_bytes()).hexdigest()
                       if args.init else None)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.SubprocessError):
        sha, dirty = None, None
    return {"version": "constructor-quality-v2", "source_commit": sha, "source_dirty": dirty,
            "initial_checkpoint_sha256": checkpoint_hash,
            "options": options, "train_fingerprints": [signature(t) for t in train],
            "heldout_fingerprints": [signature(t) for t in heldout]}


def validation_score(evaluation, objective):
    """Failures contribute zero to quality utility; no qubit or chain penalty."""
    held = evaluation["heldout"]
    if objective == "quality":
        return float(held["deployment_utility"]["policy"])
    if objective == "deployment":
        return float(held["coverage"]["policy"])
    return float(np.mean(list(held.values())))


class ValidationCheckpoint:
    def __init__(self, path, metadata, objective, heldout_role="validation"):
        self.path = Path(path) if path else None
        self.metadata = metadata
        self.objective, self.heldout_role = objective, heldout_role
        self.best_score = -float("inf")
        self.best_tag = None

    def consider(self, actor, evaluation, tag):
        if self.heldout_role != "validation":
            return False
        score = validation_score(evaluation, self.objective)
        if not np.isfinite(score):
            raise ValueError("checkpoint validation score must be finite")
        if score <= self.best_score:
            return False
        self.best_score, self.best_tag = score, tag
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            target = self.path.with_name(self.path.stem + ".best" + self.path.suffix)
            payload = dict(self.metadata, state=actor.state_dict(),
                           selected_on="validation", selection_metric=self.objective,
                           selected_tag=tag, selected_score=score)
            # Atomic replacement, so an interrupted write cannot destroy a good checkpoint.
            temporary = target.with_suffix(target.suffix + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(target)
        return True
