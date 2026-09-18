"""Track cumulative constructor training lineages across warm-start checkpoints."""
import hashlib
from pathlib import Path

import torch


def _lineages(tasks):
    return {t if isinstance(t, str) else t.lineage or t.name for t in tasks}


def training_provenance(init_path, current_train, heldout, *, training=True,
                        forbidden_lineages=()):
    """Reject known train/holdout overlap and carry unknown ancestry forward.

    Task sequences or explicit lineage strings are accepted. ``training=False``
    preserves the initialization's history for evaluation-only runs without claiming
    that the current train pool updated the model. ``complete`` is false for legacy
    checkpoints whose complete training ancestry cannot be recovered.
    """
    known, complete, checkpoint_hash = set(), True, None
    if init_path:
        blob = torch.load(init_path, map_location="cpu", weights_only=False)
        contract = blob.get("contract") or {}
        provenance = blob.get("training_provenance")
        if not isinstance(contract, dict) or (provenance is not None
                                              and not isinstance(provenance, dict)):
            raise ValueError("checkpoint training provenance must be an object")
        options = contract.get("options") or {}
        evaluation_only = options.get("iterations") == 0
        found_history = False
        for source in (blob, contract, provenance or {}):
            for key in ("training_lineages", "train_lineages"):
                # run_contract records its declared data pools even on an
                # evaluation-only pass. That pool did not update the model;
                # cumulative training_provenance remains authoritative history.
                if source is contract and key == "train_lineages" and evaluation_only:
                    continue
                if key in source:
                    values = source[key]
                    if (not isinstance(values, (list, tuple))
                            or any(not isinstance(v, str) for v in values)):
                        raise ValueError("checkpoint training lineages must be a list of strings")
                    known.update(values)
                    found_history = True
        if provenance is not None:
            complete = (provenance.get("complete") is True
                        and "training_lineages" in provenance)
        else:
            # A direct lineage list covers this run only. If it used a warm start,
            # inherited lineage metadata is needed to certify the full ancestry.
            complete = (found_history
                        and not contract.get("initial_checkpoint_sha256")
                        and not options.get("init"))
        # Older constructor trainers already persisted an explicit ancestry flag.
        # Never turn an explicitly unverified history into a verified one.
        if blob.get("lineage_provenance_verified") is False:
            complete = False
        checkpoint_hash = hashlib.sha256(Path(init_path).read_bytes()).hexdigest()

    reserved = _lineages(heldout) | _lineages(forbidden_lineages)
    overlap = sorted(known & reserved)
    if overlap:
        raise ValueError("initial checkpoint training lineages overlap held-out lineages: "
                         + ", ".join(overlap))
    if training:
        current = _lineages(current_train)
        if current & reserved:
            raise ValueError("current training lineages overlap held-out lineages: "
                             + ", ".join(sorted(current & reserved)))
        known.update(current)
    return {"training_lineages": sorted(known), "complete": bool(complete),
            "initial_checkpoint_sha256": checkpoint_hash}
