"""What a label cache was built from, so a trainer can refuse one that no longer matches.

A cache key that is a path and a few integers lets two corpora at one path, a change of
annealing schedule, or a change of compiler share labels silently; the GPT-6 astra review
lists this among the routes to a favourable artefact. Provenance here covers the corpus
contents, the Context fields that affect a label, and the source of the modules that turn an
embedding into a measured number.
"""
import hashlib
import json
import os
from pathlib import Path

CODE_FILES = ("rl/program.py", "rl/evaluator.py", "rl/env.py", "rl/evaluate.py")


class CacheProvenanceError(RuntimeError):
    pass


def _sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def label_provenance(corpus, ctx) -> dict:
    corpus = Path(corpus)
    src = Path(os.environ["ISINGFOLD_SRC"]) / "isingfold"
    return {
        "corpus_sha256": _sha([corpus / "instances.jsonl"]),
        "context": {
            "qubit_cap": int(ctx.qubit_cap),
            "beta_range": list(ctx.beta_range) if ctx.beta_range is not None else None,
            "strength_ratios": list(ctx.strength_ratios),
            "epsilon_strength": float(ctx.epsilon_strength),
        },
        "code_sha256": _sha([src / f for f in CODE_FILES]),
    }


def check_provenance(blob, corpus, ctx) -> None:
    """Raise unless the cache records the provenance the caller would produce now."""
    recorded = blob.get("provenance")
    if recorded is None:
        raise CacheProvenanceError("cache carries no provenance; it predates ADR-001/ADR-002 "
                                   "and its observations do not match its labels")
    current = label_provenance(corpus, ctx)
    if recorded != current:
        raise CacheProvenanceError("cache provenance differs from the current corpus, context "
                                   "or code:\n  recorded %s\n  current  %s"
                                   % (json.dumps(recorded, sort_keys=True),
                                      json.dumps(current, sort_keys=True)))
