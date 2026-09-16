"""A label cache records what produced it, and a trainer refuses a cache that does not match.

The old key was the corpus path and a few integers, so two corpora at one path, a change of
schedule, or a change of compiler shared a cache silently. Provenance covers the corpus
contents, the Context fields that affect labels, and the source of the compiler, evaluator and
environment.
"""
import json
from pathlib import Path

import pytest

PROBES = Path(__file__).resolve().parents[2] / "probes"
SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(autouse=True)
def _paths(monkeypatch):
    monkeypatch.setenv("ISINGFOLD_SRC", str(SRC))
    monkeypatch.syspath_prepend(str(PROBES))
    monkeypatch.syspath_prepend(str(SRC))


def _corpus(tmp_path, text):
    d = tmp_path / "corpus"
    d.mkdir(exist_ok=True)
    (d / "instances.jsonl").write_text(text)
    return d


def test_provenance_changes_when_the_corpus_content_changes(tmp_path):
    from _context import host_context
    from _provenance import label_provenance

    ctx = host_context(120)
    a = label_provenance(_corpus(tmp_path, '{"name": "a"}\n'), ctx)
    b = label_provenance(_corpus(tmp_path, '{"name": "b"}\n'), ctx)
    assert a["corpus_sha256"] != b["corpus_sha256"]
    assert a["code_sha256"] == b["code_sha256"]


def test_provenance_changes_with_the_schedule_and_the_ratios(tmp_path):
    from _context import host_context
    from _provenance import label_provenance

    corpus = _corpus(tmp_path, '{"name": "a"}\n')
    registered = label_provenance(corpus, host_context(120))
    auto = label_provenance(corpus, host_context(120, beta_range=None))
    assert registered["context"]["beta_range"] == [0.1, 2.0]
    assert auto["context"]["beta_range"] is None
    assert registered != auto


def test_a_stale_cache_is_refused_and_a_matching_one_accepted(tmp_path):
    from _context import host_context
    from _provenance import CacheProvenanceError, check_provenance, label_provenance

    corpus = _corpus(tmp_path, '{"name": "a"}\n')
    ctx = host_context(120)
    blob = {"provenance": label_provenance(corpus, ctx)}
    check_provenance(blob, corpus, ctx)
    stale = {"provenance": label_provenance(_corpus(tmp_path, '{"name": "b"}\n'), ctx)}
    with pytest.raises(CacheProvenanceError):
        check_provenance(stale, corpus, ctx)
    with pytest.raises(CacheProvenanceError):
        check_provenance({"key": ["old"]}, corpus, ctx)
