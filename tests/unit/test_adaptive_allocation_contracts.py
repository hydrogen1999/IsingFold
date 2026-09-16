"""Adaptive read selection must preserve budgets, candidate support and assessment independence."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setenv("ISINGFOLD_SRC", str(ROOT / "src"))
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    monkeypatch.syspath_prepend(str(ROOT / "probes"))
    import adaptive_allocation
    return adaptive_allocation


@pytest.mark.parametrize("size,budget", [(4, 7), (5, 10), (6, 769), (7, 899), (8, 1024)])
def test_halving_uses_exact_equal_budget_on_full_and_prior_pools(probe, size, budget):
    for candidates in (list(range(size)), list(range((size + 1) // 2))):
        calls = []
        def sample(j, reads):
            calls.append(reads)
            return (reads if j == 0 else 0), reads
        pick, spent = probe.successive_halving(candidates, budget, sample)
        assert pick == 0
        assert spent == sum(calls) == budget
        assert all(reads > 0 for reads in calls)


def test_halving_rejects_insufficient_budget_before_measurement(probe):
    def sample(*args):
        pytest.fail("must reject before spending reads")
    with pytest.raises(ValueError, match="too small"):
        probe.successive_halving(range(8), 8, sample)


@pytest.mark.parametrize("failed", [False, True])
def test_ucb_consumes_requested_budget_including_failed_calls(probe, failed):
    calls = []
    def sample(j, reads):
        calls.append(reads)
        return None if failed else (reads if j == 0 else 0, reads)
    _, spent = probe.ucb_select(6, 101, 32, sample)
    assert spent == sum(calls) == 101
    assert max(calls) <= 32


def test_seed_namespaces_stay_disjoint_over_thousands_of_states(probe):
    seeds = probe.EvaluationSeeds()
    select = {seeds.selection() for _ in range(150_000)}
    assess = {seeds.assessment(state, candidate) for state in range(2_000) for candidate in range(8)}
    assert len(select) == 150_000
    assert len(assess) == 16_000
    assert not select & assess
    assert max(select | assess) < 2**31


def test_stale_override_accepts_only_code_hash_changes(probe, monkeypatch):
    current = {"code_sha256": "new", "corpus_sha256": "same", "context": {"beta_range": [.1, 2.]}}
    monkeypatch.setattr(probe, "label_provenance", lambda *args: current)
    probe.validate_cache({"provenance": current}, None, None)
    old = {**current, "code_sha256": "old"}
    probe.validate_cache({"provenance": old}, None, None, True)
    for recorded in (None, {**old, "corpus_sha256": "different"}, {**old, "context": {}}):
        with pytest.raises(probe.CacheProvenanceError):
            probe.validate_cache({"provenance": recorded}, None, None, True)
    with pytest.raises(probe.CacheProvenanceError):
        probe.validate_cache({"provenance": old}, None, None)


def test_prior_requires_saved_feature_contract_and_split(probe, tmp_path):
    path = tmp_path / "scorer.pt"
    meta = {"model": "SuccessorScorer", "coords": True, "physics": True,
            "space": False, "width": 32, "training_lineages": ["train"],
            "selection_lineages": ["validation"]}
    path.with_suffix(".json").write_text(json.dumps(meta))
    assert probe.prior_config(path) == meta
    with pytest.raises(ValueError, match="width"):
        probe.prior_config(path, width=64)
    probe.verify_prior_split(meta, [{"lineage": "test"}])
    for lineage in ("train", "validation"):
        with pytest.raises(ValueError, match="overlaps"):
            probe.verify_prior_split(meta, [{"lineage": lineage}], exploratory=True)
    with pytest.raises(ValueError, match="provenance"):
        probe.verify_prior_split({}, [{"lineage": "test"}])
    probe.verify_prior_split({}, [{"lineage": "test"}], exploratory=True)


def test_prior_cannot_silently_drop_or_reorder_candidates(probe):
    states = [{"state_id": "s", "rows": [{"succ": {0: {1}}}, {"succ": {0: {2}}}]}]
    probe.check_prior_rows(states, states)
    for built in ([], [{"state_id": "s", "rows": states[0]["rows"][:1]}],
                  [{"state_id": "s", "rows": states[0]["rows"][::-1]}]):
        with pytest.raises(ValueError, match="prior compilation"):
            probe.check_prior_rows(states, built)


def test_main_reports_exact_budgets_and_uses_fresh_shared_assessment(probe, monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace
    import pickle
    import sys

    current = {"corpus_sha256": "fixture", "code_sha256": "fixture", "context": {}}
    states = [{"lineage": "heldout", "state_id": "s", "task": SimpleNamespace(name="instance", lineage="heldout"),
               "rows": [{"succ": {0: {j}}} for j in range(5)]}]
    cache = tmp_path / "cache.pkl"
    cache.write_bytes(pickle.dumps({"key": ["unused"], "provenance": current, "eval": states}))
    monkeypatch.setattr(probe, "label_provenance", lambda *args: current)
    monkeypatch.setattr(probe, "host_context", lambda *args: SimpleNamespace(beta_range=(.1, 2.)))
    calls = []
    def run_controller(tasks, ctx, controller, *, initializer, reward_reads, seed, **kwargs):
        j = next(iter(initializer(None, None, None)[0]))
        calls.append((seed, reward_reads, j))
        return [SimpleNamespace(returned_valid=True, utility=float(j == 0),
                                evaluator_reads=reward_reads,
                                evaluator_hits=reward_reads if j == 0 else 0,
                                evaluator_seed=probe._derived_seed(seed, "final-evaluator", tasks[0], 0))]
    monkeypatch.setattr(probe, "run_controller", run_controller)
    monkeypatch.setattr(sys, "argv", ["adaptive_allocation", "--cache", str(cache),
                                    "--reads", "5", "--block", "32", "--assess-reads", "17"])
    assert probe.main() == 0
    selection = [r for r in calls if r[0] < 2**30]
    assessment = [r for r in calls if r[0] >= 2**30]
    assert sum(r[1] for r in selection) == 25 + 12 + 12
    assert len({r[0] for r in calls}) == len(calls)
    assert all(r[1] == 17 for r in assessment)
    assert len({r[2] for r in assessment}) == len(assessment)
    output = capsys.readouterr().out
    assert "vs halving" in output and "ADAPTIVE ALLOCATION DONE" in output


def test_prior_rejects_swapped_weights_with_same_architecture(probe, tmp_path):
    import hashlib
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"replacement")
    checkpoint.with_suffix(".json").write_text(json.dumps({
        "model": "SuccessorScorer", "width": 64, "coords": False,
        "physics": False, "space": False,
        "checkpoint_sha256": hashlib.sha256(b"original").hexdigest(),
    }))
    with pytest.raises(ValueError, match="fingerprint"):
        probe.prior_config(checkpoint)
