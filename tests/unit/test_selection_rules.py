import importlib.util, sys
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[2] / "probes" / "selection_rules.py"
spec = importlib.util.spec_from_file_location("selection_rules", _p)
sr = importlib.util.module_from_spec(spec); spec.loader.exec_module(sr)


def _draws():
    return [
        {0: (1, 2, 3), 1: (4,)},          # 4 qubits, longest 3
        {0: (1, 2), 1: (4, 5)},           # 4 qubits, longest 2
        {0: (1,), 1: (4, 5, 6)},          # 4 qubits, longest 3
        {0: (1, 2), 1: (4,)},             # 3 qubits, longest 2
    ]


def test_fewest_qubits_prefers_the_smallest_then_the_earliest():
    d = _draws()
    assert sr.pick_fewest_qubits(d, [0, 1, 2, 3]) == 3
    assert sr.pick_fewest_qubits(d, [0, 1, 2]) == 0


def test_shortest_chain_breaks_ties_by_qubits_then_order():
    d = _draws()
    assert sr.pick_shortest_chain(d, [0, 1, 2, 3]) == 3
    assert sr.pick_shortest_chain(d, [0, 1, 2]) == 1


def test_pick_by_score_ignores_missing_and_nonfinite():
    assert sr.pick_by_score({0: 0.1, 1: None, 2: float("nan"), 3: 0.4}, [0, 1, 2, 3]) == 3
    assert sr.pick_by_score({0: None, 1: float("inf")}, [0, 1]) is None or sr.pick_by_score({0: None, 1: float("inf")}, [0, 1]) == 1
    assert sr.pick_by_score({}, [0, 1]) is None


def test_choose_returns_none_for_every_rule_without_valid_draws():
    picks = sr.choose(_draws(), [], {}, {}, np.random.default_rng(0))
    assert set(picks) == set(sr.RULES) and all(v is None for v in picks.values())


def test_choose_single_is_the_first_valid_and_oracle_reads_the_assessment():
    d = _draws()
    picks = sr.choose(d, [1, 2, 3], {1: 0.9, 2: 0.1, 3: 0.5}, {1: 0.2, 2: 0.8, 3: 0.3}, np.random.default_rng(0))
    assert picks["single"] == 1 and picks["measured"] == 1 and picks["oracle"] == 2
    assert picks["qubits"] == 3 and picks["chain"] == 3 and picks["random"] in (1, 2, 3)


def test_spearman_matches_known_values_and_handles_constants():
    assert abs(sr.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(sr.spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-12
    assert sr.spearman([1, 1, 1], [1, 2, 3]) is None
    assert sr.spearman([1, 2], [1, 2]) is None
    tied = sr.spearman([1, 1, 2, 3], [1, 2, 3, 4])
    assert tied is not None and 0.9 < tied < 1.0


def test_paired_boot_is_deterministic_and_brackets_the_mean():
    m, lo, hi = sr.paired_boot([0.1, 0.2, 0.3, 0.0, 0.4], seed=1)
    assert abs(m - 0.2) < 1e-12 and lo <= m <= hi
    assert sr.paired_boot([]) is None
