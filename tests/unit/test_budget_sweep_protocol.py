"""A noisy winning growth draw must not report its own selection noise as quality."""
import json
import sys
from types import SimpleNamespace

import pytest


def test_selected_growth_is_assessed_on_fresh_reads():
    from budget_sweep import select_and_assess

    calls = []
    outcomes = iter([0.95, 0.80, 0.20])

    def measure(candidate, reads):
        calls.append((candidate, reads))
        return next(outcomes)

    result = select_and_assess(["A", "B"], measure,
                               selection_reads=32, assessment_reads=512)
    assert result == ("A", 0.20)
    assert calls == [("A", 32), ("B", 32), ("A", 512)]


def test_single_draw_skips_selection_and_failed_assessment_is_not_replaced_by_pilot():
    from budget_sweep import select_and_assess

    calls = []

    def measure(candidate, reads):
        calls.append((candidate, reads))
        return None if reads == 512 else 0.9

    assert select_and_assess(["A"], measure,
                             selection_reads=32, assessment_reads=512) is None
    assert calls == [("A", 512)]
    calls.clear()
    assert select_and_assess(["A", "B"], measure,
                             selection_reads=32, assessment_reads=512) is None
    assert calls[-1] == ("A", 512)


def test_sweep_reports_independent_quality_and_unambiguous_provenance(monkeypatch, capsys):
    import budget_sweep as sweep

    task = SimpleNamespace(logical=object(), host=object())
    base = {0: frozenset({0})}
    monkeypatch.setattr(sweep, "load_instances", lambda _: [task] * 5)
    monkeypatch.setattr(sweep, "host_context", lambda _: SimpleNamespace(beta_range=(0.1, 2.0)))
    monkeypatch.setattr(sweep, "minorminer_initializer", lambda _: lambda *args: base)
    monkeypatch.setattr(sweep, "grown_embedding", lambda *args, **kw: ({0: frozenset({0, 1})}, 0))
    monkeypatch.setattr(sweep, "realises", lambda *args: True)
    measurements = []

    def run_controller(tasks, ctx, controller, **kw):
        chains = kw["initializer"](None, None, None)
        measurements.append((kw["seed"], kw["reward_reads"], chains))
        # Pilots look excellent, independent grown assessment is worse than the start.
        utility = 0.95 if kw["reward_reads"] == 32 else (0.4 if chains == base else 0.2)
        return [SimpleNamespace(returned_valid=True, utility=utility)]

    monkeypatch.setattr(sweep, "run_controller", run_controller)
    monkeypatch.setattr(sys, "argv", ["budget_sweep", "--corpus", "unused", "--alphas", "2",
                                     "--modes", "length", "--repeats", "2",
                                     "--selection-reads", "32", "--reads", "512"])
    assert sweep.main() == 0
    output = capsys.readouterr().out
    provenance = json.loads(output.splitlines()[0])
    assert provenance["beta_range"] == [0.1, 2.0]
    assert provenance["selection_reads"] == 32
    assert provenance["assessment_reads"] == 512
    assert "independent_assessment" in provenance["protocol"]
    assert len(measurements) == 20  # base + two pilots + one grown assessment per instance
    assert len({seed for seed, _, _ in measurements}) == len(measurements)
    row = next(line for line in output.splitlines() if line.strip().startswith("length "))
    cells = row.split()
    assert cells[:6] == ["length", "2.0", "2.0", "2.00", "2.00", "0.2000"]
    assert float(cells[6]) == pytest.approx(-0.2)
    assert "rows are not nested" in output


@pytest.mark.parametrize("option", ["--reads", "--selection-reads", "--repeats"])
def test_nonpositive_measurement_budget_is_rejected(monkeypatch, option):
    import budget_sweep as sweep

    monkeypatch.setattr(sys, "argv", ["budget_sweep", "--corpus", "unused", option, "0"])
    with pytest.raises(SystemExit) as exc:
        sweep.main()
    assert exc.value.code == 2
