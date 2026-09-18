"""Exercise the v4 CLI with a deterministic toy backend, never an annealing run."""
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest


@pytest.fixture(autouse=True)
def _paths(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("ISINGFOLD_SRC", str(root / "src"))
    monkeypatch.syspath_prepend(str(root / "probes"))
    monkeypatch.syspath_prepend(str(root / "src"))


def test_contextual_training_checkpoint_and_no_validation_witness(monkeypatch, tmp_path, capsys):
    import torch
    import train_hybrid_rl as hybrid

    class Task:
        def __init__(self, index, training):
            self.name = f"toy-fill80-s{index}"
            self.lineage = f"toy-fill80-medium-s{index}"
            self.logical, self.host = nx.path_graph(2), nx.path_graph(5)
            self.problem = SimpleNamespace(h={0: 0.1}, j={(0, 1): -1.0})
            self.training = training

        @property
        def witness(self):
            assert self.training, "validation must not read a witness"
            return {0: {0, 1}, 1: {2}}

    tasks = [Task(0, True), Task(1, False)]
    monkeypatch.setattr(hybrid, "load_instances", lambda _: tasks)
    monkeypatch.setattr(hybrid, "split_by_lineage", lambda *_: (tasks[:1], tasks[1:]))

    def complete(task, roots, *args, **kwargs):
        complete.last = roots
        return True, 0.001, 1

    monkeypatch.setattr(hybrid, "complete", complete)
    monkeypatch.setattr(hybrid, "measure_residual", lambda task, chains, *args, **kw:
                        sum(sum(c) for c in chains.values()) / 100)
    proposals = []

    def evaluate(task, propose, **kwargs):
        if propose is not None:
            roots = propose(1)
            assert set(roots) == set(task.logical)
            assert len(set.union(*(set(c) for c in roots.values()))) == 2
            proposals.append(roots)
        return {"valid": True, "residual": 0.05, "attempts": 1, "candidates": 1}

    monkeypatch.setattr(hybrid, "evaluate_arm", evaluate)
    output = tmp_path / "model.pt"
    monkeypatch.setattr("sys.argv", ["train_hybrid_rl", "--corpus", "fake", "--out", str(output),
                                   "--fast", "--actor", "contextual", "--features", "capacity",
                                   "--root-support", "all_free", "--baseline", "value",
                                   "--warmstart-epochs", "1", "--iterations", "1", "--eval-every", "1",
                                   "--width", "8", "--objective", "quality", "--entropy-coef", "0.01"])
    assert hybrid.main() == 0
    assert len(proposals) == 2
    checkpoint = torch.load(str(output) + ".last", weights_only=True)
    assert checkpoint["model_spec"] == {"actor": "contextual", "feature_version": "layout-v2",
                                        "in_dim": 65, "width": 8}
    assert all(torch.isfinite(x).all() for x in checkpoint["state"].values())
    log = capsys.readouterr().out
    assert '"evaluation": "post_warmstart"' in log
    assert '"gradient_norm_before_clip"' in log and '"teacher_coverage": 1.0' in log

    # Feature semantics cannot silently change under a checkpoint.
    monkeypatch.setattr("sys.argv", ["train_hybrid_rl", "--corpus", "fake", "--out", str(output),
                                   "--init", str(output), "--fast", "--width", "8"])
    with pytest.raises(SystemExit):
        hybrid.main()


@pytest.mark.parametrize("options", [
    ["--baseline", "value"], ["--features", "capacity"], ["--entropy-coef", "nan"],
    ["--learning-rate", "0"], ["--warmstart-epochs", "-1"],
])
def test_invalid_v4_cli_contract_rejected_before_reading_corpus(monkeypatch, options):
    import train_hybrid_rl as hybrid
    monkeypatch.setattr(hybrid, "load_instances", lambda _: pytest.fail("invalid config read data"))
    monkeypatch.setattr("sys.argv", ["train_hybrid_rl", "--corpus", "fake", "--out", "unused", *options])
    with pytest.raises(SystemExit):
        hybrid.main()
