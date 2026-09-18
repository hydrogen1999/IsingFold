"""Behaviour cloning: a set-valued teacher, train-only witnesses, evaluation from empty."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_p = Path(__file__).resolve().parents[2] / "probes" / "constructor_clone.py"
spec = importlib.util.spec_from_file_location("constructor_clone", _p)
cl = importlib.util.module_from_spec(spec); spec.loader.exec_module(cl)
import constructor_curriculum as cc  # noqa: E402


def test_clone_loss_is_the_negative_log_mass_of_the_good_rows():
    torch.manual_seed(0)
    actor = cc.make_actor("linear", 8, cc.FEATURE_WIDTHS["tiny"])
    rows = np.random.default_rng(0).normal(size=(6, cc.FEATURE_WIDTHS["tiny"])).astype(np.float32)
    for good in ([0], [1, 3], [0, 1, 2, 3, 4, 5]):
        loss = cl.clone_loss(actor, rows, good)
        with torch.no_grad():
            probs = torch.softmax(actor(torch.as_tensor(rows)), dim=0)
            expected = -torch.log(probs[torch.as_tensor(good)].sum())
        assert abs(float(loss) - float(expected)) < 1e-5
    all_good = cl.clone_loss(actor, rows, list(range(6)))
    assert abs(float(all_good)) < 1e-5


def test_the_loss_descends_on_a_fixed_state():
    torch.manual_seed(1)
    actor = cc.make_actor("linear", 8, cc.FEATURE_WIDTHS["tiny"])
    opt = torch.optim.Adam(actor.parameters(), lr=0.1)
    rows = np.random.default_rng(1).normal(size=(8, cc.FEATURE_WIDTHS["tiny"])).astype(np.float32)
    good = [2, 5]
    first = float(cl.clone_loss(actor, rows, good))
    for _ in range(20):
        opt.zero_grad(); loss = cl.clone_loss(actor, rows, good); loss.backward(); opt.step()
    assert float(cl.clone_loss(actor, rows, good)) < first / 2


def test_teacher_walks_a_small_instance_and_evaluation_never_sees_a_witness():
    train, heldout = cc.build_sets("b", 1, 1, seed=51)
    task = train[0]
    features = cc.make_features("tiny", task)
    witness = {v: frozenset(c) for v, c in task.prefix_source.items()}
    with cc.no_completion_solver():
        steps, reason, valid, progress, secs = cl.teacher_steps(task, witness, features, 200, 60.)
    assert steps and all(good for _, good in steps)
    assert reason in ("valid", "stuck", "terminal", "HORIZON", "DEADLINE")
    for rows, good in steps:
        assert rows.shape[1] == cc.FEATURE_WIDTHS["tiny"] and max(good) < len(rows)
    assert heldout[0].prefix_source is None
    with pytest.raises(AssertionError):
        heldout[0].witness


def test_cli_rejects_bad_arguments():
    for argv in (["--corpus", "x", "--epochs", "0"], ["--corpus", "x", "--learning-rate", "0"],
                 ["--corpus", "x", "--train", "0"]):
        with pytest.raises(SystemExit):
            cl.main(argv)
