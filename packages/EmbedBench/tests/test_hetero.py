"""Heterogeneous scorer: encoding round trip, candidate conditioning, and the checkpoint path."""
import json, glob, os, random
import numpy as np, pytest


def _rec():
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "LAC", "runs", "chain2", "chimera5_rnd.jsonl")))
    if not files:
        pytest.skip("chain corpus not present")
    return json.loads(open(files[0]).readline())


def test_encode_shapes_and_conditioning():
    from embedbench.models_hetero import encode_hetero, QF, VF, EF
    r = _rec(); e = encode_hetero(r)
    assert e.xq.shape[1] == QF and e.xv.shape[1] == VF and e.Eqq.shape[2] == EF
    assert e.cand_masks.shape[0] == len(r["candidates"]) and e.cand_masks.sum(1).min() >= 1
    assert e.Mvq[e.focus_index].sum() == 0  # the focus row is filled per candidate
    assert all(len(c) >= 0 for c in e.cand_contacts)


def test_forward_is_invariant_to_qubit_relabelling():
    import torch
    from embedbench.models_hetero import encode_hetero, build_hetero
    r = _rec(); torch.manual_seed(0); m = build_hetero(); m.eval()
    s1 = m(encode_hetero(r)).detach().numpy()
    perm = {q: 100000 + k for k, q in enumerate(reversed(r["window_nodes"]))}
    def relabel(q): return perm.get(q, q)
    r2 = dict(r); r2["window_nodes"] = sorted(relabel(q) for q in r["window_nodes"]); r2["window_edges"] = [[relabel(a), relabel(b)] for a, b in r["window_edges"]]
    r2["candidates"] = [[relabel(q) for q in c] for c in r["candidates"]]
    r2["frozen_adjacency"] = {str(relabel(int(q))): v for q, v in r.get("frozen_adjacency", {}).items()}
    s2 = m(encode_hetero(r2)).detach().numpy()
    assert np.allclose(s1, s2, atol=1e-4)


def test_checkpoint_round_trip(tmp_path):
    import torch
    from embedbench.models_hetero import build_hetero, load_scorer, encode_hetero
    r = _rec(); m = build_hetero(); m.eval()
    torch.save({"state": m.state_dict(), "meta": {"arch": "hetero"}}, tmp_path / "h.pt")
    sc = load_scorer(str(tmp_path / "h.pt"))
    assert np.allclose(sc(r), m(encode_hetero(r)).detach().numpy(), atol=1e-6)
