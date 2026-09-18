"""Warm starts must preserve training ancestry across successive constructor runs."""
import hashlib
from types import SimpleNamespace

import pytest
import torch

from constructor_provenance import training_provenance


def checkpoint(tmp_path, metadata, name="initial.pt"):
    path = tmp_path / name
    torch.save(metadata, path)
    return path


def test_fresh_training_tracks_lineages_and_falls_back_to_task_names():
    train = [SimpleNamespace(lineage="train-a", name="a-variant"),
             SimpleNamespace(lineage="train-a", name="a-other-variant"),
             SimpleNamespace(lineage=None, name="unlineaged-task")]
    result = training_provenance(None, train, ["heldout"])
    assert result["training_lineages"] == ["train-a", "unlineaged-task"]
    assert result["complete"] is True
    assert result["initial_checkpoint_sha256"] is None


@pytest.mark.parametrize("metadata", [
    {"training_provenance": {"training_lineages": ["reserved"], "complete": True}},
    {"training_provenance": {"training_lineages": ["reserved"], "complete": False}},
    {"contract": {"train_lineages": ["reserved"]}},
    {"training_lineages": ["reserved"]},
])
@pytest.mark.parametrize("training", [True, False])
def test_known_inherited_overlap_is_rejected_even_for_incomplete_history_or_eval(
        tmp_path, metadata, training):
    initial = checkpoint(tmp_path, metadata)
    with pytest.raises(ValueError, match="initial checkpoint.*reserved"):
        training_provenance(initial, ["fresh"], ["reserved"], training=training)


@pytest.mark.parametrize("reservation", ["heldout", "forbidden"])
def test_current_training_cannot_use_either_reserved_pool(reservation):
    heldout = ["reserved"] if reservation == "heldout" else []
    forbidden = ["reserved"] if reservation == "forbidden" else []
    with pytest.raises(ValueError, match="current training.*reserved"):
        training_provenance(None, ["reserved"], heldout, forbidden_lineages=forbidden)


def test_inherited_training_cannot_use_explicitly_forbidden_lineages(tmp_path):
    initial = checkpoint(tmp_path, {"train_lineages": ["test-lineage"]})
    with pytest.raises(ValueError, match="initial checkpoint.*test-lineage"):
        training_provenance(initial, ["fresh"], [], forbidden_lineages=["test-lineage"])


def test_known_ancestry_accumulates_and_binds_the_actual_checkpoint(tmp_path):
    initial = checkpoint(tmp_path, {
        "training_provenance": {"training_lineages": ["ancestor"], "complete": True},
        "contract": {"train_lineages": ["previous-stage"]},
    })
    result = training_provenance(initial, ["fresh", "ancestor"], ["heldout"])
    assert result["training_lineages"] == ["ancestor", "fresh", "previous-stage"]
    assert result["complete"] is True
    assert result["initial_checkpoint_sha256"] == hashlib.sha256(initial.read_bytes()).hexdigest()


def test_unknown_ancestry_survives_two_further_training_stages(tmp_path):
    legacy = checkpoint(tmp_path, {"state": {}}, "legacy.pt")
    first = training_provenance(legacy, ["stage-one"], ["heldout"])
    assert first["training_lineages"] == ["stage-one"]
    assert first["complete"] is False
    descendant = checkpoint(tmp_path, {"training_provenance": first}, "descendant.pt")
    second = training_provenance(descendant, ["stage-two"], ["heldout"])
    assert second["training_lineages"] == ["stage-one", "stage-two"]
    assert second["complete"] is False
    with pytest.raises(ValueError, match="initial checkpoint.*stage-one"):
        training_provenance(descendant, [], ["stage-one"], training=False)


def test_evaluation_does_not_claim_training_on_the_supplied_current_pool(tmp_path):
    initial = checkpoint(tmp_path, {
        "training_provenance": {"training_lineages": ["trained"], "complete": True}})
    result = training_provenance(initial, ["heldout", "unused"], ["heldout"], training=False)
    assert result["training_lineages"] == ["trained"]
    assert result["complete"] is True
    fresh = training_provenance(None, ["heldout"], ["heldout"], training=False)
    assert fresh["training_lineages"] == []


@pytest.mark.parametrize("complete", [True, False])
def test_saved_eval_only_pool_is_not_training_but_real_history_is_preserved(tmp_path, complete):
    initial = checkpoint(tmp_path, {
        "contract": {"train_lineages": ["declared-unused"],
                     "options": {"iterations": 0, "init": "ancestor.pt"}},
        "training_provenance": {"training_lineages": ["actually-trained"], "complete": complete},
    })
    result = training_provenance(initial, ["new-train"], ["declared-unused"])
    assert result["training_lineages"] == ["actually-trained", "new-train"]
    assert result["complete"] is complete
    with pytest.raises(ValueError, match="initial checkpoint.*actually-trained"):
        training_provenance(initial, [], ["actually-trained"], training=False)


@pytest.mark.parametrize("init", ["", "ancestor.pt"])
def test_legacy_eval_only_contract_pool_cannot_certify_missing_history(tmp_path, init):
    initial = checkpoint(tmp_path, {
        "contract": {"train_lineages": ["declared-unused"],
                     "options": {"iterations": 0, "init": init}},
    })
    result = training_provenance(initial, [], ["declared-unused"], training=False)
    assert result["training_lineages"] == []
    assert result["complete"] is False


@pytest.mark.parametrize("contract, complete", [
    ({"train_lineages": ["known"]}, True),
    ({"train_lineages": ["known"], "options": {"init": ""}}, True),
    ({"train_lineages": ["known"], "options": {"init": "ancestor.pt"}}, False),
    ({"train_lineages": ["known"], "initial_checkpoint_sha256": "ancestor-hash"}, False),
    ({"options": {"init": ""}}, False),
])
def test_legacy_contract_only_certifies_ancestry_when_it_accounts_for_initialization(
        tmp_path, contract, complete):
    initial = checkpoint(tmp_path, {"contract": contract})
    result = training_provenance(initial, ["new-stage"], ["heldout"])
    assert result["complete"] is complete
    expected = ["known", "new-stage"] if "train_lineages" in contract else ["new-stage"]
    assert result["training_lineages"] == expected


@pytest.mark.parametrize("verified", [True, False])
def test_legacy_rl_explicit_verification_marker_is_preserved(tmp_path, verified):
    # train_constructor_rl already emits this marker when legacy ancestry is imported.
    initial = checkpoint(tmp_path, {
        "training_lineages": ["previous-stage"],
        "lineage_provenance_verified": verified,
        "init_lineage_provenance_verified": verified,
    })
    result = training_provenance(initial, ["new-stage"], ["heldout"])
    assert result["training_lineages"] == ["new-stage", "previous-stage"]
    assert result["complete"] is verified


@pytest.mark.parametrize("metadata", [
    {"training_lineages": "not-a-list"},
    {"contract": {"train_lineages": [1]}},
    {"training_provenance": {"training_lineages": [None], "complete": True}},
    {"training_provenance": ["not-an-object"]},
])
def test_malformed_lineage_metadata_is_rejected(tmp_path, metadata):
    initial = checkpoint(tmp_path, metadata)
    with pytest.raises(ValueError, match="provenance|lineages"):
        training_provenance(initial, ["fresh"], ["heldout"])
