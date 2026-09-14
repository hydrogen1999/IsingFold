"""Adversarial regression cases for the action-Q warm-start slice."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from isingfold.rl.cli import (
    _load_transfer,
    _model_identity,
    _representation_checkpoint_payload_digest,
)
from isingfold.rl.model import IFCore
from isingfold.rl.ppo import (
    WARM_START_ACTION_VALUE_COEFFICIENT,
    WARM_START_ACTION_VALUE_TARGET,
    WARM_START_ACTION_VALUE_WEIGHTING,
    WARM_START_ACTOR_CRITIC_LOSS,
    WARM_START_COMMIT_DELTA_COEFFICIENT,
    WARM_START_COMMIT_DELTA_TARGET,
    WARM_START_FULL_LOSS_PROFILE,
    WARM_START_RANK_COEFFICIENT,
    WARM_START_RANK_VALUE_CONTROL_PROFILE,
    WARM_START_REDUCTION,
    WARM_START_UTILITY_COEFFICIENT,
    WARM_START_UTILITY_TARGET,
    WARM_START_UTILITY_WEIGHTING,
    WarmStartActionValueTarget,
    WarmStartCorpusDenominators,
    warm_start_actor_critic_loss,
)
from tests.unit.test_rl_ifcore_normative import _conflict_observation

torch = pytest.importorskip("torch")


def test_action_q_uses_count_over_propensity_then_equal_state_reduction() -> None:
    torch.manual_seed(260915)
    model = IFCore(improvement_mode=True).eval()
    observation = _conflict_observation()
    target = WarmStartActionValueTarget(
        action_indices=(0, 1),
        q_mu=(0.2, 0.8),
        continuation_counts=(2, 8),
        inclusion_probabilities=(0.25, 0.5),
        legal_action_count=2,
    )
    output = model([observation])
    logits = output.action_quality_logit[0, :2]
    elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        torch.tensor((0.2, 0.8)),
        reduction="none",
    )
    # C/pi gives weights 8 and 16.  Their sum normalizes inside this state.
    expected = (8.0 * elementwise[0] + 16.0 * elementwise[1]) / 24.0

    losses = warm_start_actor_critic_loss(
        model,
        [observation],
        [(0, 1)],
        [(0, 1)],
        [0.5],
        [10.0],
        action_value_targets=[target],
        corpus_denominators=WarmStartCorpusDenominators(
            actor_ranking_records=0,
            utility_effective_count=10.0,
            action_value_records=1,
            commit_delta_records=0,
        ),
    )

    torch.testing.assert_close(losses.action_value, expected)
    assert losses.action_effective_count == pytest.approx(24.0)


def _warm_contract(
    model: IFCore,
    *,
    quality_authority: object,
    target_access: object,
    ground_partition_receipt: object,
    top_preflight_sha256: str,
    top_preflight_digest: str,
    hyper_preflight_sha256: str,
    hyper_preflight_digest: str,
) -> dict[str, object]:
    return {
        "phase": "representation",
        "method": "supervised-ranking",
        "model_family": "if-core",
        "model": _model_identity(model, "if-core"),
        "seed": 1103,
        "grid_cell": "repr-000-if-core-s1103",
        "grid_manifest_sha256": "d" * 64,
        "corpus_manifest_sha256": "a" * 64,
        "quality_authority": quality_authority,
        "target_access": target_access,
        "ground_partition_receipt": ground_partition_receipt,
        "selector_digest": "b" * 64,
        "normalizer_digest": "c" * 64,
        "quality_preflight_receipt_sha256": top_preflight_sha256,
        "quality_preflight_record_digest": top_preflight_digest,
        "hyperparameters": {
            "warm_start_loss": WARM_START_ACTOR_CRITIC_LOSS,
            "warm_start_loss_profile": WARM_START_FULL_LOSS_PROFILE.contract(),
            "warm_start_rank_coefficient": WARM_START_RANK_COEFFICIENT,
            "warm_start_action_value_target": WARM_START_ACTION_VALUE_TARGET,
            "warm_start_action_value_weighting": WARM_START_ACTION_VALUE_WEIGHTING,
            "warm_start_action_value_coefficient": WARM_START_ACTION_VALUE_COEFFICIENT,
            "warm_start_commit_delta_target": WARM_START_COMMIT_DELTA_TARGET,
            "warm_start_commit_delta_coefficient": WARM_START_COMMIT_DELTA_COEFFICIENT,
            "warm_start_reduction": WARM_START_REDUCTION,
            "warm_start_utility_target": WARM_START_UTILITY_TARGET,
            "warm_start_utility_weighting": WARM_START_UTILITY_WEIGHTING,
            "warm_start_utility_coefficient": WARM_START_UTILITY_COEFFICIENT,
            "warm_start_diagnostic_binding": None,
            "warm_start_actor_ranking_records": 1,
            "warm_start_utility_critic_records": 1,
            "warm_start_action_value_records": 1,
            "warm_start_commit_delta_records": 1,
            "warm_start_corpus_denominators": {
                "actor_ranking_records": 1,
                "utility_effective_count": 1.0,
                "action_value_records": 1,
                "commit_delta_records": 1,
            },
            "quality_preflight_receipt_sha256": hyper_preflight_sha256,
            "quality_preflight_record_digest": hyper_preflight_digest,
        },
    }


def _exercise_untrusted_transfer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    contract: dict[str, object],
    bundle: object,
    expected_checkpoint_payload_digest: str = "9" * 64,
) -> None:
    record = {
        "schema": "isingfold.training-run",
        "schema_version": 4,
        "phase": "representation",
        "model_family": "if-core",
        "method": "supervised-ranking",
        "complete": True,
        "experiment_contract": contract,
        "feature_normalizer": bundle.normalizer,
        "training_lineages": ["train-lineage"],
        "selector_digest": bundle.selector_digest,
        "proposal_version": "proposal-v1",
        "runtime_implementation_digest": "f" * 64,
        "checkpoint_payload_digest": "9" * 64,
    }
    monkeypatch.setattr("isingfold.rl.cli._strict_json", lambda _path: record)
    monkeypatch.setattr("isingfold.rl.cli._verify_record", lambda *_args: None)
    monkeypatch.setattr(
        "isingfold.rl.cli._corpus_manifest_digest", lambda _corpus: "a" * 64
    )
    monkeypatch.setattr("isingfold.rl.cli._proposal_version", lambda: "proposal-v1")

    def forbidden_checkpoint_load(*_args, **_kwargs):
        raise AssertionError("untrusted warm artifact reached checkpoint deserialization")

    monkeypatch.setattr(
        "isingfold.rl.checkpoint.load_checkpoint", forbidden_checkpoint_load
    )
    _load_transfer(
        tmp_path / "checkpoint.pt",
        model=IFCore(improvement_mode=True),
        bundle=bundle,
        corpus=tmp_path / "prepared",
        model_family="if-core",
        expected_seed=1103,
        expected_grid_cell="repr-000-if-core-s1103",
        expected_grid_manifest_sha256="d" * 64,
        expected_quality_preflight_receipt_sha256="5" * 64,
        expected_quality_preflight_record_digest="6" * 64,
        expected_checkpoint_payload_digest=expected_checkpoint_payload_digest,
    )


def test_transfer_rejects_a_different_quality_authority_before_checkpoint_load(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = IFCore(improvement_mode=True)
    bundle = SimpleNamespace(
        selector_digest="b" * 64,
        normalizer={},
        normalizer_digest="c" * 64,
        quality_authority={"record_digest": "1" * 64},
        target_access={"record_digest": "2" * 64},
        ground_partition_receipt={"record_digest": "3" * 64},
    )
    contract = _warm_contract(
        model,
        quality_authority={"record_digest": "4" * 64},
        target_access=bundle.target_access,
        ground_partition_receipt=bundle.ground_partition_receipt,
        top_preflight_sha256="5" * 64,
        top_preflight_digest="6" * 64,
        hyper_preflight_sha256="5" * 64,
        hyper_preflight_digest="6" * 64,
    )

    with pytest.raises(ValueError, match="quality authorit"):
        _exercise_untrusted_transfer(
            monkeypatch,
            tmp_path,
            contract=contract,
            bundle=bundle,
        )


def test_transfer_rejects_inconsistent_quality_preflight_pins_before_load(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = IFCore(improvement_mode=True)
    bundle = SimpleNamespace(
        selector_digest="b" * 64,
        normalizer={},
        normalizer_digest="c" * 64,
        quality_authority={"record_digest": "1" * 64},
        target_access={"record_digest": "2" * 64},
        ground_partition_receipt={"record_digest": "3" * 64},
    )
    contract = _warm_contract(
        model,
        quality_authority=bundle.quality_authority,
        target_access=bundle.target_access,
        ground_partition_receipt=bundle.ground_partition_receipt,
        top_preflight_sha256="5" * 64,
        top_preflight_digest="6" * 64,
        hyper_preflight_sha256="7" * 64,
        hyper_preflight_digest="8" * 64,
    )

    with pytest.raises(ValueError, match="preflight"):
        _exercise_untrusted_transfer(
            monkeypatch,
            tmp_path,
            contract=contract,
            bundle=bundle,
        )


def test_transfer_rejects_inconsistent_corpus_denominators_before_load(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = IFCore(improvement_mode=True)
    bundle = SimpleNamespace(
        selector_digest="b" * 64,
        normalizer={},
        normalizer_digest="c" * 64,
        quality_authority={"record_digest": "1" * 64},
        target_access={"record_digest": "2" * 64},
        ground_partition_receipt={"record_digest": "3" * 64},
    )
    contract = _warm_contract(
        model,
        quality_authority=bundle.quality_authority,
        target_access=bundle.target_access,
        ground_partition_receipt=bundle.ground_partition_receipt,
        top_preflight_sha256="5" * 64,
        top_preflight_digest="6" * 64,
        hyper_preflight_sha256="5" * 64,
        hyper_preflight_digest="6" * 64,
    )
    contract["hyperparameters"]["warm_start_corpus_denominators"][
        "action_value_records"
    ] = 2

    with pytest.raises(ValueError, match="corpus denominators"):
        _exercise_untrusted_transfer(
            monkeypatch,
            tmp_path,
            contract=contract,
            bundle=bundle,
        )


@pytest.mark.parametrize(
    ("contract_field", "bad_value", "message"),
    [
        ("seed", 2207, "seed"),
        ("grid_cell", "repr-003-if-core-s2207", "grid cell"),
        ("grid_manifest_sha256", "e" * 64, "grid manifest"),
        ("quality_preflight_receipt_sha256", "7" * 64, "quality preflight"),
        ("quality_preflight_record_digest", "8" * 64, "quality preflight"),
    ],
)
def test_transfer_rejects_wrong_external_provenance_before_checkpoint_load(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    contract_field: str,
    bad_value: object,
    message: str,
) -> None:
    model = IFCore(improvement_mode=True)
    bundle = SimpleNamespace(
        selector_digest="b" * 64,
        normalizer={},
        normalizer_digest="c" * 64,
        quality_authority={"record_digest": "1" * 64},
        target_access={"record_digest": "2" * 64},
        ground_partition_receipt={"record_digest": "3" * 64},
    )
    contract = _warm_contract(
        model,
        quality_authority=bundle.quality_authority,
        target_access=bundle.target_access,
        ground_partition_receipt=bundle.ground_partition_receipt,
        top_preflight_sha256="5" * 64,
        top_preflight_digest="6" * 64,
        hyper_preflight_sha256="5" * 64,
        hyper_preflight_digest="6" * 64,
    )
    contract[contract_field] = bad_value

    with pytest.raises(ValueError, match=message):
        _exercise_untrusted_transfer(
            monkeypatch,
            tmp_path,
            contract=contract,
            bundle=bundle,
        )


def test_transfer_rejects_wrong_selected_payload_before_checkpoint_load(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = IFCore(improvement_mode=True)
    bundle = SimpleNamespace(
        selector_digest="b" * 64,
        normalizer={},
        normalizer_digest="c" * 64,
        quality_authority={"record_digest": "1" * 64},
        target_access={"record_digest": "2" * 64},
        ground_partition_receipt={"record_digest": "3" * 64},
    )
    contract = _warm_contract(
        model,
        quality_authority=bundle.quality_authority,
        target_access=bundle.target_access,
        ground_partition_receipt=bundle.ground_partition_receipt,
        top_preflight_sha256="5" * 64,
        top_preflight_digest="6" * 64,
        hyper_preflight_sha256="5" * 64,
        hyper_preflight_digest="6" * 64,
    )

    with pytest.raises(ValueError, match="selected checkpoint payload"):
        _exercise_untrusted_transfer(
            monkeypatch,
            tmp_path,
            contract=contract,
            bundle=bundle,
            expected_checkpoint_payload_digest="8" * 64,
        )


def test_transfer_rejects_diagnostic_q_ablation_before_checkpoint_load(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = IFCore(improvement_mode=True)
    bundle = SimpleNamespace(
        selector_digest="b" * 64,
        normalizer={},
        normalizer_digest="c" * 64,
        quality_authority={"record_digest": "1" * 64},
        target_access={"record_digest": "2" * 64},
        ground_partition_receipt={"record_digest": "3" * 64},
    )
    contract = _warm_contract(
        model,
        quality_authority=bundle.quality_authority,
        target_access=bundle.target_access,
        ground_partition_receipt=bundle.ground_partition_receipt,
        top_preflight_sha256="5" * 64,
        top_preflight_digest="6" * 64,
        hyper_preflight_sha256="5" * 64,
        hyper_preflight_digest="6" * 64,
    )
    hyperparameters = contract["hyperparameters"]
    hyperparameters["warm_start_loss_profile"] = (
        WARM_START_RANK_VALUE_CONTROL_PROFILE.contract()
    )
    hyperparameters["warm_start_action_value_coefficient"] = 0.0
    hyperparameters["warm_start_commit_delta_coefficient"] = 0.0
    hyperparameters["warm_start_diagnostic_binding"] = {"control": True}

    with pytest.raises(ValueError, match="current actor-critic loss contract"):
        _exercise_untrusted_transfer(
            monkeypatch,
            tmp_path,
            contract=contract,
            bundle=bundle,
        )


def test_transfer_accepts_exact_external_provenance_before_checkpoint_load(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = IFCore(improvement_mode=True)
    bundle = SimpleNamespace(
        selector_digest="b" * 64,
        normalizer={},
        normalizer_digest="c" * 64,
        quality_authority={"record_digest": "1" * 64},
        target_access={"record_digest": "2" * 64},
        ground_partition_receipt={"record_digest": "3" * 64},
    )
    contract = _warm_contract(
        model,
        quality_authority=bundle.quality_authority,
        target_access=bundle.target_access,
        ground_partition_receipt=bundle.ground_partition_receipt,
        top_preflight_sha256="5" * 64,
        top_preflight_digest="6" * 64,
        hyper_preflight_sha256="5" * 64,
        hyper_preflight_digest="6" * 64,
    )

    with pytest.raises(AssertionError, match="checkpoint deserialization"):
        _exercise_untrusted_transfer(
            monkeypatch,
            tmp_path,
            contract=contract,
            bundle=bundle,
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "expected_seed",
        "expected_grid_cell",
        "expected_grid_manifest_sha256",
        "expected_quality_preflight_receipt_sha256",
        "expected_quality_preflight_record_digest",
    ],
)
def test_transfer_requires_every_external_provenance_pin_before_reading_receipt(
    tmp_path, monkeypatch: pytest.MonkeyPatch, missing_field: str
) -> None:
    def forbidden_receipt_read(*_args, **_kwargs):
        raise AssertionError("missing external provenance reached receipt loading")

    monkeypatch.setattr("isingfold.rl.cli._strict_json", forbidden_receipt_read)
    expected = {
        "expected_seed": 1103,
        "expected_grid_cell": "rep-003-if-dual-s1103",
        "expected_grid_manifest_sha256": "1" * 64,
        "expected_quality_preflight_receipt_sha256": "2" * 64,
        "expected_quality_preflight_record_digest": "3" * 64,
    }
    expected[missing_field] = None

    with pytest.raises(ValueError, match="requires an expected"):
        _load_transfer(
            tmp_path / "checkpoint.pt",
            model=IFCore(improvement_mode=True),
            bundle=SimpleNamespace(),
            corpus=tmp_path / "prepared",
            model_family="if-core",
            **expected,
        )


def test_representation_selection_payload_lookup_requires_one_exact_cell(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = {
        "schema": "isingfold.representation-selection",
        "schema_version": 4,
        "record_digest": "2" * 64,
        "source_reports": [
            {
                "cell_id": "rep-003-if-dual-s1103",
                "model_family": "if-dual",
                "seed": 1103,
                "checkpoint_payload_digest": "3" * 64,
            }
        ],
    }
    monkeypatch.setattr("isingfold.rl.cli._strict_json", lambda _path: selection)
    monkeypatch.setattr("isingfold.rl.cli._verify_record", lambda *_args: None)
    monkeypatch.setattr("isingfold.rl.cli._sha256_file", lambda _path: "1" * 64)

    digest = _representation_checkpoint_payload_digest(
        tmp_path / "selection.json",
        expected_selection_sha256="1" * 64,
        expected_selection_record_digest="2" * 64,
        grid_cell="rep-003-if-dual-s1103",
        model_family="if-dual",
        seed=1103,
    )
    assert digest == "3" * 64

    with pytest.raises(ValueError, match="one warm-start checkpoint"):
        _representation_checkpoint_payload_digest(
            tmp_path / "selection.json",
            expected_selection_sha256="1" * 64,
            expected_selection_record_digest="2" * 64,
            grid_cell="rep-004-if-dual-s2207",
            model_family="if-dual",
            seed=2207,
        )
