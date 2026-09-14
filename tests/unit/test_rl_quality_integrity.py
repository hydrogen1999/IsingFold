from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from isingfold.rl import cli
from isingfold.rl.contracts import Context
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.quality import run_continuation
from isingfold.rl.ppo import WarmStartActionValueTarget, warm_start_utility_target
from tests.unit.test_rl_selector_labels import (
    _authority,
    _global_authority,
    _ground_partition_receipt,
    _patch_prepared,
    _pin as _publisher_pin,
    _prepared_task,
    _source_manifest,
    _target_access,
)
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.proposal import LEGACY_ONLINE_INITIALIZER_RESTARTS_V1
from tests.unit.test_rl_training_contracts import _tiny
from tests.unit.quality_receipt_support import fake_continuation_result


def _quality_cli_args(path: str = "publisher-attestation.json") -> list[str]:
    return [
        "--allow-legacy-pilot",
        "--quality-attestation",
        path,
        "--expected-quality-attestation-digest",
        "a" * 64,
        "--expected-quality-publisher-id",
        "test-publisher",
        "--ground-certificate-root",
        "ground-certificate-root.json",
        "--expected-ground-certificate-root-sha256",
        "b" * 64,
    ]


def _pin(tmp_path: Path, *, corpus: Path | None = None) -> cli._VerifiedQualityAttestationPin:
    publisher_pin = _publisher_pin(tmp_path)
    return cli._VerifiedQualityAttestationPin(
        path=publisher_pin.path,
        expected_digest=publisher_pin.expected_digest,
        expected_publisher_id=publisher_pin.expected_publisher_id,
        ground_certificate_root_path=str(tmp_path / "ground-certificate-root.json"),
        ground_certificate_root_sha256="b" * 64,
        ground_certificate_root_record_digest="e" * 64,
        corpus_directory=str(corpus or (tmp_path / "prepared")),
        global_quality_authority=_global_authority(),
    )


def _verified_authority(*, train_count: int) -> dict[str, object]:
    return _authority(train_count=train_count)


def test_warm_start_utility_target_is_uniform_action_value_and_count_aware() -> None:
    full = warm_start_utility_target(
        q_mu=[0.2, 0.8],
        continuation_counts=[4, 4],
        inclusion_probabilities=[1.0, 1.0],
    )
    assert full.value == pytest.approx(0.5)
    assert full.effective_count == pytest.approx(8.0)
    assert full.continuation_trajectories == 8
    assert full.evaluated_actions == 2

    partial = warm_start_utility_target(
        q_mu=[0.2, 0.8],
        continuation_counts=[2, 8],
        inclusion_probabilities=[0.5, 0.5],
    )
    # Equal action weighting preserves the uniform-action estimand.  Unequal Monte
    # Carlo counts and partial action coverage affect confidence, never the target.
    assert partial.value == pytest.approx(0.5)
    assert partial.effective_count == pytest.approx(1.0 / (0.625 / 4.0 + 0.5 / 2.0))
    assert partial.continuation_trajectories == 10


@pytest.mark.parametrize(
    ("q_mu", "counts", "probabilities", "message"),
    [
        ([0.2], [2, 2], [1.0], "same nonzero length"),
        ([0.2], [0], [1.0], "positive integers"),
        ([0.2, 0.8], [2, 2], [0.5, 1.0], "legal_action_count"),
    ],
)
def test_warm_start_utility_target_rejects_ambiguous_sampling_contracts(
    q_mu: list[float],
    counts: list[int],
    probabilities: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        warm_start_utility_target(
            q_mu=q_mu,
            continuation_counts=counts,
            inclusion_probabilities=probabilities,
        )


def test_warm_start_utility_target_supports_protected_commit_sampling() -> None:
    target = warm_start_utility_target(
        q_mu=[0.4, 0.2, 0.8],
        continuation_counts=[4, 4, 4],
        inclusion_probabilities=[1.0, 0.5, 0.5],
        legal_action_count=5,
    )

    assert target.value == pytest.approx(0.48)
    assert target.effective_count == pytest.approx(4.0)
    assert target.inclusion_probabilities == (1.0, 0.5, 0.5)
    assert target.legal_action_count == 5


def test_continuation_policy_rng_is_replayed_from_its_domain_separated_seed() -> None:
    task, chains = _tiny()

    def draw(seed: int) -> tuple[float, ...]:
        samples: list[float] = []

        def policy(decision, rng: random.Random) -> int:
            samples.append(rng.random())
            return next(index for index, legal in enumerate(decision.legal_mask) if legal)

        run_continuation(
            task,
            Context(qubit_cap=4),
            initializer=lambda *_: chains,
            selector=fixed_strength_selector(),
            prefix=(),
            policy=policy,
            seed=17,
            continuation_seed=seed,
            reward_reads=4,
            max_steps=2,
        )
        return tuple(samples)

    first = draw(101)
    replay = draw(101)
    independent = draw(102)

    assert first
    assert first == replay
    assert first != independent


def test_quality_instances_count_independent_lineages_and_cap_tasks() -> None:
    tasks = (
        [
            SimpleNamespace(task_id=f"a-{index}", task=SimpleNamespace(lineage="lineage-a"))
            for index in range(4)
        ]
        + [
            SimpleNamespace(task_id=f"b-{index}", task=SimpleNamespace(lineage="lineage-b"))
            for index in range(3)
        ]
        + [SimpleNamespace(task_id="c-0", task=SimpleNamespace(lineage="lineage-c"))]
    )

    selected = cli._quality_sampling_plan(
        tasks,
        lineage_count=2,
        tasks_per_lineage=2,
        seed=1701,
    )
    replay = cli._quality_sampling_plan(
        list(reversed(tasks)),
        lineage_count=2,
        tasks_per_lineage=2,
        seed=1701,
    )

    assert [[item.task_id for item in group] for group in selected] == [
        [item.task_id for item in group] for group in replay
    ]
    assert len(selected) == 2
    assert all(1 <= len(group) <= 2 for group in selected)
    assert len({group[0].task.lineage for group in selected}) == 2

    all_lineages = cli._quality_sampling_plan(
        tasks,
        lineage_count=0,
        tasks_per_lineage=1,
        seed=1701,
    )
    assert len(all_lineages) == 3

    with pytest.raises(ValueError, match="independent lineages"):
        cli._quality_sampling_plan(
            tasks,
            lineage_count=4,
            tasks_per_lineage=1,
            seed=1701,
        )


def test_quality_cli_uses_unambiguous_per_lineage_caps() -> None:
    args = cli.build_parser().parse_args(
        [
            "label-quality",
            "--corpus",
            "prepared",
            "--selector",
            "selector",
            *_quality_cli_args(),
            "--out",
            "labels",
            "--instances",
            "12",
            "--tasks-per-lineage",
            "3",
            "--states-per-lineage",
            "4",
        ]
    )

    assert args.instances == 12
    assert args.tasks_per_lineage == 3
    assert args.states_per_lineage == 4


def test_quality_label_rejects_validation_before_opening_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli.build_parser().parse_args(
        [
            "label-quality",
            "--corpus",
            "prepared",
            "--selector",
            "selector",
            *_quality_cli_args(),
            "--out",
            "labels",
            "--partition",
            "validation",
        ]
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("validation targets were opened by the train-only labeler")

    monkeypatch.setattr(cli, "_quality_attestation_pin", lambda _args: object())
    monkeypatch.setattr(cli, "_load_quality_partition", forbidden)
    with pytest.raises(ValueError, match="train partition only"):
        args.func(args)


def test_quality_shards_partition_the_deterministic_global_lineage_plan() -> None:
    tasks = [
        SimpleNamespace(task_id=f"task-{index}", task=SimpleNamespace(lineage=f"lineage-{index}"))
        for index in range(7)
    ]
    plan = cli._quality_sampling_plan(
        tasks,
        lineage_count=7,
        tasks_per_lineage=1,
        seed=1701,
    )

    shards = [cli._quality_plan_shard(plan, shard_index=index, shard_count=3) for index in range(3)]

    assert [group for shard in shards for group in shard] == [
        group for offset in range(3) for group in plan[offset::3]
    ]
    assigned = [group[0].task.lineage for shard in shards for group in shard]
    assert len(assigned) == len(set(assigned)) == 7
    with pytest.raises(ValueError, match="smaller than or equal"):
        cli._quality_plan_shard(plan, shard_index=0, shard_count=8)
    with pytest.raises(ValueError, match="strictly less"):
        cli._quality_plan_shard(plan, shard_index=3, shard_count=3)


class _Selector:
    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, _programs, _features):
        return 0


def _make_quality_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    evaluated_actions: int = 73,
    force_reward: float | None = None,
) -> tuple[Path, Path, cli.SelectorBundle, Context]:
    corpus = _source_manifest(tmp_path / "prepared")
    item = _prepared_task("quality-task", "quality-lineage", "train")
    _patch_prepared(monkeypatch, [item])
    verified_pin = _pin(tmp_path, corpus=corpus)
    monkeypatch.setattr(cli, "_quality_attestation_pin", lambda _args: verified_pin)
    selector_model = _Selector()
    bundle = cli.SelectorBundle(
        model=selector_model,
        selector_digest="a" * 64,
        normalizer={},
        normalizer_digest="b" * 64,
        coefficient_scale=1.0,
        corpus_manifest_sha256=hashlib.sha256((corpus / "manifest.json").read_bytes()).hexdigest(),
        quality_authority=_verified_authority(train_count=1),
        target_access=_target_access("train", 1),
        ground_partition_receipt=_ground_partition_receipt("train", 1),
        root=tmp_path / "selector",
    )
    monkeypatch.setattr(cli, "_load_selector_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(cli, "_seed_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_resolve_device", lambda *args, **kwargs: "cpu")

    from isingfold.rl.data import quality

    decision = cli._quality_prefixes(item, Context(qubit_cap=4), selector_model, count=1, seed=202)
    assert decision == ((),)
    environment = __import__("isingfold.rl.env", fromlist=["EmbeddingEnv"]).EmbeddingEnv(
        item.task,
        Context(qubit_cap=4),
        initializer=item.initializer(),
        selector=selector_model,
        reward_reads=256,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=202,
    )
    state = environment.reset(202)
    rewarded = max(index for index, legal in enumerate(state.legal_mask) if legal)

    def continuation(*args, **kwargs):
        action = kwargs["prefix"][-1]
        return fake_continuation_result(
            continuation_seed=kwargs["continuation_seed"],
            reward=(float(action == rewarded) if force_reward is None else force_reward),
            reward_reads=kwargs["reward_reads"],
            prefix=kwargs["prefix"],
            chains=item.task.initial_embedding,
            num_sweeps=args[1].num_sweeps,
        )

    monkeypatch.setattr(quality, "run_continuation", continuation)
    output = tmp_path / "quality"
    args = cli.build_parser().parse_args(
        [
            "label-quality",
            "--corpus",
            str(corpus),
            "--selector",
            "selector",
            *_quality_cli_args(str(tmp_path / "publisher-attestation.json")),
            "--out",
            str(output),
            "--instances",
            "1",
            "--tasks-per-lineage",
            "1",
            "--states-per-lineage",
            "1",
            "--actions",
            str(evaluated_actions),
            "--continuations",
            "32",
            "--seed",
            "202",
            "--device",
            "cpu",
        ]
    )
    args.func(args)
    return corpus, output, bundle, Context(qubit_cap=4)


def _make_quality_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    lineage_count: int = 4,
    shard_count: int = 2,
    continuations: int = 32,
) -> tuple[Path, list[Path], cli.SelectorBundle, Context]:
    corpus = _source_manifest(tmp_path / "prepared-shards")
    items = [
        _prepared_task(f"quality-task-{index}", f"quality-lineage-{index}", "train")
        for index in range(lineage_count)
    ]
    _patch_prepared(monkeypatch, items)
    verified_pin = _pin(tmp_path, corpus=corpus)
    monkeypatch.setattr(cli, "_quality_attestation_pin", lambda _args: verified_pin)
    selector_model = _Selector()
    bundle = cli.SelectorBundle(
        model=selector_model,
        selector_digest="c" * 64,
        normalizer={},
        normalizer_digest="d" * 64,
        coefficient_scale=1.0,
        corpus_manifest_sha256=hashlib.sha256((corpus / "manifest.json").read_bytes()).hexdigest(),
        quality_authority=_verified_authority(train_count=lineage_count),
        target_access=_target_access("train", lineage_count),
        ground_partition_receipt=_ground_partition_receipt("train", lineage_count),
        root=tmp_path / "selector-shards",
    )
    monkeypatch.setattr(cli, "_load_selector_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(cli, "_seed_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_resolve_device", lambda *args, **kwargs: "cpu")

    from isingfold.rl.data import quality

    environment_type = __import__("isingfold.rl.env", fromlist=["EmbeddingEnv"]).EmbeddingEnv
    rewarded_by_task: dict[str, int] = {}
    for item in items:
        environment_seed = cli._domain_seed(
            202, "quality-environment", item.task.lineage, item.task_id
        )
        environment = environment_type(
            item.task,
            Context(qubit_cap=4),
            initializer=item.initializer(),
            selector=selector_model,
            reward_reads=256,
            improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
            seed=environment_seed,
        )
        state = environment.reset(environment_seed)
        rewarded_by_task[item.task_id] = max(
            index for index, legal in enumerate(state.legal_mask) if legal
        )

    def continuation(*args, **kwargs):
        task = args[0]
        action = kwargs["prefix"][-1]
        return fake_continuation_result(
            continuation_seed=kwargs["continuation_seed"],
            reward=float(action == rewarded_by_task[task.name]),
            reward_reads=kwargs["reward_reads"],
            prefix=kwargs["prefix"],
            chains=items[0].task.initial_embedding,
            num_sweeps=args[1].num_sweeps,
        )

    monkeypatch.setattr(quality, "run_continuation", continuation)
    outputs: list[Path] = []
    for shard_index in range(shard_count):
        output = tmp_path / f"quality-shard-{shard_index}"
        args = cli.build_parser().parse_args(
            [
                "label-quality",
                "--corpus",
                str(corpus),
                "--selector",
                "selector",
                *_quality_cli_args(str(tmp_path / "publisher-attestation.json")),
                "--out",
                str(output),
                "--instances",
                str(lineage_count),
                "--tasks-per-lineage",
                "1",
                "--states-per-lineage",
                "1",
                "--actions",
                "1000",
                "--continuations",
                str(continuations),
                "--seed",
                "202",
                "--device",
                "cpu",
                "--shard-index",
                str(shard_index),
                "--shard-count",
                str(shard_count),
            ]
        )
        args.func(args)
        outputs.append(output)
    return corpus, outputs, bundle, Context(qubit_cap=4)


def test_quality_shards_bind_global_plan_and_exact_nonoverlapping_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus, shards, _bundle, _context = _make_quality_shards(tmp_path, monkeypatch)
    manifests = [json.loads((shard / "manifest.json").read_text()) for shard in shards]

    assert {manifest["schema"] for manifest in manifests} == {cli.QUALITY_SHARD_SCHEMA}
    assert {manifest["schema_version"] for manifest in manifests} == {5}
    assert {manifest["sharding_receipt"]["shard_index"] for manifest in manifests} == {0, 1}
    assert (
        len({manifest["sharding_receipt"]["global_sampling_plan_digest"] for manifest in manifests})
        == 1
    )
    subsets = [set(manifest["sharding_receipt"]["selected_lineages"]) for manifest in manifests]
    assert subsets[0].isdisjoint(subsets[1])
    assert set.union(*subsets) == set(manifests[0]["sampling_receipt"]["selected_lineages"])

    with pytest.raises(ValueError, match="must be merged"):
        cli._load_quality_labels(
            shards[0],
            corpus=_corpus,
            selector=_bundle,
            context=_context,
            enforce_resolution=False,
            quality_attestation_pin=_pin(tmp_path),
            allow_diagnostic_legacy=True,
        )


def test_quality_merge_is_order_independent_complete_and_warm_start_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, shards, bundle, context = _make_quality_shards(tmp_path, monkeypatch)
    merged = tmp_path / "quality-merged"
    args = cli.build_parser().parse_args(
        [
            "merge-quality-labels",
            "--corpus",
            str(corpus),
            "--selector",
            "selector",
            *_quality_cli_args(str(tmp_path / "publisher-attestation.json")),
            "--shard",
            str(shards[1]),
            "--shard",
            str(shards[0]),
            "--out",
            str(merged),
            "--device",
            "cpu",
        ]
    )
    args.func(args)

    replay_merged = tmp_path / "quality-merged-replayed"
    replay_args = _merge_args(corpus, shards, replay_merged)
    replay_args.func(replay_args)
    assert (merged / "records.jsonl").read_bytes() == (replay_merged / "records.jsonl").read_bytes()
    assert (merged / "manifest.json").read_bytes() == (replay_merged / "manifest.json").read_bytes()

    manifest = json.loads((merged / "manifest.json").read_text())
    assert manifest["schema"] == cli.QUALITY_MERGED_SCHEMA
    assert manifest["schema_version"] == 5
    assert manifest["record_count"] == 4
    assert manifest["independent_denominators"]["selected_lineages"] == 4
    assert [shard["shard_index"] for shard in manifest["merge_receipt"]["source_shards"]] == [0, 1]
    source_task_union = {
        task_id
        for shard in manifest["merge_receipt"]["source_shards"]
        for task_id in shard["selected_task_ids"]
    }
    assert source_task_union == set(manifest["sampling_receipt"]["selected_task_ids"])
    records, loaded = cli._load_quality_labels(
        merged,
        corpus=corpus,
        selector=bundle,
        context=context,
        min_resolved_rows=4,
        min_resolved_lineages=4,
        quality_attestation_pin=_pin(tmp_path),
        allow_diagnostic_legacy=True,
    )
    assert len(records) == 4
    assert loaded["loader_summary"]["passes_resolution"] is True


def test_quality_merge_rejects_missing_or_duplicate_shards_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, shards, _bundle, _context = _make_quality_shards(tmp_path, monkeypatch)
    missing_output = tmp_path / "missing-merge"
    base = [
        "merge-quality-labels",
        "--corpus",
        str(corpus),
        "--selector",
        "selector",
        *_quality_cli_args(str(tmp_path / "publisher-attestation.json")),
        "--out",
        str(missing_output),
        "--device",
        "cpu",
    ]
    missing = cli.build_parser().parse_args([*base, "--shard", str(shards[0])])
    with pytest.raises(ValueError, match="missing shard indices"):
        missing.func(missing)
    assert not missing_output.exists()

    duplicate_output = tmp_path / "duplicate-merge"
    duplicate_args = cli.build_parser().parse_args(
        [
            "merge-quality-labels",
            "--corpus",
            str(corpus),
            "--selector",
            "selector",
            *_quality_cli_args(str(tmp_path / "publisher-attestation.json")),
            "--out",
            str(duplicate_output),
            "--device",
            "cpu",
            "--shard",
            str(shards[0]),
            "--shard",
            str(shards[0]),
        ]
    )
    with pytest.raises(ValueError, match="duplicate shard index"):
        duplicate_args.func(duplicate_args)
    assert not duplicate_output.exists()


def _refresh_diagnostic_receipt(root: Path, manifest: dict[str, object]) -> None:
    diagnostic_path = root / "DIAGNOSTIC_ONLY" / "receipt.json"
    if diagnostic_path.is_file():
        previous = json.loads(diagnostic_path.read_text())
        receipt = cli._quality_diagnostic_receipt(
            continuations=int(previous["effective_continuations"]),
            continuation_source=str(previous["continuation_source"]),
            manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
            manifest_record_digest=str(manifest["record_digest"]),
            records_sha256=hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest(),
        )
        diagnostic_path.write_bytes(canonical_json_bytes(receipt) + b"\n")


def _rehash_manifest(root: Path, mutate) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    mutate(manifest)
    payload = {key: value for key, value in manifest.items() if key != "record_digest"}
    manifest = {**payload, "record_digest": content_digest(payload)}
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    _refresh_diagnostic_receipt(root, manifest)


def _merge_args(corpus: Path, shards: list[Path], output: Path) -> SimpleNamespace:
    values = [
        "merge-quality-labels",
        "--corpus",
        str(corpus),
        "--selector",
        "selector",
        *_quality_cli_args(str(output.parent / "publisher-attestation.json")),
        "--out",
        str(output),
        "--device",
        "cpu",
    ]
    for shard in shards:
        values.extend(("--shard", str(shard)))
    return cli.build_parser().parse_args(values)


def test_quality_merge_rejects_protocol_or_lineage_subset_disagreement_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, shards, _bundle, _context = _make_quality_shards(tmp_path, monkeypatch)
    mismatch = tmp_path / "protocol-mismatch"
    _rehash_manifest(
        shards[1],
        lambda manifest: manifest["label_protocol"].__setitem__("continuations", 16),
    )
    mismatch_args = _merge_args(corpus, shards, mismatch)
    with pytest.raises(ValueError, match="label protocol"):
        mismatch_args.func(mismatch_args)
    assert not mismatch.exists()

    corpus, shards, _bundle, _context = _make_quality_shards(tmp_path / "lineage-case", monkeypatch)
    first = json.loads((shards[0] / "manifest.json").read_text())["sharding_receipt"]
    overlap = tmp_path / "lineage-overlap"

    def duplicate_subset(manifest) -> None:
        manifest["sharding_receipt"]["selected_lineages"] = first["selected_lineages"]
        manifest["sharding_receipt"]["selected_task_ids"] = first["selected_task_ids"]

    _rehash_manifest(shards[1], duplicate_subset)
    overlap_args = _merge_args(corpus, shards, overlap)
    with pytest.raises(ValueError, match="lineage subset"):
        overlap_args.func(overlap_args)
    assert not overlap.exists()


def test_quality_merge_reports_missing_common_identity_as_controlled_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, shards, _bundle, _context = _make_quality_shards(tmp_path, monkeypatch)
    _rehash_manifest(shards[1], lambda manifest: manifest.pop("selector_digest"))
    output = tmp_path / "missing-identity"
    args = _merge_args(corpus, shards, output)

    with pytest.raises(ValueError, match="common identity field"):
        args.func(args)
    assert not output.exists()


def test_quality_merge_semantically_reauthenticates_every_v7_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, shards, _bundle, _context = _make_quality_shards(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in (shards[1] / "records.jsonl").read_text().splitlines()]
    rows[0]["evaluated"][0]["q_mu"] = 0.125
    payload = {key: value for key, value in rows[0].items() if key != "record_digest"}
    rows[0] = {**payload, "record_digest": content_digest(payload)}
    raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    (shards[1] / "records.jsonl").write_bytes(raw)
    _rehash_manifest(
        shards[1],
        lambda manifest: manifest.__setitem__("records_sha256", hashlib.sha256(raw).hexdigest()),
    )

    output = tmp_path / "tampered-row-merge"
    args = _merge_args(corpus, shards, output)
    with pytest.raises(ValueError, match="mean"):
        args.func(args)
    assert not output.exists()


def test_merged_quality_loader_rejects_rehashed_partial_merge_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, shards, bundle, context = _make_quality_shards(tmp_path, monkeypatch)
    merged = tmp_path / "quality-merged-partial"
    args = _merge_args(corpus, shards, merged)
    args.func(args)
    _rehash_manifest(
        merged,
        lambda manifest: manifest["merge_receipt"]["source_shards"].pop(),
    )

    with pytest.raises(ValueError, match="complete shard census"):
        cli._load_quality_labels(
            merged,
            corpus=corpus,
            selector=bundle,
            context=context,
            enforce_resolution=False,
            quality_attestation_pin=_pin(tmp_path),
            allow_diagnostic_legacy=True,
        )


def test_merged_quality_loader_rejects_source_record_count_or_task_union_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, shards, bundle, context = _make_quality_shards(tmp_path, monkeypatch)
    merged = tmp_path / "quality-merged-count-forgery"
    args = _merge_args(corpus, shards, merged)
    args.func(args)
    _rehash_manifest(
        merged,
        lambda manifest: manifest["merge_receipt"]["source_shards"][0].__setitem__(
            "record_count",
            manifest["merge_receipt"]["source_shards"][0]["record_count"] + 1,
        ),
    )
    with pytest.raises(ValueError, match="source record counts"):
        cli._load_quality_labels(
            merged,
            corpus=corpus,
            selector=bundle,
            context=context,
            enforce_resolution=False,
            quality_attestation_pin=_pin(tmp_path),
            allow_diagnostic_legacy=True,
        )

    corpus, shards, bundle, context = _make_quality_shards(
        tmp_path / "task-union-case", monkeypatch
    )
    merged = tmp_path / "quality-merged-task-forgery"
    args = _merge_args(corpus, shards, merged)
    args.func(args)
    _rehash_manifest(
        merged,
        lambda manifest: manifest["merge_receipt"]["source_shards"][0].__setitem__(
            "selected_task_ids", []
        ),
    )
    with pytest.raises(ValueError, match="source task subset"):
        cli._load_quality_labels(
            merged,
            corpus=corpus,
            selector=bundle,
            context=context,
            enforce_resolution=False,
            quality_attestation_pin=_pin(tmp_path),
            allow_diagnostic_legacy=True,
        )


def test_quality_artifact_is_row_authenticated_and_reports_independent_denominators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, output, bundle, context = _make_quality_artifact(tmp_path, monkeypatch)
    row = json.loads((output / "records.jsonl").read_text())
    manifest = json.loads((output / "manifest.json").read_text())

    assert row["schema"] == "isingfold.quality-counterfactual"
    assert row["schema_version"] == 7
    assert manifest["schema"] == cli.QUALITY_SCHEMA
    assert manifest["schema_version"] == 7
    assert len(row["record_digest"]) == 64
    assert len(row["action_envelope_record_digest"]) == 64
    assert all(len(action["selected_payload_digest"]) == 64 for action in row["evaluated"])
    assert "structural_action_bridge" in manifest["label_protocol"]["implementation_contract"]
    assert row["task_id"] == "quality-task"
    assert row["instance_id"] == "instance-quality-task"
    assert row["partition"] == "train"
    assert manifest["independent_denominators"]["selected_lineages"] == 1
    assert manifest["independent_denominators"]["selected_tasks"] == 1
    assert manifest["independent_denominators"]["resolved_lineages"] == 1

    records, loaded = cli._load_quality_labels(
        output,
        corpus=corpus,
        selector=bundle,
        context=context,
        quality_attestation_pin=_pin(tmp_path),
        allow_diagnostic_legacy=True,
    )
    assert len(records) == 1
    expected_target = warm_start_utility_target(
        q_mu=[float(action["q_mu"]) for action in row["evaluated"]],
        continuation_counts=[action["continuations"] for action in row["evaluated"]],
        inclusion_probabilities=[
            float(action["inclusion_probability"]) for action in row["evaluated"]
        ],
        action_value_target=WarmStartActionValueTarget(
            action_indices=tuple(action["action_index"] for action in row["evaluated"]),
            q_mu=tuple(float(action["q_mu"]) for action in row["evaluated"]),
            continuation_counts=tuple(action["continuations"] for action in row["evaluated"]),
            inclusion_probabilities=tuple(
                float(action["inclusion_probability"]) for action in row["evaluated"]
            ),
            legal_action_count=sum(bool(item["legal"]) for item in row["support"]),
            commit_index=next(
                (
                    action["action_index"]
                    for action in row["evaluated"]
                    if action["opcode"] == "COMMIT"
                    and row["support"][action["action_index"]]["archive_ref"] == 0
                ),
                None,
            ),
            support_size=len(row["support"]),
        ),
    )
    assert records[0][3] == expected_target
    assert loaded["loader_summary"]["resolved_lineages"] == 1


def test_diagnostic_quality_artifact_requires_explicit_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, output, bundle, context = _make_quality_artifact(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="diagnostic-only.*explicit diagnostic opt-in"):
        cli._load_quality_labels(
            output,
            corpus=corpus,
            selector=bundle,
            context=context,
            quality_attestation_pin=_pin(tmp_path),
        )


def test_publication_quality_loader_replays_from_live_bank_without_legacy_initializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from isingfold.rl.data import quality
    from tests.unit.test_rl_quality_initializer_bank import (
        _publication_contract_and_binding,
        _sealed_test_bank,
    )

    corpus, output, bundle, context = _make_quality_artifact(tmp_path, monkeypatch)
    item = _prepared_task("quality-task", "quality-lineage", "train")
    row = json.loads((output / "records.jsonl").read_text())
    replayed = quality.replay_decision_with_action_envelope(
        item.task,
        context,
        initializer=item.initializer(),
        selector=bundle.model,
        prefix=tuple(row["prefix"]),
        seed=int(row["environment_seed"]),
        reward_reads=context.n_est_reads,
        provenance_fingerprint=cli._quality_action_provenance_fingerprint(
            item,
            source_corpus_manifest_sha256=hashlib.sha256(
                (corpus / "manifest.json").read_bytes()
            ).hexdigest(),
        ),
    )
    assert replayed is not None

    public, bank, bank_manifest_sha256 = _sealed_test_bank(tmp_path / "bank")
    _receipt, bank_contract, binding = _publication_contract_and_binding(
        public,
        bank,
        bank_manifest_sha256,
    )

    def bind_receipts(record: dict[str, object]) -> None:
        for action in record["evaluated"]:
            for receipt in action["continuation_receipts"]:
                body = {
                    **{
                        key: value
                        for key, value in receipt.items()
                        if key != "record_digest"
                    },
                    "initializer_binding": binding,
                }
                receipt.clear()
                receipt.update({**body, "record_digest": content_digest(body)})

    _rewrite_quality_artifact(output, bind_receipts)
    resolution_plan_identity = {
        "raw_sha256": "8" * 64,
        "record_digest": "9" * 64,
    }

    def promote(manifest: dict[str, object]) -> None:
        manifest["schema_version"] = cli.QUALITY_SCHEMA_VERSION
        manifest["initializer_bank"] = bank_contract
        manifest["quality_resolution_plan"] = resolution_plan_identity

    _rehash_manifest(output, promote)
    shutil.rmtree(output / "DIAGNOSTIC_ONLY")
    manifest = json.loads((output / "manifest.json").read_text())
    monkeypatch.setattr(
        cli,
        "_quality_implementation_contract",
        lambda _context: manifest["label_protocol"]["implementation_contract"],
    )
    trusted_preflight = {
        "advance": True,
        "source_quality_manifest_sha256": hashlib.sha256(
            (output / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_quality_manifest_record_digest": manifest["record_digest"],
        "source_quality_records_sha256": manifest["records_sha256"],
        "initializer_bank": bank_contract,
        "quality_resolution_plan": resolution_plan_identity,
    }

    with pytest.raises(ValueError, match="requires the live initializer bank"):
        cli._load_quality_labels(
            output,
            corpus=corpus,
            selector=bundle,
            context=context,
            quality_attestation_pin=_pin(tmp_path),
            trusted_preflight=trusted_preflight,
        )

    live_contract_calls: list[object] = []

    def live_contract(live_bank, *, expected_manifest_sha256: str):
        assert live_bank is bank
        assert expected_manifest_sha256 == bank_manifest_sha256
        live_contract_calls.append(live_bank)
        return bank_contract

    replay_calls: list[dict[str, object]] = []

    def banked_replay(*_args, **kwargs):
        assert kwargs["initializer"] is None
        assert kwargs["initializer_bank"] is bank
        assert kwargs["expected_initializer_bank_manifest_sha256"] == bank_manifest_sha256
        assert kwargs["initializer_bank_episode_index"] == 0
        replay_calls.append(kwargs)
        return replayed

    monkeypatch.setattr(quality, "quality_initializer_bank_contract", live_contract)
    monkeypatch.setattr(quality, "replay_decision_with_action_envelope", banked_replay)
    monkeypatch.setattr(
        "isingfold.rl.data.quality_resolution_plan.quality_action_provenance_fingerprint",
        lambda *_args, **_kwargs: "7" * 64,
    )

    def forbidden_initializer(_self):
        raise AssertionError("publication replay called the legacy per-task initializer")

    monkeypatch.setattr(type(item), "initializer", forbidden_initializer)
    records, loaded = cli._load_quality_labels(
        output,
        corpus=corpus,
        selector=bundle,
        context=context,
        quality_attestation_pin=_pin(tmp_path),
        trusted_preflight=trusted_preflight,
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=bank_manifest_sha256,
    )

    assert records
    assert live_contract_calls == [bank]
    assert replay_calls
    assert loaded["initializer_bank"] == bank_contract


def test_quality_loader_rejects_rehashed_detached_applied_action_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selected workspace action must replay from the row's exact live envelope."""

    corpus, output, bundle, context = _make_quality_artifact(tmp_path, monkeypatch)
    row = json.loads((output / "records.jsonl").read_text())
    workspace_action = next(
        action for action in row["evaluated"] if action["opcode"] not in {"COMMIT", "STOP"}
    )
    workspace_action["applied_action_record_digest"] = "0" * 64
    payload = {key: value for key, value in row.items() if key != "record_digest"}
    forged = {**payload, "record_digest": content_digest(payload)}
    _bind_raw_to_manifest(output, canonical_json_bytes(forged) + b"\n")

    with pytest.raises(ValueError, match="successor binding"):
        cli._load_quality_labels(
            output,
            corpus=corpus,
            selector=bundle,
            context=context,
            quality_attestation_pin=_pin(tmp_path),
            allow_diagnostic_legacy=True,
        )


def test_quality_loader_rejects_rehashed_detached_action_envelope_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, output, bundle, context = _make_quality_artifact(tmp_path, monkeypatch)
    row = json.loads((output / "records.jsonl").read_text())
    row["action_envelope_record_digest"] = "0" * 64
    payload = {key: value for key, value in row.items() if key != "record_digest"}
    forged = {**payload, "record_digest": content_digest(payload)}
    _bind_raw_to_manifest(output, canonical_json_bytes(forged) + b"\n")

    with pytest.raises(ValueError, match="state/action envelope"):
        cli._load_quality_labels(
            output,
            corpus=corpus,
            selector=bundle,
            context=context,
            quality_attestation_pin=_pin(tmp_path),
            allow_diagnostic_legacy=True,
        )


def test_one_pinned_quality_preflight_replaces_all_later_stochastic_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, labels, bundle, context = _make_quality_artifact(tmp_path, monkeypatch)
    preflight_path = tmp_path / "quality-preflight.json"
    args = cli.build_parser().parse_args(
        [
            "quality-preflight",
            "--corpus",
            str(corpus),
            "--selector",
            "selector",
            *_quality_cli_args(str(tmp_path / "publisher-attestation.json")),
            "--quality-labels",
            str(labels),
            "--out",
            str(preflight_path),
            "--device",
            "cpu",
        ]
    )
    args.func(args)
    pinned_sha256 = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    preflight = cli._load_quality_preflight_receipt(
        preflight_path,
        expected_sha256=pinned_sha256,
        quality_labels=labels,
        corpus=corpus,
        selector=bundle,
        context=context,
        min_resolved_rows=1,
        min_resolved_lineages=1,
        allow_diagnostic_legacy=True,
    )
    assert preflight["full_replay"]["continuation_replays_executed"] > 0
    assert (
        preflight["full_replay"]["continuation_replay_matches"]
        == preflight["full_replay"]["continuation_trajectories"]
    )

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("scientific training repeated an already authenticated SA replay")

    monkeypatch.setattr("isingfold.rl.data.quality.run_continuation", forbidden)
    # Keep the authenticated implementation identity fixed while replacing only the
    # test callable with a tripwire.  A real source change must still invalidate the pin.
    monkeypatch.setattr(
        cli,
        "_quality_implementation_contract",
        lambda _context: preflight["implementation_contract"],
    )
    records, loaded = cli._load_quality_labels(
        labels,
        corpus=corpus,
        selector=bundle,
        context=context,
        quality_attestation_pin=_pin(tmp_path),
        allow_diagnostic_legacy=True,
        trusted_preflight=preflight,
    )
    assert records
    summary = loaded["loader_summary"]
    assert summary["continuation_replay_mode"] == "pinned-full-replay-receipt"
    assert summary["continuation_replays_executed"] == 0
    assert summary["continuation_replay_matches"] == 0
    assert summary["continuation_trajectories_authenticated"] > 0


def test_quality_preflight_requires_external_pin_and_current_evaluator_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, labels, bundle, context = _make_quality_artifact(tmp_path, monkeypatch)
    preflight_path = tmp_path / "quality-preflight.json"
    args = cli.build_parser().parse_args(
        [
            "quality-preflight",
            "--corpus",
            str(corpus),
            "--selector",
            "selector",
            *_quality_cli_args(str(tmp_path / "publisher-attestation.json")),
            "--quality-labels",
            str(labels),
            "--out",
            str(preflight_path),
            "--device",
            "cpu",
        ]
    )
    args.func(args)
    pinned_sha256 = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    kwargs = {
        "quality_labels": labels,
        "corpus": corpus,
        "selector": bundle,
        "context": context,
        "min_resolved_rows": 1,
        "min_resolved_lineages": 1,
        "allow_diagnostic_legacy": True,
    }
    with pytest.raises(ValueError, match="externally pinned"):
        cli._load_quality_preflight_receipt(
            preflight_path,
            expected_sha256="0" * 64,
            **kwargs,
        )

    original_contract = cli._quality_implementation_contract

    def changed_contract(ctx):
        contract = original_contract(ctx)
        return {**contract, "evaluator": {"forged": True}}

    monkeypatch.setattr(cli, "_quality_implementation_contract", changed_contract)
    with pytest.raises(ValueError, match="another scientific experiment"):
        cli._load_quality_preflight_receipt(
            preflight_path,
            expected_sha256=pinned_sha256,
            **kwargs,
        )


def test_quality_loader_rejects_rehashed_legal_action_cherry_pick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid-looking alternative arm cannot replace the registered random draw."""

    from isingfold.rl.data import quality

    corpus, output, bundle, context = _make_quality_artifact(
        tmp_path,
        monkeypatch,
        evaluated_actions=2,
    )
    row = json.loads((output / "records.jsonl").read_text())
    sampled = {action["action_index"] for action in row["evaluated"]}
    replace_position = next(
        position
        for position, action in enumerate(row["evaluated"])
        if action["inclusion_probability"] < 1.0
    )
    alternative = next(
        support["index"]
        for support in row["support"]
        if support["legal"] and support["index"] not in sampled
    )
    support = row["support"][alternative]
    action = row["evaluated"][replace_position]
    action["action_index"] = alternative
    action["payload_key"] = support["payload_key"]
    action["opcode"] = support["opcode"]
    action["continuation_seeds"] = [
        quality.continuation_seed(
            row["environment_seed"],
            row["task_id"],
            row["state_fingerprint"],
            alternative,
            continuation_index,
        )
        for continuation_index in range(action["continuations"])
    ]
    rewarded = max(item["index"] for item in row["support"] if item["legal"])
    reward = float(alternative == rewarded)
    action["continuation_rewards"] = [reward] * action["continuations"]
    action["continuation_valid"] = [True] * action["continuations"]
    action["valid_returns"] = action["continuations"]
    action["q_mu"] = reward
    row["evaluated"].sort(key=lambda item: item["action_index"])
    row["best_actions"] = [alternative]
    row_payload = {key: value for key, value in row.items() if key != "record_digest"}
    row = {**row_payload, "record_digest": content_digest(row_payload)}
    raw = canonical_json_bytes(row) + b"\n"
    _bind_raw_to_manifest(output, raw)

    with pytest.raises(ValueError, match="receipt|deterministic draw"):
        cli._load_quality_labels(
            output,
            corpus=corpus,
            selector=bundle,
            context=context,
            quality_attestation_pin=_pin(tmp_path),
            allow_diagnostic_legacy=True,
        )


def _rewrite_quality_artifact(
    root: Path,
    mutate,
) -> None:
    row = json.loads((root / "records.jsonl").read_text())
    mutate(row)
    row_payload = {key: value for key, value in row.items() if key != "record_digest"}
    row = {**row_payload, "record_digest": content_digest(row_payload)}
    raw = canonical_json_bytes(row) + b"\n"
    (root / "records.jsonl").write_bytes(raw)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["records_sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_payload = {key: value for key, value in manifest.items() if key != "record_digest"}
    manifest = {**manifest_payload, "record_digest": content_digest(manifest_payload)}
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    _refresh_diagnostic_receipt(root, manifest)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row["evaluated"][0].__setitem__("q_mu", 0.25), "mean"),
        (
            lambda row: row["evaluated"][0].__setitem__(
                "continuations", row["evaluated"][0]["continuations"] + 1
            ),
            "arrays",
        ),
        (
            lambda row: row["evaluated"][0].__setitem__("valid_returns", 999),
            "valid-return",
        ),
        (
            lambda row: row["evaluated"][0].__setitem__(
                "continuation_seeds", [7] * row["evaluated"][0]["continuations"]
            ),
            "seed",
        ),
        (
            lambda row: row["evaluated"][0].__setitem__(
                "continuation_rewards",
                [bool(value) for value in row["evaluated"][0]["continuation_rewards"]],
            ),
            "numeric",
        ),
        (
            lambda row: row["evaluated"][0].__setitem__("inclusion_probability", 0.125),
            "inclusion",
        ),
        (lambda row: row.__setitem__("best_actions", []), "tied-best"),
        (
            lambda row: row["evaluated"][0].__setitem__("payload_key", "forged"),
            "payload",
        ),
        (
            lambda row: row["evaluated"][0].__setitem__("opcode", "STOP"),
            "opcode",
        ),
        (lambda row: row["support"][0].__setitem__("legal", False), "support"),
        (lambda row: row.__setitem__("lineage", "held-out-lineage"), "lineage"),
        (lambda row: row.__setitem__("context_version", "other"), "context"),
        (lambda row: row.__setitem__("state_fingerprint", "0" * 64), "fingerprint"),
        (lambda row: row.__setitem__("support_fingerprint", "0" * 64), "fingerprint"),
        (lambda row: row.__setitem__("label_version", "other"), "label"),
    ],
)
def test_quality_loader_rejects_semantically_tampered_but_rehashed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    corpus, output, bundle, context = _make_quality_artifact(tmp_path, monkeypatch)
    _rewrite_quality_artifact(output, mutate)

    with pytest.raises(ValueError, match=message):
        cli._load_quality_labels(
            output,
            corpus=corpus,
            selector=bundle,
            context=context,
            quality_attestation_pin=_pin(tmp_path),
            allow_diagnostic_legacy=True,
        )


def _bind_raw_to_manifest(root: Path, raw: bytes) -> None:
    (root / "records.jsonl").write_bytes(raw)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["records_sha256"] = hashlib.sha256(raw).hexdigest()
    payload = {key: value for key, value in manifest.items() if key != "record_digest"}
    manifest = {**payload, "record_digest": content_digest(payload)}
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    _refresh_diagnostic_receipt(root, manifest)


def test_quality_loader_rejects_duplicate_json_keys_even_with_matching_file_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, output, bundle, context = _make_quality_artifact(tmp_path, monkeypatch)
    raw = (output / "records.jsonl").read_bytes()
    forged = b'{"schema":"duplicate",' + raw[1:]
    _bind_raw_to_manifest(output, forged)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        cli._load_quality_labels(
            output,
            corpus=corpus,
            selector=bundle,
            context=context,
            quality_attestation_pin=_pin(tmp_path),
            allow_diagnostic_legacy=True,
        )


def test_quality_loader_rejects_nonfinite_json_even_with_matching_file_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, output, bundle, context = _make_quality_artifact(tmp_path, monkeypatch)
    row = json.loads((output / "records.jsonl").read_text())
    row["evaluated"][0]["q_mu"] = float("nan")
    forged = (json.dumps(row, allow_nan=True, separators=(",", ":")) + "\n").encode()
    _bind_raw_to_manifest(output, forged)

    with pytest.raises(ValueError, match="non-finite"):
        cli._load_quality_labels(
            output,
            corpus=corpus,
            selector=bundle,
            context=context,
            quality_attestation_pin=_pin(tmp_path),
            allow_diagnostic_legacy=True,
        )


def test_quality_resolution_threshold_fails_before_training_with_exact_denominators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, output, bundle, context = _make_quality_artifact(tmp_path, monkeypatch)

    with pytest.raises(
        ValueError,
        match=r"resolved rows 1/1 \(required 2\).*resolved independent lineages 1/1",
    ):
        cli._load_quality_labels(
            output,
            corpus=corpus,
            selector=bundle,
            context=context,
            min_resolved_rows=2,
            min_resolved_lineages=2,
            quality_attestation_pin=_pin(tmp_path),
            allow_diagnostic_legacy=True,
        )


def test_quality_preflight_retains_fully_unresolved_rows_without_inventing_winners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, output, bundle, context = _make_quality_artifact(
        tmp_path,
        monkeypatch,
        force_reward=0.0,
    )

    records, loaded = cli._load_quality_labels(
        output,
        corpus=corpus,
        selector=bundle,
        context=context,
        enforce_resolution=False,
        quality_attestation_pin=_pin(tmp_path),
        allow_diagnostic_legacy=True,
    )

    assert len(records) == 1
    _observation, best, evaluated, utility_target = records[0]
    assert best == evaluated
    assert 0.0 <= utility_target.value <= 1.0
    assert utility_target.effective_count > 0.0
    assert utility_target.action_value is not None
    assert utility_target.action_value.action_indices == evaluated
    assert loaded["loader_summary"]["fully_unresolved_skipped"] == 1
    assert loaded["loader_summary"]["records_used"] == 0
    assert loaded["loader_summary"]["resolved_lineages"] == 0
    assert loaded["loader_summary"]["passes_resolution"] is False
    assert len(loaded["lineages"]) == 1


def test_quality_preflight_writes_an_authenticated_failure_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, output, _bundle, _context = _make_quality_artifact(tmp_path, monkeypatch)
    report = tmp_path / "preflight.json"
    args = cli.build_parser().parse_args(
        [
            "quality-preflight",
            "--corpus",
            str(corpus),
            "--selector",
            "selector",
            *_quality_cli_args(str(tmp_path / "publisher-attestation.json")),
            "--quality-labels",
            str(output),
            "--out",
            str(report),
            "--min-resolved-rows",
            "2",
            "--min-resolved-lineages",
            "2",
            "--device",
            "cpu",
        ]
    )

    with pytest.raises(ValueError, match="resolution preflight failed"):
        args.func(args)

    receipt = json.loads(report.read_text())
    assert receipt["schema"] == "isingfold.quality-resolution-preflight"
    assert receipt["advance"] is False
    assert receipt["resolution"]["records_used"] == 1
    assert len(receipt["record_digest"]) == 64


def test_scientific_grid_preregisters_nontrivial_quality_resolution_minima() -> None:
    root = Path(__file__).resolve().parents[2]
    grid = json.loads((root / "configs" / "rl_grid_hybrid_v1.json").read_text())

    assert grid["quality_resolution"] == {
        "min_resolved_rows": 128,
        "min_resolved_lineages": 128,
    }
