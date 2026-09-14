from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from embedbench.apps import graphcut_mrf, jobshop
from embedbench.candidate_bank import (
    assign_split,
    canonical_json_bytes,
    content_digest,
    stable_seed,
    validate_group_against_instance,
)
from embedbench.isingfold_corpus_shard import (
    CorpusShard,
    ExactReference,
    LineageRequest,
    ShardConfig,
    _generation_provenance,
    derive_exact_reference,
    generate_corpus_shard,
    lineage_shard,
    prospect_lineage,
    reference_problem_digest,
)
from embedbench.reference_export import export_problem_references


def _exact_energy(prospect) -> float:
    nodes = [node for node, _ in prospect.h]
    best = float("inf")
    for values in itertools.product((-1, 1), repeat=len(nodes)):
        spins = dict(zip(nodes, values, strict=True))
        energy = sum(coefficient * spins[node] for node, coefficient in prospect.h)
        energy += sum(
            coefficient * spins[left] * spins[right] for left, right, coefficient in prospect.j
        )
        best = min(best, energy)
    return float(best)


def _config(**overrides: object) -> ShardConfig:
    values: dict[str, object] = {
        "root_seed": 20260912,
        "shard_index": 0,
        "shard_count": 1,
        "attempt_slots": 10,
        "incumbent_search_slots": 4,
        "max_split_search": 128,
        "strengths": (1.0,),
        "reads": 2,
        "sweeps": 5,
    }
    values.update(overrides)
    return ShardConfig(**values)


def _requests() -> tuple[LineageRequest, ...]:
    return (
        LineageRequest(
            lineage_id="application-a",
            application_family="graphcut",
            origin="application-derived",
            partition="train",
            distribution_regime="iid",
            topology="chimera",
            host_size=1,
            n_variables=3,
        ),
        LineageRequest(
            lineage_id="ink-drop-b",
            application_family="ink-drop-quotient",
            origin="synthetic-ink-drop",
            partition="test",
            distribution_regime="ood",
            topology="chimera",
            host_size=2,
            n_variables=3,
            ink_chain_size=3,
            qubit_fraction=0.05,
            coupler_fraction=0.05,
        ),
    )


def test_generates_deterministic_candidate_records_and_outcome_blind_design_facts() -> None:
    config = _config()
    requests = _requests()

    first = generate_corpus_shard(config, requests)
    second = generate_corpus_shard(config, tuple(reversed(requests)))

    assert first.to_dict() == second.to_dict()
    assert len(first.lineages) == 2
    for generated in first.lineages:
        validate_group_against_instance(generated.group, generated.instance)
        assert len(generated.group.candidates) >= 2
        assert [attempt.slot for attempt in generated.group.attempts] == list(
            range(config.attempt_slots)
        )
        assert sum(attempt.status == "valid" for attempt in generated.group.attempts) == len(
            generated.group.candidates
        )
        assert generated.group.incumbent.decision.partition == "decision"
        assert generated.group.incumbent.audit.partition == "audit"
        assert not set(generated.group.incumbent.decision.seeds).intersection(
            generated.group.incumbent.audit.seeds
        )
        assert assign_split(generated.instance.split_unit_id) == generated.request.partition
        assert generated.task_fact.distribution["learning_partition"] == (
            generated.request.partition
        )
        assert generated.task_fact.distribution["source_partition"] == "isingfold-corpus-v4"
        fact_bytes = canonical_json_bytes(
            {
                "lineage": generated.lineage_fact,
                "task": generated.task_fact,
            }
        )
        for forbidden in (b"p_solve", b"residual", b"reference_energy", b"audit"):
            assert forbidden not in fact_bytes
        assert {name for name, _ in generated.lineage_fact.measurements} == {
            "candidate_resource_spread",
            "host_fill_fraction",
            "logical_edge_density",
        }


@pytest.mark.parametrize(
    ("topology", "size"),
    [("chimera", 1), ("pegasus", 2), ("zephyr", 1)],
)
def test_prospects_on_verified_realized_dwave_hosts_with_optional_faults(
    topology: str,
    size: int,
) -> None:
    config = _config()
    request = LineageRequest(
        lineage_id=f"host-{topology}",
        application_family="graphcut",
        origin="application-derived",
        partition="val",
        distribution_regime="iid",
        topology=topology,
        host_size=size,
        n_variables=3,
        qubit_fraction=0.0,
        coupler_fraction=0.0,
    )

    prospect = prospect_lineage(config, request)

    assert prospect.realized_host_artifact["topology"] == topology
    assert prospect.realized_host_artifact["nodes"]
    assert prospect.realized_host_artifact["host_sha256"]
    assert assign_split(prospect.split_unit_id) == "val"


def test_reference_source_row_is_accepted_by_authenticated_exact_export(
    tmp_path: Path,
) -> None:
    config = _config()
    request = _requests()[0]
    generated = generate_corpus_shard(config, (request,)).lineages[0]
    release = tmp_path / "release"
    release.mkdir()
    corpus = release / "quality_isingfold_shard.jsonl"
    corpus.write_bytes(generated.reference_source.canonical_bytes + b"\n")
    corpus_sha256 = hashlib.sha256(corpus.read_bytes()).hexdigest()
    manifest = release / f"{corpus.name}.manifest.json"
    manifest.write_bytes(
        canonical_json_bytes({"file": f"runs/isingfold/{corpus.name}", "sha256": corpus_sha256})
        + b"\n"
    )
    checksums = release / "SHA256SUMS"
    checksums.write_text(
        f"{corpus_sha256}  {corpus.name}\n"
        f"{hashlib.sha256(manifest.read_bytes()).hexdigest()}  {manifest.name}\n"
    )

    result = export_problem_references(
        [corpus],
        checksums_path=checksums,
        output_dir=tmp_path / "references",
    )

    assert result["certified_problems"] == 1
    assert (
        generated.reference_source.record_sha256
        == hashlib.sha256(generated.reference_source.canonical_bytes).hexdigest()
    )
    row = json.loads(generated.reference_source.canonical_bytes)
    assert row["problem"]["e0"] == generated.exact_reference.energy


def test_jobshop_origin_contains_real_precedence_and_machine_contention() -> None:
    config = _config()
    request = LineageRequest(
        lineage_id="application-jobshop-n12",
        application_family="jobshop",
        origin="application-derived",
        partition="train",
        distribution_regime="iid",
        topology="chimera",
        host_size=2,
        n_variables=12,
    )

    prospect = prospect_lineage(config, request)

    assert prospect.origin_metadata["kwargs"] == {
        "horizon": 3,
        "n_jobs": 2,
        "n_machines": 2,
    }
    assert prospect.origin_metadata["derived"]["proc"]
    assert len(prospect.logical_edges) > request.n_variables


def test_jobshop_seed_entropy_survives_old_processing_time_and_gauge_collision() -> None:
    """Distinct requests that collided under the v1 Jobshop objective stay distinct."""

    first_id = "if-prod-v1-test-0436-364b5cbd93e457f3"
    second_id = "if-prod-v1-test-0744-b873e69ed26835e1"
    first_slot, second_slot = 7, 19
    first_app_seed = stable_seed(
        260912,
        "isingfold-corpus-application-v1",
        first_id,
        "jobshop",
        first_slot,
    )
    second_app_seed = stable_seed(
        260912,
        "isingfold-corpus-application-v1",
        second_id,
        "jobshop",
        second_slot,
    )
    assert (first_app_seed, second_app_seed) == (
        7065353393835017404,
        950464185220619136,
    )
    # The old generator consumed only these two discrete processing times, then
    # these requests happened to receive the same six gauge signs.
    def old_processing_times(seed: int) -> dict[str, int]:
        rng = np.random.default_rng(seed)
        return {f"{job}-0": int(rng.integers(1, 3)) for job in range(2)}

    assert old_processing_times(first_app_seed) == old_processing_times(second_app_seed) == {
        "0-0": 2,
        "1-0": 1,
    }
    first_gauge_seed = stable_seed(
        260912,
        "isingfold-corpus-prospective-split-v1",
        first_id,
        first_slot,
    )
    second_gauge_seed = stable_seed(
        260912,
        "isingfold-corpus-prospective-split-v1",
        second_id,
        second_slot,
    )
    first_gauges = tuple(-1 if (first_gauge_seed >> node) & 1 else 1 for node in range(6))
    second_gauges = tuple(-1 if (second_gauge_seed >> node) & 1 else 1 for node in range(6))
    assert first_gauges == second_gauges

    first = jobshop(n_jobs=2, n_machines=1, horizon=3, seed=first_app_seed)
    second = jobshop(n_jobs=2, n_machines=1, horizon=3, seed=second_app_seed)
    first_gauged_h = tuple(
        sorted((node, value * first_gauges[node]) for node, value in first.problem.h.items())
    )
    second_gauged_h = tuple(
        sorted((node, value * second_gauges[node]) for node, value in second.problem.h.items())
    )
    first_gauged_j = tuple(
        sorted(
            (left, right, value * first_gauges[left] * first_gauges[right])
            for (left, right), value in first.problem.j.items()
        )
    )
    second_gauged_j = tuple(
        sorted(
            (left, right, value * second_gauges[left] * second_gauges[right])
            for (left, right), value in second.problem.j.items()
        )
    )

    assert (first_gauged_h, first_gauged_j) != (second_gauged_h, second_gauged_j)
    assert first.derived["objective"]["protocol"] == "weighted-completion-tardiness-v1"
    denominator = first.derived["encoding"]["ising_coefficient_denominator"]
    assert denominator == 16384
    assert all(float(value * denominator).is_integer() for value in first.problem.h.values())
    assert all(float(value * denominator).is_integer() for value in first.problem.j.values())


def test_shard_assignment_is_stable_disjoint_and_complete() -> None:
    ids = [f"lineage-{index}" for index in range(30)]
    shards = [{item for item in ids if lineage_shard(item, 3) == index} for index in range(3)]

    assert set().union(*shards) == set(ids)
    assert all(
        left.isdisjoint(right) for index, left in enumerate(shards) for right in shards[index + 1 :]
    )
    assert [lineage_shard(item, 3) for item in ids] == [lineage_shard(item, 3) for item in ids]


def test_rejects_ood_outside_test() -> None:
    with pytest.raises(ValueError, match="OOD.*test"):
        LineageRequest(
            lineage_id="invalid-ood",
            application_family="ink-drop-quotient",
            origin="synthetic-ink-drop",
            partition="train",
            distribution_regime="ood",
            topology="chimera",
            host_size=1,
            n_variables=3,
        )


def test_fails_closed_when_fixed_slots_do_not_yield_two_alternatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import embedbench.isingfold_corpus_shard as shard_module

    config = _config(attempt_slots=2)
    request = LineageRequest(
        lineage_id="too-constrained",
        application_family="graphcut",
        origin="application-derived",
        partition="train",
        distribution_regime="iid",
        topology="chimera",
        host_size=1,
        n_variables=4,
    )
    real_find_embedding = shard_module.minorminer.find_embedding
    incumbent: dict[int, list[int]] | None = None

    def repeated_proposal(*args, **kwargs):
        nonlocal incumbent
        assert kwargs["random_seed"] > 0
        assert kwargs["threads"] == 1
        assert "timeout" not in kwargs
        if incumbent is None:
            incumbent = real_find_embedding(*args, **kwargs)
        return {node: list(chain) for node, chain in incumbent.items()}

    monkeypatch.setattr(shard_module.minorminer, "find_embedding", repeated_proposal)

    with pytest.raises(ValueError, match="two unique valid alternatives"):
        generate_corpus_shard(config, (request,))


def test_minorminer_zero_seed_is_projected_without_a_wall_clock_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import embedbench.isingfold_corpus_shard as shard_module

    config = _config()
    prospect = prospect_lineage(config, _requests()[0])
    host = shard_module._load_prospect_host(config, prospect)
    observed: dict[str, object] = {}

    def capture_proposal(*args, **kwargs):
        observed.update(kwargs)
        return {}

    monkeypatch.setattr(shard_module.minorminer, "find_embedding", capture_proposal)

    assert shard_module._proposal(host=host, prospect=prospect, seed=0) is None
    assert observed["random_seed"] == 1
    assert observed["threads"] == 1
    assert "timeout" not in observed


def test_strict_shard_loader_revalidates_nested_records_and_provenance() -> None:
    config = _config()
    requests = _requests()
    shard = generate_corpus_shard(config, requests)

    loaded = CorpusShard.from_dict(shard.to_dict())
    loaded_from_json = CorpusShard.from_dict(json.loads(json.dumps(shard.to_dict())))

    assert loaded.to_dict() == shard.to_dict()
    assert canonical_json_bytes(loaded_from_json.to_dict()) == canonical_json_bytes(shard.to_dict())

    tampered = json.loads(json.dumps(shard.to_dict()))
    tampered["lineages"][0]["group"]["incumbent"]["total_qubits"] += 1
    with pytest.raises(ValueError, match="resource metrics|digest"):
        CorpusShard.from_dict(tampered)

    unknown = json.loads(json.dumps(shard.to_dict()))
    unknown["unexpected"] = True
    with pytest.raises(ValueError, match="schema fields"):
        CorpusShard.from_dict(unknown)

    config_tamper = json.loads(json.dumps(shard.to_dict()))
    config_tamper["config"]["reads"] += 1
    config_tamper["record_digest"] = content_digest(
        {key: value for key, value in config_tamper.items() if key != "record_digest"}
    )
    with pytest.raises(ValueError, match="evaluation protocol"):
        CorpusShard.from_dict(config_tamper)

    exact_forgery = json.loads(json.dumps(shard.to_dict()))
    forged_reference = exact_forgery["lineages"][0]["exact_reference"]
    forged_reference["evaluator_protocol"]["forged_authority"] = True
    forged_reference["authority_sha256"] = content_digest(forged_reference["evaluator_protocol"])
    exact_forgery["lineages"][0]["reference_authority_sha256"] = forged_reference[
        "authority_sha256"
    ]
    exact_forgery["record_digest"] = content_digest(
        {key: value for key, value in exact_forgery.items() if key != "record_digest"}
    )
    with pytest.raises(ValueError, match="exact-replay authority"):
        CorpusShard.from_dict(exact_forgery)


def test_application_origin_uses_the_registered_application_generator() -> None:
    config = _config()
    request = LineageRequest(
        lineage_id="real-graphcut",
        application_family="graphcut",
        origin="application-derived",
        partition="test",
        distribution_regime="iid",
        topology="chimera",
        host_size=1,
        n_variables=3,
    )

    prospect = prospect_lineage(config, request)
    app_seed = stable_seed(
        config.root_seed,
        "isingfold-corpus-application-v1",
        request.lineage_id,
        request.application_family,
        prospect.prospective_slot,
    )
    expected = graphcut_mrf(rows=1, cols=3, seed=app_seed)

    assert sorted(abs(value) for _, value in prospect.h) == sorted(
        abs(value) for value in expected.problem.h.values()
    )
    assert sorted(abs(value) for *_, value in prospect.j) == sorted(
        abs(value) for value in expected.problem.j.values()
    )
    generated = generate_corpus_shard(config, (request,)).lineages[0]
    assert generated.lineage_fact.generator_id.endswith("application-graphcut-v1")
    assert "maxcut" not in generated.lineage_fact.generator_id.casefold()
    with pytest.raises(ValueError, match="registered application family"):
        LineageRequest(
            lineage_id="fake-maxcut",
            application_family="maxcut-cycle",
            origin="application-derived",
            partition="train",
            distribution_regime="iid",
            topology="chimera",
            host_size=1,
            n_variables=3,
        )


def test_synthetic_prospect_rejects_unusable_quotients_after_fixed_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import embedbench.isingfold_corpus_shard as shard_module

    monkeypatch.setattr(
        shard_module,
        "ink_drop",
        lambda *args, **kwargs: SimpleNamespace(logical=nx.empty_graph(3)),
    )
    request = LineageRequest(
        lineage_id="edgeless-ink",
        application_family="ink-drop-quotient",
        origin="synthetic-ink-drop",
        partition="train",
        distribution_regime="iid",
        topology="chimera",
        host_size=1,
        n_variables=3,
    )

    with pytest.raises(ValueError, match="usable connected quotient"):
        prospect_lineage(_config(max_split_search=3), request)


def test_exact_reference_is_internally_enumerated_and_authority_bound() -> None:
    config = _config()
    prospect = prospect_lineage(config, _requests()[0])

    reference = derive_exact_reference(prospect)

    assert reference.energy == _exact_energy(prospect)
    assert reference.problem_digest == reference_problem_digest(prospect)
    assert reference.energy_integer * 2.0**reference.energy_power_of_two == reference.energy
    assert reference.certificate_sha256
    assert reference.proof_artifact_sha256
    assert set(reference.evaluator_protocol) == {
        "algorithm",
        "certificate_sha256",
        "checked_state_count",
        "energy",
        "environment_sha256",
        "execution_mode",
        "problem_sha256",
        "proof_artifact_sha256",
        "quality_eligible",
        "replay_entry_point",
        "schema",
        "schema_version",
        "source_sha256",
        "status",
        "variable_count",
    }
    assert reference.evaluator_protocol["schema"] == (
        "embedbench.isingfold-exact-reference-authority"
    )
    assert reference.evaluator_protocol["schema_version"] == 2
    assert reference.evaluator_protocol["execution_mode"] == "in_process"
    assert reference.evaluator_protocol["replay_entry_point"] == (
        "embedbench.ground_certificate.verify_ground_state_certificate"
    )
    assert reference.evaluator_protocol["checked_state_count"] == 1 << len(
        prospect.logical_nodes
    )
    assert reference.evaluator_protocol["status"] == "exact_enumeration"
    assert reference.evaluator_protocol["quality_eligible"] is True
    assert reference.evaluator_protocol["certificate_sha256"] == reference.certificate_sha256
    assert reference.evaluator_protocol["proof_artifact_sha256"] == (
        reference.proof_artifact_sha256
    )
    assert reference.evaluator_protocol["problem_sha256"] == reference.problem_digest
    assert reference.evaluator_protocol["energy"] == {
        "integer": reference.energy_integer,
        "power_of_two": reference.energy_power_of_two,
    }
    assert reference.authority_sha256 == content_digest(reference.evaluator_protocol)

    malformed_protocol = dict(reference.evaluator_protocol)
    malformed_protocol["checked_state_count"] -= 1
    with pytest.raises(ValueError, match="checked state count"):
        ExactReference(
            energy=reference.energy,
            energy_integer=reference.energy_integer,
            energy_power_of_two=reference.energy_power_of_two,
            problem_digest=reference.problem_digest,
            authority_sha256=content_digest(malformed_protocol),
            certificate_sha256=reference.certificate_sha256,
            proof_artifact_sha256=reference.proof_artifact_sha256,
            evaluator_protocol=malformed_protocol,
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        generate_corpus_shard(
            config,
            (_requests()[0],),
            exact_references={"application-a": object()},
        )


def test_lineage_scientific_records_do_not_depend_on_shard_coordinates() -> None:
    request = _requests()[0]
    unsharded = generate_corpus_shard(_config(), (request,)).lineages[0]
    shard_count = 7
    shard_index = lineage_shard(request.lineage_id, shard_count)
    sharded = generate_corpus_shard(
        _config(shard_count=shard_count, shard_index=shard_index),
        (request,),
    ).lineages[0]

    assert unsharded.instance.to_dict() == sharded.instance.to_dict()
    assert unsharded.group.to_dict() == sharded.group.to_dict()
    assert unsharded.lineage_fact == sharded.lineage_fact
    assert unsharded.task_fact == sharded.task_fact
    assert unsharded.exact_reference == sharded.exact_reference
    assert unsharded.reference_source == sharded.reference_source


def test_provenance_binds_sources_dependencies_and_application_parameters() -> None:
    shard = generate_corpus_shard(_config(), (_requests()[0],))
    generated = shard.lineages[0]
    provenance = shard.provenance

    assert shard.provenance_digest == content_digest(provenance)
    assert {
        "apps",
        "candidate_bank",
        "embedding",
        "ground_certificate",
        "hard_ood_schema",
        "inkdrop",
        "isingfold_corpus_shard",
        "isingfold_design",
        "programming",
        "realized_host",
        "structural",
        "surrogate",
    } == set(provenance["source_modules"])
    assert {
        "dimod",
        "dwave-networkx",
        "dwave-samplers",
        "embedbench",
        "minorminer",
        "networkx",
        "numpy",
        "scipy",
    } == set(provenance["dependency_versions"])
    metadata = dict(generated.instance.metadata)
    assert metadata["application_seed"] == generated.prospect.origin_metadata["seed"]
    assert dict(metadata["application_kwargs"]) == generated.prospect.origin_metadata["kwargs"]
    assert dict(metadata["application_derived"]) == generated.prospect.origin_metadata["derived"]


def test_generation_provenance_binds_exact_thread_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = {
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    for name, value in controls.items():
        monkeypatch.setenv(name, value)

    provenance = _generation_provenance()

    assert provenance["protocol"] == "embedbench.isingfold-corpus-generator-v4"
    assert provenance["thread_controls"] == controls


def test_evaluation_budget_requires_explicit_values_with_named_smoke_factory() -> None:
    with pytest.raises(TypeError, match="required positional argument"):
        ShardConfig(root_seed=1, shard_index=0, shard_count=1)

    smoke = ShardConfig.smoke(root_seed=1, shard_index=0, shard_count=1)

    assert smoke.strengths == (1.0,)
    assert smoke.reads == 2
    assert smoke.sweeps == 5
