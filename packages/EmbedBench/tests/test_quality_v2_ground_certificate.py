from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

import pytest
from embedbench.ground_certificate import (
    CheckerRun,
    Dyadic,
    IsingProblem,
    PlantedTerm,
    SolverRun,
    build_exact_enumeration_certificate,
    build_planted_certificate,
)

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_quality_v2 import (  # noqa: E402
    _validate_ground_reference,
    _validate_min_cut_ground_certificate,
)
from quality_v2_paper_contract import load_paper_audit_contract  # noqa: E402
from rescore_quality import (  # noqa: E402
    _atomic_write,
    _certified_ground_reference,
    _load_resumable_paper_row,
    _paper_row_payload,
    _records_for_audit,
    audit_shard_index,
    canonical_audit_identity,
)


def _energy(h: dict[int, float], j: dict[tuple[int, int], float], spins: tuple[int, ...]) -> float:
    value = sum(h[node] * spins[node] for node in h)
    value += sum(weight * spins[u] * spins[v] for (u, v), weight in j.items())
    return float(value)


def _exact_energy(h: dict[int, float], j: dict[tuple[int, int], float]) -> float:
    return min(_energy(h, j, spins) for spins in itertools.product((-1, 1), repeat=len(h)))


def _record(h: dict[int, float], j: dict[tuple[int, int], float], e0: float) -> dict:
    return {
        "instance_id": "ground-certificate-test",
        "problem": {
            "h": {str(node): value for node, value in h.items()},
            "J": [[u, v, value] for (u, v), value in j.items()],
            "e0": e0,
        },
    }


def _paper_contract():
    path = ROOT / "configs" / "quality_v2_paper_audit_v2.json"
    digest = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    return load_paper_audit_contract(path, digest)


def test_legacy_ground_paths_reject_aliasing_noncanonical_variable_ids() -> None:
    malformed = _record({1: -1.0}, {}, -1.0)
    malformed["problem"]["h"] = {"1": -1.0, "01": -1.0}

    with pytest.raises(ValueError, match="canonical integer strings"):
        _certified_ground_reference(malformed, {}, exhaustive_max_n=0)

    valid = _certified_ground_reference(
        _record({1: -1.0}, {}, -1.0),
        {},
        exhaustive_max_n=0,
    )
    registration = _paper_contract().evaluation["accepted_ground_references"][
        "ferromagnetic_s_t_min_cut"
    ]
    with pytest.raises(ValueError, match="canonical integer strings"):
        _validate_min_cut_ground_certificate(
            {"1": -1.0, "01": -1.0},
            [],
            valid,
            registration,
        )


def _checker_run() -> CheckerRun:
    return CheckerRun(
        source_sha256="1" * 64,
        environment_sha256="2" * 64,
        command=("python3", "verify_ground_state.py", "proof.json"),
        exit_code=0,
        log_sha256="3" * 64,
    )


def _certificate_envelope(bundle) -> dict[str, object]:
    return {
        "certificate": bundle.certificate.to_dict(),
        "proof": bundle.proof.to_dict(),
        "certificate_sha256": bundle.certificate.digest,
        "proof_artifact_sha256": bundle.proof.digest,
    }


def _exact_certificate_envelope() -> tuple[dict[str, object], dict[str, object]]:
    certified_problem = IsingProblem(
        variables=(0, 1),
        linear=((0, -1.0), (1, 0.25)),
        quadratic=((0, 1, 0.5),),
    )
    bundle = build_exact_enumeration_certificate(
        certified_problem,
        solver=SolverRun(
            name="embedbench-gray-code-enumerator",
            version="1",
            command=("python3", "certify_ground_state.py", "problem.json"),
            seed=None,
            deterministic_work_limit=4,
            safety_timeout_seconds=None,
        ),
        checker=_checker_run(),
    )
    record = _record(
        {0: -1.0, 1: 0.25},
        {(0, 1): 0.5},
        bundle.certificate.energy.to_float(),
    )
    return record["problem"], _certificate_envelope(bundle)


def test_evaluator_accepts_closed_embedded_exact_enumeration_certificate() -> None:
    problem, ground = _exact_certificate_envelope()

    _validate_ground_reference(
        problem,
        ground,
        _paper_contract(),
        problem_digest="1" * 64,
        exhaustive_cache={},
    )


def test_evaluator_accepts_closed_embedded_planted_certificate() -> None:
    certified_problem = IsingProblem(
        variables=(0,),
        linear=((0, -1.0),),
        quadratic=(),
    )
    bundle = build_planted_certificate(
        certified_problem,
        assignment=((0, 1),),
        terms=(
            PlantedTerm(
                term_index=0,
                linear=((0, Dyadic(-1, 0)),),
                quadratic=(),
                lower_bound=Dyadic(-1, 0),
            ),
        ),
        method="single_term_planting",
        checker=_checker_run(),
    )
    problem = _record({0: -1.0}, {}, -1.0)["problem"]

    _validate_ground_reference(
        problem,
        _certificate_envelope(bundle),
        _paper_contract(),
        problem_digest="2" * 64,
        exhaustive_cache={},
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda envelope: envelope.update(alias="exact_enumeration"),
        lambda envelope: envelope.update(certificate_sha256="0" * 64),
        lambda envelope: envelope["certificate"].update(status="certified_optimal"),
        lambda envelope: envelope["proof"].update(unregistered=True),
    ],
)
def test_embedded_ground_certificate_envelope_rejects_aliases_forgery_and_untrusted_status(
    mutation,
) -> None:
    problem, ground = _exact_certificate_envelope()
    mutation(ground)

    with pytest.raises((TypeError, ValueError)):
        _validate_ground_reference(
            problem,
            ground,
            _paper_contract(),
            problem_digest="3" * 64,
            exhaustive_cache={},
        )


def test_embedded_ground_certificate_cache_never_skips_artifact_digest_recomputation() -> None:
    problem, ground = _exact_certificate_envelope()
    cache: dict[str, tuple[str, float]] = {}
    _validate_ground_reference(
        problem,
        ground,
        _paper_contract(),
        problem_digest="4" * 64,
        exhaustive_cache=cache,
    )
    ground["proof"]["unregistered"] = True

    with pytest.raises(ValueError, match="proof artifact digest"):
        _validate_ground_reference(
            problem,
            ground,
            _paper_contract(),
            problem_digest="4" * 64,
            exhaustive_cache=cache,
        )


def test_evaluator_independently_rejects_a_forged_exhaustive_ground_energy() -> None:
    problem = _record({0: -1.0, 1: 0.25}, {(0, 1): 0.5}, 123.0)["problem"]
    ground = {
        "status": "certified_exact",
        "method": "exhaustive_enumeration",
        "energy": 123.0,
        "n_variables": 2,
    }

    with pytest.raises(ValueError, match="independent enumeration"):
        _validate_ground_reference(
            problem,
            ground,
            _paper_contract(),
            problem_digest="f" * 64,
            exhaustive_cache={},
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda ground: ground.update(n_variables=True),
        lambda ground: ground.update(unregistered=True),
    ],
)
def test_evaluator_exhaustive_ground_schema_is_closed_and_type_exact(mutation) -> None:
    problem = _record({0: -1.0}, {}, -1.0)["problem"]
    ground = {
        "status": "certified_exact",
        "method": "exhaustive_enumeration",
        "energy": -1.0,
        "n_variables": 1,
    }
    mutation(ground)

    with pytest.raises(ValueError):
        _validate_ground_reference(
            problem,
            ground,
            _paper_contract(),
            problem_digest="e" * 64,
            exhaustive_cache={},
        )


class _ForbiddenReleaseLabel:
    def __iter__(self):
        raise AssertionError("paper audit inspected a release label")


def test_audit_shard_identity_has_a_cross_process_golden_vector() -> None:
    identity = canonical_audit_identity(
        "quality.jsonl",
        {"instance_id": "instance-α", "focus": 7},
    )

    assert identity == b'["quality.jsonl","instance-\xce\xb1",7]'
    assert audit_shard_index(identity, 64) == 43


@pytest.mark.parametrize(
    ("filename", "record"),
    [
        ("../quality.jsonl", {"instance_id": "x", "focus": 0}),
        ("quality.jsonl", {"instance_id": "", "focus": 0}),
        ("quality.jsonl", {"instance_id": "x", "focus": True}),
    ],
)
def test_audit_shard_identity_rejects_ambiguous_fields(
    filename: str,
    record: dict,
) -> None:
    with pytest.raises(ValueError, match="audit identity"):
        canonical_audit_identity(filename, record)

    def __eq__(self, other: object) -> bool:
        raise AssertionError("paper audit inspected a release label")


def test_paper_record_order_never_reads_release_quality_labels() -> None:
    first = _record({0: -1.0}, {}, -1.0)
    first.update(instance_id="b", focus=1, p_solve=_ForbiddenReleaseLabel())
    second = _record({0: 1.0}, {}, -1.0)
    second.update(
        instance_id="a",
        focus=0,
        p_solve=_ForbiddenReleaseLabel(),
        best_index=_ForbiddenReleaseLabel(),
        stage=_ForbiddenReleaseLabel(),
    )

    selected = _records_for_audit(
        [first, second],
        per_file=2,
        low_margin=1.0,
        rng=random.Random(0),
        paper_mode=True,
    )

    assert selected == [second, first]


def test_paper_record_selection_requires_full_fixed_support() -> None:
    record = _record({0: -1.0}, {}, -1.0)

    with pytest.raises(ValueError, match="complete fixed-test record support"):
        _records_for_audit(
            [record, record],
            per_file=1,
            low_margin=0.1,
            rng=random.Random(0),
            paper_mode=True,
        )


def test_paper_audit_row_never_serializes_release_quality_labels(tmp_path: Path) -> None:
    from rescore_quality import build_audit_record

    record = {
        **_record({0: -1.0}, {}, -1.0),
        "focus": 0,
        "candidates": [[0], [1]],
        "resource_index": 0,
        "original_index": -1,
        "p_solve": _ForbiddenReleaseLabel(),
        "best_index": _ForbiddenReleaseLabel(),
        "stage": _ForbiddenReleaseLabel(),
    }
    corpus = tmp_path / "quality.jsonl"
    corpus.write_text("immutable bytes\n", encoding="utf-8")

    row = build_audit_record(
        corpus,
        record,
        scores=[0.25, 0.5],
        reads=4000,
        base_seed=0,
        paper_binding={"selection": {"sha256": "a" * 64}},
        realized_strengths=[0.5],
        realized_seeds=[[0], [1]],
        strength_p_solve=[[0.25], [0.5]],
        strength_success_counts=[[1000], [2000]],
        ground_reference={"status": "certified_exact"},
    )

    assert "corpus_best" not in row
    assert "corpus_margin" not in row
    assert "corpus_scores" not in row


def test_partial_paper_audit_cache_is_never_reused_even_when_canonical(tmp_path: Path) -> None:
    record = {
        **_record({0: -1.0}, {}, -1.0),
        "focus": 0,
        "candidates": [[0], [1]],
        "resource_index": 0,
        "original_index": -1,
        "p_solve": _ForbiddenReleaseLabel(),
        "best_index": _ForbiddenReleaseLabel(),
    }
    corpus = tmp_path / "quality.jsonl"
    corpus.write_text("immutable bytes\n", encoding="utf-8")
    expected = {
        "reads": 4000,
        "base_seed": 0,
        "corpus_digest": "a" * 64,
        "paper_binding": {"selection": {"sha256": "b" * 64}},
        "strengths": [0.5],
        "realized_seeds": [[0], [1]],
        "ground_reference": {"status": "certified_exact"},
    }
    row, payload = _paper_row_payload(
        corpus,
        record,
        scores=[0.25, 0.5],
        strength_p_solve=[[0.25], [0.5]],
        strength_success_counts=[[1000], [2000]],
        **expected,
    )
    destination = tmp_path / "records" / "row.json"
    _atomic_write(destination, payload)

    resumed = _load_resumable_paper_row(
        destination,
        corpus,
        record,
        **expected,
    )

    assert resumed is None
    tampered = dict(row)
    tampered["high_read_scores"] = [0.5, 0.25]
    destination.write_text(
        json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    assert (
        _load_resumable_paper_row(
            destination,
            corpus,
            record,
            **expected,
        )
        is None
    )


@pytest.mark.parametrize("seed", range(12))
def test_ferromagnetic_min_cut_matches_exhaustive_random_instances(seed: int) -> None:
    rng = random.Random(seed)
    n = 3 + seed % 5
    h = {node: rng.uniform(-2.0, 2.0) for node in range(n)}
    j = {
        (u, v): -rng.uniform(0.0, 2.0)
        for u in range(n)
        for v in range(u + 1, n)
        if rng.random() < 0.45
    }
    e0 = _exact_energy(h, j)

    result = _certified_ground_reference(
        _record(h, j, e0),
        {},
        exhaustive_max_n=0,
    )

    assert result["status"] == "certified_exact"
    assert result["method"] == "ferromagnetic_s_t_min_cut"
    assert result["energy"] == pytest.approx(e0, abs=1e-12)
    certificate = result["certificate"]
    assert certificate["algorithm"] == "networkx_preflow_push_integer_capacities_v1"
    assert certificate["capacity_scale"] > 0
    assert certificate["cut_value_scaled"] >= 0
    assert sorted(certificate["source_variables"] + certificate["sink_variables"]) == list(range(n))
    spins = tuple(-1 if node in certificate["source_variables"] else 1 for node in range(n))
    assert _energy(h, j, spins) == pytest.approx(e0, abs=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_registered_exhaustive_certifier_matches_brute_force(seed: int) -> None:
    rng = random.Random(f"nonferromagnetic:{seed}")
    n = 2 + seed % 6
    h = {node: rng.uniform(-1.5, 1.5) for node in range(n)}
    j = {
        (u, v): rng.uniform(-1.5, 1.5)
        for u in range(n)
        for v in range(u + 1, n)
        if rng.random() < 0.6
    }
    e0 = _exact_energy(h, j)

    result = _certified_ground_reference(_record(h, j, e0), {})

    assert result == {
        "status": "certified_exact",
        "method": "exhaustive_enumeration",
        "energy": e0,
        "n_variables": n,
    }


def test_large_ferromagnetic_problem_is_certified_without_exhaustive_enumeration() -> None:
    n = 25
    h = {node: -1.0 for node in range(n)}
    j = {(node, node + 1): -0.7 for node in range(n - 1)}
    # Every term is minimized by the all-positive state, while the certifier still has to
    # establish global optimality through the registered min-cut reduction.
    e0 = sum(h.values()) + sum(j.values())
    record = _record(h, j, e0)

    result = _certified_ground_reference(record, {})

    assert result["status"] == "certified_exact"
    assert result["method"] == "ferromagnetic_s_t_min_cut"
    assert result["n_variables"] == n


def test_large_nonferromagnetic_problem_is_rejected_fail_closed() -> None:
    h = {node: 0.0 for node in range(23)}
    j = {(0, 1): 0.25}

    result = _certified_ground_reference(_record(h, j, -0.25), {})

    assert result == {
        "status": "uncertified_best_known",
        "method": "not_exactly_verifiable_by_registered_auditor",
        "energy": -0.25,
        "n_variables": 23,
    }


def test_min_cut_certificate_rejects_a_stale_reference_energy() -> None:
    h = {node: float(node % 2) - 0.25 for node in range(23)}
    j = {(node, node + 1): -0.5 for node in range(22)}

    with pytest.raises(ValueError, match="problem.e0 disagrees"):
        _certified_ground_reference(_record(h, j, 123.0), {})


@pytest.mark.parametrize(
    ("h", "j"),
    [
        ({0: 0.0}, {}),
        ({0: -1e-12, 1: 1e12}, {(0, 1): -1e-9}),
        ({0: -0.5, 1: 0.25, 2: 0.0}, {(0, 1): 0.0}),
        ({0: 0.4, 1: -0.6, 2: 0.1, 3: -0.3}, {(0, 1): -2.0, (2, 3): -0.1}),
    ],
)
def test_min_cut_handles_zero_disconnected_and_scaled_coefficients(
    h: dict[int, float],
    j: dict[tuple[int, int], float],
) -> None:
    e0 = _exact_energy(h, j)

    result = _certified_ground_reference(
        _record(h, j, e0),
        {},
        exhaustive_max_n=0,
    )

    assert result["status"] == "certified_exact"
    assert result["energy"] == pytest.approx(e0, rel=1e-12, abs=1e-12)
