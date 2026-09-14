from __future__ import annotations

from collections import UserDict
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import embedbench.hard_ood_schema as hard_ood_schema
import numpy as np
import pytest
from embedbench.hard_ood_schema import (
    SEED_REGISTRY_ROOT_FILENAME,
    ArtifactEntry,
    ReleaseIdentity,
    SeedRegistration,
    SeedRegistry,
    SeedRegistryArtifactManifest,
    SeedRegistryRoot,
    SeedRegistryShard,
    SeedRegistryStageArtifact,
    SeedRegistryVerification,
    SeedRequest,
    VerifiedSeedResolver,
    allocate_seed32,
    canonical_bytes,
    canonical_sha256,
    canonical_value,
    require_exact_keys,
    require_sharded_seed_request,
    rng_for_sharded_seed_request,
    third_party_seed32_for_sharded_seed_request,
    validate_seed_registry_artifact_manifest,
    validate_seed_resolver,
    validate_versioned_object,
    verify_seed_registry_artifact_chain,
    verify_seed_registry_shards,
    write_seed_registry_shards,
)

OPTIONAL_SEED_FIELDS = frozenset(
    {
        "task_type",
        "problem_sha256",
        "state_sha256",
        "candidate_index",
        "strength_index",
        "label_stage",
    }
)
REQUIRED_SEED_FIELDS = {
    "problem": frozenset(),
    "host": frozenset(),
    "witness": frozenset({"problem_sha256"}),
    "solver_attempt": frozenset({"problem_sha256"}),
    "mechanism": frozenset({"problem_sha256", "state_sha256"}),
    "focus": frozenset({"problem_sha256"}),
    "candidate_sample": frozenset({"problem_sha256"}),
    "screen_label": OPTIONAL_SEED_FIELDS,
    "refine_label": OPTIONAL_SEED_FIELDS,
    "locked_label": OPTIONAL_SEED_FIELDS,
    "decode_tie": OPTIONAL_SEED_FIELDS,
    "audit_label": OPTIONAL_SEED_FIELDS,
    "audit_bootstrap": frozenset(),
}
NULL_SEED_FIELDS = {
    purpose: OPTIONAL_SEED_FIELDS - required for purpose, required in REQUIRED_SEED_FIELDS.items()
}


def _valid_seed_values(purpose: str, *, replicate: int = 0) -> dict[str, object]:
    required = REQUIRED_SEED_FIELDS[purpose]
    task_type = "terminal_quality"
    label_stages = {
        "screen_label": "screen",
        "refine_label": "refine",
        "locked_label": "locked",
        "decode_tie": "locked",
        "audit_label": "audit",
    }
    return {
        "release_id": "embedbench-hard-ood-v1.0.0",
        "purpose": purpose,
        "partition": "locked_ood" if purpose in {"locked_label", "decode_tie"} else "hard_dev",
        "panel": "terminal_quality",
        "cell": "cell-001",
        "task_type": task_type if "task_type" in required else None,
        "problem_sha256": "1" * 64 if "problem_sha256" in required else None,
        "state_sha256": "2" * 64 if "state_sha256" in required else None,
        "candidate_index": 3 if "candidate_index" in required else None,
        "strength_index": 1 if "strength_index" in required else None,
        "label_stage": label_stages.get(purpose) if "label_stage" in required else None,
        "replicate": replicate,
    }


def test_canonical_bytes_and_digest_match_golden_vector() -> None:
    value = {
        "weight": -0.0,
        "nested": [1.5, True, None],
        "label": "cafe\u0301",
    }

    expected = (
        b'{"label":"caf\\u00e9","nested":['
        b'{"__float64_hex__":"0x1.8000000000000p+0"},true,null],'
        b'"weight":{"__float64_hex__":"0x0.0p+0"}}'
    )
    assert canonical_bytes(value) == expected
    assert canonical_sha256(value) == (
        "3fd926d1c8ad5452e8e17391a7cd96691ce4c0af970005901e3c39eba7916af2"
    )


def test_canonical_bytes_are_order_and_sequence_representation_independent() -> None:
    left = {"z": (3, 2, 1), "a": {"y": "value", "x": 4}}
    right = {"a": {"x": 4, "y": "value"}, "z": [3, 2, 1]}

    assert canonical_bytes(left) == canonical_bytes(right)


def test_canonical_value_normalizes_negative_zero() -> None:
    assert canonical_value(-0.0) == {"__float64_hex__": "0x0.0p+0"}
    assert canonical_bytes(-0.0) == canonical_bytes(0.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_value_rejects_nonfinite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite float"):
        canonical_value(value)


def test_canonical_value_normalizes_unicode_strings_and_keys_to_nfc() -> None:
    assert canonical_bytes({"e\u0301": "cafe\u0301"}) == b'{"\\u00e9":"caf\\u00e9"}'


def test_canonical_value_rejects_unicode_normalized_key_collision() -> None:
    with pytest.raises(ValueError, match="duplicate normalized or reserved key"):
        canonical_value({"\u00e9": 1, "e\u0301": 2})


@pytest.mark.parametrize("value", ["\ud800", "prefix\udfff", {"\ud800": "value"}])
def test_canonical_value_rejects_unicode_surrogates(value: object) -> None:
    with pytest.raises(ValueError, match="Unicode surrogate"):
        canonical_value(value)


def test_canonical_value_rejects_reserved_float_tag_from_input() -> None:
    with pytest.raises(ValueError, match="duplicate normalized or reserved key"):
        canonical_value({"__float64_hex__": "forged"})


def test_canonical_value_rejects_non_string_mapping_key() -> None:
    with pytest.raises(TypeError, match="canonical object keys must be strings"):
        canonical_value({1: "value"})


@pytest.mark.parametrize("value", [b"bytes", {1, 2}, object()])
def test_canonical_value_rejects_unsupported_types(value: object) -> None:
    with pytest.raises(TypeError, match="unsupported canonical type"):
        canonical_value(value)


class _FloatSubclass(float):
    pass


class _IntSubclass(int):
    pass


class _StringSubclass(str):
    pass


class _ListSubclass(list[object]):
    pass


class _TupleSubclass(tuple[object, ...]):
    pass


class _DictSubclass(dict[str, object]):
    pass


@pytest.mark.parametrize(
    "value",
    [
        _FloatSubclass(1.0),
        _IntSubclass(1),
        _StringSubclass("value"),
        _ListSubclass([1]),
        _TupleSubclass((1,)),
        _DictSubclass(value=1),
        UserDict({"value": 1}),
        MappingProxyType({"value": 1}),
        np.float64(1.0),
        np.int64(1),
    ],
)
def test_canonical_value_rejects_subclasses_and_adversarial_mappings(value: object) -> None:
    with pytest.raises(TypeError, match="unsupported canonical type"):
        canonical_value(value)


def test_exact_key_validator_accepts_only_a_plain_object_with_exact_fields() -> None:
    value = {"first": 1, "second": 2}

    assert require_exact_keys(value, {"first", "second"}, "fixture") is value

    with pytest.raises(ValueError, match=r"missing=\['second'\], unknown=\[\]"):
        require_exact_keys({"first": 1}, {"first", "second"}, "fixture")
    with pytest.raises(ValueError, match=r"missing=\[\], unknown=\['extra'\]"):
        require_exact_keys(
            {"first": 1, "second": 2, "extra": 3},
            {"first", "second"},
            "fixture",
        )
    with pytest.raises(TypeError, match="fixture must be a JSON object"):
        require_exact_keys([], {"first"}, "fixture")
    with pytest.raises(TypeError, match="fixture keys must be strings"):
        require_exact_keys({1: "value"}, {"first"}, "fixture")


def test_versioned_object_validator_rejects_schema_version_and_field_drift() -> None:
    value = {
        "schema": "embedbench.fixture",
        "schema_version": 1,
        "payload": "ok",
    }
    assert (
        validate_versioned_object(
            value,
            schema="embedbench.fixture",
            schema_version=1,
            fields={"payload"},
            name="fixture",
        )
        is value
    )

    for drifted, message in (
        ({**value, "unknown": 1}, "schema fields differ"),
        ({"schema": value["schema"], "schema_version": 1}, "schema fields differ"),
        ({**value, "schema": "embedbench.other"}, "requires schema"),
        ({**value, "schema_version": 2}, "requires schema_version 1"),
        ({**value, "schema_version": True}, "requires schema_version 1"),
        ({**value, "schema_version": _IntSubclass(1)}, "requires schema_version 1"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_versioned_object(
                drifted,
                schema="embedbench.fixture",
                schema_version=1,
                fields={"payload"},
                name="fixture",
            )


class _EqualSchemaProxy:
    def __eq__(self, other: object) -> bool:
        return other == "embedbench.fixture"

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


@pytest.mark.parametrize(
    "schema_value",
    [_StringSubclass("embedbench.fixture"), _EqualSchemaProxy()],
)
def test_versioned_object_validator_requires_exact_string_discriminator(
    schema_value: object,
) -> None:
    with pytest.raises(ValueError, match="requires schema"):
        validate_versioned_object(
            {
                "schema": schema_value,
                "schema_version": 1,
                "payload": "ok",
            },
            schema="embedbench.fixture",
            schema_version=1,
            fields={"payload"},
            name="fixture",
        )


def test_release_identity_is_frozen_normalized_and_strictly_round_trips() -> None:
    release = ReleaseIdentity("embedbench-hard-ood-v1.0.0")

    assert release.to_dict() == {"release_id": "embedbench-hard-ood-v1.0.0"}
    assert ReleaseIdentity.from_dict(release.to_dict()) == release
    with pytest.raises(FrozenInstanceError):
        release.release_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="release identity schema fields differ"):
        ReleaseIdentity.from_dict({**release.to_dict(), "extra": True})
    with pytest.raises(ValueError, match="release_id must be a non-empty string"):
        ReleaseIdentity("")


def test_artifact_entry_is_frozen_strict_and_can_bind_raw_file_bytes() -> None:
    entry = ArtifactEntry.from_payload(
        relative_path="records/hard_dev/part-000.jsonl",
        payload=b"one\ntwo\n",
        record_count=2,
        schema_version=1,
    )

    assert entry.to_dict() == {
        "relative_path": "records/hard_dev/part-000.jsonl",
        "sha256": "c3f9c8c283a2b1f2f1896f27a01cbe3cddc0c9d93f752e4639035a0f5b36f6e8",
        "byte_count": 8,
        "record_count": 2,
        "schema_version": 1,
    }
    assert ArtifactEntry.from_dict(entry.to_dict()) == entry
    with pytest.raises(FrozenInstanceError):
        entry.record_count = 3  # type: ignore[misc]
    with pytest.raises(ValueError, match="artifact entry schema fields differ"):
        ArtifactEntry.from_dict({**entry.to_dict(), "rows": 2})


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"relative_path": "."}, "relative POSIX path"),
        ({"relative_path": "/absolute.json"}, "relative POSIX path"),
        ({"relative_path": "records\\part.jsonl"}, "relative POSIX path"),
        ({"relative_path": "records/../escape.json"}, "relative POSIX path"),
        ({"relative_path": "records/part\x00.jsonl"}, "relative POSIX path"),
        ({"relative_path": "records/part\nforged.jsonl"}, "control character"),
        ({"relative_path": "records/part\rforged.jsonl"}, "control character"),
        ({"relative_path": "records/part\tforged.jsonl"}, "control character"),
        ({"relative_path": "records/part\u202eforged.jsonl"}, "control character"),
        ({"relative_path": "records/\ud800.jsonl"}, "Unicode surrogate"),
        ({"relative_path": "C:/absolute-on-windows.jsonl"}, "Windows drive"),
        ({"relative_path": "C:drive-relative.jsonl"}, "Windows drive"),
        ({"relative_path": "records/cafe\u0301.jsonl"}, "NFC-normalized"),
        ({"sha256": "A" * 64}, "lowercase SHA-256"),
        ({"byte_count": -1}, "byte_count must be a non-negative integer"),
        ({"record_count": True}, "record_count must be a non-negative integer"),
        ({"record_count": _IntSubclass(1)}, "record_count must be a non-negative integer"),
        ({"schema_version": 0}, "schema_version must be a positive integer"),
    ],
)
def test_artifact_entry_rejects_invalid_identity_fields(
    updates: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "relative_path": "records/part.jsonl",
        "sha256": "a" * 64,
        "byte_count": 10,
        "record_count": 1,
        "schema_version": 1,
    }
    values.update(updates)

    with pytest.raises(ValueError, match=message):
        ArtifactEntry.from_dict(values)


def test_seed_request_round_trip_and_seed_key_match_golden_vector() -> None:
    request = SeedRequest(
        release_id="embedbench-hard-ood-v1.0.0",
        purpose="locked_label",
        partition="locked_ood",
        panel="terminal_quality",
        cell="pegasus/p3/heavy",
        task_type="terminal_quality",
        problem_sha256="01" * 32,
        state_sha256="ab" * 32,
        candidate_index=7,
        strength_index=2,
        label_stage="locked",
        replicate=3,
    )

    expected_preimage = (
        b'{"candidate_index":7,"cell":"pegasus/p3/heavy","label_stage":"locked",'
        b'"panel":"terminal_quality","partition":"locked_ood","problem_sha256":"'
        + b"01"
        * 32
        + b'","purpose":"locked_label","release_id":"embedbench-hard-ood-v1.0.0",'
        b'"replicate":3,"state_sha256":"'
        + b"ab" * 32
        + b'","strength_index":2,"task_type":"terminal_quality"}'
    )
    assert request.canonical_preimage() == expected_preimage
    assert request.seed_key_hex == (
        "c01af2269fe81007e60c8300957cb89e51fb796e5d4e06b4f99d84cbab0944de"
    )
    assert request.seed_words == (
        3222991398,
        2682785799,
        3859579648,
        2507978910,
        1375435118,
        1565394612,
        4187849931,
        2869511390,
    )
    assert SeedRequest.from_dict(request.to_dict()) == request
    with pytest.raises(FrozenInstanceError):
        request.replicate = 4  # type: ignore[misc]


def test_structure_solvers_share_one_registered_attempt_seed_namespace() -> None:
    shared = SeedRequest.from_dict(_valid_seed_values("solver_attempt", replicate=7))

    assert shared.purpose == "solver_attempt"
    assert shared.seed_key_hex == SeedRequest.from_dict(shared.to_dict()).seed_key_hex
    for obsolete_solver_specific_purpose in ("minorminer", "cpp_baseline"):
        values = shared.to_dict()
        values["purpose"] = obsolete_solver_specific_purpose
        with pytest.raises(ValueError, match="unregistered seed purpose"):
            SeedRequest.from_dict(values)


@pytest.mark.parametrize("purpose", sorted(REQUIRED_SEED_FIELDS))
def test_every_seed_purpose_has_one_valid_required_null_field_shape(purpose: str) -> None:
    request = SeedRequest.from_dict(_valid_seed_values(purpose))

    assert request.purpose == purpose
    for field in REQUIRED_SEED_FIELDS[purpose]:
        assert getattr(request, field) is not None
    for field in NULL_SEED_FIELDS[purpose]:
        assert getattr(request, field) is None


@pytest.mark.parametrize(
    ("purpose", "field"),
    [
        (purpose, field)
        for purpose, required in REQUIRED_SEED_FIELDS.items()
        for field in sorted(required)
    ],
)
def test_seed_purpose_rejects_each_missing_required_field(purpose: str, field: str) -> None:
    values = _valid_seed_values(purpose)
    values[field] = None

    with pytest.raises(ValueError, match=rf"{field} must be non-null for seed purpose"):
        SeedRequest.from_dict(values)


@pytest.mark.parametrize(
    ("purpose", "field"),
    [
        (purpose, field)
        for purpose, null_fields in NULL_SEED_FIELDS.items()
        for field in sorted(null_fields)
    ],
)
def test_seed_purpose_rejects_each_irrelevant_nonnull_field(purpose: str, field: str) -> None:
    values = _valid_seed_values(purpose)
    nonnull_values: dict[str, object] = {
        "task_type": "terminal_quality",
        "problem_sha256": "3" * 64,
        "state_sha256": "4" * 64,
        "candidate_index": 0,
        "strength_index": 0,
        "label_stage": "screen",
    }
    values[field] = nonnull_values[field]

    with pytest.raises(ValueError, match=rf"{field} must be null for seed purpose"):
        SeedRequest.from_dict(values)


@pytest.mark.parametrize(
    ("purpose", "task_type"),
    [
        ("screen_label", "partial_structural"),
        ("refine_label", "partial_structural"),
        ("locked_label", "partial_structural"),
        ("decode_tie", "partial_structural"),
        ("audit_label", "partial_structural"),
    ],
)
def test_seed_purpose_rejects_inappropriate_task_type(purpose: str, task_type: str) -> None:
    values = _valid_seed_values(purpose)
    values["task_type"] = task_type

    with pytest.raises(ValueError, match="task_type is invalid for seed purpose"):
        SeedRequest.from_dict(values)


@pytest.mark.parametrize(
    ("purpose", "wrong_stage"),
    [
        ("screen_label", "refine"),
        ("refine_label", "screen"),
        ("locked_label", "audit"),
        ("audit_label", "locked"),
    ],
)
def test_label_seed_purpose_rejects_wrong_label_stage(purpose: str, wrong_stage: str) -> None:
    values = _valid_seed_values(purpose)
    values["label_stage"] = wrong_stage

    with pytest.raises(ValueError, match="label_stage is invalid for seed purpose"):
        SeedRequest.from_dict(values)


@pytest.mark.parametrize("label_stage", ["screen", "refine", "locked", "audit"])
def test_decode_tie_accepts_each_registered_label_stage(label_stage: str) -> None:
    values = _valid_seed_values("decode_tie")
    values["label_stage"] = label_stage

    assert SeedRequest.from_dict(values).label_stage == label_stage


def test_seed_request_constructs_pcg64dxsm_from_all_eight_big_endian_words() -> None:
    request = SeedRequest(
        release_id="embedbench-hard-ood-v1.0.0",
        purpose="locked_label",
        partition="locked_ood",
        panel="terminal_quality",
        cell="pegasus/p3/heavy",
        task_type="terminal_quality",
        problem_sha256="01" * 32,
        state_sha256="ab" * 32,
        candidate_index=7,
        strength_index=2,
        label_stage="locked",
        replicate=3,
    )

    registry = SeedRegistry.build([request], stage_id="locked-labels")
    rng = registry.rng_for(request)
    assert not hasattr(request, "make_rng")
    assert isinstance(rng.bit_generator, np.random.PCG64DXSM)
    assert rng.bit_generator.random_raw(6).tolist() == [
        10604001438409506517,
        4019337015554564660,
        14975131130004900796,
        14973193223735593065,
        759005771706853311,
        16305519620496373129,
    ]


def test_seed32_allocation_matches_direct_and_forced_collision_golden_vectors() -> None:
    direct = bytes.fromhex("11223344" + "00" * 28)
    colliding = bytes.fromhex("11223344" + "ff" * 28)

    allocations = allocate_seed32([colliding, direct])

    assert [allocation.seed_key for allocation in allocations] == [direct, colliding]
    assert [allocation.seed32 for allocation in allocations] == [287454020, 4056338324]
    assert [allocation.collision_counter for allocation in allocations] == [0, 1]
    with pytest.raises(FrozenInstanceError):
        allocations[0].seed32 = 0  # type: ignore[misc]


def test_seed32_allocation_preserves_inherited_values_when_a_later_stage_sorts_earlier() -> None:
    inherited_key = bytes.fromhex("11223344" + "ff" * 28)
    later_stage_key = bytes.fromhex("11223344" + "00" * 28)
    inherited = allocate_seed32([inherited_key])

    extension = allocate_seed32(
        [later_stage_key],
        occupied_seed32=[inherited[0].seed32],
    )

    assert inherited[0].seed32 == 287454020
    assert inherited[0].collision_counter == 0
    assert extension[0].seed32 == 2763398642
    assert extension[0].collision_counter == 1


def test_seed32_allocation_rejects_duplicate_or_malformed_keys() -> None:
    key = bytes.fromhex("11223344" + "00" * 28)

    with pytest.raises(ValueError, match="duplicate seed key"):
        allocate_seed32([key, key])
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        allocate_seed32([b"short"])
    with pytest.raises(TypeError, match="seed keys must be exact bytes"):
        allocate_seed32([bytearray(key)])  # type: ignore[list-item]


def test_seed_registry_is_deterministic_frozen_and_canonically_persistable() -> None:
    assert "SeedRegistry" not in hard_ood_schema.__all__
    first = SeedRequest.from_dict(_valid_seed_values("problem", replicate=0))
    second = SeedRequest.from_dict(_valid_seed_values("problem", replicate=1))

    registry = SeedRegistry.build(
        [second, first],
        stage_id="foundation",
        third_party_32_requests=[second, first],
    )
    same_registry = SeedRegistry.build(
        [first, second],
        stage_id="foundation",
        third_party_32_requests=[first, second],
    )
    payload = registry.to_bytes()

    assert registry == same_registry
    assert registry.to_dict()["schema"] == "embedbench.seed-registry-stage"
    assert registry.to_bytes() == canonical_bytes(registry.to_dict())
    assert SeedRegistry.from_bytes(payload) == registry
    assert registry.sha256 == canonical_sha256(registry.to_dict())
    assert registry.require(first).request == first
    assert registry.third_party_seed32_for(first) == registry.require(first).seed32
    assert (
        registry.rng_for(first).bit_generator.random_raw()
        == registry.rng_for(first).bit_generator.random_raw()
    )
    with pytest.raises(FrozenInstanceError):
        registry.release_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        registry.entries[0].seed32 = 0  # type: ignore[misc]


def test_seed_registry_stage_extension_preserves_parent_collision_assignment() -> None:
    requests = []
    for replicate in (1451, 38751):
        values = _valid_seed_values("problem", replicate=replicate)
        values["cell"] = "collision-golden"
        requests.append(SeedRequest.from_dict(values))
    inherited_request, earlier_sorting_request = requests
    parent = SeedRegistry.build(
        [inherited_request],
        stage_id="structure-solvers",
        third_party_32_requests=[inherited_request],
    )

    child = SeedRegistry.build(
        [earlier_sorting_request],
        stage_id="quality-labels",
        parent=parent,
        third_party_32_requests=[earlier_sorting_request],
    )

    assert child.parent_registry_sha256 == parent.sha256
    assert child.require(inherited_request).seed32 == 2630587898
    assert child.require(inherited_request).collision_counter == 0
    assert child.require(earlier_sorting_request).seed32 == 1210714687
    assert child.require(earlier_sorting_request).collision_counter == 1
    assert SeedRegistry.from_bytes(child.to_bytes(), parent=parent) == child
    with pytest.raises(ValueError, match="parent seed registry is required"):
        SeedRegistry.from_bytes(child.to_bytes())


def test_pcg_only_stage_requests_do_not_consume_or_perturb_seed32() -> None:
    parent_request = SeedRequest.from_dict(_valid_seed_values("problem", replicate=1451))
    pcg_request = SeedRequest.from_dict(_valid_seed_values("problem", replicate=38751))
    parent = SeedRegistry.build(
        [parent_request],
        stage_id="structure-solvers",
        third_party_32_requests=[parent_request],
    )

    child = SeedRegistry.build(
        [pcg_request],
        stage_id="quality-labels",
        parent=parent,
    )

    entry = child.require(pcg_request)
    assert entry.seed32_required is False
    assert entry.seed32 is None
    assert entry.collision_counter is None
    assert child.occupied_seed32 == parent.occupied_seed32
    with pytest.raises(ValueError, match="does not require a 32-bit seed"):
        child.third_party_seed32_for(pcg_request)


def test_seed_registry_rejects_duplicate_request_across_parent_stage() -> None:
    request = SeedRequest.from_dict(_valid_seed_values("problem"))
    parent = SeedRegistry.build([request], stage_id="foundation")

    with pytest.raises(ValueError, match="already registered in a parent stage"):
        SeedRegistry.build([request], stage_id="structure", parent=parent)


def test_seed_registry_rejects_stage_id_reused_from_any_ancestor() -> None:
    first = SeedRequest.from_dict(_valid_seed_values("problem", replicate=0))
    second = SeedRequest.from_dict(_valid_seed_values("problem", replicate=1))
    third = SeedRequest.from_dict(_valid_seed_values("problem", replicate=2))
    foundation = SeedRegistry.build([first], stage_id="foundation")
    structure = SeedRegistry.build([second], stage_id="structure", parent=foundation)

    with pytest.raises(ValueError, match="stage_id is already present in its ancestor chain"):
        SeedRegistry.build([third], stage_id="foundation", parent=structure)


def _collision_seed_requests() -> tuple[SeedRequest, SeedRequest]:
    requests = []
    for replicate in (1451, 38751):
        values = _valid_seed_values("problem", replicate=replicate)
        values["cell"] = "collision-golden"
        requests.append(SeedRequest.from_dict(values))
    return requests[0], requests[1]


def _read_stage_payload_files(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file()}


def _verify_stage(
    directory: Path,
    root: SeedRegistryRoot,
    *,
    parent: SeedRegistryVerification | None = None,
) -> SeedRegistryVerification:
    return verify_seed_registry_shards(
        directory,
        expected_root_sha256=root.sha256,
        parent=parent,
    )


def test_sharded_seed_registry_is_deterministic_under_input_and_chunk_order(
    tmp_path: Path,
) -> None:
    inherited_request, _ = _collision_seed_requests()
    requests = [
        SeedRequest.from_dict(_valid_seed_values("problem", replicate=index)) for index in range(7)
    ]
    registrations = [
        SeedRegistration(request=inherited_request, seed32_required=True),
        *(SeedRegistration(request=request, seed32_required=False) for request in requests),
    ]

    first_root = write_seed_registry_shards(
        iter(reversed(registrations)),
        directory=tmp_path / "first",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="foundation",
        shard_entry_limit=3,
        sort_chunk_limit=2,
    )
    second_root = write_seed_registry_shards(
        iter(registrations[::2] + registrations[1::2]),
        directory=tmp_path / "second",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="foundation",
        shard_entry_limit=3,
        sort_chunk_limit=5,
    )
    first = _verify_stage(tmp_path / "first", first_root)
    second = _verify_stage(tmp_path / "second", second_root)

    assert first_root == second_root
    assert first.root == first_root
    assert second.root == second_root
    assert first_root.entry_count == len(registrations)
    assert [shard.record_count for shard in first_root.shards] == [3, 3, 2]
    assert first.occupied_seed32 == frozenset({2630587898})
    assert _read_stage_payload_files(tmp_path / "first") == _read_stage_payload_files(
        tmp_path / "second"
    )
    assert (
        SeedRegistryRoot.from_bytes((tmp_path / "first" / SEED_REGISTRY_ROOT_FILENAME).read_bytes())
        == first_root
    )


def test_sharded_child_stage_binds_parent_and_preserves_collision_assignment(
    tmp_path: Path,
) -> None:
    inherited_request, earlier_sorting_request = _collision_seed_requests()
    parent_root = write_seed_registry_shards(
        [SeedRegistration(inherited_request, seed32_required=True)],
        directory=tmp_path / "parent",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="structure-solvers",
        shard_entry_limit=2,
        sort_chunk_limit=2,
    )
    parent = _verify_stage(tmp_path / "parent", parent_root)
    pcg_requests = [
        SeedRequest.from_dict(_valid_seed_values("audit_bootstrap", replicate=index))
        for index in range(5)
    ]

    child_root = write_seed_registry_shards(
        [
            SeedRegistration(earlier_sorting_request, seed32_required=True),
            *(SeedRegistration(request, seed32_required=False) for request in pcg_requests),
        ],
        directory=tmp_path / "child",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="quality-labels",
        parent=parent,
        shard_entry_limit=2,
        sort_chunk_limit=2,
    )
    child = _verify_stage(tmp_path / "child", child_root, parent=parent)

    assert child_root.parent_root_sha256 == parent_root.sha256
    assert parent.occupied_seed32 == frozenset({2630587898})
    assert child.occupied_seed32 == frozenset({2630587898, 1210714687})
    assert not hasattr(child, "stage_allocations")
    resolver = VerifiedSeedResolver(
        child,
        expected_terminal_root_sha256=child_root.sha256,
    )
    assert (
        validate_seed_resolver(
            resolver,
            expected_terminal_root_sha256=child_root.sha256,
        )
        is resolver
    )
    assert require_sharded_seed_request(resolver, inherited_request).seed32 == 2630587898
    assert require_sharded_seed_request(resolver, earlier_sorting_request).seed32 == 1210714687
    assert (
        third_party_seed32_for_sharded_seed_request(resolver, earlier_sorting_request) == 1210714687
    )
    assert isinstance(
        rng_for_sharded_seed_request(resolver, pcg_requests[0]).bit_generator,
        np.random.PCG64DXSM,
    )
    with pytest.raises(TypeError, match="VerifiedSeedResolver"):
        rng_for_sharded_seed_request(child, pcg_requests[0])
    unplanned = SeedRequest.from_dict(_valid_seed_values("audit_bootstrap", replicate=99))
    with pytest.raises(KeyError, match="not registered"):
        require_sharded_seed_request(resolver, unplanned)


def test_pcg_only_child_reuses_parent_occupied_seed32_inventory(tmp_path: Path) -> None:
    parent_request, _ = _collision_seed_requests()
    parent_root = write_seed_registry_shards(
        [SeedRegistration(parent_request, seed32_required=True)],
        directory=tmp_path / "parent",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="parent",
    )
    parent = _verify_stage(tmp_path / "parent", parent_root)
    child_request = SeedRequest.from_dict(_valid_seed_values("audit_bootstrap"))
    child_root = write_seed_registry_shards(
        [SeedRegistration(child_request, seed32_required=False)],
        directory=tmp_path / "child",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="child",
        parent=parent,
    )

    child = _verify_stage(tmp_path / "child", child_root, parent=parent)

    assert child.occupied_seed32 is parent.occupied_seed32


def test_sharded_seed_registry_verifier_rejects_parent_and_payload_tampering(
    tmp_path: Path,
) -> None:
    request = SeedRequest.from_dict(_valid_seed_values("problem"))
    write_seed_registry_shards(
        [SeedRegistration(request, seed32_required=False)],
        directory=tmp_path / "root",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="foundation",
        shard_entry_limit=2,
        sort_chunk_limit=2,
    )
    root_artifact = SeedRegistryRoot.from_bytes(
        (tmp_path / "root" / SEED_REGISTRY_ROOT_FILENAME).read_bytes()
    )
    root = _verify_stage(tmp_path / "root", root_artifact)
    other_request = SeedRequest.from_dict(_valid_seed_values("host"))
    write_seed_registry_shards(
        [SeedRegistration(other_request, seed32_required=False)],
        directory=tmp_path / "child",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="structure",
        parent=root,
        shard_entry_limit=2,
        sort_chunk_limit=2,
    )

    with pytest.raises(ValueError, match="parent seed-registry root is required"):
        verify_seed_registry_shards(
            tmp_path / "child",
            expected_root_sha256=SeedRegistryRoot.from_bytes(
                (tmp_path / "child" / SEED_REGISTRY_ROOT_FILENAME).read_bytes()
            ).sha256,
        )

    shard_path = tmp_path / "root" / root.root.shards[0].relative_path
    shard_path.write_bytes(shard_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="bytes|descriptor"):
        verify_seed_registry_shards(
            tmp_path / "root",
            expected_root_sha256=root_artifact.sha256,
        )


def test_sharded_seed_registry_rejects_request_repeated_from_parent(tmp_path: Path) -> None:
    request = SeedRequest.from_dict(_valid_seed_values("problem"))
    write_seed_registry_shards(
        [SeedRegistration(request, seed32_required=False)],
        directory=tmp_path / "parent",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="foundation",
    )
    parent_root = SeedRegistryRoot.from_bytes(
        (tmp_path / "parent" / SEED_REGISTRY_ROOT_FILENAME).read_bytes()
    )
    parent = _verify_stage(tmp_path / "parent", parent_root)

    with pytest.raises(ValueError, match="already registered in an ancestor stage"):
        write_seed_registry_shards(
            [SeedRegistration(request, seed32_required=False)],
            directory=tmp_path / "child",
            release_id="embedbench-hard-ood-v1.0.0",
            stage_id="structure",
            parent=parent,
        )


def test_sharded_seed_registry_rejects_stage_id_reused_from_ancestor(tmp_path: Path) -> None:
    requests = [
        SeedRequest.from_dict(_valid_seed_values("problem", replicate=index)) for index in range(3)
    ]
    write_seed_registry_shards(
        [SeedRegistration(requests[0], seed32_required=False)],
        directory=tmp_path / "foundation",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="foundation",
    )
    foundation_root = SeedRegistryRoot.from_bytes(
        (tmp_path / "foundation" / SEED_REGISTRY_ROOT_FILENAME).read_bytes()
    )
    foundation = _verify_stage(tmp_path / "foundation", foundation_root)
    structure_root = write_seed_registry_shards(
        [SeedRegistration(requests[1], seed32_required=False)],
        directory=tmp_path / "structure",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="structure",
        parent=foundation,
    )
    structure = _verify_stage(tmp_path / "structure", structure_root, parent=foundation)

    with pytest.raises(ValueError, match="stage_id is already present in its ancestor chain"):
        write_seed_registry_shards(
            [SeedRegistration(requests[2], seed32_required=False)],
            directory=tmp_path / "invalid",
            release_id="embedbench-hard-ood-v1.0.0",
            stage_id="foundation",
            parent=structure,
        )


def test_sharded_extension_rechecks_parent_bytes_after_prior_verification(
    tmp_path: Path,
) -> None:
    request = SeedRequest.from_dict(_valid_seed_values("problem"))
    write_seed_registry_shards(
        [SeedRegistration(request, seed32_required=False)],
        directory=tmp_path / "parent",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="foundation",
    )
    parent_root = SeedRegistryRoot.from_bytes(
        (tmp_path / "parent" / SEED_REGISTRY_ROOT_FILENAME).read_bytes()
    )
    parent = _verify_stage(tmp_path / "parent", parent_root)
    replacement = SeedRequest.from_dict(_valid_seed_values("problem", replicate=20))
    replacement_entry = SeedRegistry.build(
        [replacement],
        stage_id="fixture",
    ).entries[0]
    parent_shard = tmp_path / "parent" / parent.root.shards[0].relative_path
    parent_shard.write_bytes(canonical_bytes(replacement_entry.to_dict()) + b"\n")
    child_request = SeedRequest.from_dict(_valid_seed_values("host"))

    with pytest.raises(ValueError, match="changed after validation"):
        write_seed_registry_shards(
            [SeedRegistration(child_request, seed32_required=False)],
            directory=tmp_path / "child",
            release_id="embedbench-hard-ood-v1.0.0",
            stage_id="structure",
            parent=parent,
        )


def test_sharded_lookup_rechecks_selected_shard_after_verification(tmp_path: Path) -> None:
    request = SeedRequest.from_dict(_valid_seed_values("problem"))
    write_seed_registry_shards(
        [SeedRegistration(request, seed32_required=False)],
        directory=tmp_path / "registry",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="foundation",
    )
    root = SeedRegistryRoot.from_bytes(
        (tmp_path / "registry" / SEED_REGISTRY_ROOT_FILENAME).read_bytes()
    )
    verified = _verify_stage(tmp_path / "registry", root)
    shard_path = tmp_path / "registry" / verified.root.shards[0].relative_path
    shard_path.write_bytes(shard_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="changed after validation"):
        require_sharded_seed_request(
            VerifiedSeedResolver(verified, expected_terminal_root_sha256=root.sha256),
            request,
        )


def test_sharded_registry_root_stays_small_for_streamed_pcg_plan(tmp_path: Path) -> None:
    def registrations() -> Iterator[SeedRegistration]:
        for replicate in range(1_000):
            request = SeedRequest.from_dict(
                _valid_seed_values("audit_bootstrap", replicate=replicate)
            )
            yield SeedRegistration(request, seed32_required=False)

    root = write_seed_registry_shards(
        registrations(),
        directory=tmp_path / "large",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="audit-bootstrap",
        shard_entry_limit=100,
        sort_chunk_limit=37,
    )
    verified = _verify_stage(tmp_path / "large", root)

    assert root.entry_count == 1_000
    assert len(root.shards) == 10
    assert (tmp_path / "large" / SEED_REGISTRY_ROOT_FILENAME).stat().st_size < 10_000
    assert verified.occupied_seed32 == frozenset()


def test_verified_resolver_batches_one_shard_read_and_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = [
        SeedRequest.from_dict(_valid_seed_values("audit_bootstrap", replicate=index))
        for index in range(2_000)
    ]
    root = write_seed_registry_shards(
        (SeedRegistration(request, seed32_required=False) for request in reversed(requests)),
        directory=tmp_path / "registry",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="bootstrap",
        shard_entry_limit=2_000,
        sort_chunk_limit=127,
    )
    verified = _verify_stage(tmp_path / "registry", root)
    resolver = VerifiedSeedResolver(
        verified,
        expected_terminal_root_sha256=root.sha256,
        max_cached_shards=1,
    )
    shard_reads = 0
    original = hard_ood_schema._read_verified_shard_entries

    def counted_read(*args, **kwargs):
        nonlocal shard_reads
        shard_reads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(hard_ood_schema, "_read_verified_shard_entries", counted_read)
    selected = [requests[-1], requests[1_500], requests[17], requests[0]]

    resolved = resolver.resolve_many(selected)

    assert [entry.request for entry in resolved] == selected
    assert shard_reads == 1
    assert resolver.resolve(selected[-1]).request == selected[-1]
    assert isinstance(rng_for_sharded_seed_request(resolver, selected[0]), np.random.Generator)
    assert shard_reads == 1


def test_verified_resolver_whole_rebase_cannot_replace_external_root(tmp_path: Path) -> None:
    requests = [
        SeedRequest.from_dict(_valid_seed_values("audit_bootstrap", replicate=index))
        for index in range(2)
    ]
    resolvers = []
    roots = []
    for index, request in enumerate(requests):
        directory = tmp_path / f"registry-{index}"
        root = write_seed_registry_shards(
            [SeedRegistration(request, seed32_required=False)],
            directory=directory,
            release_id="embedbench-hard-ood-v1.0.0",
            stage_id=f"stage-{index}",
        )
        verified = _verify_stage(directory, root)
        roots.append(root)
        resolvers.append(
            VerifiedSeedResolver(
                verified,
                expected_terminal_root_sha256=root.sha256,
            )
        )

    honest, attacker = resolvers
    for attribute in ("_stages", "_last_keys", "_terminal_root_sha256"):
        setattr(honest, attribute, getattr(attacker, attribute))

    with pytest.raises(ValueError, match="external terminal-root commitment"):
        validate_seed_resolver(
            honest,
            expected_terminal_root_sha256=roots[0].sha256,
        )


def test_verified_resolver_rejects_a_spliced_parent_chain(tmp_path: Path) -> None:
    requests = [
        SeedRequest.from_dict(_valid_seed_values("problem", replicate=index)) for index in range(3)
    ]
    honest_parent_root = write_seed_registry_shards(
        [SeedRegistration(requests[0], seed32_required=False)],
        directory=tmp_path / "honest-parent",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="honest-parent",
    )
    honest_parent = _verify_stage(tmp_path / "honest-parent", honest_parent_root)
    unrelated_root = write_seed_registry_shards(
        [SeedRegistration(requests[1], seed32_required=False)],
        directory=tmp_path / "unrelated-parent",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="unrelated-parent",
    )
    unrelated_parent = _verify_stage(tmp_path / "unrelated-parent", unrelated_root)
    child_root = write_seed_registry_shards(
        [SeedRegistration(requests[2], seed32_required=False)],
        directory=tmp_path / "child",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="child",
        parent=honest_parent,
    )
    child = _verify_stage(tmp_path / "child", child_root, parent=honest_parent)
    object.__setattr__(child, "_parent", unrelated_parent)

    with pytest.raises(ValueError, match="parent chain"):
        VerifiedSeedResolver(
            child,
            expected_terminal_root_sha256=child_root.sha256,
        )


def test_verification_requires_external_root_digest_and_cannot_be_forged(
    tmp_path: Path,
) -> None:
    request = SeedRequest.from_dict(_valid_seed_values("problem"))
    root = write_seed_registry_shards(
        [SeedRegistration(request, seed32_required=False)],
        directory=tmp_path / "registry",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="foundation",
    )

    with pytest.raises(ValueError, match="expected root SHA-256"):
        verify_seed_registry_shards(
            tmp_path / "registry",
            expected_root_sha256="0" * 64,
        )

    forged_root = SeedRegistryRoot(
        release_id=root.release_id,
        stage_id=root.stage_id,
        parent_root_sha256=None,
        entry_count=root.entry_count,
        shard_entry_limit=root.shard_entry_limit,
        occupied_seed32_count=1,
        occupied_seed32_sha256=canonical_sha256([42]),
        shards=root.shards,
    )
    with pytest.raises(TypeError):
        SeedRegistryVerification(
            root=forged_root,
            occupied_seed32=frozenset({42}),
            directory=tmp_path / "registry",
            parent=None,
        )

    verified = _verify_stage(tmp_path / "registry", root)
    with pytest.raises(ValueError, match="terminal root SHA-256"):
        VerifiedSeedResolver(
            verified,
            expected_terminal_root_sha256="0" * 64,
        )


def test_child_paths_reverify_bytes_even_for_privately_forged_parent_summary(
    tmp_path: Path,
) -> None:
    parent_request = SeedRequest.from_dict(_valid_seed_values("problem"))
    honest_root = write_seed_registry_shards(
        [SeedRegistration(parent_request, seed32_required=False)],
        directory=tmp_path / "parent",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="foundation",
    )
    honest_parent = _verify_stage(tmp_path / "parent", honest_root)
    child_request = SeedRequest.from_dict(_valid_seed_values("host"))
    child_root = write_seed_registry_shards(
        [SeedRegistration(child_request, seed32_required=False)],
        directory=tmp_path / "honest-child",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="structure",
        parent=honest_parent,
    )
    forged_root = SeedRegistryRoot(
        release_id=honest_root.release_id,
        stage_id=honest_root.stage_id,
        parent_root_sha256=None,
        entry_count=honest_root.entry_count,
        shard_entry_limit=honest_root.shard_entry_limit,
        occupied_seed32_count=1,
        occupied_seed32_sha256=canonical_sha256([42]),
        shards=honest_root.shards,
    )
    forged_parent = SeedRegistryVerification._verified(
        root=forged_root,
        occupied_seed32={42},
        directory=tmp_path / "parent",
        parent=None,
        expected_root_sha256=forged_root.sha256,
    )

    with pytest.raises(ValueError, match="root changed after validation"):
        write_seed_registry_shards(
            [SeedRegistration(child_request, seed32_required=False)],
            directory=tmp_path / "forged-child",
            release_id="embedbench-hard-ood-v1.0.0",
            stage_id="structure",
            parent=forged_parent,
        )
    with pytest.raises(ValueError, match="root changed after validation"):
        verify_seed_registry_shards(
            tmp_path / "honest-child",
            expected_root_sha256=child_root.sha256,
            parent=forged_parent,
        )


@pytest.mark.parametrize("limit_name", ["shard_entry_limit", "sort_chunk_limit"])
def test_sharded_writer_rejects_limits_above_fifty_thousand(
    tmp_path: Path,
    limit_name: str,
) -> None:
    request = SeedRequest.from_dict(_valid_seed_values("problem"))
    kwargs = {limit_name: 50_001}

    with pytest.raises(ValueError, match="at most 50000"):
        write_seed_registry_shards(
            [SeedRegistration(request, seed32_required=False)],
            directory=tmp_path / limit_name,
            release_id="embedbench-hard-ood-v1.0.0",
            stage_id="foundation",
            **kwargs,
        )


@pytest.mark.parametrize("stage_id", [".", "..", "../unsafe", "unsafe/name", "unsafe\\name"])
def test_sharded_writer_rejects_unsafe_stage_id(tmp_path: Path, stage_id: str) -> None:
    request = SeedRequest.from_dict(_valid_seed_values("problem"))

    with pytest.raises(ValueError, match="safe path component"):
        write_seed_registry_shards(
            [SeedRegistration(request, seed32_required=False)],
            directory=tmp_path / "registry",
            release_id="embedbench-hard-ood-v1.0.0",
            stage_id=stage_id,
        )


def test_seed_registry_root_constructor_rejects_oversized_output() -> None:
    shards = tuple(
        SeedRegistryShard(
            relative_path=f"part-{index:06d}.jsonl",
            sha256=f"{index:064x}",
            byte_count=1,
            record_count=1,
            schema_version=1,
            first_seed_key_sha256=f"{index:064x}",
            last_seed_key_sha256=f"{index:064x}",
        )
        for index in range(30_000)
    )

    with pytest.raises(ValueError, match="8 MiB"):
        SeedRegistryRoot(
            release_id="embedbench-hard-ood-v1.0.0",
            stage_id="oversized",
            parent_root_sha256=None,
            entry_count=len(shards),
            shard_entry_limit=1,
            occupied_seed32_count=0,
            occupied_seed32_sha256=canonical_sha256([]),
            shards=shards,
        )


def test_seed_registry_root_rejects_shard_limit_above_fifty_thousand() -> None:
    shard = SeedRegistryShard(
        relative_path="part-000000.jsonl",
        sha256="1" * 64,
        byte_count=1,
        record_count=1,
        schema_version=1,
        first_seed_key_sha256="2" * 64,
        last_seed_key_sha256="2" * 64,
    )

    with pytest.raises(ValueError, match="at most 50000"):
        SeedRegistryRoot(
            release_id="embedbench-hard-ood-v1.0.0",
            stage_id="invalid-limit",
            parent_root_sha256=None,
            entry_count=1,
            shard_entry_limit=50_001,
            occupied_seed32_count=0,
            occupied_seed32_sha256=canonical_sha256([]),
            shards=(shard,),
        )


def test_verifier_retains_only_required_occupied_seed32_state(tmp_path: Path) -> None:
    requests = [
        SeedRequest.from_dict(_valid_seed_values("problem", replicate=index))
        for index in range(5_000)
    ]
    root = write_seed_registry_shards(
        (SeedRegistration(request, seed32_required=True) for request in requests),
        directory=tmp_path / "registry",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="foundation",
        shard_entry_limit=5_000,
        sort_chunk_limit=500,
    )

    verified = _verify_stage(tmp_path / "registry", root)

    assert len(verified.occupied_seed32) == 5_000
    assert not hasattr(verified, "stage_allocations")


def test_verifier_rejects_unexpected_file_before_opening_any_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SeedRequest.from_dict(_valid_seed_values("problem"))
    root = write_seed_registry_shards(
        [SeedRegistration(request, seed32_required=False)],
        directory=tmp_path / "registry",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="foundation",
    )
    (tmp_path / "registry" / "UNBOUND").write_bytes(b"unexpected")
    original = hard_ood_schema._open_regular_file_at

    def reject_shard_open(directory_fd: int, name: str) -> int:
        if name.startswith("part-"):
            pytest.fail("verifier opened a shard before rejecting an unexpected file")
        return original(directory_fd, name)

    monkeypatch.setattr(hard_ood_schema, "_open_regular_file_at", reject_shard_open)

    with pytest.raises(ValueError, match="unexpected artifact"):
        verify_seed_registry_shards(
            tmp_path / "registry",
            expected_root_sha256=root.sha256,
        )


def test_registry_io_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    request = SeedRequest.from_dict(_valid_seed_values("problem"))

    with pytest.raises(ValueError, match="symlink|secure directory"):
        write_seed_registry_shards(
            [SeedRegistration(request, seed32_required=False)],
            directory=alias / "new-registry",
            release_id="embedbench-hard-ood-v1.0.0",
            stage_id="foundation",
        )

    root = write_seed_registry_shards(
        [SeedRegistration(request, seed32_required=False)],
        directory=real / "registry",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="foundation",
    )
    with pytest.raises(ValueError, match="symlink|secure directory"):
        verify_seed_registry_shards(
            alias / "registry",
            expected_root_sha256=root.sha256,
        )


@pytest.mark.parametrize("artifact_name", [SEED_REGISTRY_ROOT_FILENAME, "shard"])
def test_registry_verifier_rejects_symlinked_root_or_shard(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    request = SeedRequest.from_dict(_valid_seed_values("problem"))
    root = write_seed_registry_shards(
        [SeedRegistration(request, seed32_required=False)],
        directory=tmp_path / "registry",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="foundation",
    )
    name = (
        SEED_REGISTRY_ROOT_FILENAME
        if artifact_name == SEED_REGISTRY_ROOT_FILENAME
        else root.shards[0].relative_path
    )
    artifact = tmp_path / "registry" / name
    copied = tmp_path / f"copy-{artifact_name}"
    copied.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(copied)

    with pytest.raises(ValueError, match="linked|regular file"):
        verify_seed_registry_shards(
            tmp_path / "registry",
            expected_root_sha256=root.sha256,
        )


def test_registry_verifier_closes_check_open_symlink_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SeedRequest.from_dict(_valid_seed_values("problem"))
    root = write_seed_registry_shards(
        [SeedRegistration(request, seed32_required=False)],
        directory=tmp_path / "registry",
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="foundation",
    )
    shard_name = root.shards[0].relative_path
    shard_path = tmp_path / "registry" / shard_name
    malicious = tmp_path / "malicious.jsonl"
    malicious.write_bytes(shard_path.read_bytes())
    original_open = hard_ood_schema.os.open
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == shard_name and not swapped:
            swapped = True
            shard_path.unlink()
            shard_path.symlink_to(malicious)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(hard_ood_schema.os, "open", racing_open)

    with pytest.raises(ValueError, match="linked|unreadable"):
        verify_seed_registry_shards(
            tmp_path / "registry",
            expected_root_sha256=root.sha256,
        )
    assert swapped


def test_registry_publication_never_replaces_racing_target(tmp_path: Path) -> None:
    target = tmp_path / "registry"
    request = SeedRequest.from_dict(_valid_seed_values("problem"))

    def racing_registration() -> Iterator[SeedRegistration]:
        target.mkdir()
        yield SeedRegistration(request, seed32_required=False)

    with pytest.raises(FileExistsError):
        write_seed_registry_shards(
            racing_registration(),
            directory=target,
            release_id="embedbench-hard-ood-v1.0.0",
            stage_id="foundation",
        )

    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_registry_publication_rejects_a_renamed_parent_during_staging(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    moved_parent = tmp_path / "moved-parent"
    parent.mkdir()
    target = parent / "registry"
    request = SeedRequest.from_dict(_valid_seed_values("problem"))

    def racing_registration() -> Iterator[SeedRegistration]:
        parent.rename(moved_parent)
        parent.mkdir()
        staged = [
            path for path in moved_parent.iterdir() if path.name.startswith(".seed-registry-")
        ]
        assert len(staged) == 1
        replacement = parent / staged[0].name
        (replacement / "sort").mkdir(parents=True)
        (replacement / "payload").mkdir()
        yield SeedRegistration(request, seed32_required=False)

    with pytest.raises(ValueError, match="publication parent changed"):
        write_seed_registry_shards(
            racing_registration(),
            directory=target,
            release_id="embedbench-hard-ood-v1.0.0",
            stage_id="foundation",
        )

    assert not target.exists()
    assert not (moved_parent / "registry").exists()
    assert not any(path.name.startswith(".seed-registry-") for path in parent.iterdir())
    assert not any(path.name.startswith(".seed-registry-") for path in moved_parent.iterdir())


def _stage_artifact(directory: Path, root: SeedRegistryRoot) -> SeedRegistryStageArtifact:
    prefix = f"seed_registry/{root.stage_id}"
    root_path = directory / SEED_REGISTRY_ROOT_FILENAME
    return SeedRegistryStageArtifact(
        stage_id=root.stage_id,
        visibility_class="public",
        parent_root_sha256=root.parent_root_sha256,
        root_sha256=root.sha256,
        root_artifact=ArtifactEntry.from_payload(
            relative_path=f"{prefix}/{SEED_REGISTRY_ROOT_FILENAME}",
            payload=root_path.read_bytes(),
            record_count=1,
            schema_version=1,
        ),
        shard_artifacts=tuple(
            ArtifactEntry.from_payload(
                relative_path=f"{prefix}/{shard.relative_path}",
                payload=(directory / shard.relative_path).read_bytes(),
                record_count=shard.record_count,
                schema_version=shard.schema_version,
            )
            for shard in root.shards
        ),
    )


def test_stage_artifact_manifest_round_trip_and_terminal_chain_validation(
    tmp_path: Path,
) -> None:
    requests = [
        SeedRequest.from_dict(_valid_seed_values("problem", replicate=index)) for index in range(3)
    ]
    foundation_directory = tmp_path / "seed_registry" / "foundation"
    structure_directory = tmp_path / "seed_registry" / "structure"
    parent_root = write_seed_registry_shards(
        [SeedRegistration(requests[0], seed32_required=False)],
        directory=foundation_directory,
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="foundation",
    )
    parent = _verify_stage(foundation_directory, parent_root)
    child_root = write_seed_registry_shards(
        [
            SeedRegistration(requests[1], seed32_required=False),
            SeedRegistration(requests[2], seed32_required=False),
        ],
        directory=structure_directory,
        release_id="embedbench-hard-ood-v1.0.0",
        stage_id="structure",
        parent=parent,
        shard_entry_limit=1,
    )
    manifest = SeedRegistryArtifactManifest(
        release_id="embedbench-hard-ood-v1.0.0",
        terminal_root_sha256=child_root.sha256,
        stages=(
            _stage_artifact(foundation_directory, parent_root),
            _stage_artifact(structure_directory, child_root),
        ),
    )

    restored = validate_seed_registry_artifact_manifest(
        manifest.to_dict(),
        expected_manifest_sha256=manifest.sha256,
        expected_terminal_root_sha256=child_root.sha256,
    )

    assert restored == manifest
    assert SeedRegistryArtifactManifest.from_bytes(manifest.to_bytes()) == manifest
    terminal = verify_seed_registry_artifact_chain(
        manifest,
        artifact_root=tmp_path,
        expected_manifest_sha256=manifest.sha256,
        expected_terminal_root_sha256=child_root.sha256,
    )
    assert terminal.root == child_root
    incomplete = manifest.to_dict()
    incomplete["stages"][1]["shard_artifacts"] = incomplete["stages"][1]["shard_artifacts"][:1]
    with pytest.raises(ValueError, match="every seed-registry shard"):
        verify_seed_registry_artifact_chain(
            incomplete,
            artifact_root=tmp_path,
            expected_manifest_sha256=canonical_sha256(incomplete),
            expected_terminal_root_sha256=child_root.sha256,
        )
    with pytest.raises(ValueError, match="expected terminal root SHA-256"):
        validate_seed_registry_artifact_manifest(
            manifest.to_dict(),
            expected_manifest_sha256=manifest.sha256,
            expected_terminal_root_sha256="0" * 64,
        )
    broken = manifest.to_dict()
    broken["stages"][1]["parent_root_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="ordered parent chain"):
        validate_seed_registry_artifact_manifest(
            broken,
            expected_manifest_sha256=canonical_sha256(broken),
            expected_terminal_root_sha256=child_root.sha256,
        )
    wrong_release = manifest.to_dict()
    wrong_release["release_id"] = "another-release"
    with pytest.raises(ValueError, match="release_id"):
        verify_seed_registry_artifact_chain(
            wrong_release,
            artifact_root=tmp_path,
            expected_manifest_sha256=canonical_sha256(wrong_release),
            expected_terminal_root_sha256=child_root.sha256,
        )
    invalid_visibility = manifest.to_dict()
    invalid_visibility["stages"][0]["visibility_class"] = "secret"
    with pytest.raises(ValueError, match="visibility_class"):
        validate_seed_registry_artifact_manifest(
            invalid_visibility,
            expected_manifest_sha256=canonical_sha256(invalid_visibility),
            expected_terminal_root_sha256=child_root.sha256,
        )

    rewritten_visibility = manifest.to_dict()
    rewritten_visibility["stages"][0]["visibility_class"] = "custodian_private"
    with pytest.raises(ValueError, match="artifact manifest SHA-256"):
        verify_seed_registry_artifact_chain(
            rewritten_visibility,
            artifact_root=tmp_path,
            expected_manifest_sha256=manifest.sha256,
            expected_terminal_root_sha256=child_root.sha256,
        )


def test_seed_registry_rejects_duplicate_requests_and_mixed_releases() -> None:
    request = SeedRequest.from_dict(_valid_seed_values("problem"))
    other_release = SeedRequest.from_dict(
        {**_valid_seed_values("problem", replicate=1), "release_id": "other-release"}
    )

    with pytest.raises(ValueError, match="duplicate seed request"):
        SeedRegistry.build([request, request], stage_id="foundation")
    with pytest.raises(ValueError, match="one release_id"):
        SeedRegistry.build([request, other_release], stage_id="foundation")
    with pytest.raises(ValueError, match="at least one seed request"):
        SeedRegistry.build([], stage_id="foundation")


def test_seed_registry_persists_forced_collision_for_real_request_golden_pair() -> None:
    requests = []
    for replicate in (1451, 38751):
        values = _valid_seed_values("problem", replicate=replicate)
        values["cell"] = "collision-golden"
        requests.append(SeedRequest.from_dict(values))

    registry = SeedRegistry.build(
        requests,
        stage_id="structure-solvers",
        third_party_32_requests=requests,
    )

    assert [entry.request.replicate for entry in registry.entries] == [38751, 1451]
    assert [entry.seed_key_sha256 for entry in registry.entries] == [
        "9ccb95fa61eab3e18275f9fbf4d7d73422b81d27798275b5999b5264148ecbce",
        "9ccb95fabd95767174caf214c0672285c09c82bf832ffca6d6d27b670cfd3737",
    ]
    assert [entry.seed32 for entry in registry.entries] == [2630587898, 2533497468]
    assert [entry.collision_counter for entry in registry.entries] == [0, 1]
    assert SeedRegistry.from_bytes(registry.to_bytes()) == registry


def test_seed_registry_rejects_unplanned_requests_and_persistence_tampering() -> None:
    request = SeedRequest.from_dict(_valid_seed_values("problem"))
    unplanned = SeedRequest.from_dict(_valid_seed_values("problem", replicate=1))
    registry = SeedRegistry.build(
        [request],
        stage_id="foundation",
        third_party_32_requests=[request],
    )

    with pytest.raises(KeyError, match="seed request is not registered"):
        registry.require(unplanned)
    with pytest.raises(ValueError, match="canonical seed-registry bytes"):
        SeedRegistry.from_bytes(b" " + registry.to_bytes())

    document = registry.to_dict()
    document["unknown"] = True
    with pytest.raises(ValueError, match="seed registry schema fields differ"):
        SeedRegistry.from_dict(document)

    document = registry.to_dict()
    document["entries"][0]["seed32"] ^= 1  # type: ignore[index,operator]
    with pytest.raises(ValueError, match="seed registry allocation fields do not match"):
        SeedRegistry.from_dict(document)


def test_seed_request_key_changes_when_namespace_field_changes() -> None:
    base = SeedRequest.from_dict(_valid_seed_values("problem"))
    changed = SeedRequest.from_dict({**base.to_dict(), "replicate": 1})

    assert base.seed_key != changed.seed_key


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"purpose": "unregistered"}, "unregistered seed purpose"),
        ({"replicate": -1}, "replicate must be a non-negative integer"),
        ({"replicate": _IntSubclass(0)}, "replicate must be a non-negative integer"),
        ({"candidate_index": True}, "candidate_index must be a non-negative integer or null"),
        ({"problem_sha256": "ABC"}, "problem_sha256 must be a lowercase SHA-256"),
        ({"panel": ""}, "panel must be a non-empty string"),
    ],
)
def test_seed_request_rejects_invalid_namespace_fields(
    updates: dict[str, object], message: str
) -> None:
    purpose = "locked_label" if "candidate_index" in updates else "problem"
    values = _valid_seed_values(purpose)
    values.update(updates)

    with pytest.raises(ValueError, match=message):
        SeedRequest.from_dict(values)


def test_seed_request_rejects_unknown_or_missing_fields() -> None:
    request = SeedRequest(
        release_id="embedbench-hard-ood-v1.0.0",
        purpose="host",
        partition="hard_dev",
        panel="partial_structural",
        cell="cell-001",
        task_type=None,
        problem_sha256=None,
        state_sha256=None,
        candidate_index=None,
        strength_index=None,
        label_stage=None,
        replicate=0,
    ).to_dict()

    with pytest.raises(ValueError, match="seed request schema fields differ"):
        SeedRequest.from_dict({**request, "unknown": 1})
    del request["cell"]
    with pytest.raises(ValueError, match="seed request schema fields differ"):
        SeedRequest.from_dict(request)
