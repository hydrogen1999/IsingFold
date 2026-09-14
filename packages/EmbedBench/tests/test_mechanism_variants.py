"""Adversarial contract tests for certified single-mechanism variants."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import embedbench.mechanism_variants as mechanism_variants
import pytest
from embedbench.handtests import Motif, build_motif, certify_motif
from embedbench.hard_ood_schema import (
    SeedRegistration,
    VerifiedSeedResolver,
    canonical_bytes,
    verify_seed_registry_shards,
    write_seed_registry_shards,
)
from embedbench.mechanism_variants import (
    MECHANISM_FAMILIES,
    MECHANISM_PARAMETER_SCHEMA,
    MECHANISM_PARAMETER_SCHEMA_VERSION,
    MechanismCollisionError,
    MechanismGenerationResult,
    MechanismIdentityRegistry,
    MechanismParameterRow,
    MechanismVariant,
    generate_single_mechanism_variant,
    host_automorphism_count,
    logical_symmetry_count,
    make_parameter_row,
    unsupported_composed_mechanism,
    validate_variant,
    verify_byte_reconstruction,
)


def _seed_hex(offset: int) -> str:
    return bytes((offset + index) % 256 for index in range(32)).hex()


def _row(
    family: str,
    *,
    partition: str = "mechanism_dev",
    ordinal: int = 0,
    offset: int = 0,
    **kwargs: object,
) -> MechanismParameterRow:
    return make_parameter_row(
        release_id="embedbench-hard-ood-v1.0.0",
        partition=partition,
        family=family,
        parent_ordinal=ordinal,
        parent_seed_key=_seed_hex(offset),
        transformation_seed_key=_seed_hex(offset + 64),
        **kwargs,
    )


@pytest.mark.parametrize("family", MECHANISM_FAMILIES)
def test_all_seven_registered_families_recertify_with_positive_margin(family: str) -> None:
    row = _row(family)

    result = generate_single_mechanism_variant(row)

    assert result.status == "accepted"
    variant = result.require_variant()
    assert variant.parameter_row.family == family
    assert variant.certificate.certified_winner_index == variant.hypothesized_winner_index
    assert variant.certificate.margin_kind in {"feasibility", "qubits", "max_chain"}
    assert variant.certificate.margin_value > 0
    assert verify_byte_reconstruction(variant)


def test_short_chain_trap_is_an_exact_terminal_max_chain_choice() -> None:
    certificate = certify_motif(build_motif("short_chain_trap"))

    assert certificate.sample.actions == ((1, 4), (1, 2))
    assert certificate.sample.values == ((1, -8, -3), (1, -8, -2))
    assert certificate.certified_winner == (1, 2)
    assert certificate.margin == ("max_chain", 1)


def test_multi_neighbour_advantage_reverses_under_either_neighbour_ablation() -> None:
    motif = build_motif("multi_neighbour")
    certificate = certify_motif(motif)

    assert certificate.sample.actions == ((0, 10), (0, 0))
    assert certificate.sample.values == ((1, -9, -3), (1, -8, -4))
    assert certificate.certified_winner == (0, 0)
    assert certificate.margin == ("qubits", 1)

    for removed_neighbour in (1, 2):
        ablated_logical = motif.logical.copy()
        ablated_logical.remove_node(removed_neighbour)
        ablated = certify_motif(
            Motif(
                name="multi_neighbour",
                mechanism="single-neighbour causal ablation",
                host=motif.host,
                logical=ablated_logical,
                cores={
                    logical: nodes
                    for logical, nodes in motif.cores.items()
                    if logical != removed_neighbour
                },
                actions=motif.actions,
                hypothesis_winner=(0, 10),
                l_cap=motif.l_cap,
                q_cap=motif.q_cap,
            )
        )
        assert ablated.sample.values == ((1, -5, -3), (1, -6, -3))
        assert ablated.certified_winner == (0, 10)
        assert ablated.margin == ("qubits", 1)


def test_parameter_record_has_pinned_schema_and_canonical_json_round_trip() -> None:
    row = _row("dead_end")
    document = json.loads(canonical_bytes(row.to_dict()))

    assert document["schema"] == MECHANISM_PARAMETER_SCHEMA
    assert document["schema_version"] == MECHANISM_PARAMETER_SCHEMA_VERSION == 1
    parsed = MechanismParameterRow.from_dict(document)
    assert parsed == row
    assert parsed.to_bytes() == row.to_bytes()
    assert parsed.sha256 == row.sha256


def test_release_identity_is_nfc_normalized_before_namespace_derivation() -> None:
    row = make_parameter_row(
        release_id="cafe\u0301",
        partition="mechanism_dev",
        family="dead_end",
        parent_ordinal=0,
        parent_seed_key=_seed_hex(28),
        transformation_seed_key=_seed_hex(92),
    )

    assert row.release_id == "caf\u00e9"
    assert row.parent_seed_namespace.startswith("caf\u00e9/")


def test_schema_v1_golden_identities_pin_all_canonicalization_layers() -> None:
    row = _row("dead_end")
    variant = generate_single_mechanism_variant(row).require_variant()

    assert row.motif_template_sha256 == (
        "e4e9c98a08db6371646aa0d3f1801e33b03cbf819259d9c8100c519e8755d094"
    )
    assert row.parent_scientific_sha256 == (
        "924cf8f5a03c5f3db7557f4df7259e44aced1e4cee13e3bee839d00f59878395"
    )
    assert row.sha256 == "62b0bdcc666b19ca4d85d13f7c206c08e58ba848318db92442afffc0dd811ba6"
    assert variant.state_sha256 == (
        "13b54618c95d97c087237f6a01b00c43bc464ca508eeb38449dc59f490ad31be"
    )
    assert variant.diagnostic_iso_sha256 == (
        "cce4e7b8e18c6be5e3bc229fdbe675a41491b8d507de3a5d11e7957f0399be6f"
    )
    assert variant.certificate.certificate_sha256 == (
        "ea71a1eace0b837ce766f2bd7a40827bbc79ef270846a4812f70710ce3194dda"
    )
    assert hashlib.sha256(variant.to_bytes()).hexdigest() == (
        "d2dc7f966e761310c6fd07997e5885442aa557a62b7fb7d3330f8c57ff1e3d55"
    )


@pytest.mark.parametrize(
    ("family", "expected"),
    (
        (
            "multi_neighbour",
            (
                "e7857249a179099cedf7e5bbcffa0faf2b1bc6700a3c04dfde51e898138d5c06",
                "065296dcd3b65366e98c0603fd17f27bab2cc5f1057de23b6ff011982fa422d8",
                "c1652a1f954548141a1a91e8a98dc5a0c8dfdc9d934f8a5de06085bad6c772f7",
                "feac6233985dc595867676f02c5ec57adca56b2a9b2da8d97fbd1ceb320a2882",
                "1f9fc1d4454fcfa6ebd86f087c9d91a8fc9990c2d07adfed53678dc64d75d60d",
                "80b9a01cc8194d14281b40f797e6217f5b18903d91ba9bc1b55bb0d87fc50cbc",
                "f7abf361d8e3db05d2d22c4134ef9b4b3a3c993260b11bd94c31765893ad08b7",
            ),
        ),
        (
            "short_chain_trap",
            (
                "a11e4acaf0c078b634d9625f1bb979e3c071187af5f4710bc2f95ad62f17f62e",
                "9a5cfc581f359f76388f923b20a850a51fcd51695fcae53956f318898d044b81",
                "4a4349a9bf1c4cd0afebf01ff663883e4ca854fc8b47f6d9000b527d987680df",
                "b535a2774d54d4511b7c4af36864574c077a4bde118cea0e6d5daca74cbe9792",
                "6684c7f673fa97212a347e9dae708431d5c22efbd132173dc0570b9080356cc4",
                "93d69359208fbcc7f35a65ae887c40b6fce2284440c3fcfaa780bb923003569f",
                "e792a53220955b843c7f39a4a72d72e4cefdea0f46a1095dc6e97ed45ac8ecc9",
            ),
        ),
    ),
)
def test_repaired_motif_golden_identities_pin_scientific_content(
    family: str, expected: tuple[str, ...]
) -> None:
    row = _row(family)
    variant = generate_single_mechanism_variant(row).require_variant()

    assert (
        row.motif_template_sha256,
        row.parent_scientific_sha256,
        row.sha256,
        variant.state_sha256,
        variant.diagnostic_iso_sha256,
        variant.certificate.certificate_sha256,
        hashlib.sha256(variant.to_bytes()).hexdigest(),
    ) == expected


def test_explicit_relabel_symmetry_orientation_and_decoration_are_deterministic() -> None:
    base = _row("dead_end", offset=1)
    host_map = tuple((source, source + 100) for source, _ in base.host_node_relabeling)
    logical_map = tuple((source, source + 20) for source, _ in base.logical_role_relabeling)
    # Node 1000 is an explicit, label-free leaf decoration. Exact certification decides
    # whether it preserved the precommitted mechanism; no outcome was used to choose it.
    transformed = _row(
        "dead_end",
        ordinal=1,
        offset=2,
        host_automorphism_index=1,
        logical_symmetry_index=1,
        orientation="reverse",
        host_node_relabeling=host_map,
        logical_role_relabeling=logical_map,
        decoration_nodes=(1000,),
        decoration_edges=(),
    )

    first = generate_single_mechanism_variant(transformed).require_variant()
    second = generate_single_mechanism_variant(transformed).require_variant()

    assert first.to_bytes() == second.to_bytes()
    assert first.parameter_row.sha256 == transformed.sha256
    assert first.host_nodes[-1] == 1000
    assert first.original_candidate_index == 1
    assert first.diagnostic_iso_sha256 != (
        generate_single_mechanism_variant(base).require_variant().diagnostic_iso_sha256
    )
    assert verify_byte_reconstruction(first)
    assert base.sha256 != transformed.sha256


def test_pure_relabel_and_candidate_orientation_preserve_diagnostic_isomorphism() -> None:
    identity = generate_single_mechanism_variant(_row("dead_end", offset=3)).require_variant()
    base_row = _row("dead_end", ordinal=1, offset=4)
    relabelled_row = _row(
        "dead_end",
        ordinal=1,
        offset=4,
        orientation="reverse",
        host_node_relabeling=tuple(
            (source, 100 + 7 * source) for source, _ in base_row.host_node_relabeling
        ),
        logical_role_relabeling=tuple(
            (source, 50 - source) for source, _ in base_row.logical_role_relabeling
        ),
    )
    relabelled = generate_single_mechanism_variant(relabelled_row).require_variant()

    assert relabelled.state_sha256 != identity.state_sha256
    assert relabelled.diagnostic_iso_sha256 == identity.diagnostic_iso_sha256
    assert relabelled.original_candidate_index == 1
    assert relabelled.actions[relabelled.original_candidate_index] != relabelled.actions[0]


@pytest.mark.parametrize(
    "family",
    MECHANISM_FAMILIES,
)
def test_every_family_isomorphism_digest_survives_all_permitted_relabel_classes(
    family: str,
) -> None:
    identity_row = _row(family, offset=23)
    identity = generate_single_mechanism_variant(identity_row).require_variant()
    transformed_row = _row(
        family,
        ordinal=1,
        offset=24,
        host_automorphism_index=min(1, host_automorphism_count(family) - 1),
        logical_symmetry_index=min(1, logical_symmetry_count(family) - 1),
        orientation="reverse",
        host_node_relabeling=tuple(
            (source, 10_000 + 17 * index)
            for index, (source, _) in enumerate(reversed(identity_row.host_node_relabeling))
        )[::-1],
        logical_role_relabeling=tuple(
            (source, -100 - 11 * index)
            for index, (source, _) in enumerate(identity_row.logical_role_relabeling)
        ),
    )
    transformed = generate_single_mechanism_variant(transformed_row).require_variant()

    assert transformed.state_sha256 != identity.state_sha256
    assert transformed.diagnostic_iso_sha256 == identity.diagnostic_iso_sha256
    assert transformed.certificate.margin_value == identity.certificate.margin_value
    assert verify_byte_reconstruction(transformed)


def test_state_digest_preserves_original_candidate_role() -> None:
    variant = generate_single_mechanism_variant(_row("dead_end", offset=5)).require_variant()
    forged = variant.to_dict()
    forged["original_candidate_index"] = 1

    with pytest.raises(ValueError, match="state_sha256|original candidate|reconstruction"):
        MechanismVariant.from_dict(forged)


def test_isomorphism_digest_preserves_original_role_and_numeric_caps() -> None:
    variant = generate_single_mechanism_variant(_row("dead_end", offset=29)).require_variant()
    state = variant._state()
    original_role_flipped = replace(state, original_candidate_index=1)
    cap_changed = replace(state, l_cap=state.l_cap + 1)

    assert mechanism_variants._diagnostic_iso_sha256(original_role_flipped) != (
        variant.diagnostic_iso_sha256
    )
    assert mechanism_variants._diagnostic_iso_sha256(cap_changed) != (
        variant.diagnostic_iso_sha256
    )


def test_forged_transformation_metadata_fails_byte_reconstruction() -> None:
    variant = generate_single_mechanism_variant(_row("articulation", offset=6)).require_variant()
    forged = variant.to_dict()
    forged["parameter_row"]["orientation"] = "reverse"

    with pytest.raises(ValueError, match="parameter|reconstruct|digest"):
        validate_variant(forged)


def test_forged_exact_certificate_is_rejected_even_with_rehashed_metadata() -> None:
    variant = generate_single_mechanism_variant(_row("cut_capacity", offset=7)).require_variant()
    forged = variant.to_dict()
    certificate = forged["certificate"]
    certificate["candidates"][0]["outcome"] = [1, -1, -1]
    certificate["certificate_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="certificate|completion|outcome"):
        validate_variant(forged)


def test_rehashed_but_false_completion_is_rejected_by_independent_exact_replay() -> None:
    variant = generate_single_mechanism_variant(_row("articulation", offset=25)).require_variant()
    forged = variant.to_dict()
    certificate = forged["certificate"]
    winner = certificate["candidates"][certificate["certified_winner_index"]]
    completion = winner["completion"]
    completion[0][1][-1] = 999
    digest_payload = {
        key: value for key, value in certificate.items() if key != "certificate_sha256"
    }
    certificate["certificate_sha256"] = hashlib.sha256(canonical_bytes(digest_payload)).hexdigest()

    with pytest.raises(ValueError, match="reconstruct|byte-for-byte"):
        MechanismVariant.from_dict(forged)


def test_non_positive_exact_margin_is_retained_as_a_closed_failure_status() -> None:
    # A leaf beyond dead_end's pocket opens a completion for the bad action and creates a tie.
    row = _row(
        "dead_end",
        offset=8,
        decoration_nodes=(6,),
        decoration_edges=((5, 6),),
    )

    result = generate_single_mechanism_variant(row)

    assert result.status == "non_positive_margin"
    assert result.variant is None


def test_hypothesized_winner_mismatch_is_retained_as_a_closed_failure_status() -> None:
    # The extra hub shortcut makes short_chain_trap's precommitted losing action strictly better.
    row = _row(
        "short_chain_trap",
        offset=9,
        decoration_nodes=(9,),
        decoration_edges=((0, 9), (4, 9), (7, 9)),
    )

    result = generate_single_mechanism_variant(row)

    assert result.status == "hypothesized_winner_not_certified"
    assert result.variant is None


def test_identity_registry_has_no_public_caller_variant_admission() -> None:
    assert not hasattr(MechanismIdentityRegistry.empty(), "admit")


@pytest.mark.parametrize("logical", [False, True])
def test_parameter_row_rejects_non_bijective_relabeling(logical: bool) -> None:
    base = _row("dead_end", offset=26)
    field = "logical_role_relabeling" if logical else "host_node_relabeling"
    mapping = list(getattr(base, field))
    mapping[-1] = (mapping[-1][0], mapping[0][1])
    kwargs = {field: tuple(mapping)}

    with pytest.raises(ValueError, match="duplicate-free"):
        _row("dead_end", ordinal=1, offset=27, **kwargs)


def test_empty_identity_registry_round_trips() -> None:
    empty = MechanismIdentityRegistry.empty()
    assert MechanismIdentityRegistry.from_dict(empty.to_dict()) == empty


def test_parent_and_transform_namespaces_are_partition_scoped_and_independent() -> None:
    dev = _row("dead_end", offset=15)
    locked = _row("dead_end", partition="mechanism_locked", offset=16)

    assert dev.parent_seed_namespace != dev.transformation_seed_namespace
    assert locked.parent_seed_namespace != locked.transformation_seed_namespace
    assert dev.parent_seed_namespace != locked.parent_seed_namespace
    assert dev.transformation_seed_namespace != locked.transformation_seed_namespace

    forged = dev.to_dict()
    forged["transformation_seed_namespace"] = dev.parent_seed_namespace
    with pytest.raises(ValueError, match="namespace"):
        MechanismParameterRow.from_dict(forged)


def test_records_detach_mutable_json_and_return_fresh_mutable_views() -> None:
    variant = generate_single_mechanism_variant(
        _row("high_degree_trap", offset=17)
    ).require_variant()
    document = variant.to_dict()
    parsed = validate_variant(document)
    document["host_nodes"].append(9999)
    document["certificate"]["candidates"][0]["outcome"][0] = 0
    emitted = parsed.to_dict()
    emitted["actions"][0][1] = 9999

    assert parsed == variant
    assert 9999 not in parsed.host_nodes
    assert parsed.actions == variant.actions
    with pytest.raises(FrozenInstanceError):
        parsed.l_cap = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parent_ordinal", True),
        ("host_automorphism_index", 0.0),
        ("decoration_nodes", [True]),
        ("q_cap", False),
        ("orientation", 1),
    ],
)
def test_parameter_parser_rejects_type_confusion(field: str, value: object) -> None:
    document = _row("dead_end", offset=18).to_dict()
    document[field] = value

    with pytest.raises((TypeError, ValueError)):
        MechanismParameterRow.from_dict(document)


@pytest.mark.parametrize(
    "forbidden", ["p_solve", "quality_label", "minorminer_choice", "baseline_output"]
)
def test_parameter_parser_rejects_quality_labels_and_baseline_outputs(forbidden: str) -> None:
    document = _row("dead_end", offset=19).to_dict()
    document[forbidden] = 0

    with pytest.raises(ValueError, match="unknown"):
        MechanismParameterRow.from_dict(document)


def test_all_versioned_parsers_reject_unknown_keys() -> None:
    row = _row("dead_end", offset=20)
    variant = generate_single_mechanism_variant(row).require_variant()
    row_doc = row.to_dict()
    variant_doc = variant.to_dict()
    registry_doc = MechanismIdentityRegistry.empty().to_dict()
    row_doc["unknown"] = None
    variant_doc["unknown"] = None
    registry_doc["unknown"] = None

    with pytest.raises(ValueError, match="unknown"):
        MechanismParameterRow.from_dict(row_doc)
    with pytest.raises(ValueError, match="unknown"):
        MechanismVariant.from_dict(variant_doc)
    with pytest.raises(ValueError, match="unknown"):
        MechanismIdentityRegistry.from_dict(registry_doc)


def test_result_and_registry_round_trip_with_self_digests_intact() -> None:
    result = generate_single_mechanism_variant(_row("dead_end", offset=30))
    result.require_variant()
    registry = MechanismIdentityRegistry.empty()

    assert MechanismGenerationResult.from_dict(result.to_dict()) == result
    assert MechanismIdentityRegistry.from_dict(registry.to_dict()) == registry

    forged_registry = registry.to_dict()
    forged_registry["registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="registry digest"):
        MechanismIdentityRegistry.from_dict(forged_registry)


def test_generic_hardware_placement_and_composition_fail_closed() -> None:
    generic = _row("dead_end", offset=21, placement_mode="registered_hardware_subgraph")
    result = generate_single_mechanism_variant(generic)

    assert result.status == "unsupported_generic_hardware_placement"
    assert result.variant is None
    composed = unsupported_composed_mechanism(
        parameter_row_sha256=generic.sha256,
        constituent_families=("dead_end", "articulation"),
    )
    assert composed.status == "unsupported_composed_mechanism"
    assert composed.variant is None


def test_dataclass_replacement_cannot_forge_parameter_self_identity() -> None:
    row = _row("dead_end", offset=22)

    sibling = replace(row, parent_ordinal=row.parent_ordinal + 1)
    assert sibling.mechanism_parent_id == row.mechanism_parent_id
    with pytest.raises(ValueError, match="parent_scientific_sha256"):
        replace(row, parent_scientific_sha256="0" * 64)


# WP7 release-admission regressions.  These deliberately exercise the public atomic
# boundary rather than treating a successfully constructed diagnostic variant as released.


def _verified_release_plan(
    tmp_path,
    family: str = "dead_end",
    *,
    partition: str = "mechanism_dev",
    parent_ordinal: int = 0,
    transformation_ordinal: int = 0,
    authoritative_manifest: str | None = None,
    suffix: str = "0",
    **row_kwargs: object,
):
    parent_request, transformation_request = mechanism_variants.mechanism_seed_requests(
        release_id="embedbench-hard-ood-v1.0.0",
        partition=partition,
        family=family,
        parent_ordinal=parent_ordinal,
        transformation_ordinal=transformation_ordinal,
    )
    row = make_parameter_row(
        release_id=parent_request.release_id,
        partition=parent_request.partition,
        family=family,
        parent_ordinal=parent_request.replicate,
        transformation_ordinal=transformation_request.replicate,
        parent_seed_key=parent_request.seed_key_hex,
        transformation_seed_key=transformation_request.seed_key_hex,
        **row_kwargs,
    )
    registry_dir = tmp_path / (
        f"seed-registry-{family}-{partition}-{parent_ordinal}-"
        f"{transformation_ordinal}-{suffix}"
    )
    root = write_seed_registry_shards(
        (
            SeedRegistration(parent_request, seed32_required=False),
            SeedRegistration(transformation_request, seed32_required=False),
        ),
        directory=registry_dir,
        release_id=row.release_id,
        stage_id=f"mechanism-{family}-{partition}-{transformation_ordinal}-{suffix}",
    )
    verified_registry = verify_seed_registry_shards(
        registry_dir,
        expected_root_sha256=root.sha256,
    )
    resolver = VerifiedSeedResolver(
        verified_registry,
        expected_terminal_root_sha256=root.sha256,
    )
    document = mechanism_variants.make_mechanism_plan_document(
        row=row,
        parent_seed_request=parent_request,
        transformation_seed_request=transformation_request,
        seed_registry_terminal_root_sha256=root.sha256,
        authoritative_isomorphism_manifest_sha256=authoritative_manifest,
    )
    plan_sha256 = hashlib.sha256(canonical_bytes(document)).hexdigest()
    return mechanism_variants.verify_mechanism_plan(
        document,
        expected_plan_sha256=plan_sha256,
        resolver=resolver,
        expected_seed_registry_terminal_root_sha256=root.sha256,
    ), resolver, document, plan_sha256, root.sha256


class _PinnedTestEngine:
    """Hostile callback that can self-report any committed provider metadata."""

    engine_name = "test-exact-canonicalizer"
    engine_version = "1.0.0"
    executable_sha256 = hashlib.sha256(b"test-executable").hexdigest()
    canonicalization_profile = "embedbench-test-profile-v1"

    def canonical_sha256(self, scientific_state_bytes: bytes) -> str:
        del scientific_state_bytes
        return "0" * 64


def _test_engine_manifest():
    provider = _PinnedTestEngine()
    manifest = {
        "schema": mechanism_variants.AUTHORITATIVE_ISOMORPHISM_MANIFEST_SCHEMA,
        "schema_version": (
            mechanism_variants.AUTHORITATIVE_ISOMORPHISM_MANIFEST_SCHEMA_VERSION
        ),
        "engine_name": provider.engine_name,
        "engine_version": provider.engine_version,
        "executable_sha256": provider.executable_sha256,
        "canonicalization_profile": provider.canonicalization_profile,
    }
    digest = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    return provider, manifest, digest


def test_parent_identity_uses_only_canonical_pretransform_scientific_content() -> None:
    dev = _row("dead_end", partition="mechanism_dev", ordinal=0, offset=31)
    locked = make_parameter_row(
        release_id="another-release",
        partition="mechanism_locked",
        family="dead_end",
        parent_ordinal=99,
        parent_seed_key=_seed_hex(32),
        transformation_seed_key=_seed_hex(96),
    )

    assert dev.parent_scientific_sha256 == locked.parent_scientific_sha256
    assert dev.mechanism_parent_id == locked.mechanism_parent_id


def test_parent_partition_map_allows_siblings_but_rejects_cross_partition_parent() -> None:
    parent_id = _row("dead_end", offset=33).mechanism_parent_id
    committed = mechanism_variants.make_parent_partition_map(
        release_id="release",
        assignments=((parent_id, "mechanism_dev"), (parent_id, "mechanism_dev")),
    )
    assert len(committed.entries) == 1

    with pytest.raises(MechanismCollisionError, match="crosses mechanism partitions"):
        mechanism_variants.make_parent_partition_map(
            release_id="release",
            assignments=((parent_id, "mechanism_dev"), (parent_id, "mechanism_locked")),
        )


def test_custodian_parent_map_comparison_is_bound_to_independent_commitments() -> None:
    dev_parent = _row("dead_end", offset=34).mechanism_parent_id
    locked_parent = _row(
        "articulation", partition="mechanism_locked", offset=35
    ).mechanism_parent_id
    public = mechanism_variants.make_parent_partition_map(
        release_id="release",
        assignments=((dev_parent, "mechanism_dev"),),
    )
    locked = mechanism_variants.make_parent_partition_map(
        release_id="release",
        assignments=((locked_parent, "mechanism_locked"),),
    )
    verified_public = mechanism_variants.verify_parent_partition_map(
        public.to_dict(), expected_parent_map_sha256=public.parent_map_sha256
    )
    verified_locked = mechanism_variants.verify_parent_partition_map(
        locked.to_dict(), expected_parent_map_sha256=locked.parent_map_sha256
    )
    mechanism_variants.compare_committed_parent_maps(verified_public, verified_locked)

    with pytest.raises(ValueError, match="commitment"):
        mechanism_variants.verify_parent_partition_map(
            public.to_dict(), expected_parent_map_sha256="0" * 64
        )

    colliding_locked = mechanism_variants.make_parent_partition_map(
        release_id="release",
        assignments=((dev_parent, "mechanism_locked"),),
    )
    verified_collision = mechanism_variants.verify_parent_partition_map(
        colliding_locked.to_dict(),
        expected_parent_map_sha256=colliding_locked.parent_map_sha256,
    )
    with pytest.raises(MechanismCollisionError, match="committed parent maps"):
        mechanism_variants.compare_committed_parent_maps(
            verified_public, verified_collision
        )


def test_atomic_attempt_withholds_release_digest_without_authoritative_engine(tmp_path) -> None:
    plan, _resolver, _document, _digest, _root = _verified_release_plan(tmp_path)
    registry = MechanismIdentityRegistry.empty()
    ledger = mechanism_variants.MechanismAttemptLedger.empty()

    outcome = mechanism_variants.execute_mechanism_attempt(
        plan=plan,
        registry=registry,
        ledger=ledger,
        authoritative_engine=None,
    )

    assert outcome.status == "unsupported_authoritative_isomorphism"
    assert outcome.variant is None
    assert outcome.release_mechanism_iso_sha256 is None
    assert outcome.registry == registry
    assert outcome.registry.registry_sha256 == registry.registry_sha256
    assert len(outcome.ledger.entries) == 1
    assert outcome.ledger.entries[0].status == outcome.status
    assert outcome.ledger.entries[0].diagnostic_iso_sha256 is not None
    assert outcome.ledger.entries[0].release_mechanism_iso_sha256 is None


@pytest.mark.parametrize("family", ("multi_neighbour", "short_chain_trap"))
def test_repaired_causal_motifs_reach_authoritative_boundary_without_release_digest(
    tmp_path, family: str
) -> None:
    plan, _resolver, _document, _digest, _root = _verified_release_plan(tmp_path, family)

    outcome = mechanism_variants.execute_mechanism_attempt(
        plan=plan,
        registry=MechanismIdentityRegistry.empty(),
        ledger=mechanism_variants.MechanismAttemptLedger.empty(),
        authoritative_engine=None,
    )

    assert outcome.status == "unsupported_authoritative_isomorphism"
    assert outcome.variant is None
    assert outcome.release_mechanism_iso_sha256 is None
    assert outcome.ledger.entries[0].diagnostic_iso_sha256 is not None
    assert outcome.ledger.entries[0].release_mechanism_iso_sha256 is None


def test_verified_plan_rejects_post_commit_outcome_shopping(tmp_path) -> None:
    _plan, resolver, document, plan_digest, root_digest = _verified_release_plan(tmp_path)
    document["parameter_row"]["orientation"] = "reverse"

    with pytest.raises(ValueError, match="plan SHA-256|parameter-row digest"):
        mechanism_variants.verify_mechanism_plan(
            document,
            expected_plan_sha256=plan_digest,
            resolver=resolver,
            expected_seed_registry_terminal_root_sha256=root_digest,
        )


def test_atomic_boundary_rejects_unverified_plan_dataclass() -> None:
    with pytest.raises(TypeError, match="VerifiedMechanismPlan"):
        mechanism_variants.execute_mechanism_attempt(
            plan=_row("dead_end", offset=36),
            registry=MechanismIdentityRegistry.empty(),
            ledger=mechanism_variants.MechanismAttemptLedger.empty(),
            authoritative_engine=None,
        )


@pytest.mark.parametrize("family", MECHANISM_FAMILIES)
def test_atomic_release_path_has_status_coverage_for_every_family(
    tmp_path, family: str
) -> None:
    plan, *_ = _verified_release_plan(tmp_path, family, suffix="coverage")
    outcome = mechanism_variants.execute_mechanism_attempt(
        plan=plan,
        registry=MechanismIdentityRegistry.empty(),
        ledger=mechanism_variants.MechanismAttemptLedger.empty(),
        authoritative_engine=None,
    )

    assert outcome.status == "unsupported_authoritative_isomorphism"
    assert outcome.release_mechanism_iso_sha256 is None
    assert len(outcome.registry.entries) == 0
    assert len(outcome.ledger.entries) == 1


def test_callback_only_authority_verifier_requires_external_execution_evidence() -> None:
    provider, manifest, manifest_sha256 = _test_engine_manifest()

    with pytest.raises(
        mechanism_variants.AuthoritativeIsomorphismExecutionUnavailableError,
        match="execution evidence",
    ):
        mechanism_variants.verify_authoritative_isomorphism_engine(
            provider,
            manifest,
            expected_manifest_sha256=manifest_sha256,
        )


def test_self_reported_engine_cannot_fabricate_release_digest(tmp_path) -> None:
    provider, _manifest, manifest_sha256 = _test_engine_manifest()
    plan, *_ = _verified_release_plan(
        tmp_path,
        authoritative_manifest=manifest_sha256,
        suffix="callback-engine",
    )

    outcome = mechanism_variants.execute_mechanism_attempt(
        plan=plan,
        registry=MechanismIdentityRegistry.empty(),
        ledger=mechanism_variants.MechanismAttemptLedger.empty(),
        authoritative_engine=provider,
    )

    assert outcome.status == "unsupported_authoritative_isomorphism"
    assert outcome.release_mechanism_iso_sha256 is None
    assert outcome.variant is None
    assert len(outcome.registry.entries) == 0
    assert outcome.ledger.entries[0].release_mechanism_iso_sha256 is None


def test_forged_seed_request_fails_even_under_a_new_plan_digest(tmp_path) -> None:
    _plan, resolver, document, _plan_digest, root_digest = _verified_release_plan(
        tmp_path, suffix="seed-forgery"
    )
    document["parent_seed_request"]["replicate"] = 1
    forged_digest = hashlib.sha256(canonical_bytes(document)).hexdigest()

    with pytest.raises(ValueError, match="canonical request|registered parent seed"):
        mechanism_variants.verify_mechanism_plan(
            document,
            expected_plan_sha256=forged_digest,
            resolver=resolver,
            expected_seed_registry_terminal_root_sha256=root_digest,
        )


def test_contextual_replay_failure_is_logged_without_registry_mutation(
    tmp_path, monkeypatch
) -> None:
    plan, *_ = _verified_release_plan(tmp_path, suffix="replay-failure")
    registry = MechanismIdentityRegistry.empty()

    def fail_reconstruction(_value):
        raise ValueError("adversarial reconstruction failure")

    monkeypatch.setattr(MechanismVariant, "from_dict", fail_reconstruction)
    outcome = mechanism_variants.execute_mechanism_attempt(
        plan=plan,
        registry=registry,
        ledger=mechanism_variants.MechanismAttemptLedger.empty(),
        authoritative_engine=None,
    )

    assert outcome.status == "contextual_replay_failed"
    assert outcome.registry == registry
    assert outcome.ledger.entries[0].status == "contextual_replay_failed"


def test_diagnostic_digest_is_explicitly_not_a_release_digest() -> None:
    variant = generate_single_mechanism_variant(_row("dead_end", offset=37)).require_variant()
    document = variant.to_dict()

    assert "diagnostic_iso_sha256" in document
    assert "mechanism_iso_sha256" not in document
    assert "release_mechanism_iso_sha256" not in document


def test_exact_certificate_ties_use_canonical_action_not_input_order() -> None:
    candidates = (
        mechanism_variants.ExactCandidateRecord((0, 9), (0, 0, 0), None),
        mechanism_variants.ExactCandidateRecord((0, 3), (0, 0, 0), None),
    )
    certificate = mechanism_variants.MechanismExactCertificate(
        family="dead_end",
        state_sha256="1" * 64,
        hypothesized_winner_index=1,
        certified_winner_index=1,
        margin_kind="none",
        margin_value=0,
        candidates=candidates,
    )
    assert certificate.certified_winner_index == 1

    with pytest.raises(ValueError, match="certified_winner_index"):
        replace(certificate, certified_winner_index=0)


@pytest.mark.parametrize("bad_version", (True, 1.0, "1"))
def test_parent_map_schema_version_rejects_bool_float_and_string(bad_version: object) -> None:
    value = mechanism_variants.make_parent_partition_map(
        release_id="release",
        assignments=((_row("dead_end", offset=38).mechanism_parent_id, "mechanism_dev"),),
    ).to_dict()
    value["schema_version"] = bad_version

    with pytest.raises((TypeError, ValueError)):
        mechanism_variants.MechanismParentPartitionMap.from_dict(value)
