"""Versioned, fail-closed Quality V2 paper-audit protocol contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

PAPER_AUDIT_CONTRACT_SCHEMA = "embedbench.quality-v2-paper-audit-contract"
PAPER_AUDIT_CONTRACT_SCHEMA_VERSION = 2
PAPER_AUDIT_PROTOCOL_ID = "quality-v2-paper-audit-v2"
AUDIT_SOURCE_HASH_ALGORITHM = (
    "sha256_length_prefixed_single_snapshot_transitive_local_python_import_closure_v5"
)
PAPER_PREREGISTRATION_SCHEMA = "embedbench.quality-v2-paper-preregistration"
PAPER_PREREGISTRATION_SCHEMA_VERSION = 1
PAPER_PREREGISTRATION_TRUST_MODEL = "sha256_supplied_out_of_band_before_audit_label_generation"
PAPER_PREREGISTRATION_COMMITMENT_SCOPE = (
    "audit_contract_audit_source_selection_four_checkpoints_four_policy_freezes"
)
CORE_RUNTIME_DISTRIBUTIONS = (
    "embedbench",
    "numpy",
    "scipy",
    "networkx",
    "dimod",
    "dwave-samplers",
    "dwave-networkx",
    "minorminer",
)
PAPER_PIPELINE_SCRIPTS = (
    "aggregate_quality_v2_paper.py",
    "evaluate_quality_v2.py",
    "quality_v2_paper_contract.py",
    "rescore_quality.py",
    "run_training_grid.py",
    "select_training_grid.py",
    "train_quality_v2.py",
    "training_artifacts.py",
    "training_splits.py",
)
PAPER_AUDIT_ENTRYPOINTS = (
    "aggregate_quality_v2_paper.py",
    "evaluate_quality_v2.py",
    "quality_v2_paper_contract.py",
    "rescore_quality.py",
    "select_training_grid.py",
)
PAPER_VERIFIER_PACKAGE_FILES = (
    "paper_verifier/__init__.py",
    "paper_verifier/ground_certificate.py",
    "paper_verifier/schema.py",
)
_REGISTERED_LOCAL_NAMESPACE_ROOTS = {
    "embedbench": "src/embedbench/__init__.py",
    "paper_verifier": "scripts/paper_verifier/__init__.py",
    **{Path(name).stem: f"scripts/{name}" for name in PAPER_PIPELINE_SCRIPTS},
}
_PROTECTED_LOCAL_NAMESPACES = frozenset(_REGISTERED_LOCAL_NAMESPACE_ROOTS)
MAX_JSON_INTEGER = (1 << 63) - 1


def strict_json_loads(payload: bytes | str, *, location: str) -> Any:
    """Parse untrusted JSON with duplicate, non-finite, and overflow rejection."""

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = value
        return result

    def parse_int(raw: str) -> int:
        value = int(raw)
        if not -MAX_JSON_INTEGER - 1 <= value <= MAX_JSON_INTEGER:
            raise ValueError("integer exceeds signed 64-bit range")
        return value

    def parse_float(raw: str) -> float:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("floating-point value is not finite")
        return value

    def parse_constant(raw: str) -> object:
        raise ValueError(f"non-standard numeric constant {raw!r}")

    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        return json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_int=parse_int,
            parse_float=parse_float,
            parse_constant=parse_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{location} is not strict JSON: {error}") from error


EXPECTED_PAPER_AUDIT_CONTRACT: dict[str, object] = {
    "schema": PAPER_AUDIT_CONTRACT_SCHEMA,
    "schema_version": PAPER_AUDIT_CONTRACT_SCHEMA_VERSION,
    "protocol_id": PAPER_AUDIT_PROTOCOL_ID,
    "partition": "test",
    "selection": {
        "artifact_schema": "embedbench.training-grid-selection",
        "artifact_schema_version": 3,
        "grid_id": "isingfold-quality-value-v2-screen",
        "grid_path": "configs/training_grid_quality_v2.json",
        "stage": "quality_value_v2_screen",
        "registered_cell_count": 80,
        "registered_seeds": [0, 1, 2, 3],
        "validation_only": True,
    },
    "training_registration": {
        "grid_sha256": "10be13fe54669e3577c9c6a32f972d1cd94640f091bb262b03f302bd59bdc270",
        "source_sha256": "294caa56d0bb04f716163b143a995ee176c8e72dd209c5f5f77caef92d72a264",
        "selection": {
            "unit": "logical problem group",
            "primary_metric": "mean_finite_budget_regret",
            "primary_metric_description": (
                "Validation quality regret under exact validity and B/Q_MM qubit-budget "
                "constraints."
            ),
            "auxiliary_policy": (
                "Residual-connectivity and robustness losses may regularize representation "
                "learning but never select the winning architecture."
            ),
            "test_policy": "locked during every screen cell",
            "seeds": [0, 1, 2, 3],
            "paper_checkpoint_policy": (
                "Freeze and evaluate all four seed checkpoints of the winning configuration; "
                "report the aggregate across seeds."
            ),
            "deployment_checkpoint_policy": (
                "The lowest-validation-regret seed is the deployment checkpoint only, not "
                "the sole paper estimate."
            ),
        },
        "stage": {
            "trainer": "scripts/train_quality_v2.py",
            "inputs": "runs/release_v1_1/quality_*.jsonl",
            "splits": "data/release_v1/splits_quality_problem_v2.json",
            "output_dir": "runs/training_grid_quality_v2/screen",
            "checkpoint_dir": "runs/training_grid_quality_v2/checkpoints/screen",
            "epochs": 25,
            "axis_order": ["objective_variant", "arch", "seed"],
            "axes": {
                "arch": ["mpnn", "gin", "gatv2", "gps", "hetero"],
                "objective_variant": [
                    "p_only",
                    "p_connectivity",
                    "p_robustness",
                    "full",
                ],
                "seed": [0, 1, 2, 3],
            },
            "extra_args": [
                "--hidden",
                "64",
                "--layers",
                "3",
                "--heads",
                "4",
                "--lr",
                "0.001",
                "--tol",
                "0.05",
                "--rank-margin",
                "0.05",
                "--stage1-rank-margin",
                "0.10",
                "--lcb-z",
                "1.0",
                "--stage1-weight",
                "0.25",
                "--lambda-connectivity",
                "0.1",
                "--lambda-robustness",
                "0.1",
                "--deploy-max-free",
                "8",
                "--evaluation-support",
                "full",
                "--deploy-view",
            ],
            "execution": {
                "assignment": "cell_index_modulo",
                "modulus": 2,
                "remainder_to_site": {"0": "apollo", "1": "goose"},
                "required_device_type": "cuda",
                "slurm_required_sites": ["goose"],
                "slurm_forbidden_sites": ["apollo"],
                "require_each_configuration_all_sites": True,
                "exact_balance_axes": ["arch", "objective_variant"],
                "seeds_per_configuration_per_site": 2,
                "design": (
                    "With four seeds as the innermost axis, cell-index parity assigns seeds "
                    "0 and 2 to Apollo and seeds 1 and 3 to Goose for every architecture-"
                    "objective configuration. Every architecture and objective variant is "
                    "balanced 50/50 across sites, and every configuration has exactly two "
                    "seeds on each site."
                ),
            },
        },
        "data_provenance": {
            "split_manifest": {
                "file": "splits_quality_problem_v2.json",
                "sha256": "cf3b7b44a9d3e95e1dc82b11870a9056492d4d3d7029f181e7737d07906a78f0",
                "schema": "embedbench.split-manifest",
                "schema_version": 2,
            },
            "corpus_inputs": [
                {
                    "file": "quality_chimera5_app.jsonl",
                    "sha256": "8b11660c068b626e4eae55c6d8d3ed31fdd335a48ebd682bd06727cc99b5410e",
                },
                {
                    "file": "quality_chimera5_inkdrop.jsonl",
                    "sha256": "15cc622b3d8d4cb4fd66a378a83d2d15a886f076344817515d2c250bedaa2cce",
                },
                {
                    "file": "quality_chimera5_random.jsonl",
                    "sha256": "1a4403535fa4d3ecf15c4c9d89d60456b1bba2d0bf85f2012af9b9090d196368",
                },
                {
                    "file": "quality_pegasus3_app.jsonl",
                    "sha256": "f1d911990fed7773f624dcf1badf2c806c7a3d42a0fb1c33aeb246ff56a9e9ed",
                },
                {
                    "file": "quality_pegasus3_inkdrop.jsonl",
                    "sha256": "d2ffc17be0ab3770d24436d1e28335ed91711971303c35200183783bc4b9dfb3",
                },
                {
                    "file": "quality_pegasus3_random.jsonl",
                    "sha256": "07871cdae0d8fb2858bc5c21af14dbee1c7e59807f724c54df52131d9dd1cd32",
                },
                {
                    "file": "quality_zephyr2_app.jsonl",
                    "sha256": "3bd996cef358e4b924bdc3d051e5815af34ec168b55af8abdd4db5fb4280e48d",
                },
                {
                    "file": "quality_zephyr2_inkdrop.jsonl",
                    "sha256": "b1b62bcf1a8637eb9bbb2ed2aefddbffb387fb451dc6145e37fdc1331e91cd7a",
                },
                {
                    "file": "quality_zephyr2_random.jsonl",
                    "sha256": "3b82118599bfb10fe3300db11504d0dd15577574c12d08724b6d667c37278aca",
                },
            ],
        },
        "validation_replay": {
            "partition": "val",
            "cell_coverage": "exactly_all_80_registered_frozen_checkpoints",
            "metric": "mean_finite_budget_regret",
            "evidence": (
                "per_record_problem_digest_exact_masks_total_qubits_qmm_budgets_and_choices"
            ),
            "authoritative_cross_cell_selection_source": ("fresh_canonical_cpu_checkpoint_replay"),
            "stored_training_validation_role": (
                "required_training_provenance_and_best_epoch_binding_only"
            ),
            "cross_device_model_dependent_comparison": (
                "record_only_never_reject_rank_exclude_or_break_ties"
            ),
            "model_independent_replay_action": "reject_unless_exact_match",
            "test_access": "forbidden_until_all_80_replays_and_selection_are_frozen",
            "canonical_runtime": {
                "schema": "embedbench.canonical-validation-replay-policy",
                "schema_version": 1,
                "device": "cpu",
                "model_parameter_dtype": "float32",
                "intraop_threads": 1,
                "interop_threads": 1,
                "deterministic_algorithms": True,
                "float32_matmul_precision": "highest",
                "selection_metric_source": "canonical_cpu_checkpoint_replay",
                "training_device_metric_role": (
                    "authenticated_training_time_best_epoch_provenance_only_never_cross_cell_selection"
                ),
                "cross_device_sweep_equality_required": False,
            },
        },
    },
    "audit_labels": {
        "schema": "embedbench.independent-quality-audit",
        "schema_version": 2,
        "generator": "EmbedBench/scripts/rescore_quality.py",
        "reads_per_strength": 4000,
        "num_sweeps": 200,
        "base_seed": 0,
        "seed_schedule": ("(base_seed + 7919*candidate_index + 31*strength_index) % 2**31"),
        "registered_strength_count": 4,
        "strength_schedule": "default_strength_grid(problem, 4)",
        "aggregation": "max_p_solve_over_strength_schedule",
        "score_evidence": "integer_success_counts_per_candidate_strength",
        "probability_derivation": "success_count_divided_by_reads_per_strength_binary64",
    },
    "audit_execution": {
        "shard_count": 64,
        "assignment": "sha256(canonical_audit_identity) mod shard_count",
        "canonical_audit_identity": "json_compact_utf8_array[file,instance_id,focus]",
        "record_output": "atomic_recomputed_one_record_per_audit_identity_no_partial_reuse",
        "merge_coverage": "exactly_once_complete_fixed_test_partition",
        "merge_order": "canonical_audit_identity_lexicographic",
        "release_commitment": "exact_jsonl_and_shard_manifest_bytes",
        "release_trust_anchor": "sha256_registered_out_of_band_before_phase_2",
    },
    "evaluation_phases": {
        "phase_1": "atomic_full_label_free_policy_freeze_without_audit_paths",
        "phase_2": (
            "registered_sha256_byte_exact_label_free_replay_then_external_audit_release_"
            "validation_and_scoring"
        ),
        "freeze_registration": (
            "single_preregistration_manifest_external_sha256_required_before_audit_label_generation"
        ),
        "audit_release_registration": "sha256_retained_out_of_band_before_phase_2",
        "aggregator_replay": (
            "all80_then_all4_phase1_then_externally_committed_all64_audit_validation_"
            "and_metric_recomputation"
        ),
    },
    "preregistration": {
        "schema": PAPER_PREREGISTRATION_SCHEMA,
        "schema_version": PAPER_PREREGISTRATION_SCHEMA_VERSION,
        "commitment_scope": PAPER_PREREGISTRATION_COMMITMENT_SCOPE,
        "audit_source_hash_algorithm": AUDIT_SOURCE_HASH_ALGORITHM,
        "required_checkpoint_count": 4,
        "required_policy_freeze_count": 4,
        "trust_model": PAPER_PREREGISTRATION_TRUST_MODEL,
        "colocated_checksum_sidecar_authoritative": False,
        "verification_order": (
            "manifest_sha256_then_manifest_schema_and_bindings_then_all_policy_freeze_"
            "bytes_then_audit_inputs"
        ),
    },
    "evaluation": {
        "provisional": True,
        "evaluation_design": "exploratory_legacy_iid",
        "legacy_test_exposure": ("membership_problem_sizes_and_e0_inspected_without_audit_p_solve"),
        "record_support": "complete_fixed_test_partition",
        "candidate_support": "full",
        "support_uses_labels": False,
        "ground_reference_policy": "certified_exact_only",
        "uncertified_ground_reference_action": "reject",
        "top_tolerance": 0.02,
        "random_seed": 0,
        "random_baseline_policy": "sha256_record_key_modulo_eligible_v1",
        "lcb_z": 1.0,
        "device": "cpu",
        "runtime_environment_schema": "embedbench.runtime-environment/v1-complete",
        "runtime_consistency": "exact_across_model_seed_evaluations",
        "selection_statistic": "mean",
        "reported_selection_statistics": ["mean", "lcb"],
        "budget_contract": "B/Q_MM",
        "budget_reference": "stock_minorminer_original_total_qubits",
        "primary_record_policy": "valid_stock_minorminer_original_reference_only",
        "exact_mask": "minor_embedding_feasible_and_full_embedding_total_qubits",
        "registered_budget_ratios": [1.0, 1.1, 1.25, 1.5, None],
        "primary_metric": "mean_finite_budget_regret",
        "accepted_ground_references": {
            "embedded_ground_state_certificate_v1": {
                "format_version": 1,
                "envelope_fields": [
                    "certificate",
                    "proof",
                    "certificate_sha256",
                    "proof_artifact_sha256",
                ],
                "certificate_schema": "embedbench.ground-state-certificate",
                "certificate_schema_version": 1,
                "accepted_statuses": ["planted_proof", "exact_enumeration"],
                "certified_optimal_action": "reject_until_executable_proof_checker",
                "artifact_storage": "embedded_closed_json_envelope",
            },
            "exhaustive_enumeration": {
                "format_version": "legacy_v1_1_v1_2",
                "certificate": "not_required",
                "max_variables": 22,
                "batch_states": 131072,
                "enumeration": "all_spin_assignments_vectorized_float64_v1",
            },
            "ferromagnetic_s_t_min_cut": {
                "format_version": "legacy_v1_1_v1_2",
                "certificate": "required",
                "algorithm": "networkx_preflow_push_integer_capacities_v1",
            },
        },
    },
    "secondary_endpoints": {
        "selection_role": "report_only_after_label_free_policy_freeze",
        "availability_policy": (
            "fail_closed_on_malformed_authenticated_input_else_status_and_coverage"
        ),
        "selectors": ["mean", "lcb", "resource", "original", "random"],
        "exact_feasibility_budget_survival": {
            "source": "independent_complete_embedding_validation",
            "metrics": [
                "candidate_feasibility_rate",
                "record_survival_rate",
                "selection_survival_rate",
            ],
        },
        "terminal_resources": {
            "source": "independent_complete_embedding_validation",
            "reference": "stock_minorminer_original",
            "metrics": [
                "mean_total_qubits_delta_vs_stock_original",
                "mean_max_chain_delta_vs_stock_original",
            ],
        },
        "residual_connectivity": {
            "source": "authenticated_realized_host_minus_complete_embedding",
            "reference": "stock_minorminer_original",
            "metrics": [
                "mean_remaining_free_qubits",
                "mean_largest_remaining_component",
                "mean_largest_component_fraction",
                "mean_remaining_component_count",
                "mean_remaining_free_qubits_delta_vs_stock_original",
                "mean_largest_remaining_component_delta_vs_stock_original",
                "mean_largest_component_fraction_delta_vs_stock_original",
                "mean_remaining_component_count_delta_vs_stock_original",
            ],
            "missing_input_action": "unavailable_with_explicit_coverage",
        },
        "strength_robustness": {
            "source": "authenticated_four_registered_strength_p_solve_outcomes",
            "registered_strength_count": 4,
            "support": "complete_fixed_test_and_registered_bqmm_budgets",
            "metrics": [
                "mean_worst_case_p_solve",
                "mean_p_solve_spread",
                "mean_worst_case_p_solve_delta_vs_stock_original",
                "mean_p_solve_spread_delta_vs_stock_original",
            ],
            "missing_input_action": "unavailable_with_explicit_coverage",
        },
    },
    "secondary_endpoint_inference": {
        "selection_role": "report_only_never_selection_or_primary_pass_fail",
        "selector": "mean",
        "reference": "stock_minorminer_original",
        "cluster_unit": "quality_problem_digest",
        "model_seed_reduction": "arithmetic_mean_within_record",
        "record_reduction": "arithmetic_mean_within_problem_digest",
        "missing_pair_action": "unavailable_or_partial_with_explicit_coverage",
        "bootstrap": {
            "method": "percentile_problem_cluster_bootstrap",
            "resamples": 10000,
            "confidence_level": 0.95,
            "seed": 260912,
            "quantile_method": "linear",
            "namespace_seed": "sha256_first_8_bytes_big_endian_added_to_seed",
            "minimum_problem_clusters": 20,
            "interval_role": "pointwise_descriptive_not_hypothesis_test",
            "inference_scope": "conditional_on_four_registered_model_checkpoints",
        },
        "endpoints": {
            "residual_connectivity": {
                "supports": ["1.00x", "1.10x", "1.25x", "1.50x", "uncapped"],
                "metrics": [
                    "remaining_free_qubits_delta_vs_stock_original",
                    "largest_remaining_component_delta_vs_stock_original",
                    "largest_component_fraction_delta_vs_stock_original",
                    "remaining_component_count_delta_vs_stock_original",
                ],
            },
            "strength_robustness": {
                "supports": [
                    "full_support",
                    "1.00x",
                    "1.10x",
                    "1.25x",
                    "1.50x",
                    "uncapped",
                ],
                "metrics": [
                    "worst_case_p_solve_delta_vs_stock_original",
                    "p_solve_spread_delta_vs_stock_original",
                ],
            },
        },
    },
    "model_seed_aggregation": {
        "registered_seed_count": 4,
        "seed_source": "validation_selection.registered_seeds",
        "coverage": "exactly_every_registered_seed_checkpoint_once",
        "aggregation": "arithmetic_mean",
        "dispersion": "sample_standard_deviation_ddof_1",
    },
    "secondary_inference": {
        "status": "secondary_distribution_free_bounded_mean_inference",
        "estimand": ("problem_equal_weighted_p_solve_delta_learned_mean_minus_stock_original"),
        "cluster_unit": "quality_problem_digest",
        "model_seed_reduction": "arithmetic_mean_within_record",
        "record_reduction": "arithmetic_mean_within_problem_digest",
        "inference_scope": "conditional_on_four_registered_model_checkpoints",
        "sampling_assumption": "independent_problem_clusters",
        "mean_inference": {
            "method": "two_sided_hoeffding_confidence_interval",
            "bounded_support": [-1.0, 1.0],
            "confidence_level": 0.95,
            "minimum_problem_clusters": 20,
        },
        "paired_test": {
            "method": "one_sided_hoeffding_bounded_mean_test",
            "null": "population_mean_p_solve_delta_less_than_or_equal_to_zero",
            "alternative": "learned_mean_p_solve_greater_than_stock_original",
        },
        "per_budget_multiplicity": {
            "family": "four_finite_registered_budgets",
            "method": "holm_bonferroni",
            "familywise_alpha": 0.05,
        },
    },
}


@dataclass(frozen=True)
class PaperAuditContract:
    path: Path
    sha256: str
    checksum_path: Path
    checksum_sha256: str
    document: Mapping[str, Any]

    @property
    def audit_labels(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.document["audit_labels"])

    @property
    def training_registration(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.document["training_registration"])

    @property
    def selection(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.document["selection"])

    @property
    def audit_execution(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.document["audit_execution"])

    @property
    def evaluation(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.document["evaluation"])

    @property
    def model_seed_aggregation(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.document["model_seed_aggregation"])

    @property
    def secondary_endpoints(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.document["secondary_endpoints"])

    @property
    def secondary_endpoint_inference(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.document["secondary_endpoint_inference"])

    @property
    def secondary_inference(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.document["secondary_inference"])

    @property
    def evaluation_phases(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.document["evaluation_phases"])

    @property
    def preregistration(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.document["preregistration"])

    def public_binding(self) -> dict[str, object]:
        return {
            "file": self.path.name,
            "sha256": self.sha256,
            "schema": self.document["schema"],
            "schema_version": self.document["schema_version"],
            "protocol_id": self.document["protocol_id"],
            "checksum_sidecar": {
                "file": self.checksum_path.name,
                "sha256": self.checksum_sha256,
            },
        }

    def validate_audit_provenance(
        self,
        value: object,
        *,
        location: str,
    ) -> None:
        expected = {
            key: expected_value
            for key, expected_value in self.audit_labels.items()
            if key not in {"schema", "schema_version"}
        }
        mismatches = _mismatch_paths(value, expected, "audit_labels")
        if mismatches:
            raise ValueError(
                f"{location}: audit provenance differs from the frozen paper contract at: "
                + ", ".join(mismatches)
            )


@dataclass(frozen=True)
class PaperPreregistration:
    """A byte-bound paper registration whose digest came from outside the run."""

    path: Path
    sha256: str
    document: Mapping[str, Any]

    @property
    def policy_freezes(self) -> Sequence[Mapping[str, Any]]:
        return cast(Sequence[Mapping[str, Any]], self.document["policy_freezes"])

    def public_binding(self) -> dict[str, object]:
        return {
            "file": self.path.name,
            "sha256": self.sha256,
            "schema": self.document["schema"],
            "schema_version": self.document["schema_version"],
            "protocol_id": self.document["protocol_id"],
            "trust_model": self.document["trust_model"],
            "commitment_scope": self.document["commitment_scope"],
        }


_DIRECT_DYNAMIC_CODE_LOADERS = frozenset(
    {
        "__import__",
        "__builtins__",
        "builtins.__import__",
        "builtins.compile",
        "builtins.eval",
        "builtins.exec",
        "builtins.globals",
        "builtins.locals",
        "builtins.vars",
        "compile",
        "eval",
        "exec",
        "globals",
        "importlib.import_module",
        "importlib.machinery.ExtensionFileLoader",
        "importlib.machinery.SourceFileLoader",
        "importlib.machinery.SourcelessFileLoader",
        "importlib.util.module_from_spec",
        "importlib.util.spec_from_file_location",
        "locals",
        "object.__getattribute__",
        "operator.attrgetter",
        "pkgutil.resolve_name",
        "runpy.run_module",
        "runpy.run_path",
        "sys.meta_path",
        "sys.modules",
        "sys.path",
        "sys.path_hooks",
        "sys.path_importer_cache",
        "type.__getattribute__",
        "vars",
    }
)
_DYNAMIC_CODE_LOADER_MODULES = frozenset(
    {"builtins", "importlib", "importlib.machinery", "importlib.util", "pkgutil", "runpy"}
)
_REFLECTIVE_NAMESPACE_ATTRIBUTES = frozenset(
    {
        "__bases__",
        "__class__",
        "__closure__",
        "__code__",
        "__dict__",
        "__func__",
        "__getattr__",
        "__getattribute__",
        "__globals__",
        "__mro__",
        "__self__",
        "__subclasses__",
    }
)
_DANGEROUS_LOADER_ATTRIBUTE_NAMES = frozenset(
    {
        "ExtensionFileLoader",
        "SourceFileLoader",
        "SourcelessFileLoader",
        "__import__",
        "import_module",
        "module_from_spec",
        "resolve_name",
        "run_module",
        "run_path",
        "spec_from_file_location",
    }
)
_FORBIDDEN_REFLECTIVE_IMPORT_ROOTS = frozenset(
    {"builtins", "importlib", "operator", "pkgutil", "runpy"}
)
_ALLOWED_IMPORTLIB_MEMBERS = frozenset({"metadata", "resources"})
_ALLOWED_IMPORTLIB_MODULE_ATTRIBUTES = {
    "importlib.metadata": frozenset({"PackageNotFoundError", "version"}),
    "importlib.resources": frozenset({"as_file", "files"}),
}
_DORMANT_LEARNED_RESOURCE_MODULES = frozenset(
    {"embedbench.evaluate", "embedbench.objective_embedder"}
)
_SECURE_NOFOLLOW_AVAILABLE = hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY")
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_SOURCE_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


@dataclass(frozen=True)
class _AuditSourceSnapshot:
    source_root: Path
    payloads: Mapping[str, bytes]
    inventory: Mapping[str, tuple[int, int, int, int, int, int]]


def _source_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        stat.S_IFMT(metadata.st_mode),
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Track directory replacement without treating unrelated cache writes as source drift."""

    return (stat.S_IFMT(metadata.st_mode), metadata.st_dev, metadata.st_ino, 0, 0, 0)


def _open_source_root(root: str | Path) -> tuple[Path, int]:
    if not _SECURE_NOFOLLOW_AVAILABLE:
        raise ValueError("paper-audit source verification requires O_NOFOLLOW and O_DIRECTORY")
    source_root = Path(os.path.abspath(os.fspath(root)))
    try:
        descriptor = os.open(source_root.anchor or os.sep, _DIRECTORY_OPEN_FLAGS)
    except OSError as error:
        raise ValueError(f"cannot securely open paper-audit source root {source_root}") from error
    traversed = Path(source_root.anchor or os.sep)
    try:
        for component in source_root.parts[1:]:
            traversed /= component
            try:
                metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise ValueError(
                    f"cannot securely inspect paper-audit source-root ancestor {traversed}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"paper-audit source root has a symlinked ancestor: {traversed}")
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(
                    f"paper-audit source-root ancestor is not a directory: {traversed}"
                )
            try:
                child = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=descriptor)
            except OSError as error:
                raise ValueError(
                    f"cannot securely open paper-audit source-root ancestor {traversed}"
                ) from error
            opened = os.fstat(child)
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                os.close(child)
                raise ValueError(
                    f"paper-audit source-root ancestor changed while opening {traversed}"
                )
            os.close(descriptor)
            descriptor = child
    except Exception:
        os.close(descriptor)
        raise
    return source_root, descriptor


def _read_source_descriptor(
    directory_descriptor: int,
    name: str,
    expected: os.stat_result,
    *,
    relative: str,
) -> bytes:
    try:
        descriptor = os.open(name, _SOURCE_OPEN_FLAGS, dir_fd=directory_descriptor)
    except OSError as error:
        raise ValueError(f"cannot securely open paper-audit source {relative}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (
            before.st_dev,
            before.st_ino,
        ) != (expected.st_dev, expected.st_ino):
            raise ValueError(f"paper-audit source changed while opening {relative}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _source_identity(before) != _source_identity(after):
            raise ValueError(f"paper-audit source changed while reading {relative}")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise ValueError(f"paper-audit source changed while reading {relative}")
        return payload
    finally:
        os.close(descriptor)


def _scan_source_tree(
    root_descriptor: int,
    *,
    capture_payloads: bool,
) -> tuple[dict[str, bytes], dict[str, tuple[int, int, int, int, int, int]]]:
    payloads: dict[str, bytes] = {}
    inventory: dict[str, tuple[int, int, int, int, int, int]] = {}

    def visit(directory_descriptor: int, relative_directory: str) -> None:
        try:
            with os.scandir(directory_descriptor) as entries:
                ordered = sorted(entries, key=lambda entry: entry.name)
        except OSError as error:
            raise ValueError(
                f"cannot securely enumerate paper-audit source {relative_directory}"
            ) from error
        for entry in ordered:
            relative = f"{relative_directory}/{entry.name}"
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ValueError(f"cannot inspect paper-audit source {relative}") from error
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"paper-audit source tree contains a symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                inventory[f"{relative}/"] = _directory_identity(metadata)
                try:
                    child = os.open(entry.name, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_descriptor)
                except OSError as error:
                    raise ValueError(
                        f"cannot securely open paper-audit source directory {relative}"
                    ) from error
                try:
                    opened = os.fstat(child)
                    if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                        raise ValueError(
                            f"paper-audit source directory changed while opening {relative}"
                        )
                    visit(child, relative)
                finally:
                    os.close(child)
            elif entry.name.endswith(".py"):
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError(f"paper-audit Python source is not a regular file: {relative}")
                inventory[relative] = _source_identity(metadata)
                if capture_payloads:
                    payloads[relative] = _read_source_descriptor(
                        directory_descriptor,
                        entry.name,
                        metadata,
                        relative=relative,
                    )

    for import_root in ("scripts", "src"):
        try:
            descriptor = os.open(import_root, _DIRECTORY_OPEN_FLAGS, dir_fd=root_descriptor)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(
                f"cannot securely open paper-audit import root {import_root}"
            ) from error
        try:
            metadata = os.fstat(descriptor)
            inventory[f"{import_root}/"] = _directory_identity(metadata)
            visit(descriptor, import_root)
        finally:
            os.close(descriptor)
    return payloads, inventory


def _capture_audit_source_snapshot(root: str | Path) -> tuple[_AuditSourceSnapshot, int]:
    source_root, root_descriptor = _open_source_root(root)
    try:
        payloads, inventory = _scan_source_tree(root_descriptor, capture_payloads=True)
    except Exception:
        os.close(root_descriptor)
        raise
    return _AuditSourceSnapshot(source_root, payloads, inventory), root_descriptor


def _verify_source_inventory_unchanged(
    snapshot: _AuditSourceSnapshot,
    root_descriptor: int,
) -> None:
    _, current = _scan_source_tree(root_descriptor, capture_payloads=False)
    if current != snapshot.inventory:
        raise ValueError("paper-audit source inventory changed during closure verification")


def _local_python_modules(snapshot: _AuditSourceSnapshot) -> dict[str, tuple[Path, ...]]:
    """Index importable modules from the retained paper-source byte snapshot."""

    candidates: dict[str, list[Path]] = {}
    for relative_text in snapshot.payloads:
        relative = Path(relative_text)
        if relative.parts[0] not in {"scripts", "src"}:
            continue
        import_relative = Path(*relative.parts[1:])
        if relative.name == "__init__.py":
            parts = import_relative.parent.parts
        else:
            parts = (*import_relative.parent.parts, relative.stem)
        if not parts:
            continue
        module = ".".join(parts)
        candidates.setdefault(module, []).append(snapshot.source_root / relative)
    return {
        module: tuple(
            sorted(paths, key=lambda path: path.relative_to(snapshot.source_root).as_posix())
        )
        for module, paths in candidates.items()
    }


def _module_for_path(path: Path, source_root: Path) -> tuple[str, bool]:
    for import_root in (source_root / "scripts", source_root / "src"):
        try:
            relative = path.relative_to(import_root)
        except ValueError:
            continue
        if path.name == "__init__.py":
            return ".".join(relative.parent.parts), True
        return ".".join((*relative.parent.parts, path.stem)), False
    raise ValueError(f"paper-audit source is outside its local import roots: {path}")


def _absolute_from_import(
    node: ast.ImportFrom,
    *,
    importer_module: str,
    importer_is_package: bool,
    importer: Path,
) -> str:
    if node.level == 0:
        return node.module or ""
    package = importer_module if importer_is_package else importer_module.rpartition(".")[0]
    package_parts = package.split(".") if package else []
    parents_to_remove = node.level - 1
    if parents_to_remove > len(package_parts):
        raise ValueError(f"unresolved relative local import in {importer}")
    prefix = package_parts[: len(package_parts) - parents_to_remove]
    if node.module:
        prefix.extend(node.module.split("."))
    return ".".join(prefix)


def _dotted_expression(
    value: ast.expr,
    aliases: Mapping[str, str],
) -> str | None:
    if isinstance(value, ast.Name):
        resolved = value.id
        seen: set[str] = set()
        while resolved in aliases and resolved not in seen:
            seen.add(resolved)
            candidate = aliases[resolved]
            if candidate == resolved:
                break
            resolved = candidate
        return resolved
    if isinstance(value, ast.Attribute):
        parent = _dotted_expression(value.value, aliases)
        return f"{parent}.{value.attr}" if parent else None
    return None


def _reject_dynamic_code_loading(tree: ast.AST, *, source: Path) -> None:
    aliases: dict[str, str] = {}
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_root = alias.name.partition(".")[0]
                if import_root in _FORBIDDEN_REFLECTIVE_IMPORT_ROOTS:
                    raise ValueError(
                        f"dynamic import module access {alias.name!r} is forbidden in {source}"
                    )
                bound = alias.asname or alias.name.partition(".")[0]
                aliases[bound] = alias.name if alias.asname else bound
        elif isinstance(node, ast.ImportFrom) and node.module:
            import_root = node.module.partition(".")[0]
            imported_names = {alias.name for alias in node.names}
            if imported_names & _DANGEROUS_LOADER_ATTRIBUTE_NAMES:
                dangerous = sorted(imported_names & _DANGEROUS_LOADER_ATTRIBUTE_NAMES)
                raise ValueError(
                    f"dynamic import loader aliases {dangerous!r} are forbidden in {source}"
                )
            allowed_members = _ALLOWED_IMPORTLIB_MODULE_ATTRIBUTES.get(node.module)
            allowed_importlib_api = (
                allowed_members is not None
                and imported_names <= allowed_members
                and "*" not in imported_names
            ) or (
                node.module == "importlib"
                and imported_names <= _ALLOWED_IMPORTLIB_MEMBERS
                and "*" not in imported_names
            )
            if import_root in _FORBIDDEN_REFLECTIVE_IMPORT_ROOTS and not allowed_importlib_api:
                raise ValueError(
                    f"dynamic import module access {node.module!r} is forbidden in {source}"
                )
            for alias in node.names:
                if alias.name != "*":
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    assignments: list[tuple[str, ast.expr]] = []
    for node in ast.walk(tree):
        targets: tuple[ast.expr, ...]
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
            value = node.value
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            targets = (node.target,)
            value = node.value
        else:
            continue
        assignments.extend((target.id, value) for target in targets if isinstance(target, ast.Name))

    for _ in range(len(assignments) + 1):
        changed = False
        for bound, value in assignments:
            resolved = _dotted_expression(value, aliases)
            if resolved in _DIRECT_DYNAMIC_CODE_LOADERS:
                raise ValueError(
                    f"dynamic code loading alias {bound!r} for {resolved!r} "
                    f"is forbidden in {source}"
                )
            if resolved is not None and bound not in aliases:
                aliases[bound] = resolved
                changed = True
        if not changed:
            break

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced_module = _dotted_expression(node, aliases)
            allowed_attributes = _ALLOWED_IMPORTLIB_MODULE_ATTRIBUTES.get(referenced_module or "")
            if allowed_attributes is not None:
                parent = parents.get(node)
                if not (
                    isinstance(parent, ast.Attribute)
                    and parent.value is node
                    and parent.attr in allowed_attributes
                ):
                    raise ValueError(
                        f"dynamic import module object {referenced_module!r} may only be "
                        f"used through registered attributes in {source}"
                    )
        if isinstance(node, ast.expr):
            referenced = _dotted_expression(node, aliases)
            if referenced in _DIRECT_DYNAMIC_CODE_LOADERS:
                raise ValueError(
                    f"dynamic code loading reference {referenced!r} is forbidden in {source}"
                )
        if isinstance(node, ast.Attribute):
            target = _dotted_expression(node.value, aliases)
            allowed_attributes = _ALLOWED_IMPORTLIB_MODULE_ATTRIBUTES.get(target or "")
            if allowed_attributes is not None and node.attr not in allowed_attributes:
                raise ValueError(
                    f"dynamic import access through unregistered attribute "
                    f"{target}.{node.attr} is forbidden in {source}"
                )
            if node.attr in _REFLECTIVE_NAMESPACE_ATTRIBUTES:
                raise ValueError(
                    f"dynamic import reflection through {target}.{node.attr} "
                    f"is forbidden in {source}"
                )
            if node.attr in _DANGEROUS_LOADER_ATTRIBUTE_NAMES:
                raise ValueError(
                    f"dynamic import loader attribute {node.attr!r} is forbidden in {source}"
                )
        if not isinstance(node, ast.Call):
            continue
        called = _dotted_expression(node.func, aliases)
        if called in _DIRECT_DYNAMIC_CODE_LOADERS:
            raise ValueError(f"dynamic code loading {called!r} is forbidden in {source}")
        if called == "getattr" and node.args:
            target = _dotted_expression(node.args[0], aliases)
            requested = (
                node.args[1].value
                if len(node.args) > 1
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                else None
            )
            if (
                target in _DYNAMIC_CODE_LOADER_MODULES
                or requested in _DANGEROUS_LOADER_ATTRIBUTE_NAMES
                or requested in _REFLECTIVE_NAMESPACE_ATTRIBUTES
            ):
                raise ValueError(
                    f"dynamic import dispatch through getattr is forbidden in {source}"
                )


def _reject_learned_resource_execution_path(
    tree: ast.AST,
    *,
    importer_module: str,
    source: Path,
) -> None:
    """Keep ObjectiveEmbedder's packaged joblib files outside the paper trust boundary.

    The historical package initializer imports ``embedbench.evaluate`` and our conservative
    static import closure therefore includes its dormant ObjectiveEmbedder convenience
    function and the ObjectiveEmbedder module.  Paper entrypoints never call that function.
    Rejecting every other normal import or call edge into it makes the two packaged learned
    resources unreachable under the registered, non-reflective source policy.
    """

    if importer_module in _DORMANT_LEARNED_RESOURCE_MODULES:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == "embedbench.objective_embedder"
            or alias.name.startswith("embedbench.objective_embedder.")
            for alias in node.names
        ):
            raise ValueError(
                f"paper verifier must not execute packaged learned-model resources from {source}"
            )
        if (
            isinstance(node, ast.ImportFrom)
            and node.module in {"embedbench.objective_embedder", "embedbench.evaluate"}
            and any(
                alias.name in {"ObjectiveEmbedder", "evaluate_objective_embedder"}
                for alias in node.names
            )
        ):
            raise ValueError(
                f"paper verifier must not execute packaged learned-model resources from {source}"
            )
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in {"ObjectiveEmbedder", "evaluate_objective_embedder"}
        ) or (
            isinstance(node, ast.Attribute)
            and node.attr in {"ObjectiveEmbedder", "evaluate_objective_embedder"}
        ):
            raise ValueError(
                f"paper verifier must not execute packaged learned-model resources from {source}"
            )
        if isinstance(node, ast.Call):
            called = _dotted_expression(node.func, {})
            if called and called.rpartition(".")[2] in {
                "ObjectiveEmbedder",
                "evaluate_objective_embedder",
            }:
                raise ValueError(
                    "paper verifier must not execute packaged learned-model resources "
                    f"from {source}"
                )


def _audit_source_closure(root: str | Path) -> tuple[Path, tuple[tuple[Path, bytes], ...]]:
    snapshot, root_descriptor = _capture_audit_source_snapshot(root)
    source_root = snapshot.source_root
    try:
        modules = _local_python_modules(snapshot)
        local_namespaces = {
            *{module.partition(".")[0] for module in modules},
            *_PROTECTED_LOCAL_NAMESPACES,
        }
        entrypoints = tuple(source_root / "scripts" / name for name in PAPER_AUDIT_ENTRYPOINTS)
        entrypoint_relatives = {path.relative_to(source_root).as_posix() for path in entrypoints}
        if not entrypoints or not entrypoint_relatives <= set(snapshot.payloads):
            raise ValueError("staged paper-audit source tree is incomplete")
        verifier_relatives = {f"scripts/{relative}" for relative in PAPER_VERIFIER_PACKAGE_FILES}
        if not verifier_relatives <= set(snapshot.payloads):
            raise ValueError("staged paper-audit verifier package is incomplete")

        closure: set[Path] = set()
        pending = list(entrypoints)

        def resolve(module: str, *, importer: Path) -> Path | None:
            matches = modules.get(module, ())
            if len(matches) > 1:
                rendered = ", ".join(path.relative_to(source_root).as_posix() for path in matches)
                raise ValueError(f"ambiguous local import {module!r} in {importer}: {rendered}")
            return matches[0] if matches else None

        def enqueue(module: str, *, importer: Path) -> bool:
            path = resolve(module, importer=importer)
            if path is None:
                return False
            pending.append(path)
            parts = module.split(".")
            for index in range(1, len(parts)):
                initializer = resolve(".".join(parts[:index]), importer=importer)
                if initializer is not None and initializer.name == "__init__.py":
                    pending.append(initializer)
            return True

        def require_registered_namespace(module: str, *, importer: Path) -> None:
            namespace = module.partition(".")[0]
            registered_root = _REGISTERED_LOCAL_NAMESPACE_ROOTS.get(namespace)
            if registered_root is not None and registered_root not in snapshot.payloads:
                raise ValueError(
                    f"registered local namespace root {registered_root!r} for {namespace!r} "
                    f"is missing in {importer}"
                )

        while pending:
            path = pending.pop()
            if path in closure:
                continue
            closure.add(path)
            relative = path.relative_to(source_root).as_posix()
            try:
                payload = snapshot.payloads[relative]
            except KeyError as error:  # pragma: no cover - snapshot/module-map invariant
                raise RuntimeError(f"paper-audit snapshot lost {relative}") from error
            try:
                tree = ast.parse(payload, filename=str(path))
            except (SyntaxError, ValueError) as error:
                raise ValueError(f"cannot parse paper-audit source {path}: {error}") from error
            importer_module, importer_is_package = _module_for_path(path, source_root)
            _reject_dynamic_code_loading(tree, source=path)
            _reject_learned_resource_execution_path(
                tree,
                importer_module=importer_module,
                source=path,
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        require_registered_namespace(alias.name, importer=path)
                        if not enqueue(alias.name, importer=path):
                            namespace = alias.name.partition(".")[0]
                            if namespace in local_namespaces:
                                raise ValueError(
                                    f"unresolved local import {alias.name!r} in {path}"
                                )
                elif isinstance(node, ast.ImportFrom):
                    base = _absolute_from_import(
                        node,
                        importer_module=importer_module,
                        importer_is_package=importer_is_package,
                        importer=path,
                    )
                    if base:
                        require_registered_namespace(base, importer=path)
                    base_resolved = enqueue(base, importer=path) if base else False
                    child_resolved = False
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        child = f"{base}.{alias.name}" if base else alias.name
                        child_resolved = enqueue(child, importer=path) or child_resolved
                    namespace = base.partition(".")[0] if base else ""
                    if (
                        not base_resolved
                        and not child_resolved
                        and (node.level > 0 or namespace in local_namespaces)
                    ):
                        raise ValueError(f"unresolved local import {base!r} in {path}")

        _verify_source_inventory_unchanged(snapshot, root_descriptor)
        ordered = tuple(
            (path, snapshot.payloads[path.relative_to(source_root).as_posix()])
            for path in sorted(
                closure,
                key=lambda candidate: candidate.relative_to(source_root).as_posix(),
            )
        )
        return source_root, ordered
    finally:
        os.close(root_descriptor)


def audit_source_files(root: str | Path) -> tuple[Path, ...]:
    """Return the complete static local-import closure of the verifier entrypoints.

    Imports are resolved without executing source. Every reachable module below ``scripts/``
    or ``src/`` is included, including package initializers. Ambiguous imports and unresolved
    imports into a known local namespace fail closed instead of silently producing a partial
    commitment.
    """

    _, closure = _audit_source_closure(root)
    return tuple(path for path, _ in closure)


def audit_source_sha256(root: str | Path) -> str:
    """Hash the verifier's local import closure separately from frozen training source."""

    source_root, closure = _audit_source_closure(root)
    digest = hashlib.sha256()
    for path, payload in closure:
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def validate_runtime_provenance(
    value: object,
    *,
    location: str,
    expected_device: str | None = None,
) -> dict[str, object]:
    """Validate the complete schema-v1 environment artifact, not a permissive subset."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "schema_version",
        "python",
        "torch",
        "device",
        "packages",
    }:
        raise ValueError(f"{location} has incomplete runtime provenance")
    python = value.get("python")
    torch = value.get("torch")
    device = value.get("device")
    packages = value.get("packages")
    if (
        value.get("schema") != "embedbench.runtime-environment"
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or not isinstance(python, Mapping)
        or set(python) != {"implementation", "version", "executable"}
        or not all(isinstance(python.get(field), str) and python[field] for field in python)
        or not isinstance(torch, Mapping)
        or set(torch) != {"version", "cuda_available", "cuda_runtime_version", "cudnn_version"}
        or not isinstance(torch.get("version"), str)
        or not torch["version"]
        or not isinstance(torch.get("cuda_available"), bool)
        or (
            torch.get("cuda_runtime_version") is not None
            and not isinstance(torch.get("cuda_runtime_version"), str)
        )
        or (
            torch.get("cudnn_version") is not None
            and (
                isinstance(torch.get("cudnn_version"), bool)
                or not isinstance(torch.get("cudnn_version"), int)
            )
        )
        or not isinstance(device, Mapping)
        or set(device) != {"selected", "type", "index", "gpu"}
        or device.get("type") not in {"cpu", "cuda"}
        or not isinstance(device.get("selected"), str)
        or not device["selected"]
        or (
            device.get("index") is not None
            and (isinstance(device.get("index"), bool) or not isinstance(device.get("index"), int))
        )
        or not isinstance(packages, Mapping)
        or set(packages) != set(CORE_RUNTIME_DISTRIBUTIONS)
        or any(value is not None and not isinstance(value, str) for value in packages.values())
    ):
        raise ValueError(f"{location} has incomplete runtime provenance")
    if expected_device is not None and (
        device.get("type") != expected_device
        or (expected_device == "cpu" and device.get("selected") != "cpu")
        or (expected_device == "cuda" and not str(device.get("selected", "")).startswith("cuda"))
    ):
        raise ValueError(f"{location} did not use the frozen {expected_device} device")
    gpu = device.get("gpu")
    if device["type"] == "cpu":
        if device.get("index") is not None or gpu is not None:
            raise ValueError(f"{location} has invalid CPU runtime provenance")
    elif (
        not isinstance(gpu, Mapping)
        or set(gpu) != {"name", "compute_capability", "total_memory_bytes"}
        or not isinstance(gpu.get("name"), str)
        or not gpu["name"]
        or not isinstance(gpu.get("compute_capability"), list)
        or len(gpu["compute_capability"]) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in gpu["compute_capability"]
        )
        or isinstance(gpu.get("total_memory_bytes"), bool)
        or not isinstance(gpu.get("total_memory_bytes"), int)
        or gpu["total_memory_bytes"] <= 0
    ):
        raise ValueError(f"{location} has invalid CUDA runtime provenance")
    return cast(dict[str, object], json.loads(json.dumps(value, sort_keys=True)))


def selection_audit_binding(
    document: Mapping[str, Any],
    *,
    selection_file: str,
    selection_sha256: str,
) -> dict[str, object]:
    """Return the immutable selection subset copied into each paper audit row."""

    winner = document.get("winner")
    if not isinstance(winner, Mapping):
        raise ValueError("selection artifact has no winner")
    checkpoints = winner.get("paper_checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("selection artifact has no paper checkpoints")
    return {
        "file": Path(selection_file).name,
        "sha256": selection_sha256,
        "artifact_schema": document.get("schema"),
        "artifact_schema_version": document.get("schema_version"),
        "grid_id": document.get("grid_id"),
        "grid_path": document.get("grid_path"),
        "grid_sha256": document.get("grid_sha256"),
        "source_sha256": document.get("source_sha256"),
        "selector_sha256": document.get("selector_sha256"),
        "stage": document.get("stage"),
        "registered_cell_count": document.get("registered_cell_count"),
        "registered_seeds": document.get("registered_seeds"),
        "winner_params": winner.get("params"),
        "paper_checkpoints": checkpoints,
        "audit_source_hash_algorithm": document.get("audit_source_hash_algorithm"),
        "audit_source_sha256": document.get("audit_source_sha256"),
    }


def selection_checkpoint_binding(
    document: Mapping[str, Any],
    *,
    selection_file: str,
    selection_sha256: str,
    checkpoint: Mapping[str, Any],
) -> dict[str, object]:
    """Return the checkpoint-specific selection binding used by a policy freeze."""

    winner = document.get("winner")
    if not isinstance(winner, Mapping) or not isinstance(winner.get("params"), Mapping):
        raise ValueError("selection artifact has no winner parameter object")
    return {
        "file": Path(selection_file).name,
        "sha256": selection_sha256,
        "artifact_schema": document.get("schema"),
        "artifact_schema_version": document.get("schema_version"),
        "grid_id": document.get("grid_id"),
        "grid_sha256": document.get("grid_sha256"),
        "source_sha256": document.get("source_sha256"),
        "selector_sha256": document.get("selector_sha256"),
        "stage": document.get("stage"),
        "registered_seeds": document.get("registered_seeds"),
        "winner_params": winner.get("params"),
        "paper_checkpoint": dict(checkpoint),
    }


def _canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _validated_preregistration_components(
    *,
    audit_contract: PaperAuditContract,
    selection_document: Mapping[str, Any],
    selection_file: str,
    selection_sha256: str,
    audit_source_root: str | Path,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    if not _valid_sha256(selection_sha256):
        raise ValueError("selection SHA-256 must be lowercase hexadecimal")
    if audit_contract.preregistration.get("audit_source_hash_algorithm") != (
        AUDIT_SOURCE_HASH_ALGORITHM
    ):
        raise ValueError("paper contract has the wrong audit-source hash algorithm")

    registered_seeds = audit_contract.selection["registered_seeds"]
    expected_selection_fields = {
        "schema": audit_contract.selection["artifact_schema"],
        "schema_version": audit_contract.selection["artifact_schema_version"],
        "grid_id": audit_contract.selection["grid_id"],
        "grid_path": audit_contract.selection["grid_path"],
        "stage": audit_contract.selection["stage"],
        "registered_cell_count": audit_contract.selection["registered_cell_count"],
        "registered_seeds": registered_seeds,
        "grid_sha256": audit_contract.training_registration["grid_sha256"],
        "source_sha256": audit_contract.training_registration["source_sha256"],
    }
    mismatches = [
        key
        for key, expected in expected_selection_fields.items()
        if selection_document.get(key) != expected
    ]
    if mismatches:
        raise ValueError("selection differs from the paper contract at: " + ", ".join(mismatches))

    current_audit_source_sha256 = audit_source_sha256(audit_source_root)
    if selection_document.get("audit_source_hash_algorithm") != AUDIT_SOURCE_HASH_ALGORITHM:
        raise ValueError("selection has the wrong audit-source hash algorithm")
    if selection_document.get("audit_source_sha256") != current_audit_source_sha256:
        raise ValueError("selection audit-source SHA-256 does not match the staged verifier tree")

    winner = selection_document.get("winner")
    if not isinstance(winner, Mapping) or not isinstance(winner.get("params"), Mapping):
        raise ValueError("selection artifact has no winner parameter object")
    raw_checkpoints = winner.get("paper_checkpoints")
    required_count = audit_contract.preregistration["required_checkpoint_count"]
    if (
        not isinstance(raw_checkpoints, list)
        or len(raw_checkpoints) != required_count
        or len(raw_checkpoints) != len(registered_seeds)
    ):
        raise ValueError("preregistration requires every registered paper checkpoint")

    checkpoints: list[dict[str, object]] = []
    checkpoint_paths: set[str] = set()
    for expected_seed, raw_checkpoint in zip(registered_seeds, raw_checkpoints, strict=True):
        if not isinstance(raw_checkpoint, Mapping) or set(raw_checkpoint) != {
            "seed",
            "cell_id",
            "checkpoint",
            "checkpoint_sha256",
        }:
            raise ValueError("selection has an invalid paper checkpoint")
        checkpoint = dict(raw_checkpoint)
        if (
            checkpoint["seed"] != expected_seed
            or isinstance(checkpoint["seed"], bool)
            or not isinstance(checkpoint["cell_id"], str)
            or not checkpoint["cell_id"]
            or not isinstance(checkpoint["checkpoint"], str)
            or not checkpoint["checkpoint"]
            or checkpoint["checkpoint"] in checkpoint_paths
            or not _valid_sha256(checkpoint["checkpoint_sha256"])
        ):
            raise ValueError("selection has an invalid paper checkpoint")
        checkpoint_paths.add(str(checkpoint["checkpoint"]))
        checkpoints.append(checkpoint)

    return (
        {
            "algorithm": AUDIT_SOURCE_HASH_ALGORITHM,
            "sha256": current_audit_source_sha256,
        },
        selection_audit_binding(
            selection_document,
            selection_file=selection_file,
            selection_sha256=selection_sha256,
        ),
        checkpoints,
    )


def _validate_policy_freeze_document(
    document: object,
    *,
    location: str,
    audit_contract_binding: Mapping[str, Any],
    selection_document: Mapping[str, Any],
    selection_file: str,
    selection_sha256: str,
    checkpoint: Mapping[str, Any],
) -> None:
    if not isinstance(document, Mapping):
        raise ValueError(f"{location} must be a JSON object")
    expected_selection = selection_checkpoint_binding(
        selection_document,
        selection_file=selection_file,
        selection_sha256=selection_sha256,
        checkpoint=checkpoint,
    )
    expected_checkpoint = {
        "seed": checkpoint["seed"],
        "file": Path(str(checkpoint["checkpoint"])).name,
        "sha256": checkpoint["checkpoint_sha256"],
    }
    actual_checkpoint = document.get("checkpoint")
    if (
        document.get("schema") != "embedbench.quality-v2-label-free-policy-freeze"
        or document.get("schema_version") != 2
        or document.get("phase") != "label_free_policy_freeze"
        or document.get("audit_paths_opened") is not False
        or document.get("release_quality_labels_consumed") is not False
        or not isinstance(actual_checkpoint, Mapping)
        or any(actual_checkpoint.get(key) != value for key, value in expected_checkpoint.items())
    ):
        raise ValueError(f"{location} is not a valid label-free policy freeze")
    require_exact_json(
        document.get("audit_contract"),
        audit_contract_binding,
        location=f"{location}.audit_contract",
    )
    require_exact_json(
        document.get("selection"),
        expected_selection,
        location=f"{location}.selection",
    )


def build_paper_preregistration(
    *,
    audit_contract: PaperAuditContract,
    selection_document: Mapping[str, Any],
    selection_file: str,
    selection_sha256: str,
    audit_source_root: str | Path,
    policy_freezes: Sequence[str | Path],
) -> tuple[dict[str, object], bytes]:
    """Build the single document that must be registered before audit labeling."""

    audit_source, selection, checkpoints = _validated_preregistration_components(
        audit_contract=audit_contract,
        selection_document=selection_document,
        selection_file=selection_file,
        selection_sha256=selection_sha256,
        audit_source_root=audit_source_root,
    )
    if len(policy_freezes) != len(checkpoints):
        raise ValueError("preregistration requires one policy freeze per paper checkpoint")

    checkpoint_by_seed = {checkpoint["seed"]: checkpoint for checkpoint in checkpoints}
    freeze_by_seed: dict[int, dict[str, object]] = {}
    for raw_path in policy_freezes:
        path = Path(raw_path)
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ValueError(f"invalid policy freeze {path}") from error
        document = strict_json_loads(payload, location=f"policy freeze {path}")
        if not isinstance(document, Mapping):
            raise ValueError(f"policy freeze {path} must be a JSON object")
        raw_checkpoint = document.get("checkpoint")
        seed = raw_checkpoint.get("seed") if isinstance(raw_checkpoint, Mapping) else None
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in checkpoint_by_seed:
            raise ValueError(f"policy freeze {path} has an unregistered checkpoint seed")
        if seed in freeze_by_seed:
            raise ValueError("preregistration repeats a policy-freeze seed")
        _validate_policy_freeze_document(
            document,
            location=f"policy freeze {path}",
            audit_contract_binding=audit_contract.public_binding(),
            selection_document=selection_document,
            selection_file=selection_file,
            selection_sha256=selection_sha256,
            checkpoint=checkpoint_by_seed[seed],
        )
        freeze_by_seed[seed] = {
            "seed": seed,
            "file": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    seeds = [checkpoint["seed"] for checkpoint in checkpoints]
    if set(freeze_by_seed) != set(seeds):
        raise ValueError("preregistration does not cover every registered policy-freeze seed")
    freeze_entries = [freeze_by_seed[cast(int, seed)] for seed in seeds]
    if len({entry["file"] for entry in freeze_entries}) != len(freeze_entries):
        raise ValueError("preregistration repeats a policy-freeze file name")

    preregistration = {
        "schema": PAPER_PREREGISTRATION_SCHEMA,
        "schema_version": PAPER_PREREGISTRATION_SCHEMA_VERSION,
        "protocol_id": audit_contract.document["protocol_id"],
        "trust_model": audit_contract.preregistration["trust_model"],
        "commitment_scope": audit_contract.preregistration["commitment_scope"],
        "audit_contract": audit_contract.public_binding(),
        "audit_source": audit_source,
        "selection": selection,
        "paper_checkpoints": checkpoints,
        "policy_freezes": freeze_entries,
    }
    return preregistration, _canonical_json_bytes(preregistration)


def load_paper_preregistration(
    path: str | Path,
    expected_sha256: str,
    *,
    audit_contract: PaperAuditContract,
    selection_document: Mapping[str, Any],
    selection_file: str,
    selection_sha256: str,
    audit_source_root: str | Path,
) -> PaperPreregistration:
    """Load a preregistration using a mandatory, independently supplied digest."""

    if not _valid_sha256(expected_sha256):
        raise ValueError(
            "preregistration requires an independently supplied SHA-256 in lowercase hexadecimal"
        )
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise ValueError(f"invalid paper preregistration {source}") from error
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("paper preregistration does not match the independently supplied SHA-256")
    document = strict_json_loads(payload, location=f"paper preregistration {source}")
    if not isinstance(document, dict):
        raise ValueError("paper preregistration must be a JSON object")
    if set(document) != {
        "schema",
        "schema_version",
        "protocol_id",
        "trust_model",
        "commitment_scope",
        "audit_contract",
        "audit_source",
        "selection",
        "paper_checkpoints",
        "policy_freezes",
    }:
        raise ValueError("paper preregistration has an incomplete or unknown top-level field")

    audit_source, selection, checkpoints = _validated_preregistration_components(
        audit_contract=audit_contract,
        selection_document=selection_document,
        selection_file=selection_file,
        selection_sha256=selection_sha256,
        audit_source_root=audit_source_root,
    )
    expected_header = {
        "schema": PAPER_PREREGISTRATION_SCHEMA,
        "schema_version": PAPER_PREREGISTRATION_SCHEMA_VERSION,
        "protocol_id": audit_contract.document["protocol_id"],
        "trust_model": audit_contract.preregistration["trust_model"],
        "commitment_scope": audit_contract.preregistration["commitment_scope"],
        "audit_contract": audit_contract.public_binding(),
        "audit_source": audit_source,
        "selection": selection,
        "paper_checkpoints": checkpoints,
    }
    for key, expected in expected_header.items():
        require_exact_json(document.get(key), expected, location=f"preregistration.{key}")

    raw_freezes = document.get("policy_freezes")
    registered_seeds = [checkpoint["seed"] for checkpoint in checkpoints]
    if (
        not isinstance(raw_freezes, list)
        or len(raw_freezes) != audit_contract.preregistration["required_policy_freeze_count"]
        or len(raw_freezes) != len(registered_seeds)
    ):
        raise ValueError("preregistration must bind exactly four policy freezes")
    freeze_names: set[str] = set()
    for expected_seed, entry in zip(registered_seeds, raw_freezes, strict=True):
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"seed", "file", "sha256"}
            or entry.get("seed") != expected_seed
            or isinstance(entry.get("seed"), bool)
            or not isinstance(entry.get("file"), str)
            or not entry["file"]
            or Path(entry["file"]).name != entry["file"]
            or entry["file"] in freeze_names
            or not _valid_sha256(entry.get("sha256"))
        ):
            raise ValueError("preregistration has an invalid policy-freeze binding")
        freeze_names.add(str(entry["file"]))

    return PaperPreregistration(
        path=source.resolve(),
        sha256=actual_sha256,
        document=document,
    )


def preregistered_policy_freeze_binding(
    preregistration: PaperPreregistration,
    seed: int,
) -> Mapping[str, Any]:
    """Return the unique authenticated freeze binding for one registered seed."""

    matches = [entry for entry in preregistration.policy_freezes if entry.get("seed") == seed]
    if len(matches) != 1:
        raise ValueError("checkpoint seed has no unique preregistered policy freeze")
    return matches[0]


def verify_preregistered_policy_freezes(
    preregistration: PaperPreregistration,
    policy_freezes: Sequence[str | Path],
) -> dict[int, Mapping[str, Any]]:
    """Verify all four freeze files against the already authenticated manifest."""

    entries = preregistration.policy_freezes
    if len(policy_freezes) != len(entries):
        raise ValueError("every preregistered policy freeze must be supplied exactly once")
    paths_by_name: dict[str, Path] = {}
    for raw_path in policy_freezes:
        path = Path(raw_path)
        if path.name in paths_by_name:
            raise ValueError("policy-freeze file name is repeated")
        paths_by_name[path.name] = path
    if set(paths_by_name) != {str(entry["file"]) for entry in entries}:
        raise ValueError("supplied policy freezes differ from the preregistered file set")

    selection = preregistration.document["selection"]
    checkpoints = preregistration.document["paper_checkpoints"]
    checkpoint_by_seed = {checkpoint["seed"]: checkpoint for checkpoint in checkpoints}
    verified: dict[int, Mapping[str, Any]] = {}
    for entry in entries:
        seed = int(entry["seed"])
        path = paths_by_name[str(entry["file"])]
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ValueError(f"invalid preregistered policy freeze {path}") from error
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("policy-freeze bytes do not match the paper preregistration")
        freeze = strict_json_loads(payload, location=f"policy freeze {path}")
        checkpoint = checkpoint_by_seed[seed]
        reconstructed_selection = {
            "schema": selection["artifact_schema"],
            "schema_version": selection["artifact_schema_version"],
            "grid_id": selection["grid_id"],
            "grid_sha256": selection["grid_sha256"],
            "source_sha256": selection["source_sha256"],
            "selector_sha256": selection["selector_sha256"],
            "stage": selection["stage"],
            "registered_seeds": selection["registered_seeds"],
            "winner": {"params": selection["winner_params"]},
        }
        _validate_policy_freeze_document(
            freeze,
            location=f"policy freeze {path}",
            audit_contract_binding=preregistration.document["audit_contract"],
            selection_document=reconstructed_selection,
            selection_file=str(selection["file"]),
            selection_sha256=str(selection["sha256"]),
            checkpoint=checkpoint,
        )
        assert isinstance(freeze, Mapping)
        verified[seed] = freeze
    return verified


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mismatch_paths(actual: object, expected: object, path: str = "") -> list[str]:
    """Return type-sensitive semantic differences from the registered document."""

    location = path or "contract"
    if type(actual) is not type(expected):
        return [location]
    if isinstance(expected, Mapping):
        actual_mapping = actual
        assert isinstance(actual_mapping, Mapping)
        mismatches = [
            f"{path}.{key}" if path else str(key)
            for key in sorted(set(actual_mapping) ^ set(expected))
        ]
        for key in sorted(set(actual_mapping) & set(expected)):
            child = f"{path}.{key}" if path else str(key)
            mismatches.extend(_mismatch_paths(actual_mapping[key], expected[key], child))
        return mismatches
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        actual_sequence = actual
        assert isinstance(actual_sequence, Sequence)
        if len(actual_sequence) != len(expected):
            return [location]
        mismatches = []
        for index, (actual_item, expected_item) in enumerate(
            zip(actual_sequence, expected, strict=True)
        ):
            mismatches.extend(_mismatch_paths(actual_item, expected_item, f"{location}[{index}]"))
        return mismatches
    return [] if actual == expected else [location]


def require_exact_json(actual: object, expected: object, *, location: str) -> None:
    """Require a type-sensitive, field-exact JSON value."""

    mismatches = _mismatch_paths(actual, expected, location)
    if mismatches:
        raise ValueError(
            f"{location} differs from its frozen registration at: " + ", ".join(mismatches)
        )


def load_paper_audit_contract(
    path: str | Path,
    expected_sha256: str,
) -> PaperAuditContract:
    """Load the pre-frozen protocol and reject any byte or semantic drift."""

    if not _valid_sha256(expected_sha256):
        raise ValueError("paper audit contract SHA-256 must be lowercase hexadecimal")
    source = Path(path)
    checksum_path = source.with_suffix(".sha256")
    try:
        checksum_payload = checksum_path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"paper audit contract checksum sidecar is missing: {checksum_path}"
        ) from error
    try:
        checksum_text = checksum_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("paper audit contract checksum sidecar is not UTF-8") from error
    checksum_fields = checksum_text.split()
    if (
        len(checksum_fields) != 2
        or checksum_fields[0] != expected_sha256
        or checksum_fields[1] != source.name
        or not _valid_sha256(checksum_fields[0])
    ):
        raise ValueError(
            "paper audit contract checksum sidecar SHA-256 is not the registered binding"
        )
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise ValueError(f"invalid paper audit contract {source}") from error
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("paper audit contract SHA-256 does not match frozen bytes")
    document = strict_json_loads(payload, location=f"paper audit contract {source}")
    if not isinstance(document, dict):
        raise ValueError("paper audit contract must be a JSON object")
    expected_static = {
        key: value
        for key, value in EXPECTED_PAPER_AUDIT_CONTRACT.items()
        if key != "training_registration"
    }
    actual_static = {
        key: value for key, value in document.items() if key != "training_registration"
    }
    mismatches = _mismatch_paths(actual_static, expected_static)
    training_registration = document.get("training_registration")
    expected_training = EXPECTED_PAPER_AUDIT_CONTRACT["training_registration"]
    assert isinstance(expected_training, Mapping)
    if not isinstance(training_registration, Mapping) or set(training_registration) != set(
        expected_training
    ):
        mismatches.append("training_registration")
    else:
        for digest_field in ("grid_sha256", "source_sha256"):
            if not _valid_sha256(training_registration.get(digest_field)):
                mismatches.append(f"training_registration.{digest_field}")
        for mapping_field in ("selection", "stage", "data_provenance"):
            if not isinstance(training_registration.get(mapping_field), Mapping):
                mismatches.append(f"training_registration.{mapping_field}")
        mismatches.extend(
            _mismatch_paths(
                training_registration.get("validation_replay"),
                expected_training["validation_replay"],
                "training_registration.validation_replay",
            )
        )
    if mismatches:
        raise ValueError(
            "paper audit contract differs from registered schema v2 at: " + ", ".join(mismatches)
        )
    return PaperAuditContract(
        path=source.resolve(),
        sha256=actual_sha256,
        checksum_path=checksum_path.resolve(),
        checksum_sha256=hashlib.sha256(checksum_payload).hexdigest(),
        document=document,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical registration that binds the selected model family and all "
            "four label-free policy freezes before independent audit labeling."
        )
    )
    parser.add_argument("--audit-contract", required=True)
    parser.add_argument("--audit-contract-sha256", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--selection-root", required=True)
    parser.add_argument("--audit-source-root", default=str(Path(__file__).parents[1]))
    parser.add_argument("--policy-freezes", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        contract = load_paper_audit_contract(
            args.audit_contract,
            args.audit_contract_sha256,
        )
        from select_training_grid import revalidate_selection_artifact

        selection = revalidate_selection_artifact(
            args.selection,
            args.selection_sha256,
            root=args.selection_root,
            audit_contract=contract,
        )
        document, payload = build_paper_preregistration(
            audit_contract=contract,
            selection_document=selection,
            selection_file=Path(args.selection).name,
            selection_sha256=args.selection_sha256,
            audit_source_root=args.audit_source_root,
            policy_freezes=args.policy_freezes,
        )
        require_exact_json(
            document["audit_contract"],
            contract.public_binding(),
            location="paper preregistration audit contract",
        )
    except ValueError as error:
        parser.error(str(error))
    destination = Path(args.out)
    _atomic_write(destination, payload)
    print(
        json.dumps(
            {
                "file": destination.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "trust_model": PAPER_PREREGISTRATION_TRUST_MODEL,
                "instruction": (
                    "retain this SHA-256 in an independent append-only record before audit "
                    "label generation"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
