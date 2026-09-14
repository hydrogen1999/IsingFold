"""Post-selection parameter-capacity controls for the fixed IsingFold model grid.

The primary nine-cell representation screen and eighteen-cell RL-value stage answer which
registered system performs best.  They do not by themselves identify whether IF-Core wins
because of its typed ownership/conflict fusion or simply because it has more parameters.

This module leaves those primary cells untouched.  It authenticates a separate diagnostic
registry, counts parameters from instantiated PyTorch modules, and exposes a fixed-width
near-parameter-matched IF-Dual control.  The control has nine local blocks and no fusion
blocks: at width 128 its trainable count differs from IF-Core by less than 0.1 percent.  It
therefore narrows the capacity objection while explicitly retaining depth and compute as
reported confounders.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from isingfold.rl.data.import_embedbench import content_digest
from isingfold.rl.model import MODEL_FAMILIES, WIDTH, IFCore, build_model

CAPACITY_CONTROL_SCHEMA = "isingfold.parameter-capacity-control"
CAPACITY_CONTROL_VERSION = 1
CAPACITY_CONTROL_STAGE = "post-representation-selection-diagnostic"
CAPACITY_PARAMETER_METRIC = "unique-trainable-numel-requires-grad"
CAPACITY_SELECTION_EFFECT = "diagnostic-only-no-main-grid-reselection"
CAPACITY_CONTROL_PROTOCOL_ID = (
    "if-core-v2-hybrid-quality-signal-post-selection-capacity-control-v1"
)
CAPACITY_CONTROL_MAIN_GRID_SHA256 = (
    "d3a7cd99c0c96c1c2f7fdd974d0856aee94ccbbe7a01f30b15d4724632712d09"
)
CAPACITY_CONTROL_V1_RECORD_DIGEST = (
    "af29b4d5a628842b520e45bdd1993af66312c6276dd7cd0baaaee2fa21658307"
)
CORE_REFERENCE_VARIANT = "if-core-v1-reference"
DUAL_DEPTH9_VARIANT = "if-dual-depth9-v1"

_PERMITTED_CLAIM = (
    "diagnostic evidence about ownership/conflict fusion versus additional within-stream "
    "depth at fixed width and near-matched trainable parameter count"
)
_PROHIBITED_CLAIMS = (
    "changing the nine-cell representation selection",
    "changing the eighteen-cell RL-value selection",
    "claiming pure architectural causality because depth and inference compute remain different",
    "claiming IF-MLP versus IF-Core is capacity-controlled",
)
_REQUIRED_REPORTS = (
    "instantiated trainable and total parameter counts",
    "wall-clock inference latency",
    "peak accelerator memory",
    "all three paired training seeds",
    "aggregate development outcomes without seed selection",
)

_HEX = frozenset("0123456789abcdef")
_EXPECTED_REPRESENTATION_FAMILIES = ("if-mlp", "if-dual", "if-core")
_EXPECTED_RL_FAMILIES = ("selected-simpler", "if-core")
_EXPECTED_RL_METHODS = ("supervised-only", "ppo-warm-start", "ppo-from-scratch")
_EXECUTION_REQUIREMENTS = (
    "fresh_retraining_required_for_both_variants",
    "paired_seed_schedule",
    "same_training_hyperparameters_as_main_representation_screen",
    "same_training_data_order",
    "same_optimizer_and_stopping_rule",
    "same_action_support_and_deployment_rule",
)


@dataclass(frozen=True, slots=True)
class ParameterCensus:
    """Observed parameter capacity of one instantiated model."""

    model_id: str
    width: int
    local_blocks: int
    fusion_blocks: int
    trainable_parameters: int
    total_parameters: int
    frozen_parameters: int
    parameter_signature_sha256: str


@dataclass(frozen=True, slots=True)
class CapacityVariant:
    """One registered post-selection diagnostic architecture."""

    variant_id: str
    base_family: str
    role: str
    width: int
    local_blocks: int
    fusion_blocks: int
    trainable_parameters: int
    total_parameters: int
    parameter_signature_sha256: str


@dataclass(frozen=True, slots=True)
class CapacityTrainingProtocol:
    """Supervised protocol shared by both diagnostic architectures."""

    objective: str
    epochs: int
    minibatch_records: int
    learning_rate: float
    weight_decay: float
    minimum_resolved_rows: int
    minimum_resolved_lineages: int


@dataclass(frozen=True, slots=True)
class CapacityControlRegistry:
    """Authenticated parameter-control registry bound to the unchanged main grid."""

    path: Path
    file_sha256: str
    record_digest: str
    protocol_id: str
    main_grid_sha256: str
    _main_grid_cell_counts: tuple[tuple[str, int], ...]
    selection_effect: str
    main_census: tuple[ParameterCensus, ...]
    variants: tuple[CapacityVariant, ...]
    maximum_relative_parameter_gap: float
    training_seeds: tuple[int, ...]
    training_protocol: CapacityTrainingProtocol
    selected_simpler_must_equal: str
    permitted_claim: str
    prohibited_claims: tuple[str, ...]
    required_reports: tuple[str, ...]
    execution_requirements: tuple[str, ...]

    @property
    def main_grid_cell_counts(self) -> dict[str, int]:
        return dict(self._main_grid_cell_counts)

    @property
    def relative_parameter_gap(self) -> float:
        by_id = {variant.variant_id: variant for variant in self.variants}
        reference = by_id[CORE_REFERENCE_VARIANT].trainable_parameters
        control = by_id[DUAL_DEPTH9_VARIANT].trainable_parameters
        return abs(control - reference) / reference


@dataclass(frozen=True, slots=True)
class CapacityDiagnosticCell:
    """One paired seed/architecture cell outside the primary selection grid."""

    cell_id: str
    variant_id: str
    seed: int
    trainable_parameters: int
    total_parameters: int
    parameter_signature_sha256: str


@dataclass(frozen=True, slots=True)
class CapacityDiagnosticPlan:
    """Post-freeze six-cell plan; its outcomes cannot revise the main selection."""

    registry_record_digest: str
    main_grid_sha256: str
    representation_selection_sha256: str
    representation_selection_record_digest: str
    selected_simpler: str
    cells: tuple[CapacityDiagnosticCell, ...]
    training_protocol: CapacityTrainingProtocol
    selection_effect: str
    seed_selection_forbidden: bool
    permitted_claim: str
    prohibited_claims: tuple[str, ...]
    required_reports: tuple[str, ...]
    execution_requirements: tuple[str, ...]
    plan_digest: str


class IFDualDepth9CapacityControl(IFCore):
    """Width-128 IF-Dual with depth chosen to near-match IF-Core's parameter count."""

    diagnostic_variant_id = DUAL_DEPTH9_VARIANT

    def __init__(self) -> None:
        super().__init__(
            width=WIDTH,
            local_blocks=9,
            fusion_blocks=0,
            improvement_mode=True,
        )
        self.model_family = "IF-Dual-Depth9-Capacity-Control"


def _reject_duplicates(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"capacity-control JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"capacity-control JSON contains non-finite number {token}")


def _strict_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    _check_finite(value, label)
    return raw, value


def _check_finite(value: object, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _check_finite(item, label)
    elif isinstance(value, list):
        for item in value:
            _check_finite(item, label)


def _exact_keys(value: Mapping[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label} schema differs: missing={sorted(keys - set(value))}, "
            f"unknown={sorted(set(value) - keys)}"
        )


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be nonempty text")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return result


def _true(value: object, label: str) -> None:
    if value is not True:
        raise ValueError(f"{label} must be true")


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    result = tuple(_text(item, label) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _verify_record(record: Mapping[str, object], label: str) -> str:
    digest = _sha256(record.get("record_digest"), f"{label} record digest")
    unsigned = {key: value for key, value in record.items() if key != "record_digest"}
    try:
        observed = content_digest(unsigned)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains non-canonical data") from exc
    if not hmac.compare_digest(digest, observed):
        raise ValueError(f"{label} record digest mismatch")
    return digest


def _parameter_census(model_id: str, model: nn.Module) -> ParameterCensus:
    parameters = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in parameters)
    trainable = sum(parameter.numel() for _, parameter in parameters if parameter.requires_grad)
    signature = content_digest(
        {
            "domain": "isingfold-parameter-signature-v1",
            "parameters": [
                {
                    "dtype": str(parameter.dtype),
                    "name": name,
                    "numel": parameter.numel(),
                    "requires_grad": parameter.requires_grad,
                    "shape": list(parameter.shape),
                }
                for name, parameter in parameters
            ],
        }
    )
    local = getattr(model, "local_l", None)
    fusion = getattr(model, "fusion", None)
    if local is None or fusion is None:
        raise ValueError(f"model {model_id!r} does not expose the registered block census")
    return ParameterCensus(
        model_id=model_id,
        width=WIDTH,
        local_blocks=len(local),
        fusion_blocks=len(fusion),
        trainable_parameters=trainable,
        total_parameters=total,
        frozen_parameters=total - trainable,
        parameter_signature_sha256=signature,
    )


def parameter_census(*, improvement_mode: bool = True) -> tuple[ParameterCensus, ...]:
    """Count the three primary families from live instantiated modules.

    Improvement mode is the registered 9+18 grid mode.  Construction-mode counts remain
    available for disclosure, but are not accepted by the capacity-control registry.
    """

    if type(improvement_mode) is not bool:
        raise ValueError("improvement_mode must be Boolean")
    return tuple(
        _parameter_census(
            family,
            build_model(family, improvement_mode=improvement_mode),
        )
        for family in MODEL_FAMILIES
    )


def _instantiate_variant(variant_id: str) -> IFCore:
    if variant_id == CORE_REFERENCE_VARIANT:
        return build_model("if-core", improvement_mode=True)
    if variant_id == DUAL_DEPTH9_VARIANT:
        return IFDualDepth9CapacityControl()
    raise ValueError(f"unknown capacity diagnostic {variant_id!r}")


def _validate_main_grid(grid: Mapping[str, object]) -> dict[str, int]:
    grid_version = grid.get("schema_version")
    if (
        grid.get("schema") != "isingfold.staged-grid"
        or isinstance(grid_version, bool)
        or grid_version != 2
        or grid.get("name") != "if-core-v2-profile-i-hybrid-chimera-registered"
    ):
        raise ValueError("capacity control requires staged-grid schema v2")
    stages = grid.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("main grid has no stages")
    representation = stages.get("representation")
    rl_value = stages.get("rl_value")
    if not isinstance(representation, list) or not isinstance(rl_value, list):
        raise ValueError("main grid stages must be lists")
    counts = {"representation": len(representation), "rl_value": len(rl_value)}
    if counts != {"representation": 9, "rl_value": 18}:
        raise ValueError("capacity control requires the fixed 9+18 main grid")

    seeds = (1103, 2207, 3301)
    representation_census = {
        (row.get("model_family"), row.get("seed"))
        for row in representation
        if isinstance(row, Mapping)
    }
    expected_representation = {
        (family, seed) for family in _EXPECTED_REPRESENTATION_FAMILIES for seed in seeds
    }
    if representation_census != expected_representation:
        raise ValueError("main representation grid differs from its fixed family/seed census")
    rl_census = {
        (row.get("model_family"), row.get("method"), row.get("seed"))
        for row in rl_value
        if isinstance(row, Mapping)
    }
    expected_rl = {
        (family, method, seed)
        for family in _EXPECTED_RL_FAMILIES
        for method in _EXPECTED_RL_METHODS
        for seed in seeds
    }
    if rl_census != expected_rl:
        raise ValueError("main RL-value grid differs from its fixed family/method/seed census")
    cell_ids = [row.get("cell_id") for row in (*representation, *rl_value)]
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in cell_ids) or len(
        set(cell_ids)
    ) != len(cell_ids):
        raise ValueError("main grid cell IDs must be nonempty and globally unique")
    return counts


def _load_training_protocol(
    raw: object,
    *,
    main_grid: Mapping[str, object],
) -> CapacityTrainingProtocol:
    fields = {
        "epochs",
        "learning_rate",
        "minibatch_records",
        "minimum_resolved_lineages",
        "minimum_resolved_rows",
        "objective",
        "weight_decay",
    }
    if not isinstance(raw, Mapping):
        raise ValueError("capacity diagnostic training protocol must be an object")
    _exact_keys(raw, fields, "capacity diagnostic training protocol")
    protocol = CapacityTrainingProtocol(
        objective=_text(raw["objective"], "capacity diagnostic objective"),
        epochs=_integer(raw["epochs"], "capacity diagnostic epochs", minimum=1),
        minibatch_records=_integer(
            raw["minibatch_records"], "capacity diagnostic minibatch", minimum=1
        ),
        learning_rate=_finite_number(
            raw["learning_rate"], "capacity diagnostic learning rate", minimum=1e-15
        ),
        weight_decay=_finite_number(raw["weight_decay"], "capacity diagnostic weight decay"),
        minimum_resolved_rows=_integer(
            raw["minimum_resolved_rows"],
            "capacity diagnostic minimum resolved rows",
            minimum=1,
        ),
        minimum_resolved_lineages=_integer(
            raw["minimum_resolved_lineages"],
            "capacity diagnostic minimum resolved lineages",
            minimum=1,
        ),
    )
    if protocol.objective != (
        "all-action-qmu-plus-commit-delta-plus-resolved-rank-plus-state-value"
    ):
        raise ValueError("capacity diagnostic objective differs from the representation screen")

    stages = main_grid["stages"]
    if not isinstance(stages, Mapping):  # Defensive guard after main-grid validation.
        raise RuntimeError("validated main grid lost its stage mapping")
    representation = stages["representation"]
    if not isinstance(representation, list):
        raise RuntimeError("validated main grid lost its representation cells")
    for row in representation:
        if not isinstance(row, Mapping):
            raise ValueError("main representation cell must be an object")
        observed = (
            row.get("epochs"),
            row.get("minibatch"),
            row.get("learning_rate"),
            row.get("weight_decay"),
        )
        expected = (
            protocol.epochs,
            protocol.minibatch_records,
            protocol.learning_rate,
            protocol.weight_decay,
        )
        if observed != expected:
            raise ValueError(
                "capacity diagnostic hyperparameters differ from the main representation screen"
            )
    resolution = main_grid.get("quality_resolution")
    if not isinstance(resolution, Mapping):
        raise ValueError("main grid has no quality-resolution threshold")
    if (
        resolution.get("min_resolved_rows") != protocol.minimum_resolved_rows
        or resolution.get("min_resolved_lineages") != protocol.minimum_resolved_lineages
    ):
        raise ValueError("capacity diagnostic resolution thresholds differ from the main grid")
    return protocol


def _validate_registered_census(
    raw: object,
    observed: tuple[ParameterCensus, ...],
) -> None:
    if not isinstance(raw, list) or len(raw) != len(observed):
        raise ValueError("capacity registry has an incomplete main-model census")
    fields = {
        "expected_frozen_parameters",
        "expected_parameter_signature_sha256",
        "expected_total_parameters",
        "expected_trainable_parameters",
        "fusion_blocks",
        "local_blocks",
        "model_id",
        "width",
    }
    for expected, actual in zip(raw, observed, strict=True):
        if not isinstance(expected, Mapping):
            raise ValueError("capacity registry main-model census row must be an object")
        _exact_keys(expected, fields, "capacity registry main-model census row")
        registered = (
            _text(expected["model_id"], "registered main model ID"),
            _integer(expected["width"], "registered width", minimum=1),
            _integer(expected["local_blocks"], "registered local-block count"),
            _integer(expected["fusion_blocks"], "registered fusion-block count"),
            _integer(
                expected["expected_trainable_parameters"],
                "registered trainable-parameter count",
                minimum=1,
            ),
            _integer(
                expected["expected_total_parameters"],
                "registered total-parameter count",
                minimum=1,
            ),
            _integer(
                expected["expected_frozen_parameters"],
                "registered frozen-parameter count",
            ),
            _sha256(
                expected["expected_parameter_signature_sha256"],
                "registered parameter signature",
            ),
        )
        observed_row = (
            actual.model_id,
            actual.width,
            actual.local_blocks,
            actual.fusion_blocks,
            actual.trainable_parameters,
            actual.total_parameters,
            actual.frozen_parameters,
            actual.parameter_signature_sha256,
        )
        if registered != observed_row:
            raise ValueError(f"instantiated parameter census drifted for {actual.model_id}")


def _load_variants(raw: object) -> tuple[CapacityVariant, ...]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError("capacity registry must contain exactly two diagnostic variants")
    fields = {
        "base_family",
        "expected_total_parameters",
        "expected_trainable_parameters",
        "expected_parameter_signature_sha256",
        "fusion_blocks",
        "local_blocks",
        "role",
        "variant_id",
        "width",
    }
    variants: list[CapacityVariant] = []
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError("capacity diagnostic variant must be an object")
        _exact_keys(row, fields, "capacity diagnostic variant")
        variant_id = _text(row["variant_id"], "capacity diagnostic variant ID")
        model = _instantiate_variant(variant_id)
        observed = _parameter_census(variant_id, model)
        expected = (
            _integer(row["width"], "capacity diagnostic width", minimum=1),
            _integer(row["local_blocks"], "capacity diagnostic local-block count"),
            _integer(row["fusion_blocks"], "capacity diagnostic fusion-block count"),
            _integer(
                row["expected_trainable_parameters"],
                "capacity diagnostic trainable-parameter count",
                minimum=1,
            ),
            _integer(
                row["expected_total_parameters"],
                "capacity diagnostic total-parameter count",
                minimum=1,
            ),
            _sha256(
                row["expected_parameter_signature_sha256"],
                "capacity diagnostic parameter signature",
            ),
        )
        actual = (
            observed.width,
            observed.local_blocks,
            observed.fusion_blocks,
            observed.trainable_parameters,
            observed.total_parameters,
            observed.parameter_signature_sha256,
        )
        if expected != actual:
            raise ValueError(f"instantiated capacity diagnostic drifted for {variant_id}")
        variants.append(
            CapacityVariant(
                variant_id=variant_id,
                base_family=_text(row["base_family"], "capacity diagnostic base family"),
                role=_text(row["role"], "capacity diagnostic role"),
                width=observed.width,
                local_blocks=observed.local_blocks,
                fusion_blocks=observed.fusion_blocks,
                trainable_parameters=observed.trainable_parameters,
                total_parameters=observed.total_parameters,
                parameter_signature_sha256=observed.parameter_signature_sha256,
            )
        )
    if tuple(variant.variant_id for variant in variants) != (
        CORE_REFERENCE_VARIANT,
        DUAL_DEPTH9_VARIANT,
    ):
        raise ValueError("capacity diagnostic variant order or identity differs")
    if variants[0].base_family != "if-core" or variants[1].base_family != "if-dual":
        raise ValueError("capacity diagnostic base-family identities differ")
    if (
        variants[0].role != "retrained-reference"
        or variants[1].role != "fixed-width-near-parameter-matched-control"
    ):
        raise ValueError("capacity diagnostic roles differ from v1")
    return tuple(variants)


def load_capacity_control_registry(
    path: str | Path,
    *,
    main_grid_path: str | Path,
) -> CapacityControlRegistry:
    """Authenticate the diagnostic registry and verify it against live model modules."""

    registry_path = Path(path)
    raw, record = _strict_json(registry_path, "capacity-control registry")
    fields = {
        "activation",
        "claim_scope",
        "execution",
        "main_census",
        "main_grid",
        "matching",
        "mode",
        "parameter_metric",
        "protocol_id",
        "record_digest",
        "schema",
        "schema_version",
        "stage",
        "training_protocol",
        "training_seeds",
        "variants",
    }
    _exact_keys(record, fields, "capacity-control registry")
    if (
        record["schema"] != CAPACITY_CONTROL_SCHEMA
        or isinstance(record["schema_version"], bool)
        or record["schema_version"] != CAPACITY_CONTROL_VERSION
        or record["stage"] != CAPACITY_CONTROL_STAGE
        or record["mode"] != "improvement"
        or record["parameter_metric"] != CAPACITY_PARAMETER_METRIC
    ):
        raise ValueError("capacity-control registry has an incompatible identity")
    digest = _verify_record(record, "capacity-control registry")
    if not hmac.compare_digest(digest, CAPACITY_CONTROL_V1_RECORD_DIGEST):
        raise ValueError("capacity-control registry differs from the pinned v1 record digest")

    grid_identity = record["main_grid"]
    if not isinstance(grid_identity, Mapping):
        raise ValueError("capacity-control main-grid identity must be an object")
    _exact_keys(
        grid_identity,
        {"cell_counts", "selection_effect", "sha256"},
        "capacity-control main-grid identity",
    )
    registered_grid_sha256 = _sha256(
        grid_identity["sha256"], "capacity-control pinned main-grid digest"
    )
    if registered_grid_sha256 != CAPACITY_CONTROL_MAIN_GRID_SHA256:
        raise ValueError("capacity registry does not name the fixed v1 main-grid digest")
    grid_path = Path(main_grid_path)
    grid_raw, grid = _strict_json(grid_path, "main grid")
    observed_grid_sha256 = hashlib.sha256(grid_raw).hexdigest()
    if not hmac.compare_digest(registered_grid_sha256, observed_grid_sha256):
        raise ValueError("main grid differs from the pinned main-grid digest")
    counts = _validate_main_grid(grid)
    registered_counts = grid_identity["cell_counts"]
    if not isinstance(registered_counts, Mapping):
        raise ValueError("capacity-control main-grid cell counts must be an object")
    _exact_keys(registered_counts, {"representation", "rl_value"}, "main-grid counts")
    parsed_counts = {
        stage: _integer(registered_counts[stage], f"registered {stage} cell count", minimum=1)
        for stage in ("representation", "rl_value")
    }
    if parsed_counts != counts:
        raise ValueError("capacity registry main-grid cell counts differ")
    selection_effect = _text(grid_identity["selection_effect"], "selection effect")
    if selection_effect != CAPACITY_SELECTION_EFFECT:
        raise ValueError("capacity diagnostic is not isolated from main-grid selection")

    observed_census = parameter_census(improvement_mode=True)
    _validate_registered_census(record["main_census"], observed_census)
    variants = _load_variants(record["variants"])
    training_protocol = _load_training_protocol(record["training_protocol"], main_grid=grid)

    matching = record["matching"]
    if not isinstance(matching, Mapping):
        raise ValueError("capacity matching contract must be an object")
    _exact_keys(
        matching,
        {
            "control_variant_id",
            "denominator",
            "maximum_relative_parameter_gap",
            "reference_variant_id",
        },
        "capacity matching contract",
    )
    if (
        matching["reference_variant_id"] != CORE_REFERENCE_VARIANT
        or matching["control_variant_id"] != DUAL_DEPTH9_VARIANT
        or matching["denominator"] != "reference-trainable-parameters"
    ):
        raise ValueError("capacity matching identities differ from the registered pair")
    maximum_gap = matching["maximum_relative_parameter_gap"]
    if isinstance(maximum_gap, bool) or not isinstance(maximum_gap, (int, float)):
        raise ValueError("maximum relative parameter gap must be numeric")
    maximum_gap = float(maximum_gap)
    if not 0.0 < maximum_gap <= 0.01:
        raise ValueError("maximum relative parameter gap must be in (0, 0.01]")
    observed_gap = (
        abs(variants[1].trainable_parameters - variants[0].trainable_parameters)
        / variants[0].trainable_parameters
    )
    if observed_gap > maximum_gap:
        raise ValueError("registered capacity diagnostic exceeds its parameter-gap tolerance")

    seeds_raw = record["training_seeds"]
    if not isinstance(seeds_raw, list):
        raise ValueError("capacity diagnostic training seeds must be a list")
    seeds = tuple(_integer(seed, "capacity diagnostic training seed") for seed in seeds_raw)
    if seeds != (1103, 2207, 3301):
        raise ValueError("capacity diagnostic must use all three paired main-grid seeds")

    activation = record["activation"]
    if not isinstance(activation, Mapping):
        raise ValueError("capacity diagnostic activation must be an object")
    _exact_keys(
        activation,
        {
            "representation_selection_record_digest_required",
            "representation_selection_sha256_required",
            "selected_simpler_must_equal",
        },
        "capacity diagnostic activation",
    )
    selected_required = _text(
        activation["selected_simpler_must_equal"], "capacity diagnostic selected family"
    )
    if selected_required != "if-dual":
        raise ValueError("capacity diagnostic must be scoped to a selected IF-Dual")
    _true(
        activation["representation_selection_sha256_required"],
        "representation selection SHA requirement",
    )
    _true(
        activation["representation_selection_record_digest_required"],
        "representation selection record-digest requirement",
    )

    execution = record["execution"]
    execution_fields = set(_EXECUTION_REQUIREMENTS)
    if not isinstance(execution, Mapping):
        raise ValueError("capacity diagnostic execution contract must be an object")
    _exact_keys(execution, execution_fields, "capacity diagnostic execution contract")
    for name in execution_fields:
        _true(execution[name], f"capacity diagnostic execution field {name}")

    claim_scope = record["claim_scope"]
    if not isinstance(claim_scope, Mapping):
        raise ValueError("capacity diagnostic claim scope must be an object")
    _exact_keys(
        claim_scope,
        {"permitted", "prohibited", "required_reports"},
        "capacity diagnostic claim scope",
    )
    prohibited = _text_tuple(claim_scope["prohibited"], "prohibited capacity claims")
    required_reports = _text_tuple(
        claim_scope["required_reports"], "required capacity diagnostic reports"
    )

    protocol_id = _text(record["protocol_id"], "capacity-control protocol ID")
    if protocol_id != CAPACITY_CONTROL_PROTOCOL_ID:
        raise ValueError("capacity-control protocol ID differs from v1")
    permitted_claim = _text(claim_scope["permitted"], "permitted capacity claim")
    if (
        permitted_claim != _PERMITTED_CLAIM
        or prohibited != _PROHIBITED_CLAIMS
        or required_reports != _REQUIRED_REPORTS
    ):
        raise ValueError("capacity diagnostic claim scope differs from v1")

    return CapacityControlRegistry(
        path=registry_path.resolve(),
        file_sha256=hashlib.sha256(raw).hexdigest(),
        record_digest=digest,
        protocol_id=protocol_id,
        main_grid_sha256=registered_grid_sha256,
        _main_grid_cell_counts=tuple(counts.items()),
        selection_effect=selection_effect,
        main_census=observed_census,
        variants=variants,
        maximum_relative_parameter_gap=maximum_gap,
        training_seeds=seeds,
        training_protocol=training_protocol,
        selected_simpler_must_equal=selected_required,
        permitted_claim=permitted_claim,
        prohibited_claims=prohibited,
        required_reports=required_reports,
        execution_requirements=_EXECUTION_REQUIREMENTS,
    )


def build_capacity_diagnostic_model(
    registry: CapacityControlRegistry,
    variant_id: str,
) -> IFCore:
    """Instantiate exactly one registry-verified diagnostic architecture."""

    if not isinstance(registry, CapacityControlRegistry):
        raise TypeError("registry must be an authenticated CapacityControlRegistry")
    registered = {variant.variant_id: variant for variant in registry.variants}
    if variant_id not in registered:
        raise ValueError(f"unknown capacity diagnostic {variant_id!r}")
    model = _instantiate_variant(variant_id)
    observed = _parameter_census(variant_id, model)
    expected = registered[variant_id]
    if (
        observed.width,
        observed.local_blocks,
        observed.fusion_blocks,
        observed.trainable_parameters,
        observed.total_parameters,
        observed.parameter_signature_sha256,
    ) != (
        expected.width,
        expected.local_blocks,
        expected.fusion_blocks,
        expected.trainable_parameters,
        expected.total_parameters,
        expected.parameter_signature_sha256,
    ):
        raise ValueError("capacity diagnostic model differs from the authenticated registry")
    return model


def build_capacity_diagnostic_plan(
    registry: CapacityControlRegistry,
    *,
    selected_simpler: str,
    representation_selection_sha256: str,
    representation_selection_record_digest: str,
) -> CapacityDiagnosticPlan:
    """Bind the six diagnostic cells to an already authenticated representation freeze."""

    if not isinstance(registry, CapacityControlRegistry):
        raise TypeError("registry must be an authenticated CapacityControlRegistry")
    if selected_simpler != registry.selected_simpler_must_equal:
        raise ValueError("capacity diagnostic is only registered when IF-Dual was selected")
    selection_sha256 = _sha256(
        representation_selection_sha256,
        "representation-selection SHA-256",
    )
    selection_record_digest = _sha256(
        representation_selection_record_digest,
        "representation-selection record digest",
    )
    cells = tuple(
        CapacityDiagnosticCell(
            cell_id=f"capacity-{variant.variant_id}-s{seed}",
            variant_id=variant.variant_id,
            seed=seed,
            trainable_parameters=variant.trainable_parameters,
            total_parameters=variant.total_parameters,
            parameter_signature_sha256=variant.parameter_signature_sha256,
        )
        for variant in registry.variants
        for seed in registry.training_seeds
    )
    payload = {
        "domain": "isingfold-capacity-diagnostic-plan-v1",
        "cells": [
            {
                "cell_id": cell.cell_id,
                "parameter_signature_sha256": cell.parameter_signature_sha256,
                "seed": cell.seed,
                "total_parameters": cell.total_parameters,
                "trainable_parameters": cell.trainable_parameters,
                "variant_id": cell.variant_id,
            }
            for cell in cells
        ],
        "execution_requirements": list(registry.execution_requirements),
        "main_grid_sha256": registry.main_grid_sha256,
        "permitted_claim": registry.permitted_claim,
        "prohibited_claims": list(registry.prohibited_claims),
        "registry_record_digest": registry.record_digest,
        "representation_selection_record_digest": selection_record_digest,
        "representation_selection_sha256": selection_sha256,
        "seed_selection_forbidden": True,
        "selected_simpler": selected_simpler,
        "selection_effect": registry.selection_effect,
        "required_reports": list(registry.required_reports),
        "training_protocol": {
            "epochs": registry.training_protocol.epochs,
            "learning_rate": registry.training_protocol.learning_rate,
            "minibatch_records": registry.training_protocol.minibatch_records,
            "minimum_resolved_lineages": (registry.training_protocol.minimum_resolved_lineages),
            "minimum_resolved_rows": registry.training_protocol.minimum_resolved_rows,
            "objective": registry.training_protocol.objective,
            "weight_decay": registry.training_protocol.weight_decay,
        },
    }
    return CapacityDiagnosticPlan(
        registry_record_digest=registry.record_digest,
        main_grid_sha256=registry.main_grid_sha256,
        representation_selection_sha256=selection_sha256,
        representation_selection_record_digest=selection_record_digest,
        selected_simpler=selected_simpler,
        cells=cells,
        training_protocol=registry.training_protocol,
        selection_effect=registry.selection_effect,
        seed_selection_forbidden=True,
        permitted_claim=registry.permitted_claim,
        prohibited_claims=registry.prohibited_claims,
        required_reports=registry.required_reports,
        execution_requirements=registry.execution_requirements,
        plan_digest=content_digest(payload),
    )


__all__ = [
    "CAPACITY_CONTROL_MAIN_GRID_SHA256",
    "CAPACITY_CONTROL_PROTOCOL_ID",
    "CAPACITY_CONTROL_SCHEMA",
    "CAPACITY_CONTROL_STAGE",
    "CAPACITY_CONTROL_V1_RECORD_DIGEST",
    "CAPACITY_CONTROL_VERSION",
    "CAPACITY_PARAMETER_METRIC",
    "CAPACITY_SELECTION_EFFECT",
    "CORE_REFERENCE_VARIANT",
    "DUAL_DEPTH9_VARIANT",
    "CapacityControlRegistry",
    "CapacityDiagnosticCell",
    "CapacityDiagnosticPlan",
    "CapacityTrainingProtocol",
    "CapacityVariant",
    "IFDualDepth9CapacityControl",
    "ParameterCensus",
    "build_capacity_diagnostic_model",
    "build_capacity_diagnostic_plan",
    "load_capacity_control_registry",
    "parameter_census",
]
