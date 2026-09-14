from __future__ import annotations

import json
import math
import random

import pytest

from lac_minorminer import SearchSession, validate_embedding
from lac_minorminer.components import WideCandidateProvider
from lac_minorminer.orchestrator import SearchOrchestrator
from lac_minorminer.quality_v2 import (
    QUALITY_V2_FEATURE_NAMES,
    LinearQualityV2Checkpoint,
    QualityV2AcceptancePolicy,
    QualityV2InferenceError,
    QualityV2Predictions,
    QualityV2Problem,
    QualityV2Scorer,
    load_quality_v2_checkpoint,
)


def _grid_edges(rows: int, columns: int) -> list[tuple[int, int]]:
    edges = []
    for row in range(rows):
        for column in range(columns):
            node = row * columns + column
            if row + 1 < rows:
                edges.append((node, node + columns))
            if column + 1 < columns:
                edges.append((node, node + 1))
    return edges


def _session() -> SearchSession:
    return SearchSession(
        [(0, 1), (0, 2), (0, 3)],
        _grid_edges(4, 4),
        random_seed=75,
        max_candidates=1,
    )


def _problem(session: SearchSession) -> QualityV2Problem:
    return QualityV2Problem.from_mappings(
        session,
        linear={0: -2.0, 1: -0.5, 2: 1.0, 3: 0.0},
        quadratic={(0, 1): -0.5, (0, 2): -0.5, (0, 3): 4.0},
    )


class FixedSelector:
    def select(self, snapshot, eligible, rng) -> int:
        assert 0 in eligible
        return 0


class CandidateIdPredictor:
    def __init__(self, logits: dict[int, float], concentrations: dict[int, float] | None = None):
        self._logits = logits
        self._concentrations = concentrations or {}
        self.last_input = None

    def predict(self, model_input):
        self.last_input = model_input
        return QualityV2Predictions(
            quality_logit=tuple(
                self._logits.get(candidate_id, -20.0) for candidate_id in model_input.candidate_ids
            ),
            quality_log_concentration=tuple(
                self._concentrations.get(candidate_id, 0.0)
                for candidate_id in model_input.candidate_ids
            ),
        )


class FailingPredictor:
    def predict(self, model_input):
        raise QualityV2InferenceError("accelerator unavailable")


class MalformedPredictor:
    def predict(self, model_input):
        return QualityV2Predictions(
            quality_logit=(0.0,),
            quality_log_concentration=(0.0,),
        )


def _orchestrator(session: SearchSession, scorer: QualityV2Scorer) -> SearchOrchestrator:
    return SearchOrchestrator(
        selector=FixedSelector(),
        candidate_provider=WideCandidateProvider(scoring_candidates=16),
        scorer=scorer,
        acceptance_policy=QualityV2AcceptancePolicy(scorer),
    )


def _linear_checkpoint() -> LinearQualityV2Checkpoint:
    feature_count = len(QUALITY_V2_FEATURE_NAMES)
    quality_weights = [0.0] * feature_count
    quality_weights[QUALITY_V2_FEATURE_NAMES.index("exact_total_qubits")] = 1.0
    return LinearQualityV2Checkpoint(
        feature_names=QUALITY_V2_FEATURE_NAMES,
        feature_mean=(0.0,) * feature_count,
        feature_scale=(1.0,) * feature_count,
        quality_weights=tuple(quality_weights),
        quality_bias=-6.5,
        concentration_weights=(0.0,) * feature_count,
        concentration_bias=0.0,
        metadata={"corpus_sha256": "development-fixture"},
    )


def test_quality_v2_selects_and_applies_quality_candidate_outside_default_prefix() -> None:
    session = _session()
    predictor = CandidateIdPredictor({0: -4.0, 1: 4.0})
    scorer = QualityV2Scorer(
        session=session,
        problem=_problem(session),
        predictor=predictor,
        total_qubit_budget=7,
        statistic="mean",
    )

    outcome = _orchestrator(session, scorer).transition(session, random.Random(75))

    assert outcome.candidate_index == 1
    assert outcome.candidate_chains[1] == (9, 12, 13, 14)
    assert outcome.after.chains[0] == [9, 12, 13, 14]
    assert sum(len(chain) for chain in outcome.after.chains) == 7
    assert outcome.after.valid
    report = validate_embedding(
        session.normalized_source,
        session.normalized_target,
        session.embedding(),
    )
    assert report.valid, report.errors
    assert predictor.last_input.candidate_ids[:2] == (0, 1)
    assert predictor.last_input.source_size == 4
    assert predictor.last_input.target_size == 16
    assert predictor.last_input.logical_fields == (-2.0, -0.5, 1.0, 0.0)
    assert predictor.last_input.logical_couplings == (
        (0, 1, -0.5),
        (0, 2, -0.5),
        (0, 3, 4.0),
    )
    assert not {
        "p_solve",
        "best_index",
        "stage",
        "label",
    } & set(predictor.last_input.__dataclass_fields__)
    assert not scorer.last_decision.used_fallback
    assert outcome.policy_mode == "learned"
    assert outcome.policy_reason is None


def test_quality_v2_masks_with_exact_total_qubit_budget_before_ranking() -> None:
    session = _session()
    scorer = QualityV2Scorer(
        session=session,
        problem=_problem(session),
        predictor=CandidateIdPredictor({0: -4.0, 1: 20.0}),
        total_qubit_budget=6,
    )

    outcome = _orchestrator(session, scorer).transition(session, random.Random(75))

    assert outcome.candidate_index == 0
    assert sum(len(chain) for chain in outcome.after.chains) == 6
    assessments = {item.candidate_id: item for item in scorer.last_decision.assessments}
    assert assessments[0].eligible
    assert assessments[0].exact_total_qubits == 6
    assert not assessments[1].eligible
    assert assessments[1].exact_total_qubits == 7


def test_quality_v2_masks_native_resource_attractive_but_invalid_overlap() -> None:
    session = SearchSession(
        [(0, 1), (0, 2), (0, 3)],
        _grid_edges(4, 4),
        random_seed=75,
        max_candidates=2,
    )
    before = session.snapshot()
    batch = session.materialize(0, [[8, 9, 10, 11, 15], [9, 13, 14]])
    assert batch.candidates[0].rank.used_target_nodes == 5
    scorer = QualityV2Scorer(
        session=session,
        problem=_problem(session),
        predictor=FailingPredictor(),
        total_qubit_budget=6,
    )

    scores = scorer.score(before, batch)
    choice = QualityV2AcceptancePolicy(scorer).choose(scores, batch)

    assessments = {item.candidate_id: item for item in scorer.last_decision.assessments}
    assert assessments[0].exact_total_qubits == 8
    assert not assessments[0].exact_valid
    assert not assessments[0].eligible
    assert assessments[1].exact_total_qubits == 6
    assert assessments[1].exact_valid
    assert choice == 1
    assert scorer.last_decision.used_fallback
    session.apply(batch, choice)
    assert session.snapshot().valid


def test_quality_v2_lcb_can_prefer_a_more_certain_lower_mean() -> None:
    predictor = CandidateIdPredictor(
        {0: math.log(0.8 / 0.2), 1: math.log(0.75 / 0.25)},
        {0: -100.0, 1: 100.0},
    )
    mean_session = _session()
    mean_scorer = QualityV2Scorer(
        session=mean_session,
        problem=_problem(mean_session),
        predictor=predictor,
        total_qubit_budget=7,
        statistic="mean",
    )
    lcb_session = _session()
    lcb_scorer = QualityV2Scorer(
        session=lcb_session,
        problem=_problem(lcb_session),
        predictor=predictor,
        total_qubit_budget=7,
        statistic="lcb",
        lcb_z=1.0,
    )

    mean = _orchestrator(mean_session, mean_scorer).transition(mean_session, random.Random(75))
    lcb = _orchestrator(lcb_session, lcb_scorer).transition(lcb_session, random.Random(75))

    assert mean.candidate_index == 0
    assert lcb.candidate_index == 1


def test_quality_v2_mean_ties_are_deterministic_in_candidate_order() -> None:
    session = _session()
    scorer = QualityV2Scorer(
        session=session,
        problem=_problem(session),
        predictor=CandidateIdPredictor({}),
        total_qubit_budget=7,
    )

    outcome = _orchestrator(session, scorer).transition(session, random.Random(75))

    assert outcome.candidate_index == 0


def test_quality_v2_falls_back_to_exact_feasible_native_order_on_model_failure() -> None:
    session = _session()
    scorer = QualityV2Scorer(
        session=session,
        problem=_problem(session),
        predictor=FailingPredictor(),
        total_qubit_budget=7,
    )

    outcome = _orchestrator(session, scorer).transition(session, random.Random(75))

    assert outcome.candidate_index == 0
    assert outcome.after.valid
    assert scorer.last_decision.used_fallback
    assert scorer.last_decision.fallback_reason == "inference_error:accelerator unavailable"
    assert outcome.policy_mode == "native_fallback"
    assert outcome.policy_reason == "inference_error:accelerator unavailable"


def test_quality_v2_returns_no_selection_when_no_exact_candidate_fits_budget() -> None:
    session = _session()
    before = session.snapshot()
    scorer = QualityV2Scorer(
        session=session,
        problem=_problem(session),
        predictor=CandidateIdPredictor({0: 20.0}),
        total_qubit_budget=5,
    )

    outcome = _orchestrator(session, scorer).transition(session, random.Random(75))

    assert outcome.candidate_index is None
    assert not outcome.accepted
    assert outcome.after.chains == before.chains
    assert not scorer.last_decision.used_fallback
    assert scorer.last_decision.fallback_reason == "no_exact_feasible_candidate"
    assert outcome.policy_mode == "no_selection"
    assert outcome.policy_reason == "no_exact_feasible_candidate"


def test_malformed_prediction_is_not_silently_converted_to_fallback() -> None:
    session = _session()
    scorer = QualityV2Scorer(
        session=session,
        problem=_problem(session),
        predictor=MalformedPredictor(),
        total_qubit_budget=7,
    )

    with pytest.raises(ValueError, match="quality_logit"):
        _orchestrator(session, scorer).transition(session, random.Random(75))

    batch = session.propose(0)
    session.discard(batch)


def test_quality_v2_policy_mode_is_persisted_in_search_diagnostics() -> None:
    session = _session()
    scorer = QualityV2Scorer(
        session=session,
        problem=_problem(session),
        predictor=CandidateIdPredictor({0: -4.0, 1: 4.0}),
        total_qubit_budget=7,
    )

    diagnostics = _orchestrator(session, scorer).run(
        session,
        random.Random(75),
        random_seed=75,
        tries=1,
        max_transitions=1,
        timeout=None,
    )

    assert diagnostics.trace[0].policy_mode == "learned"
    assert diagnostics.trace[0].policy_reason is None
    restored = type(diagnostics).from_dict(json.loads(json.dumps(diagnostics.to_dict())))
    assert restored.trace[0].policy_mode == "learned"
    assert restored == diagnostics


def test_quality_v2_scorer_cannot_be_used_with_default_greedy_acceptance() -> None:
    session = _session()
    scorer = QualityV2Scorer(
        session=session,
        problem=_problem(session),
        predictor=CandidateIdPredictor({0: 20.0}),
        total_qubit_budget=5,
    )

    with pytest.raises(ValueError, match="QualityV2AcceptancePolicy"):
        SearchOrchestrator(
            selector=FixedSelector(),
            candidate_provider=WideCandidateProvider(scoring_candidates=16),
            scorer=scorer,
        )


def test_quality_v2_rejects_a_same_size_batch_from_another_session() -> None:
    bound_session = _session()
    scorer = QualityV2Scorer(
        session=bound_session,
        problem=_problem(bound_session),
        predictor=CandidateIdPredictor({0: 20.0}),
        total_qubit_budget=16,
    )
    foreign_session = SearchSession(
        [(0, 1), (0, 2), (1, 3)],
        _grid_edges(4, 4),
        random_seed=75,
        max_candidates=1,
    )
    before = foreign_session.snapshot()

    with pytest.raises(ValueError, match="session"):
        _orchestrator(foreign_session, scorer).transition(foreign_session, random.Random(75))

    after = foreign_session.snapshot()
    assert after.chains == before.chains
    assert after.generation == before.generation


def test_quality_v2_problem_maps_noninteger_labels_to_compact_coefficients() -> None:
    session = SearchSession(
        [("left", "right")],
        [(0, 1), (1, 2)],
        random_seed=9,
        max_candidates=1,
    )

    problem = QualityV2Problem.from_mappings(
        session,
        linear={"left": -1.25, "right": 0.75},
        quadratic={("right", "left"): -2.5},
    )

    assert problem.logical_fields == (-1.25, 0.75)
    assert problem.logical_couplings == ((0, 1, -2.5),)


def test_zero_qubit_budget_is_a_valid_explicit_no_selection_policy() -> None:
    session = _session()
    scorer = QualityV2Scorer(
        session=session,
        problem=_problem(session),
        predictor=CandidateIdPredictor({0: 20.0}),
        total_qubit_budget=0,
    )

    outcome = _orchestrator(session, scorer).transition(session, random.Random(75))

    assert outcome.candidate_index is None
    assert not outcome.accepted


def test_linear_checkpoint_round_trips_without_embedbench_or_pickle(tmp_path) -> None:
    checkpoint = _linear_checkpoint()
    path = tmp_path / "quality_v2.json"

    checkpoint.save(path)
    loaded = load_quality_v2_checkpoint(path)
    session = _session()
    scorer = QualityV2Scorer(
        session=session,
        problem=_problem(session),
        predictor=loaded,
        total_qubit_budget=7,
    )
    outcome = _orchestrator(session, scorer).transition(session, random.Random(75))

    assert loaded.metadata == {"corpus_sha256": "development-fixture"}
    assert outcome.candidate_index == 1
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == "isingfold-quality-v2-linear"


def test_checkpoint_loader_rejects_unknown_formats(tmp_path) -> None:
    path = tmp_path / "foreign.json"
    _linear_checkpoint().save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["format"] = "foreign"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported"):
        load_quality_v2_checkpoint(path)


def test_checkpoint_loader_rejects_nonfinite_parameters(tmp_path) -> None:
    path = tmp_path / "nonfinite.json"
    _linear_checkpoint().save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["quality_bias"] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="quality_bias.*finite"):
        load_quality_v2_checkpoint(path)


def test_quality_problem_requires_complete_finite_coefficients() -> None:
    session = _session()

    with pytest.raises(ValueError, match="linear"):
        QualityV2Problem.from_mappings(
            session,
            linear={0: -2.0},
            quadratic={(0, 1): -0.5, (0, 2): -0.5, (0, 3): 4.0},
        )
    with pytest.raises(ValueError, match="quadratic"):
        QualityV2Problem.from_mappings(
            session,
            linear={0: -2.0, 1: -0.5, 2: 1.0, 3: 0.0},
            quadratic={(0, 1): float("nan"), (0, 2): -0.5, (0, 3): 4.0},
        )
