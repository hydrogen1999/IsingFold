"""Modular minor-embedding operations for IsingFold Track B."""

from ._core import WorkBudgetExceeded, backend_info
from ._version import __version__
from .api import find_embedding
from .components import WideCandidateProvider
from .diagnostics import (
    IncumbentRecord,
    RepairRecord,
    SearchDiagnostics,
    SearchWorkCounters,
    TerminationReason,
    TransitionRecord,
    WORK_COUNTER_SCHEMA,
    WORK_COUNTER_VERSION,
)
from .orchestrator import SearchOrchestrator
from .quality_v2 import (
    QUALITY_V2_FEATURE_NAMES,
    LinearQualityV2Checkpoint,
    QualityV2AcceptancePolicy,
    QualityV2BatchInput,
    QualityV2Decision,
    QualityV2InferenceError,
    QualityV2Predictions,
    QualityV2Predictor,
    QualityV2Problem,
    QualityV2Scorer,
    load_quality_v2_checkpoint,
)
from .session import SearchSession
from .validation import ValidationReport, validate_embedding

__all__ = [
    "QUALITY_V2_FEATURE_NAMES",
    "SearchSession",
    "SearchDiagnostics",
    "SearchWorkCounters",
    "SearchOrchestrator",
    "IncumbentRecord",
    "LinearQualityV2Checkpoint",
    "QualityV2AcceptancePolicy",
    "QualityV2BatchInput",
    "QualityV2Decision",
    "QualityV2InferenceError",
    "QualityV2Predictions",
    "QualityV2Predictor",
    "QualityV2Problem",
    "QualityV2Scorer",
    "RepairRecord",
    "TerminationReason",
    "TransitionRecord",
    "ValidationReport",
    "WideCandidateProvider",
    "WORK_COUNTER_SCHEMA",
    "WORK_COUNTER_VERSION",
    "WorkBudgetExceeded",
    "__version__",
    "backend_info",
    "find_embedding",
    "load_quality_v2_checkpoint",
    "validate_embedding",
]
