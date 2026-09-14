from __future__ import annotations

import json

from lac_minorminer import SearchDiagnostics, find_embedding


def test_diagnostics_round_trip_through_json() -> None:
    source = [(0, 1), (0, 2), (1, 2)]
    target = [(0, 1), (1, 2), (2, 3), (3, 0)]
    _, diagnostics = find_embedding(
        source, target, random_seed=7, max_transitions=100, return_diagnostics=True
    )

    payload = json.loads(json.dumps(diagnostics.to_dict()))
    restored = SearchDiagnostics.from_dict(payload)

    assert restored == diagnostics
    assert restored.structural_dict() == diagnostics.structural_dict()
