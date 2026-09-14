"""Portable artifact helpers shared by the training entry points."""

from __future__ import annotations

import platform
import sys
from importlib import metadata
from pathlib import Path

_CORE_DISTRIBUTIONS = (
    "embedbench",
    "numpy",
    "scipy",
    "networkx",
    "dimod",
    "dwave-samplers",
    "dwave-networkx",
    "minorminer",
)


def _distribution_versions() -> dict[str, str | None]:
    versions = {}
    for distribution in _CORE_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def runtime_provenance(device) -> dict[str, object]:
    """Describe the software stack and selected accelerator using JSON-safe values."""
    import torch

    selected = torch.device(device)
    cuda_available = bool(torch.cuda.is_available())
    cudnn_version = torch.backends.cudnn.version()
    gpu = None
    device_index = selected.index
    if selected.type == "cuda":
        if not cuda_available:
            raise ValueError("cannot record CUDA provenance because CUDA is unavailable")
        device_index = selected.index
        if device_index is None:
            device_index = int(torch.cuda.current_device())
        capability = torch.cuda.get_device_capability(device_index)
        properties = torch.cuda.get_device_properties(device_index)
        gpu = {
            "name": str(torch.cuda.get_device_name(device_index)),
            "compute_capability": [int(capability[0]), int(capability[1])],
            "total_memory_bytes": int(properties.total_memory),
        }

    return {
        "schema": "embedbench.runtime-environment",
        "schema_version": 1,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "torch": {
            "version": str(torch.__version__),
            "cuda_available": cuda_available,
            "cuda_runtime_version": (
                str(torch.version.cuda) if torch.version.cuda is not None else None
            ),
            "cudnn_version": int(cudnn_version) if cudnn_version is not None else None,
        },
        "device": {
            "selected": str(selected),
            "type": selected.type,
            "index": device_index,
            "gpu": gpu,
        },
        "packages": _distribution_versions(),
    }


def cpu_state_dict(model) -> dict[str, object]:
    """Clone model state onto CPU so checkpoints do not depend on the training device."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def checkpoint_path(prefix: str | Path, architecture: str, seed: int) -> Path:
    """Resolve a per-run checkpoint path and create its parent directory."""
    path = Path(f"{prefix}_{architecture}_s{seed}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
