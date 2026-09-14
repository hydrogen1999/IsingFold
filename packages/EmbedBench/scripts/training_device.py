"""Device selection shared by the standalone training entry points."""

from __future__ import annotations


def resolve_device(requested: str):
    """Resolve ``auto`` deterministically and fail early for unavailable CUDA.

    Slurm normally exposes one assigned accelerator through ``CUDA_VISIBLE_DEVICES``;
    selecting plain ``cuda`` therefore respects the scheduler allocation without
    embedding cluster-specific device indices in experiment commands.
    """
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def seed_device(seed: int, device) -> None:
    """Seed the selected accelerator in addition to PyTorch's CPU generator."""
    import torch

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
