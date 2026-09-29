"""Random seeding helpers.

``set_seed`` seeds Python's ``random``, NumPy and both torch CPU/GPU RNGs so
that a fixed seed reproduces a full run. ``task_seed`` derives per-stage RNG
streams from a sample seed.
"""
from __future__ import annotations

import random
import zlib
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int]) -> None:
    """Seed random/numpy/torch (+CUDA) for reproducibility. No-op if ``seed`` is None."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def resolve_seed(sample_seed: Optional[int], global_seed: Optional[int]) -> int:
    """Per-sample seed: sample seed > global seed > random. Never returns None."""
    if sample_seed is not None:
        return sample_seed
    if global_seed is not None:
        return global_seed
    return random.randint(0, 2**31 - 1)


def task_seed(sample_seed: int, *parts) -> int:
    """Deterministic per-task seed within ``torch.manual_seed``'s valid range.

    Both the serial and parallel paths derive every RNG-consuming task
    (geometric exploration, each sampling path) from this, so ``parallel=true``
    reproduces ``parallel=false`` exactly. NOTE: Python's builtin ``hash()`` is
    per-process randomized (PYTHONHASHSEED), so it can never be used here -- a
    crc32 of a stable string encoding is process-independent.
    """
    key = f"{sample_seed}|{'|'.join(map(str, parts))}".encode()
    return zlib.crc32(key)
