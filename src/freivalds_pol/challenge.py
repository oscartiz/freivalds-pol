"""Random challenge sampling.

Given a committed transcript, a challenger selects a small random subset of the claimed
matmuls to verify with Freivalds. Sampling is seeded so it can be made non-interactive
later (Fiat-Shamir over the commitment, or an on-chain VRF). Soundness comes from the
cheater not knowing in advance which operations will be checked: corrupting a fraction f of
the matmuls escapes a k-challenge audit with probability roughly (1 - f)^k.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class Challenge:
    matmul_index: int
    freivalds_rounds: int


def fiat_shamir_rng(*parts: bytes) -> np.random.Generator:
    """Deterministic RNG seeded by SHA-256 of the given byte strings (Fiat-Shamir).

    Used to derive the probe *after* the node publishes its commitment, so the node cannot
    target it. Include a fresh public beacon (e.g. a later block hash) as one of the parts
    to also prevent the node from grinding its commitment to a favourable probe.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return np.random.default_rng(int.from_bytes(h.digest(), "big"))


def fiat_shamir_probes(commitment: bytes, n: int, rounds: int, *,
                       beacon: bytes = b"") -> np.ndarray:
    """Unpredictable Rademacher probe matrix (n, rounds) bound to a commitment + beacon."""
    rng = fiat_shamir_rng(b"freivalds-probe", commitment, beacon)
    return rng.choice(np.array([-1.0, 1.0]), size=(n, rounds))


def sample_challenges(num_matmuls: int, k: int, *, rng=None,
                      freivalds_rounds: int = 4) -> list[Challenge]:
    rng = np.random.default_rng() if rng is None else rng
    k = min(k, num_matmuls)
    idx = rng.choice(num_matmuls, size=k, replace=False)
    return [Challenge(int(i), freivalds_rounds) for i in idx]
