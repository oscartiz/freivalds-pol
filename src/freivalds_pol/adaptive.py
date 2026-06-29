"""Adaptive adversaries against Freivalds verification.

The whole security of the spot-check turns on *when the probe is fixed relative to the cheat*:

  * **Predictable probe** (reused, leaked, or otherwise known before the node commits) -> the
    adversary places the cheat in the probe's nullspace and evades with certainty at
    unbounded impact (`nullspace_cheat`). Catastrophic; motivates commit-then-sample.

  * **Unpredictable probe** (Fiat-Shamir / VRF over the commitment + a fresh public beacon)
    -> the cheat is fixed before the probe exists, so anti-concentration applies: a probe
    ``r`` makes ``Delta r`` a random +/- sum with standard deviation ~ ||row of Delta||_2,
    which exceeds the threshold unless the cheat's impact is tiny. Detection then rises
    geometrically with the number of probe rounds (`fixed_cheat`).
"""

from __future__ import annotations

import numpy as np


def _scale_to_impact(C, Delta, rho):
    nC = float(np.linalg.norm(C))
    nD = float(np.linalg.norm(Delta))
    if nD == 0.0:
        return Delta
    return Delta * (rho * nC / nD)


def nullspace_cheat(C, probes, rho, *, rng=None, target=None):
    """High-impact cheat orthogonal to KNOWN probes: returns C + Delta with Delta @ probes = 0.

    Requires the adversary to know ``probes`` (n, k) in advance -- the predictable-probe
    regime. Each row of Delta is projected onto the orthogonal complement of the probe span,
    so every probe sees zero contribution from the cheat regardless of its impact ``rho``.
    """
    rng = np.random.default_rng() if rng is None else rng
    m, n = np.asarray(C).shape
    R = np.asarray(probes, dtype=np.float64)              # (n, k)
    if target is None:
        D = rng.normal(size=(m, n))
    else:
        D = np.broadcast_to(target, (m, n)).astype(float).copy()
    coef, *_ = np.linalg.lstsq(R, D.T, rcond=None)        # R @ coef = D^T   -> (k, m)
    D = D - (R @ coef).T                                  # remove component in span(R)
    return np.asarray(C, dtype=np.float64) + _scale_to_impact(C, D, rho)


def fixed_cheat(C, rho, *, rng=None, rank=None):
    """A cheat committed BEFORE the probe is known (unpredictable-probe regime).

    Optionally low-rank (``rank``) to test whether concentrating the impact helps it evade
    random probes -- it does not, because the probe direction is unknown.
    """
    rng = np.random.default_rng() if rng is None else rng
    m, n = np.asarray(C).shape
    if rank is None:
        D = rng.normal(size=(m, n))
    else:
        D = rng.normal(size=(m, rank)) @ rng.normal(size=(rank, n))
    return np.asarray(C, dtype=np.float64) + _scale_to_impact(C, D, rho)
