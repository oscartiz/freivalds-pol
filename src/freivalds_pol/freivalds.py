"""Probabilistic verification of matrix products (Freivalds' algorithm).

Given a claimed product ``C = A @ B``, Freivalds' algorithm checks ``A @ (B @ r) == C @ r``
for a random vector ``r``. A single round catches a wrong product with probability >= 1/2;
with several rounds the escape probability falls geometrically. Cost is O(n^2) per round
versus O(n^3) to recompute the product -- this is the whole reason the scheme can verify a
training step for far less than Psyche's current recompute-and-compare.

Floating-point note: on heterogeneous hardware an *honest* recomputation will not match
bit-for-bit, so we compare with a tolerance rather than for exact equality. Choosing that
tolerance so it cleanly separates honest numerical drift from cheating is the core open
problem (see ``docs/DESIGN.md`` -- "the FP crux"). ``freivalds_residual`` exposes the raw
residual so that study can be run directly.
"""

from __future__ import annotations

import numpy as np


def _random_vectors(n: int, rounds: int, rng) -> np.ndarray:
    # Rademacher (+/-1) vectors keep the residual scale-free and avoid overflow.
    return rng.choice(np.array([-1.0, 1.0]), size=(n, rounds))


def freivalds_residual_with(A, B, C, R) -> np.ndarray:
    """Per-probe max-abs residual ``||A(BR) - CR||_inf`` for an explicit probe matrix R.

    ``R`` is (n, rounds). Letting the caller supply the probe is what enables the adaptive
    analysis: the protocol derives ``R`` from the commitment (Fiat-Shamir) *after* the node
    commits, so an adversary cannot place its cheat in the probe's nullspace. Verifier
    recomputes in float64.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    return np.max(np.abs(A @ (B @ R) - C @ R), axis=0)


def freivalds_residual_left_with(A, B, C, L) -> np.ndarray:
    """Per-probe residual ``||(L A) B - L C||_inf`` for an explicit left probe ``L`` (rounds, m).

    The left check catches cheats hidden in a right probe's nullspace: a rank-1 cheat
    ``u v^T`` with ``v`` orthogonal to the right probes still has ``L u != 0`` for a fresh
    left probe. See ``docs/DESIGN.md`` §6.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    return np.max(np.abs((L @ A) @ B - L @ C), axis=1)


def freivalds_residual(A, B, C, *, rounds: int = 1, rng=None) -> np.ndarray:
    """Return the per-round max-abs residual ``||A(Br) - Cr||_inf`` for random r.

    Shape contract: ``A`` is (m, k), ``B`` is (k, n), ``C`` is (m, n).
    Returns an array of length ``rounds``. Verifier recomputes in float64, modelling a
    higher-precision check against the node's stored (possibly float32) output.
    """
    rng = np.random.default_rng() if rng is None else rng
    R = _random_vectors(np.asarray(B).shape[1], rounds, rng)  # (n, rounds)
    return freivalds_residual_with(A, B, C, R)


def freivalds_check(A, B, C, *, rounds: int = 1, rng=None,
                    atol: float = 1e-3, rtol: float = 1e-2) -> bool:
    """Probabilistically test whether ``C == A @ B``.

    Returns True iff every round's residual is within tolerance. The tolerance is taken
    relative to the magnitude of ``C @ r`` so that large activations do not trip false
    positives. Tightening this bound is the open research question, not a magic constant.
    """
    rng = np.random.default_rng() if rng is None else rng
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    R = _random_vectors(B.shape[1], rounds, rng)
    lhs = A @ (B @ R)
    rhs = C @ R
    bound = atol + rtol * np.max(np.abs(rhs), axis=0)
    resid = np.max(np.abs(lhs - rhs), axis=0)
    return bool(np.all(resid <= bound))


def freivalds_check_threshold(A, B, C, threshold: float, *, rounds: int = 1, rng=None) -> bool:
    """Freivalds check against an explicit residual threshold.

    The threshold should come from a floating-point error model
    (``numerics.calibrated_threshold``) rather than a guessed constant -- that is what makes
    the check sound on heterogeneous hardware. Returns True iff every round is within it.
    """
    resid = freivalds_residual(A, B, C, rounds=rounds, rng=rng)
    return bool(np.all(resid <= threshold))
