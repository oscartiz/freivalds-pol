"""Curvature probing for the worst-case sub-threshold adversary.

The §8 analysis showed a sub-threshold bias settles at an equilibrium ~ budget/curvature: the
optimizer's restoring force is weak along *flat* (low-curvature) loss directions. The worst-case
adversary therefore aims its bias along the flattest direction of the loss Hessian.

We never form the Hessian (n ~ 10^4-10^5 params). Instead we use Hessian-vector products via a
central finite difference of the gradient,  H v ~ (g(theta + eps v) - g(theta - eps v)) / 2eps,
and power iteration: directly for the top (steepest) eigenvector, and on (shift*I - H) for the
bottom (flattest) one.
"""

from __future__ import annotations

import numpy as np

from .trainer import ORDER
from .transformer import block_step


def _grad_flat(blk, X, Y) -> np.ndarray:
    _, grads, _ = block_step(blk, X, Y)
    return np.concatenate([grads[k].ravel() for k in ORDER])


def _add(blk, vec):
    i = 0
    for k in ORDER:
        p = getattr(blk, k)
        p += vec[i:i + p.size].reshape(p.shape)
        i += p.size


def _nparams(blk):
    return sum(getattr(blk, k).size for k in ORDER)


def hvp(blk, X, Y, v, eps=1e-3) -> np.ndarray:
    """Hessian-vector product H v via central difference of the gradient (params restored)."""
    _add(blk, eps * v)
    gp = _grad_flat(blk, X, Y)
    _add(blk, -eps * v)              # restore, then step the other way
    _add(blk, -eps * v)
    gm = _grad_flat(blk, X, Y)
    _add(blk, eps * v)               # restore
    return (gp - gm) / (2 * eps)


def rayleigh(blk, X, Y, v) -> float:
    """Curvature along v: v^T H v / v^T v."""
    v = v / np.linalg.norm(v)
    return float(v @ hvp(blk, X, Y, v))


def top_eigvec(blk, X, Y, *, iters=30, seed=0):
    """Largest-|eigenvalue| (steepest) direction via power iteration. Returns (eigval, vec)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=_nparams(blk))
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        Hv = hvp(blk, X, Y, v)
        lam = float(v @ Hv)
        nv = np.linalg.norm(Hv)
        if nv == 0:
            break
        v = Hv / nv
    return lam, v


def bottom_eigvec(blk, X, Y, lam_top, *, iters=30, seed=1):
    """Smallest-eigenvalue (flattest) direction via power iteration on (shift*I - H)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=_nparams(blk))
    v /= np.linalg.norm(v)
    shift = abs(lam_top) * 1.1 + 1e-9
    for _ in range(iters):
        w = shift * v - hvp(blk, X, Y, v)
        nv = np.linalg.norm(w)
        if nv == 0:
            break
        v = w / nv
    return rayleigh(blk, X, Y, v), v
