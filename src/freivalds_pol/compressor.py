"""DeMo / DisTrO-style update compression, and how to verify it.

A real Psyche node does not transmit the dense gradient. Following DeMo (Decoupled Momentum,
Peng et al., Nous Research) it: accumulates momentum with error feedback, takes a DCT of the
accumulator in tiles, transmits only the **top-k** coefficients per tile, and keeps the
residual locally. This module implements that compressor and a verifier for it.

Why it is verifiable cheaply -- a three-part decomposition:

  1. **Momentum** ``m = decay*m_prev + g`` is elementwise: O(n) to recompute from the
     already-Freivalds-verified gradient ``g`` and the committed prior accumulator.
  2. **DCT** is a *linear map* ``coeff = C @ m`` -- so transmitted-value correctness is exactly
     a (sparse) matmul check, Freivalds-amenable; here we recompute it per tile in O(tile^2).
  3. **Top-k selection** is the only non-linear part, but the verifier can recompute the full
     per-tile DCT (cheap: O(tile^2), or O(tile log tile) with an FFT) and confirm the
     transmitted indices really are the k largest.

Verification is per-tile, so a challenge over a random subset of tiles catches a node that
corrupts a fraction of them with probability 1-(1-f)^(#challenged) -- the same spot-check
structure as the Freivalds layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .verifier import VerifyResult


def dct_matrix(N: int) -> np.ndarray:
    """Orthonormal DCT-II matrix C (rows = frequencies); C @ C.T = I, so inverse DCT = C.T."""
    n = np.arange(N)
    k = n.reshape(-1, 1)
    C = np.cos(np.pi / N * (n + 0.5) * k)
    C[0, :] *= np.sqrt(1.0 / N)
    C[1:, :] *= np.sqrt(2.0 / N)
    return C


def _to_tiles(v, tile):
    pad = (-v.size) % tile
    vp = np.concatenate([v, np.zeros(pad)]) if pad else v
    return vp.reshape(-1, tile)


@dataclass
class CompressedUpdate:
    indices: np.ndarray   # (n_tiles, k) int  -- top-k DCT coefficient indices per tile
    values: np.ndarray    # (n_tiles, k) float -- their values
    n: int                # original (pre-pad) length
    tile: int
    k: int


def compress(update, m_prev, *, decay=0.9, tile=64, k=8, C=None):
    """Compress a dense update DeMo-style. Returns (CompressedUpdate, m_next, applied).

    ``m_next`` is the local error-feedback residual carried to the next step; ``applied`` is
    the dense (inverse-DCT) update actually contributed to the aggregate.
    """
    C = dct_matrix(tile) if C is None else C
    m = decay * m_prev + update
    M = _to_tiles(m, tile)
    coeffs = M @ C.T                                          # per-tile DCT, (n_tiles, tile)
    rows = np.arange(M.shape[0])[:, None]
    idx = np.sort(np.argsort(-np.abs(coeffs), axis=1)[:, :k], axis=1)
    vals = coeffs[rows, idx]
    sparse = np.zeros_like(coeffs)
    sparse[rows, idx] = vals
    time_tx = sparse @ C                                      # inverse DCT (C orthonormal)
    residual = M - time_tx
    cu = CompressedUpdate(idx, vals, update.size, tile, k)
    # m_next carries the full (padded) residual as local error-feedback state; `applied` is
    # the dense contribution at the real parameter length.
    return cu, residual.reshape(-1).copy(), time_tx.reshape(-1)[:update.size].copy()


def decompress(cu: CompressedUpdate, C=None) -> np.ndarray:
    """Reconstruct the dense (time-domain) update contributed to the aggregate."""
    C = dct_matrix(cu.tile) if C is None else C
    n_tiles = cu.indices.shape[0]
    sparse = np.zeros((n_tiles, cu.tile))
    sparse[np.arange(n_tiles)[:, None], cu.indices] = cu.values
    return (sparse @ C).reshape(-1)[:cu.n].copy()


def verify_compressed(grad, m_prev, m_next, cu: CompressedUpdate, challenged_tiles, *,
                      decay=0.9, C=None, atol=1e-8, rtol=1e-6) -> VerifyResult:
    """Verify the compressed update on a challenged subset of tiles, given the verified grad."""
    C = dct_matrix(cu.tile) if C is None else C
    G = _to_tiles(np.asarray(grad, float), cu.tile)
    MP = _to_tiles(np.asarray(m_prev, float), cu.tile)
    MN = _to_tiles(np.asarray(m_next, float), cu.tile)

    for t in challenged_tiles:
        m = decay * MP[t] + G[t]
        coeff = C @ m
        idx = cu.indices[t]

        if not np.allclose(cu.values[t], coeff[idx], atol=atol, rtol=rtol):
            return VerifyResult(False, f"tile {t}: wrong coefficient values", t)

        true_topk = np.sort(np.argsort(-np.abs(coeff))[:cu.k])
        if not np.array_equal(np.sort(idx), true_topk):
            return VerifyResult(False, f"tile {t}: not the top-k coefficients", t)

        sparse = np.zeros(cu.tile)
        sparse[idx] = cu.values[t]
        if not np.allclose(MN[t], m - C.T @ sparse, atol=atol, rtol=rtol):
            return VerifyResult(False, f"tile {t}: inconsistent error-feedback residual", t)

    return VerifyResult(True, "ok", len(list(challenged_tiles)))


# --- compression-layer cheats (one per attack in the threat model) -----------------------

def cheat_lazy(cu, m_next, *, frac=1.0, rng=None):
    """Never computed the coefficients: zero out a fraction of tiles' values."""
    rng = np.random.default_rng() if rng is None else rng
    cu = CompressedUpdate(cu.indices.copy(), cu.values.copy(), cu.n, cu.tile, cu.k)
    tiles = rng.choice(cu.values.shape[0], max(1, int(frac * cu.values.shape[0])), replace=False)
    cu.values[tiles] = 0.0
    return cu, m_next


def cheat_fake_values(cu, m_next, *, frac=0.5, rng=None):
    """Transmit plausible but wrong coefficient values."""
    rng = np.random.default_rng() if rng is None else rng
    cu = CompressedUpdate(cu.indices.copy(), cu.values.copy(), cu.n, cu.tile, cu.k)
    tiles = rng.choice(cu.values.shape[0], max(1, int(frac * cu.values.shape[0])), replace=False)
    cu.values[tiles] += rng.normal(scale=np.abs(cu.values[tiles]).mean() + 1e-9,
                                   size=cu.values[tiles].shape)
    return cu, m_next


def cheat_wrong_topk(cu, m_next, *, frac=0.5, rng=None):
    """Transmit non-top-k indices (e.g. to skip the important, expensive-to-get coefficients).

    Indices are shifted but values are left equal to the *true* coefficients at the new
    indices would not be -- so this is caught by the top-k check even if values looked sane.
    """
    rng = np.random.default_rng() if rng is None else rng
    cu = CompressedUpdate(cu.indices.copy(), cu.values.copy(), cu.n, cu.tile, cu.k)
    tiles = rng.choice(cu.values.shape[0], max(1, int(frac * cu.values.shape[0])), replace=False)
    cu.indices[tiles] = (cu.indices[tiles] + cu.k) % cu.tile
    return cu, m_next


def cheat_bad_residual(cu, m_next, *, frac=0.5, rng=None):
    """Skip the error-feedback bookkeeping: corrupt the residual carried forward."""
    rng = np.random.default_rng() if rng is None else rng
    m_next = np.asarray(m_next, float).copy()
    tile = cu.tile
    MN = _to_tiles(m_next, tile)
    tiles = rng.choice(MN.shape[0], max(1, int(frac * MN.shape[0])), replace=False)
    MN[tiles] += rng.normal(scale=np.abs(MN).mean() + 1e-9, size=MN[tiles].shape)
    return cu, MN.reshape(-1)[:m_next.size].copy()
