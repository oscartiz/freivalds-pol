"""Faithful(-er) DeMo compression: 2D per-chunk DCT + top-k + decayed error feedback.

``compressor.py`` is a simplified 1D-tiled instance; this module follows the reference DeMo
(bloc97/DeMo; Peng et al., arXiv:2411.19870) more closely for 2D weight tensors:

  delta <- compression_decay * delta + lr * grad        # decayed error-feedback accumulator
  coeff <- 2D-DCT of each (chunk x chunk) block of delta
  (idx, val) <- top-k coefficients by magnitude per block
  transmit (idx, val); applied <- inverse 2D-DCT of the sparse top-k
  delta <- delta - applied                               # error feedback (momentum subtraction)

The exact deltas vs the reference (decay/k/chunk defaults, the sign-quantization step that
happens at *aggregation*, and torch-vs-numpy bit differences) are documented in
``docs/DESIGN.md`` §7b. The per-block verifier holds for *any* orthonormal transform, so
matching DeMo is about using its transform and parameters, not about the verifiability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compressor import dct_matrix
from .verifier import VerifyResult


def _to_blocks(W, c):
    M, N = W.shape
    assert M % c == 0 and N % c == 0, "dims must be divisible by chunk (DeMo uses a divisor chunk)"
    return W.reshape(M // c, c, N // c, c).transpose(0, 2, 1, 3).reshape(-1, c, c)


def _from_blocks(blocks, shape, c):
    M, N = shape
    return blocks.reshape(M // c, N // c, c, c).transpose(0, 2, 1, 3).reshape(M, N)


def _dct2(blocks, C):
    return np.einsum("ab,tbd,ed->tae", C, blocks, C)        # C @ tile @ C^T per block


def _idct2(coeff, C):
    return np.einsum("ab,tae,ed->tbd", C, coeff, C)         # C^T @ coeff @ C per block


@dataclass
class DemoUpdate:
    idx: np.ndarray     # (n_blocks, k) flat indices into c*c
    val: np.ndarray     # (n_blocks, k)
    shape: tuple        # original (m, n)
    chunk: int
    k: int


def encode(grad, delta_prev, *, decay=0.999, lr=1.0, chunk=64, k=32, C=None):
    """One DeMo encode step on a 2D tensor. Returns (DemoUpdate, delta_next, applied)."""
    c = chunk
    C = dct_matrix(c) if C is None else C
    delta = decay * delta_prev + lr * grad
    blocks = _to_blocks(delta, c)
    coeff = _dct2(blocks, C).reshape(len(blocks), c * c)
    rows = np.arange(len(blocks))[:, None]
    idx = np.sort(np.argsort(-np.abs(coeff), axis=1)[:, :k], axis=1)
    val = coeff[rows, idx]
    sparse = np.zeros_like(coeff)
    sparse[rows, idx] = val
    applied = _from_blocks(_idct2(sparse.reshape(-1, c, c), C), grad.shape, c)
    return DemoUpdate(idx, val, grad.shape, c, k), delta - applied, applied


def decode(du: DemoUpdate, C=None) -> np.ndarray:
    """Reconstruct the dense (time-domain) update contributed to the aggregate."""
    c = du.chunk
    C = dct_matrix(c) if C is None else C
    n_blocks = du.idx.shape[0]
    sparse = np.zeros((n_blocks, c * c))
    sparse[np.arange(n_blocks)[:, None], du.idx] = du.val
    return _from_blocks(_idct2(sparse.reshape(-1, c, c), C), du.shape, c)


def verify(grad, delta_prev, delta_next, du: DemoUpdate, challenged_blocks, *,
           decay=0.999, lr=1.0, C=None, atol=1e-8, rtol=1e-6) -> VerifyResult:
    """Verify a DeMo payload on a challenged subset of blocks, given the verified grad."""
    c = du.chunk
    C = dct_matrix(c) if C is None else C
    delta = decay * np.asarray(delta_prev, float) + lr * np.asarray(grad, float)
    Gb = _to_blocks(delta, c)
    Nb = _to_blocks(np.asarray(delta_next, float), c)
    for t in challenged_blocks:
        coeff = _dct2(Gb[t:t + 1], C).reshape(c * c)
        idx = du.idx[t]
        if not np.allclose(du.val[t], coeff[idx], atol=atol, rtol=rtol):
            return VerifyResult(False, f"block {t}: wrong coefficient values", t)
        if not np.array_equal(np.sort(idx), np.sort(np.argsort(-np.abs(coeff))[:du.k])):
            return VerifyResult(False, f"block {t}: not the top-k coefficients", t)
        sparse = np.zeros(c * c)
        sparse[idx] = du.val[t]
        applied_t = _idct2(sparse.reshape(1, c, c), C).reshape(c, c)
        if not np.allclose(Nb[t], Gb[t] - applied_t, atol=atol, rtol=rtol):
            return VerifyResult(False, f"block {t}: inconsistent error-feedback residual", t)
    return VerifyResult(True, "ok", len(list(challenged_blocks)))
