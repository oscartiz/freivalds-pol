"""Floating-point error model for the Freivalds honest-noise floor.

The crux: on heterogeneous hardware an honest node's output C != A@B exactly, so the
Freivalds residual ||C r - A(B r)|| is nonzero even with no cheating. To avoid *both* false
positives (flagging honest nodes) and a wide "undetectable band" (letting cheats hide under
the noise), the acceptance threshold must come from a floating-point error model rather than
a guessed constant.

Backward-error analysis. Computing C = A@B in floating point yields
    C = A@B + E,    |E| <= gamma * (|A| @ |B|)        (elementwise)
where gamma is the effective unit roundoff of the matmul:
  - naive accumulation at unit roundoff u:            gamma ~ k * u   (k = contraction dim)
  - tensor-core (low-precision in, fp32 accumulate):  gamma ~ c * u_in (input/output
    quantization dominates; c ~ 3-4 covers both quantizations)

With a Rademacher probe r in {-1,+1}^n the honest residual on row i is (E r)_i = sum_j E_ij r_j,
so the rigorous worst-case bound is
    ||E r||_inf <= max_i sum_j |E_ij| <= gamma * || |A| @ |B| ||_inf .          (1)

Crucially, (1) is computable in **O(n^2)** -- it does NOT need the O(n^3) product |A|@|B|:
    max_i sum_j (|A| @ |B|)_ij = max_i ( |A| @ (|B| @ 1) )_i ,
two matrix-vector products. Keeping the threshold O(n^2) is what preserves Freivalds'
advantage over recompute-and-compare. The tighter 2-norm (typical-case) bound below is
O(n^3) and is provided only as an offline reference.
"""

from __future__ import annotations

import numpy as np

# Unit roundoff u = 2^-(mantissa_bits + 1) for round-to-nearest.
UNIT_ROUNDOFF = {
    "fp16": 2.0 ** -11,  # 10 mantissa bits
    "bf16": 2.0 ** -8,   # 7 mantissa bits
    "fp32": 2.0 ** -24,  # 23 mantissa bits
    "fp64": 2.0 ** -53,  # 52 mantissa bits
}


def to_bf16(x) -> np.ndarray:
    """Round a float array to bfloat16 precision (round-to-nearest-even), kept in float32.

    bf16 shares fp32's 8 exponent bits and keeps 7 mantissa bits, i.e. the top 16 bits of
    the fp32 representation.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    bias = ((u >> 16) & np.uint64(1)) + np.uint64(0x7FFF)
    u = (u + bias) & np.uint64(0xFFFF0000)
    return u.astype(np.uint32).view(np.float32)


def _as_precision(x, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        return to_bf16(x)
    np_dtype = {"fp16": np.float16, "fp32": np.float32, "fp64": np.float64}[dtype]
    return np.asarray(x, dtype=np_dtype)


def node_matmul(A, B, *, in_dtype="bf16", accum_dtype="fp32", out_dtype="bf16") -> np.ndarray:
    """Simulate a node computing C = A@B with low-precision I/O and a chosen accumulator.

    Models a modern tensor core (bf16 inputs, fp32 accumulate, bf16 output) by default.
    Returned values carry the rounding error a heterogeneous node would actually produce.
    """
    accum_np = {"fp16": np.float16, "bf16": np.float32,
                "fp32": np.float32, "fp64": np.float64}[accum_dtype]
    Aq = _as_precision(A, in_dtype).astype(accum_np)
    Bq = _as_precision(B, in_dtype).astype(accum_np)
    return _as_precision(Aq @ Bq, out_dtype)


def effective_gamma(dtype: str, *, k: int | None = None,
                    regime: str = "tensorcore", safety: float = 1.0) -> float:
    """Effective unit roundoff gamma for the chosen matmul regime (see module docstring)."""
    u = UNIT_ROUNDOFF[dtype]
    if regime == "tensorcore":
        g = u
    elif regime == "naive":
        if k is None:
            raise ValueError("naive regime needs the contraction dimension k")
        g = k * u
    else:
        raise ValueError(f"unknown regime: {regime}")
    return safety * g


def honest_bound_inf(A, B, gamma: float, *, side: str = "right") -> float:
    """Rigorous worst-case honest residual bound, computed in O(n^2). See eq. (1).

    ``side="right"`` bounds ``||E r||_inf`` for a right probe (max abs row sum of |A||B|);
    ``side="left"`` bounds ``||l E||_inf`` for a left probe (max abs column sum). Both are
    two matrix-vector products.
    """
    Aa = np.abs(np.asarray(A, dtype=np.float64))
    Ba = np.abs(np.asarray(B, dtype=np.float64))
    if side == "right":
        v = Aa @ (Ba @ np.ones(Ba.shape[1]))   # |A| @ (|B| @ 1)  -> length m
    elif side == "left":
        v = (np.ones(Aa.shape[0]) @ Aa) @ Ba   # (1 @ |A|) @ |B|  -> length n
    else:
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    return gamma * float(np.max(v))


def honest_bound_l2(A, B, gamma: float) -> float:
    """Typical-case (2-norm) honest residual bound. O(n^3); offline reference only."""
    Aa = np.abs(np.asarray(A, dtype=np.float64))
    Ba = np.abs(np.asarray(B, dtype=np.float64))
    M = Aa @ Ba  # |A| @ |B|, O(n^3)
    return gamma * float(np.max(np.sqrt((M ** 2).sum(axis=1))))


def calibrated_threshold(A, B, dtype: str, *, regime="tensorcore", mode="inf",
                         side: str = "right", k: int | None = None,
                         safety: float = 8.0) -> float:
    """Acceptance threshold for the Freivalds residual under a node computing in ``dtype``.

    ``mode="inf"`` is the cheap, rigorous, O(n^2) threshold used in the protocol (``side``
    selects the right- or left-probe bound); ``mode="l2"`` is the tighter offline reference.
    """
    if k is None:
        k = np.asarray(B).shape[0]
    gamma = effective_gamma(dtype, k=k, regime=regime, safety=safety)
    if mode == "inf":
        return honest_bound_inf(A, B, gamma, side=side)
    return honest_bound_l2(A, B, gamma)
