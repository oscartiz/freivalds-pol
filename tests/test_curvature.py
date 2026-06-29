import numpy as np

from freivalds_pol.curvature import bottom_eigvec, hvp, rayleigh, top_eigvec
from freivalds_pol.transformer import TransformerBlock, forward

D, D_FF, T = 16, 32, 8


def _setup(seed=0):
    blk = TransformerBlock.init(D, D_FF, np.random.default_rng(seed))
    teacher = TransformerBlock.init(D, D_FF, np.random.default_rng(seed + 100))
    pr = np.random.default_rng(seed + 7)
    X = pr.normal(size=(T, D))
    Y, _, _ = forward(teacher, X)
    n = sum(getattr(blk, k).size for k in
            ("Wq", "Wk", "Wv", "Wo", "W1", "W2", "g1", "g2"))
    return blk, X, Y, n, pr


def test_hvp_is_linear():
    blk, X, Y, n, pr = _setup()
    a, b = pr.normal(size=n), pr.normal(size=n)
    lhs = hvp(blk, X, Y, a + b)
    rhs = hvp(blk, X, Y, a) + hvp(blk, X, Y, b)
    assert np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs) < 1e-2


def test_top_eigvector_converges():
    blk, X, Y, n, _ = _setup()
    lam, v = top_eigvec(blk, X, Y, iters=40)
    residual = np.linalg.norm(hvp(blk, X, Y, v) - lam * v) / abs(lam)
    assert residual < 0.2


def test_steepest_curvature_exceeds_flattest():
    blk, X, Y, n, _ = _setup()
    lam_top, v_steep = top_eigvec(blk, X, Y, iters=40)
    _, v_flat = bottom_eigvec(blk, X, Y, lam_top, iters=40)
    assert rayleigh(blk, X, Y, v_steep) > rayleigh(blk, X, Y, v_flat)
    # the steepest direction is genuinely curved
    assert rayleigh(blk, X, Y, v_steep) > 0.1
