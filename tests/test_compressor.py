import numpy as np

from freivalds_pol.compressor import (
    cheat_bad_residual,
    cheat_fake_values,
    cheat_lazy,
    cheat_wrong_topk,
    compress,
    dct_matrix,
    decompress,
    verify_compressed,
)
from freivalds_pol.freivalds import freivalds_check

N, TILE, K = 200, 32, 4


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    grad = rng.normal(size=N)
    m_prev = rng.normal(size=N) * 0.1
    cu, m_next, applied = compress(grad, m_prev, decay=0.9, tile=TILE, k=K)
    return rng, grad, m_prev, cu, m_next, applied


def test_dct_is_orthonormal():
    C = dct_matrix(TILE)
    assert np.allclose(C @ C.T, np.eye(TILE), atol=1e-10)


def test_dct_value_check_is_freivalds_checkable():
    C = dct_matrix(TILE)
    m = np.random.default_rng(1).normal(size=TILE)
    coeff = C @ m
    assert freivalds_check(C, m[:, None], coeff[:, None], rounds=6)
    assert not freivalds_check(C, m[:, None], (coeff + 1.0)[:, None], rounds=12)


def test_error_feedback_invariant():
    # contributed update + carried residual == the accumulator (nothing is lost)
    _, grad, m_prev, cu, m_next, applied = _setup()
    m = 0.9 * m_prev + grad
    assert np.allclose(applied + m_next[:N], m, atol=1e-10)


def test_decompress_matches_applied():
    _, _, _, cu, _, applied = _setup()
    assert np.allclose(decompress(cu), applied, atol=1e-10)


def test_honest_compressed_update_verifies():
    _, grad, m_prev, cu, m_next, _ = _setup()
    tiles = range(cu.indices.shape[0])
    res = verify_compressed(grad, m_prev, m_next, cu, tiles, decay=0.9)
    assert res.accepted, res.reason


def test_compression_cheats_rejected():
    _, grad, m_prev, cu, m_next, _ = _setup()
    tiles = range(cu.indices.shape[0])
    for cheat, frac in [(cheat_lazy, 1.0), (cheat_fake_values, 1.0),
                        (cheat_wrong_topk, 1.0), (cheat_bad_residual, 1.0)]:
        rng = np.random.default_rng(7)
        cu_c, mn_c = cheat(cu, m_next, frac=frac, rng=rng)
        res = verify_compressed(grad, m_prev, mn_c, cu_c, tiles, decay=0.9)
        assert not res.accepted, f"{cheat.__name__} slipped through"


def test_wrong_grad_breaks_verification():
    # A node that compressed a *different* gradient than it claims is caught.
    rng, grad, m_prev, cu, m_next, _ = _setup()
    wrong_grad = grad + rng.normal(scale=0.5, size=grad.shape)
    res = verify_compressed(wrong_grad, m_prev, m_next, cu, range(cu.indices.shape[0]), decay=0.9)
    assert not res.accepted
