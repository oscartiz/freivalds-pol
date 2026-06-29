import numpy as np

from freivalds_pol.freivalds import freivalds_check, freivalds_residual


def test_honest_product_passes():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(64, 48)).astype(np.float32)
    B = rng.normal(size=(48, 80)).astype(np.float32)
    C = A @ B
    assert freivalds_check(A, B, C, rounds=8, rng=rng)


def test_wrong_product_fails():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(64, 64))
    B = rng.normal(size=(64, 64))
    C = A @ B
    C[0, 0] += 5.0  # single corrupted entry
    assert not freivalds_check(A, B, C, rounds=16, rng=rng)


def test_residual_small_for_honest_large_for_wrong():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(32, 32))
    B = rng.normal(size=(32, 32))
    C = A @ B
    honest = freivalds_residual(A, B, C, rounds=8, rng=rng)
    wrong = freivalds_residual(A, B, C + 1.0, rounds=8, rng=rng)
    assert honest.max() < wrong.min()
