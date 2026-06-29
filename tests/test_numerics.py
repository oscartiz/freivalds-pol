import numpy as np

from freivalds_pol.freivalds import freivalds_residual
from freivalds_pol.numerics import (
    UNIT_ROUNDOFF,
    calibrated_threshold,
    effective_gamma,
    honest_bound_inf,
    node_matmul,
    to_bf16,
)


def test_bf16_relative_error_within_unit_roundoff():
    rng = np.random.default_rng(0)
    x = rng.normal(size=10_000).astype(np.float32) * 1000
    q = to_bf16(x)
    rel = np.abs(q - x) / np.maximum(np.abs(x), 1e-30)
    assert rel.max() <= UNIT_ROUNDOFF["bf16"] + 1e-9


def test_on2_inf_bound_matches_on3_definition():
    # The O(n^2) formula must equal gamma * max row sum of |A| @ |B|.
    rng = np.random.default_rng(1)
    A = rng.normal(size=(40, 55))
    B = rng.normal(size=(55, 33))
    gamma = 1e-3
    brute = gamma * float(np.max((np.abs(A) @ np.abs(B)).sum(axis=1)))
    assert np.isclose(honest_bound_inf(A, B, gamma), brute, rtol=1e-12)


def test_inf_bound_holds_for_honest_lowprecision_matmul():
    # Rigorous bound must never be exceeded by an honest residual (no false positives).
    rng = np.random.default_rng(2)
    for _ in range(20):
        A = rng.normal(size=(128, 128)).astype(np.float32)
        B = rng.normal(size=(128, 128)).astype(np.float32)
        C = node_matmul(A, B, in_dtype="bf16", out_dtype="bf16")
        tau = calibrated_threshold(A, B, "bf16", mode="inf", safety=8.0)
        resid = freivalds_residual(A, B, C, rounds=32, rng=rng)
        assert resid.max() <= tau


def test_higher_precision_gives_tighter_bound():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(64, 64))
    B = rng.normal(size=(64, 64))
    bf16 = calibrated_threshold(A, B, "bf16", mode="inf", safety=8.0)
    fp32 = calibrated_threshold(A, B, "fp32", mode="inf", safety=8.0)
    assert fp32 < bf16
    # threshold scales with unit roundoff
    assert np.isclose(fp32 / bf16, UNIT_ROUNDOFF["fp32"] / UNIT_ROUNDOFF["bf16"], rtol=1e-6)


def test_effective_gamma_regimes():
    assert effective_gamma("bf16", regime="tensorcore") == UNIT_ROUNDOFF["bf16"]
    assert effective_gamma("bf16", k=512, regime="naive") == 512 * UNIT_ROUNDOFF["bf16"]
