import numpy as np

from freivalds_pol.adaptive import fixed_cheat, nullspace_cheat
from freivalds_pol.challenge import fiat_shamir_probes
from freivalds_pol.commitments import hash_array
from freivalds_pol.freivalds import freivalds_residual_left_with, freivalds_residual_with
from freivalds_pol.numerics import calibrated_threshold, node_matmul

N = 128


def _setup(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N, N)).astype(np.float32)
    B = rng.normal(size=(N, N)).astype(np.float32)
    C = node_matmul(A, B, in_dtype="fp32", out_dtype="fp32").astype(np.float64)
    tau = calibrated_threshold(A, B, "fp32", mode="inf", safety=8.0)
    return rng, A, B, C, tau


def test_nullspace_cheat_evades_known_probe_at_large_impact():
    rng, A, B, C, tau = _setup(0)
    R = rng.choice([-1.0, 1.0], size=(N, 4))
    Cp = nullspace_cheat(C, R, rho=0.5, rng=rng)
    # impact is real ...
    assert np.isclose(np.linalg.norm(Cp - C) / np.linalg.norm(C), 0.5, rtol=1e-6)
    # ... yet the known probe sees nothing above the honest threshold.
    assert freivalds_residual_with(A, B, Cp, R).max() <= tau


def test_nullspace_cheat_caught_by_fresh_probe():
    rng, A, B, C, tau = _setup(1)
    R = rng.choice([-1.0, 1.0], size=(N, 4))
    Cp = nullspace_cheat(C, R, rho=0.5, rng=rng)
    R_fresh = fiat_shamir_probes(hash_array(Cp), N, 4)
    assert freivalds_residual_with(A, B, Cp, R_fresh).max() > tau


def test_large_fixed_cheat_caught_by_fresh_probe():
    rng, A, B, C, tau = _setup(2)
    Cp = fixed_cheat(C, rho=0.1, rng=rng)
    R = fiat_shamir_probes(hash_array(Cp), N, 4)
    assert freivalds_residual_with(A, B, Cp, R).max() > tau


def test_two_sided_catches_rank1_nullspace_cheat():
    # A rank-1 cheat placed in the KNOWN right probe's nullspace evades the right check ...
    rng, A, B, C, tau = _setup(7)
    tau_l = calibrated_threshold(A, B, "fp32", mode="inf", side="left", safety=8.0)
    R = rng.choice([-1.0, 1.0], size=(N, 4))
    Cp = nullspace_cheat(C, R, rho=0.3, rng=rng, target=rng.normal(size=N))
    assert freivalds_residual_with(A, B, Cp, R).max() <= tau
    # ... but a fresh, independent left probe catches it.
    L = rng.choice([-1.0, 1.0], size=(4, N))
    assert freivalds_residual_left_with(A, B, Cp, L).max() > tau_l


def test_fiat_shamir_probes_deterministic_and_commitment_bound():
    c1 = bytes(range(32))
    c2 = bytes(range(1, 33))
    p1 = fiat_shamir_probes(c1, 64, 4)
    assert np.array_equal(p1, fiat_shamir_probes(c1, 64, 4))          # deterministic
    assert not np.array_equal(p1, fiat_shamir_probes(c2, 64, 4))      # commitment-bound
    assert not np.array_equal(p1, fiat_shamir_probes(c1, 64, 4, beacon=b"x"))  # beacon-bound
