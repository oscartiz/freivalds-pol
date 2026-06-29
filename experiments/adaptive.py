"""The adaptive adversary: does shaping the cheat to dodge the probe beat Freivalds?

Run:  python -m experiments.adaptive   (after `pip install -e .`)

Three scenarios, all at n=256 with an fp32 node and the rigorous O(n^2) threshold:

  1. PREDICTABLE probe -- the adversary knows the probe and hides a huge cheat in its
     nullspace. Detection collapses to zero at arbitrary impact. Freivalds is broken.
  2. The fix: COMMIT-THEN-SAMPLE (Fiat-Shamir over the commitment). The same huge cheat is
     now caught every time -- and near the noise floor, detection rises geometrically with
     the number of probe rounds k. Adaptivity buys nothing beyond the FP band rho* ~ tau.
  3. GRINDING -- a node could try many commitments to luck into an evading probe. The
     per-probe evasion probability is negligible for any cheat large enough to matter, so
     grinding is infeasible; a fresh public beacon removes the option entirely.
"""

from __future__ import annotations

import numpy as np

from freivalds_pol.adaptive import fixed_cheat, nullspace_cheat
from freivalds_pol.challenge import fiat_shamir_probes
from freivalds_pol.commitments import hash_array
from freivalds_pol.freivalds import freivalds_residual_left_with, freivalds_residual_with
from freivalds_pol.numerics import calibrated_threshold, node_matmul

N = 256


def setup(rng):
    A = rng.normal(size=(N, N)).astype(np.float32)
    B = rng.normal(size=(N, N)).astype(np.float32)
    C = node_matmul(A, B, in_dtype="fp32", out_dtype="fp32").astype(np.float64)
    tau = calibrated_threshold(A, B, "fp32", mode="inf", safety=8.0)
    return A, B, C, tau


def scenario_predictable_vs_fresh(trials=20, k=4, rho=0.5):
    print("=== Scenario 1+2: same impact, predictable probe vs. commit-then-sample ===\n")
    rng = np.random.default_rng(0)
    pred_caught = fresh_caught = 0
    impact = 0.0
    for _ in range(trials):
        A, B, C, tau = setup(rng)
        R_known = rng.choice([-1.0, 1.0], size=(N, k))      # leaked / reused probe
        Cp = nullspace_cheat(C, R_known, rho, rng=rng)
        impact += float(np.linalg.norm(Cp - C) / np.linalg.norm(C))
        # adversary's probe (the one it targeted)
        pred_caught += int(np.any(freivalds_residual_with(A, B, Cp, R_known) > tau))
        # honest verifier draws a fresh probe from the commitment instead
        R_fresh = fiat_shamir_probes(hash_array(Cp), N, k)
        fresh_caught += int(np.any(freivalds_residual_with(A, B, Cp, R_fresh) > tau))
    print(f"cheat impact ||Delta||/||C|| = {impact / trials:.2f}, rounds k = {k}\n")
    print(f"  predictable probe  -> detection {pred_caught / trials:.3f}   (BROKEN)")
    print(f"  commit-then-sample -> detection {fresh_caught / trials:.3f}   (FIXED)\n")


def detection_fresh(rho, k, trials=40, rank=None, seed=1):
    rng = np.random.default_rng(seed)
    caught = 0
    for _ in range(trials):
        A, B, C, tau = setup(rng)
        Cp = fixed_cheat(C, rho, rng=rng, rank=rank)
        R = fiat_shamir_probes(hash_array(Cp), N, k)
        caught += int(np.any(freivalds_residual_with(A, B, Cp, R) > tau))
    return caught / trials


def scenario_band_and_rounds():
    print("=== Scenario 2b: near the noise floor, detection vs. rounds k (fresh probe) ===\n")
    rhos = [3e-6, 1e-5, 3e-5, 1e-4]
    ks = [1, 2, 4]
    print("rho \\ k".ljust(12) + "".join(f"k={k}".rjust(9) for k in ks) + "   (rank-1)".rjust(12))
    print("-" * 56)
    for rho in rhos:
        row = f"{rho:.0e}".ljust(12)
        for k in ks:
            row += f"{detection_fresh(rho, k):9.3f}"
        r1 = detection_fresh(rho, ks[-1], rank=1)        # low-rank, k = max
        row += f"{r1:12.3f}"
        print(row)
    print("\nDetection climbs with k and with impact; the evadable region matches the FP band")
    print("rho* ~ 2.7e-5 (experiments/fp_crux.py). A rank-1 cheat evades a bit more often near")
    print("the band -- one anti-concentration draw per probe, not many -- a small but real edge")
    print("for the adversary, and a reason to also probe from the left (r^T A B vs r^T C).\n")


def scenario_grinding(samples=4000):
    print("=== Scenario 3: grinding -- per-probe evasion vs. cheat impact (k=1) ===\n")
    rng = np.random.default_rng(2)
    A, B, C, tau = setup(rng)
    print("impact rho".ljust(16) + "evasion q (1 probe)".ljust(22) + "expected grinding tries 1/q")
    print("-" * 60)
    for rho in [1e-3, 1e-4, 3e-5, 1e-5]:
        Cp = fixed_cheat(C, rho, rng=rng)
        r = rng.choice([-1.0, 1.0], size=(N, samples))
        resid = freivalds_residual_with(A, B, Cp, r)     # one residual per probe column
        q = float(np.mean(resid <= tau))
        tries = "inf" if q == 0.0 else f"{1.0 / q:.3g}"
        print(f"{rho:<16.0e}{q:<22.4g}{tries}")
    print("\nMeaningful cheats have q ~ 0, so no commitment can be ground into an evading probe;")
    print("each extra round k raises the cost to ~1/q^k, and a public beacon removes grinding.\n")


def scenario_two_sided(trials=20, k=4, rho=0.3):
    print("=== Scenario 4: two-sided probing closes the rank-1 nullspace edge ===\n")
    rng = np.random.default_rng(3)
    one_caught = two_caught = 0
    for _ in range(trials):
        A, B, C, tau = setup(rng)
        tau_l = calibrated_threshold(A, B, "fp32", mode="inf", side="left", safety=8.0)
        R_known = rng.choice([-1.0, 1.0], size=(N, k))     # adversary knows the RIGHT probe
        Cp = nullspace_cheat(C, R_known, rho, rng=rng, target=rng.normal(size=N))  # rank-1
        right = np.any(freivalds_residual_with(A, B, Cp, R_known) > tau)
        L = rng.choice([-1.0, 1.0], size=(k, N))           # a fresh, independent left probe
        left = np.any(freivalds_residual_left_with(A, B, Cp, L) > tau_l)
        one_caught += int(right)
        two_caught += int(right or left)
    print(f"rank-1 cheat in the right-probe nullspace, impact {rho}, k={k}\n")
    print(f"  one-sided (right only) -> detection {one_caught / trials:.3f}")
    print(f"  two-sided (+ left)     -> detection {two_caught / trials:.3f}\n")


if __name__ == "__main__":
    scenario_predictable_vs_fresh()
    scenario_band_and_rounds()
    scenario_grinding()
    scenario_two_sided()
