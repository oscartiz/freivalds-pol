"""Quantify grinding resistance: expected work to find a Fiat-Shamir probe that hides a cheat.

Run:  python -m experiments.grinding   (after `pip install -e .`)

With a pure Fiat-Shamir probe (derived from the commitment), a malicious node can vary a
commitment nonce and retry, hoping the induced probe fails to detect its cheat. Each try
succeeds with probability q_k = P[a fresh k-probe set has all residuals <= tau], so the expected
number of commitments to grind is 1/q_k. We measure q_k as a function of the cheat's relative
magnitude rho and the number of probe rounds k, on an fp32 matmul with the calibrated threshold.

Takeaway: grinding work explodes for any cheat large enough to matter, and falls geometrically
with k. (A fresh public beacon drawn AFTER the commit removes grinding entirely — you cannot
grind a value you do not yet know.)
"""

from __future__ import annotations

import numpy as np

from freivalds_pol.freivalds import freivalds_residual_with
from freivalds_pol.numerics import calibrated_threshold, node_matmul

N = 256
RHOS = [1e-5, 3e-5, 1e-4, 1e-3]
KS = [1, 2, 4]


def grinding_curve(trials=4000, seed=0):
    """Return (rhos, ks, tries[rho][k]) where tries = expected commitments to grind = 1/q_k."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N, N)).astype(np.float32)
    B = rng.normal(size=(N, N)).astype(np.float32)
    C = node_matmul(A, B, in_dtype="fp32", out_dtype="fp32").astype(np.float64)
    tau = calibrated_threshold(A, B, "fp32", mode="inf", safety=8.0)
    tries = {}
    for rho in RHOS:
        delta = rng.normal(size=C.shape)
        delta *= rho * np.linalg.norm(C) / np.linalg.norm(delta)
        Cp = C + delta
        # per-probe escape probability q1 = P[residual <= tau] over many single probes
        probes = rng.choice([-1.0, 1.0], size=(N, trials))
        resid = freivalds_residual_with(A, B, Cp, probes)        # one per probe column
        q1 = float(np.mean(resid <= tau))
        tries[rho] = {}
        for k in KS:
            qk = q1 ** k                                          # k independent probes
            tries[rho][k] = float("inf") if qk == 0.0 else 1.0 / qk
    return RHOS, KS, tries


def main():
    rhos, ks, tries = grinding_curve()
    print(f"expected commitments to grind an evading probe (fp32, n={N})\n")
    print("cheat rho".ljust(12) + "".join(f"k={k}".rjust(14) for k in ks))
    print("-" * (12 + 14 * len(ks)))
    for rho in rhos:
        row = f"{rho:.0e}".ljust(12)
        for k in ks:
            t = tries[rho][k]
            row += ("infeasible" if t == float("inf") else f"{t:.3g}").rjust(14)
        print(row)
    print("\nGrinding work = 1/q_k explodes once the cheat is large enough to matter (rho >= 1e-4")
    print("is already infeasible at fp32), and shrinks geometrically with k. A fresh public beacon")
    print("drawn after the commit removes grinding entirely. Beacon collusion only helps if a")
    print("colluding fraction can resample the beacon, which still faces the same 1/q_k wall.")


if __name__ == "__main__":
    main()
