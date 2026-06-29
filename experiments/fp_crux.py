"""The FP crux: how floating-point precision sets the Freivalds noise floor and the
size of the smallest cheat that can hide under it.

Run:  python -m experiments.fp_crux   (after `pip install -e .`)

A node computes C = A@B in some low precision (bf16/fp16/fp32, tensor-core style: low-prec
inputs, fp32 accumulate, low-prec output). The verifier recomputes the Freivalds probe in
fp64. We measure:

  Table 1 -- the honest noise floor vs. the model bounds, and whether the cheap O(n^2)
             worst-case threshold is even usable (it must sit below the signal scale ||C r||).
  Table 2 -- the smallest *relative* cheat rho* = ||Delta||_F / ||C||_F that is caught >=99%
             of the time, under two thresholding strategies and across node precisions.

Punchline: the rigorous O(n^2) bound is too loose to use at bf16 (threshold > signal), so
either calibrate statistically or require >= fp32 on the *challenged* layer. The undetectable
band shrinks roughly linearly with the unit roundoff -- precision is the protocol knob.
"""

from __future__ import annotations

import numpy as np

from freivalds_pol.freivalds import freivalds_residual
from freivalds_pol.numerics import (
    UNIT_ROUNDOFF,
    calibrated_threshold,
    node_matmul,
)

N = 256
PROBES = 64
TRIALS = 40
DTYPES = ["fp16", "bf16", "fp32"]


def _AB(rng):
    A = rng.normal(size=(N, N)).astype(np.float32)
    B = rng.normal(size=(N, N)).astype(np.float32)
    return A, B


def honest_residual_samples(dtype, rng, trials=TRIALS, probes=PROBES):
    """Max honest residual per trial, plus the signal scale ||C r||_inf for reference."""
    res, signal = [], []
    for _ in range(trials):
        A, B = _AB(rng)
        C = node_matmul(A, B, in_dtype=dtype, out_dtype=dtype)
        res.append(freivalds_residual(A, B, C, rounds=probes, rng=rng).max())
        Cr = C.astype(np.float64) @ rng.choice([-1.0, 1.0], size=(N, probes))
        signal.append(np.max(np.abs(Cr)))
    return np.array(res), np.array(signal)


def table1():
    print(f"=== Table 1: honest noise floor vs. model bounds  (n={N}) ===\n")
    cols = ["dtype", "emp.max", "O(n^2) inf-bound", "l2-bound", "signal ||Cr||", "usable?"]
    print("".join(c.ljust(18) for c in cols))
    print("-" * (18 * len(cols)))
    for dt in DTYPES:
        rng = np.random.default_rng(0)
        res, signal = honest_residual_samples(dt, rng)
        A, B = _AB(np.random.default_rng(0))
        inf_b = calibrated_threshold(A, B, dt, mode="inf", safety=8.0)
        l2_b = calibrated_threshold(A, B, dt, mode="l2", safety=8.0)
        sig = float(signal.mean())
        usable = "yes" if inf_b < sig else "NO (>signal)"
        vals = [dt, f"{res.max():.3g}", f"{inf_b:.3g}", f"{l2_b:.3g}", f"{sig:.3g}", usable]
        print("".join(v.ljust(18) for v in vals))
    print("\nThe O(n^2) inf-bound is rigorous (no false positives) but at bf16/fp16 it exceeds")
    print("the signal, so it cannot catch any cheat. fp32 brings it far below the signal.\n")


def min_detectable_cheat(dtype, threshold_fn, rng, trials=TRIALS, probes=PROBES):
    """Smallest relative cheat rho with detection >= 0.99 under a thresholding strategy.

    threshold_fn(A, B, C_honest, residual_samples) -> tau.
    """
    rhos = np.geomspace(1e-7, 2.0, 22)
    for rho in rhos:
        detected = 0
        for _ in range(trials):
            A, B = _AB(rng)
            C = node_matmul(A, B, in_dtype=dtype, out_dtype=dtype).astype(np.float64)
            calib = freivalds_residual(A, B, C, rounds=probes, rng=rng)  # honest samples
            tau = threshold_fn(A, B, C, calib)
            delta = rng.normal(size=C.shape)
            delta *= rho * np.linalg.norm(C) / np.linalg.norm(delta)
            r = freivalds_residual(A, B, C + delta, rounds=probes, rng=rng)
            detected += int(np.any(r > tau))
        if detected / trials >= 0.99:
            return rho
    return float("inf")


def table2():
    print("=== Table 2: smallest detectable relative cheat  rho* = ||Delta||/||C|| ===\n")

    def statistical(A, B, C, calib):
        return calib.mean() + 8.0 * calib.std()

    strategies = {
        "rigorous O(n^2) bound": "rigorous",
        "statistical (mean+8sd)": statistical,
    }
    cols = ["dtype"] + list(strategies)
    print("".join(c.ljust(26) for c in cols))
    print("-" * (26 * len(cols)))
    def rigorous_threshold(A, B, C, calib, _dt):
        return calibrated_threshold(A, B, _dt, mode="inf", safety=8.0)

    for dt in DTYPES:
        row = [dt]
        for strat in strategies.values():
            rng = np.random.default_rng(1)
            if strat == "rigorous":
                def fn(A, B, C, calib, _dt=dt):
                    return rigorous_threshold(A, B, C, calib, _dt)
            else:
                fn = strat
            rho = min_detectable_cheat(dt, fn, rng)
            row.append("undetectable" if rho == float("inf") else f"{rho:.2e}")
        print("".join(v.ljust(26) for v in row))
    print("\nLower rho* = finer cheats caught. The rigorous bound is unusable at low precision;")
    print("statistical calibration is tight but carries a small false-positive / gaming risk.")
    print("Across precisions rho* tracks the unit roundoff:")
    for dt in DTYPES:
        print(f"  u[{dt}] = {UNIT_ROUNDOFF[dt]:.2e}")


if __name__ == "__main__":
    table1()
    table2()
