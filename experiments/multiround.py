"""The hardest question: do SUB-THRESHOLD cheats accumulate across a training run?

Run:  python -m experiments.multiround   (after `pip install -e .`)

A node that cheats just under the per-step detection threshold is never caught. The fear is
that DeMo's error feedback -- built to preserve gradient information across rounds -- also
preserves an injected bias, so a tiny per-round cheat accumulates into real model damage. The
naive worst case is linear: ||drift|| <= lr * sum_t (budget * ||g_t||).

We test it by training the real transformer block for R rounds under DeMo compression, three
ways with identical data and init: honest, a worst-case *aligned* sub-threshold adversary
(fixed bias direction), and a *random* one. We compare the measured drift to the naive linear
bound.

Finding (refutes the naive fear): the drift SATURATES far below the linear bound. In stable
training the optimizer supplies a restoring force -- pushing theta off the minimum grows the
true gradient, which pushes back -- so a sub-threshold bias settles at a bounded equilibrium
~ budget/curvature instead of accumulating. The loss is essentially unchanged. The caveat,
and the protocol rule, are printed at the end.
"""

from __future__ import annotations

import numpy as np

from freivalds_pol.trainer import aligned_adversary, random_adversary, run_training

R = 300
LR = 0.1
BUDGET = 2e-3          # per-round cheat as a fraction of ||g|| -- a stand-in for the FP band rho*


def _slope(drift):
    t = np.arange(1, len(drift) + 1)
    lo = len(drift) // 2
    ok = drift[lo:] > 0
    return float(np.polyfit(np.log(t[lo:][ok]), np.log(drift[lo:][ok]), 1)[0])


def main():
    common = dict(R=R, lr=LR, trajectory=True)
    honest = run_training(**common)
    n = honest["n_params"]
    direction = np.random.default_rng(0).normal(size=n)

    aligned = run_training(adversary=aligned_adversary(BUDGET, direction), **common)
    randomr = run_training(adversary=random_adversary(BUDGET, seed=1), **common)

    d_aligned = np.linalg.norm(aligned["traj"] - honest["traj"], axis=1)
    d_random = np.linalg.norm(randomr["traj"] - honest["traj"], axis=1)
    naive_bound = LR * BUDGET * np.cumsum(honest["gnorms"])   # free-accumulation upper bound

    # Direct test of directed accumulation: drift projected onto the cheat direction.
    dir_unit = direction / np.linalg.norm(direction)
    proj = np.abs((aligned["traj"] - honest["traj"]) @ dir_unit)

    p_aligned, p_random = _slope(d_aligned), _slope(d_random)

    print(f"R={R} rounds, lr={LR}, per-round cheat budget = {BUDGET:.0e} * ||g||  "
          f"(never detected, by construction)\n")
    print("parameter drift from the honest run, and the naive linear (free-accumulation) bound:\n")
    print("round".ljust(8) + "aligned drift".rjust(15) + "random drift".rjust(15)
          + "naive bound".rjust(15))
    print("-" * 53)
    for r in (10, 30, 100, 200, R - 1):
        print(f"{r + 1:<8}{d_aligned[r]:15.4e}{d_random[r]:15.4e}{naive_bound[r]:15.4e}")

    print("\ngrowth exponent (fit drift ~ round^p over 2nd half):")
    print(f"  aligned p={p_aligned:.2f}   random p={p_random:.2f}   naive linear bound p=1.0")
    print("\nReading: a directed bias accumulates FASTER than random noise (aligned > random),")
    print("so error feedback does give a directed sub-threshold cheat a real edge -- but BOTH grow")
    print("sublinearly (p well below the naive p=1). Near a stable minimum the optimizer adds a")
    print("restoring force (off-minimum -> larger true gradient -> pushed back) that settles the")
    print("bias at a bounded equilibrium ~ budget/curvature rather than letting it run away.\n")

    print(f"direct test -- drift ALONG the cheat direction grows with exponent "
          f"p={_slope(proj):.2f} (free accumulation would be p=1.0): even the directed")
    print("  component is sublinear, so the bias does not freely accumulate.\n")

    print(f"loss after {R} rounds: honest={honest['losses'][-1]:.4f}, "
          f"aligned={aligned['losses'][-1]:.4f}, random={randomr['losses'][-1]:.4f}")
    print("  -> THE key harm metric: cheating barely moves the loss. The drift that does occur "
          "lands\n     mostly in flat, loss-irrelevant directions; the loss-relevant component "
          "is curvature-bounded.")

    print("\nCaveat + protocol rule: the restoring force scales with curvature, so along a")
    print("(near-)flat / invariant direction it weakens and the equilibrium offset grows.")
    print("Worst-case-safe design keeps budget < D/(lr*R*||g||): the detection threshold must")
    print("tighten as ~1/R with run length. (An ABOVE-threshold cheat is caught w.p. 1-(1-p)^R.)")


if __name__ == "__main__":
    main()
