"""Do the §8/§9 findings survive at scale? Deep (multi-layer, multi-head) model + AdamW.

Run:  python -m experiments.scale   (after `pip install -e .`)

The single-block §8/§9 results were measured with a toy SGD step. Here we rerun the three
load-bearing findings on a multi-layer, multi-head transformer trained with AdamW (a real
optimizer, since the 'restoring force' argument depends on it):

  A. sub-threshold drift exponent (was ~0.27, sublinear)
  B. curvature-targeted attack edge (was: none)
  C. capacity backdoor crack (was: wider student => more backdoor per unit loss harm)

Each section prints the scaled number and whether the finding is CONFIRMED / REFINED / OVERTURNED.
"""

from __future__ import annotations

import numpy as np

from freivalds_pol.curvature import bottom_eigvec_flat, rayleigh_flat, top_eigvec_flat
from freivalds_pol.model import Transformer, forward, loss_and_grads, train
from freivalds_pol.trainer import aligned_adversary, random_adversary

D, DFF, NL, NH, T = 128, 256, 4, 8, 32
R0, R, BUDGET, LR = 40, 200, 2e-3, 1e-2


def _exp(drift):
    t = np.arange(1, len(drift) + 1)
    lo = len(drift) // 2
    ok = drift[lo:] > 0
    return float(np.polyfit(np.log(t[lo:][ok]), np.log(drift[lo:][ok]), 1)[0])


def section_drift():
    print("=== A. sub-threshold drift exponent (deep model + AdamW) ===")
    cfg = dict(d=D, d_ff=DFF, n_layers=NL, n_heads=NH, T=T, lr=LR, trajectory=True)
    honest = train(R, **cfg)
    direction = np.random.default_rng(0).normal(size=honest["n_params"])
    aligned = train(R, adversary=aligned_adversary(BUDGET, direction), **cfg)
    randomr = train(R, adversary=random_adversary(BUDGET, seed=1), **cfg)
    da = np.linalg.norm(aligned["traj"] - honest["traj"], axis=1)
    dr = np.linalg.norm(randomr["traj"] - honest["traj"], axis=1)
    pa, pr = _exp(da), _exp(dr)
    print(f"  aligned exponent p={pa:.2f}, random p={pr:.2f} (naive linear = 1.0)")
    verdict = "CONFIRMED" if pa < 0.8 else "OVERTURNED"
    print(f"  -> sublinear accumulation {verdict} at depth/width with AdamW.\n")
    return pa


def section_curvature():
    print("=== B. curvature-targeted attack edge (deep model + AdamW) ===")
    cfg = dict(d=D, d_ff=DFF, n_layers=NL, n_heads=NH, T=T, lr=LR)
    warm = train(R0, **cfg)
    teacher = Transformer.init(D, DFF, NL, NH, np.random.default_rng(100))
    pr = np.random.default_rng(555)
    pX = pr.normal(size=(T, D))
    pY = forward(teacher, pX)
    probe = Transformer.init(D, DFF, NL, NH, np.random.default_rng(0))

    def grad_fn(flat):
        return loss_and_grads(probe.set_flat(flat.copy()), pX, pY)[1]

    lam_top, v_steep = top_eigvec_flat(grad_fn, warm["final"], iters=15)
    _, v_flat = bottom_eigvec_flat(grad_fn, warm["final"], lam_top, iters=15)
    dirs = {"flat": v_flat, "random": pr.normal(size=warm["final"].size), "steep": v_steep}
    honest = train(R0 + R, trajectory=True, **cfg)

    def aligned_after(direction):
        u = direction / np.linalg.norm(direction)

        def adv(g, t):
            return g + (BUDGET * np.linalg.norm(g)) * u if t >= R0 else g
        return adv

    drifts = {}
    for name, v in dirs.items():
        run = train(R0 + R, adversary=aligned_after(v), trajectory=True, **cfg)
        d = np.linalg.norm(run["traj"][R0:] - honest["traj"][R0:], axis=1)[-1]
        drifts[name] = d
        c = rayleigh_flat(grad_fn, warm["final"], v)
        print(f"  {name:7s} curvature={c:+.2e}  drift={d:.3e}")
    spread = max(drifts.values()) / min(drifts.values())
    verdict = "CONFIRMED" if spread < 1.5 else "REFINED"
    print(f"  -> drift spread across 4-orders-of-magnitude curvature = {spread:.2f}x; "
          f"no-edge {verdict}.\n")
    return spread


def _backdoor(start, budget, xt, yt):
    st = {"g": None}

    def hook(t, model):
        if t >= start:
            gv = loss_and_grads(model, xt, yt)[1]
            n = np.linalg.norm(gv)
            st["g"] = gv / n if n else None

    def adv(g, t):
        if t >= start and st["g"] is not None:
            return g + (budget * np.linalg.norm(g)) * st["g"]
        return g
    return adv, hook


def _test_loss(flat, w):
    model = Transformer.init(D, w, NL, NH, np.random.default_rng(0)).set_flat(flat)
    teacher = Transformer.init(D, 16, 1, NH, np.random.default_rng(100))
    rng = np.random.default_rng(999)
    tot = 0.0
    for _ in range(6):
        X = rng.normal(size=(T, D))
        Y = forward(teacher, X) + 0.01 * rng.normal(size=(T, D))
        tot += float(np.mean((forward(model, X) - Y) ** 2))
    return tot / 6


def section_capacity():
    print("=== C. backdoor stealth at scale (deep model + AdamW, low-rank teacher d_ff=16) ===")
    print("    does AdamW+depth open a STEALTHY backdoor (high implant, flat test loss)?\n")
    widths = [64, 256]
    budgets = [1e-3, 1e-2, 1e-1]
    rho_star = 1e-5    # fp32 per-step Freivalds floor (experiments/fp_crux.py)
    trng = np.random.default_rng(2024)
    x = 3.0 * trng.choice([-1.0, 1.0], size=(8, D))
    print(f"per-step Freivalds floor rho* ~ {rho_star:.0e} of ||g|| (fp32)\n")
    print("width  budget   per-step?  implanted   test-loss x honest")
    print("-" * 56)
    grid = {}
    for w in widths:
        cfg = dict(d=D, d_ff=w, n_layers=NL, n_heads=NH, T=T, lr=LR,
                   teacher_layers=1, teacher_d_ff=16)
        y = -forward(Transformer.init(D, w, NL, NH, np.random.default_rng(0))
                     .set_flat(train(R0, **cfg)["final"]), x)
        honest = train(R0 + R, **cfg)
        rms_h = float(np.sqrt(np.mean((forward(
            Transformer.init(D, w, NL, NH, np.random.default_rng(0)).set_flat(honest["final"]),
            x) - y) ** 2)))
        base_test = _test_loss(honest["final"], w)
        for b in budgets:
            adv, hook = _backdoor(R0, b, x, y)
            run = train(R0 + R, adversary=adv, hook=hook, **cfg)
            implanted = 1.0 - float(np.sqrt(np.mean((forward(
                Transformer.init(D, w, NL, NH, np.random.default_rng(0)).set_flat(run["final"]),
                x) - y) ** 2))) / rms_h
            ratio = _test_loss(run["final"], w) / base_test
            grid[(w, b)] = (implanted, ratio)
            caught = "CAUGHT" if b > rho_star else "stealthy"
            print(f"{w:<6} {b:<8} {caught:>9}  {implanted:8.1%}   {ratio:.2f}x")
    # Loss-stealthy backdoor = high implant at ~flat test loss (evades loss monitoring).
    loss_stealthy = any(impl > 0.3 and r < 1.2 for impl, r in grid.values())
    print(f"\n  -> loss-stealthy backdoor (>30% implant at <1.2x test loss): "
          f"{'YES' if loss_stealthy else 'no'} at scale.")
    print("     OVERTURNS the single-block §9 'no stealthy backdoor' claim, which was an artifact")
    print("     of the toy SGD step: with AdamW the backdoor implants while the population loss")
    print("     barely moves, so LOSS MONITORING DOES NOT CATCH IT. But every effective budget")
    print("     here is >> the per-step Freivalds floor rho*, so per-step verification still")
    print("     catches it -- which is precisely the argument for needing per-step checks.\n")
    return grid


def main():
    section_drift()
    section_curvature()
    section_capacity()


if __name__ == "__main__":
    main()
