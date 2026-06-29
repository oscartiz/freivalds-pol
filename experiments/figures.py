"""Regenerate every figure in figures/ from scratch.

Run:  python -m experiments.figures   (after `pip install -e ".[viz]"`)

One function per figure; each recomputes its data with the library and saves a PNG. Kept at
moderate sizes so the whole set regenerates in ~1 minute.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from freivalds_pol.adaptive import nullspace_cheat  # noqa: E402
from freivalds_pol.challenge import fiat_shamir_probes  # noqa: E402
from freivalds_pol.commitments import hash_array  # noqa: E402
from freivalds_pol.compressor import (  # noqa: E402
    cheat_bad_residual,
    cheat_fake_values,
    cheat_lazy,
    cheat_wrong_topk,
    compress,
    verify_compressed,
)
from freivalds_pol.curvature import bottom_eigvec, rayleigh, top_eigvec  # noqa: E402
from freivalds_pol.freivalds import freivalds_residual, freivalds_residual_with  # noqa: E402
from freivalds_pol.numerics import (  # noqa: E402
    UNIT_ROUNDOFF,
    calibrated_threshold,
    node_matmul,
)
from freivalds_pol.trainer import (  # noqa: E402
    aligned_adversary,
    backdoor_attack,
    block_from_flat,
    eval_loss,
    random_adversary,
    run_training,
    trigger_loss,
)
from freivalds_pol.transformer import TransformerBlock, forward  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.titlesize": 12})
BLUE, RED, GREEN, GREY = "#2c6fbb", "#c0392b", "#27ae60", "#7f8c8d"


def _save(fig, name):
    path = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# --- 1. FP crux: min detectable cheat scales with unit roundoff -------------------------
def fig_fp_crux():
    n, dtypes = 256, ["fp32", "fp16", "bf16"]
    rhos = np.geomspace(1e-7, 2.0, 20)
    xs, ys = [], []
    for dt in dtypes:
        rng = np.random.default_rng(1)
        found = np.nan
        for rho in rhos:
            det = 0
            for _ in range(30):
                A = rng.normal(size=(n, n)).astype(np.float32)
                B = rng.normal(size=(n, n)).astype(np.float32)
                C = node_matmul(A, B, in_dtype=dt, out_dtype=dt).astype(np.float64)
                calib = freivalds_residual(A, B, C, rounds=64, rng=rng)
                tau = calib.mean() + 8 * calib.std()
                d = rng.normal(size=C.shape)
                d *= rho * np.linalg.norm(C) / np.linalg.norm(d)
                det += int(np.any(freivalds_residual(A, B, C + d, rounds=64, rng=rng) > tau))
            if det / 30 >= 0.99:
                found = rho
                break
        xs.append(UNIT_ROUNDOFF[dt])
        ys.append(found)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.loglog(xs, ys, "o-", color=BLUE, label="measured $\\rho^*$")
    lo = min(xs)
    ax.loglog([lo, max(xs)], [ys[0] * 1, ys[0] * max(xs) / xs[0]], "--", color=GREY,
              label="linear in unit roundoff")
    for dt, x, y in zip(dtypes, xs, ys, strict=True):
        ax.annotate(dt, (x, y), textcoords="offset points", xytext=(6, -10))
    ax.set_xlabel("unit roundoff $u$ of checked layer")
    ax.set_ylabel("min detectable cheat $\\rho^*$ (rel.)")
    ax.set_title("FP crux: detection floor tracks precision")
    ax.legend()
    _save(fig, "fp_crux.png")


# --- 2. Adaptive: predictable probe breaks; commit-then-sample fixes --------------------
def fig_adaptive():
    n, k, rho, trials = 256, 4, 0.5, 30
    rng = np.random.default_rng(0)
    pred = fresh = 0
    for _ in range(trials):
        A = rng.normal(size=(n, n)).astype(np.float32)
        B = rng.normal(size=(n, n)).astype(np.float32)
        C = node_matmul(A, B, in_dtype="fp32", out_dtype="fp32").astype(np.float64)
        tau = calibrated_threshold(A, B, "fp32", mode="inf", safety=8.0)
        Rk = rng.choice([-1.0, 1.0], size=(n, k))
        Cp = nullspace_cheat(C, Rk, rho, rng=rng)
        pred += int(np.any(freivalds_residual_with(A, B, Cp, Rk) > tau))
        Rf = fiat_shamir_probes(hash_array(Cp), n, k)
        fresh += int(np.any(freivalds_residual_with(A, B, Cp, Rf) > tau))
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.bar(["predictable\nprobe", "commit-then-\nsample"], [pred / trials, fresh / trials],
           color=[RED, GREEN])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("detection rate")
    ax.set_title(f"Adaptive 50%-impact cheat (k={k})")
    for i, v in enumerate([pred / trials, fresh / trials]):
        ax.text(i, v + 0.03, f"{v:.0%}", ha="center")
    _save(fig, "adaptive.png")


# --- 3. Compressed update: per-tile spot-check detection --------------------------------
def fig_compressed():
    rng = np.random.default_rng(0)
    grad = rng.normal(size=20000)
    m_prev = rng.normal(size=grad.size) * 0.1
    cu, m_next, _ = compress(grad, m_prev, decay=0.9, tile=64, k=8)
    n_tiles = cu.indices.shape[0]
    cheats = {"lazy": cheat_lazy, "fake values": cheat_fake_values,
              "wrong top-k": cheat_wrong_topk, "bad residual": cheat_bad_residual}
    cs = np.arange(1, 9)
    frac = 0.25
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for name, cheat in cheats.items():
        ys = []
        for c in cs:
            caught = 0
            for s in range(120):
                r = np.random.default_rng(s)
                cu_c, mn_c = cheat(cu, m_next, frac=frac, rng=r)
                tiles = r.choice(n_tiles, c, replace=False)
                caught += int(not verify_compressed(grad, m_prev, mn_c, cu_c, tiles,
                                                    decay=0.9).accepted)
            ys.append(caught / 120)
        ax.plot(cs, ys, "o-", label=name, alpha=0.8)
    ax.plot(cs, 1 - (1 - frac) ** cs, "k--", label="$1-(1-f)^c$")
    ax.set_xlabel("challenged tiles $c$")
    ax.set_ylabel("detection rate")
    ax.set_title(f"Compressed update (DeMo), {frac:.0%} tiles corrupted")
    ax.legend(fontsize=9)
    _save(fig, "compressed.png")


# --- 4. Multi-round: sub-threshold drift is sublinear ----------------------------------
def fig_multiround():
    R, lr, budget = 300, 0.1, 2e-3
    honest = run_training(R, lr=lr, trajectory=True)
    n = honest["n_params"]
    direction = np.random.default_rng(0).normal(size=n)
    aligned = run_training(R, lr=lr, adversary=aligned_adversary(budget, direction),
                           trajectory=True)
    randomr = run_training(R, lr=lr, adversary=random_adversary(budget, seed=1),
                           trajectory=True)
    da = np.linalg.norm(aligned["traj"] - honest["traj"], axis=1)
    dr = np.linalg.norm(randomr["traj"] - honest["traj"], axis=1)
    bound = lr * budget * np.cumsum(honest["gnorms"])
    t = np.arange(1, R + 1)
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.loglog(t, da, color=RED, label="aligned cheat")
    ax.loglog(t, dr, color=BLUE, label="random cheat")
    ax.loglog(t, bound, "--", color=GREY, label="naive linear bound")
    ax.set_xlabel("training round")
    ax.set_ylabel("parameter drift from honest")
    ax.set_title("Sub-threshold cheats: sublinear, not linear")
    ax.legend(fontsize=9)
    _save(fig, "multiround.png")


# --- 5. Curvature attack: targeting flat directions gives no edge -----------------------
def fig_curvature():
    R0, R, budget = 40, 200, 2e-3
    D, DFF, T, TASK = 64, 128, 32, 100
    warm = run_training(R0, lr=0.1)
    blk = block_from_flat(warm["final"], D, DFF)
    teacher = TransformerBlock.init(D, DFF, np.random.default_rng(TASK))
    pr = np.random.default_rng(555)
    pX = pr.normal(size=(T, D))
    pY, _, _ = forward(teacher, pX)
    lam_top, v_steep = top_eigvec(blk, pX, pY, iters=22)
    _, v_flat = bottom_eigvec(blk, pX, pY, lam_top, iters=22)
    dirs = {"flat": v_flat, "random": pr.normal(size=warm["final"].size), "steep": v_steep}
    honest = run_training(R0 + R, lr=0.1, trajectory=True)
    base = eval_loss(block_from_flat(honest["final"], D, DFF))
    names, drifts, harms = [], [], []
    for name, v in dirs.items():
        v = v / np.linalg.norm(v)

        def adv(g, t, v=v):
            return g + (budget * np.linalg.norm(g)) * v if t >= R0 else g
        run = run_training(R0 + R, lr=0.1, adversary=adv, trajectory=True)
        names.append(f"{name}\n(c={rayleigh(blk, pX, pY, v):.2g})")
        drifts.append(np.linalg.norm(run["traj"][R0:] - honest["traj"][R0:], axis=1)[-1])
        harms.append(eval_loss(block_from_flat(run["final"], D, DFF)) - base)
    fig, ax = plt.subplots(figsize=(5.6, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, drifts, 0.4, color=BLUE, label="param drift")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, harms, 0.4, color=RED, label="test $\\Delta$loss")
    ax2.grid(False)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("parameter drift", color=BLUE)
    ax2.set_ylabel("test $\\Delta$loss", color=RED)
    ax.set_title("Curvature targeting gives no edge")
    _save(fig, "curvature.png")


# --- 6. Backdoor: stealth-vs-harm tradeoff --------------------------------------------
def fig_backdoor():
    R0, R = 40, 200
    D, DFF, T = 64, 128, 32
    budgets = [1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0]
    blk0 = block_from_flat(run_training(R0, lr=0.1)["final"], D, DFF)
    trng = np.random.default_rng(2024)
    x = 3.0 * trng.choice([-1.0, 1.0], size=(T, D))
    y_honest, _, _ = forward(blk0, x)
    y = -y_honest
    hb = block_from_flat(run_training(R0 + R, lr=0.1)["final"], D, DFF)
    rms_h = np.sqrt(trigger_loss(hb, x, y))
    base = eval_loss(hb)
    impl, ratio = [], []
    for b in budgets:
        adv, hook = backdoor_attack(R0, b, x, y)
        fb = block_from_flat(run_training(R0 + R, lr=0.1, adversary=adv, hook=hook)["final"],
                             D, DFF)
        impl.append(1 - np.sqrt(trigger_loss(fb, x, y)) / rms_h)
        ratio.append(eval_loss(fb) / base)
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.semilogx(budgets, impl, "o-", color=RED, label="backdoor implanted")
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.semilogx(budgets, ratio, "s--", color=BLUE, label="test loss (x honest)")
    ax.set_xlabel("per-round budget (fraction of $\\|g\\|$)")
    ax.set_ylabel("backdoor implanted", color=RED)
    ax2.set_ylabel("test loss / honest", color=BLUE)
    ax.set_title("No stealthy-and-effective backdoor")
    _save(fig, "backdoor.png")


# --- 7. Backdoor capacity: over-parameterization widens the stealth window --------------
def fig_backdoor_capacity():
    D, T, TDFF, R0, R = 64, 32, 8, 60, 180
    widths = [8, 64, 256, 1024]
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for b, col in [(0.1, BLUE), (0.3, RED)]:
        impl = []
        for w in widths:
            cfg = dict(d=D, d_ff=w, T=T, teacher_d_ff=TDFF)
            blk0 = block_from_flat(run_training(R0, lr=0.1, **cfg)["final"], D, w)
            trng = np.random.default_rng(2024)
            x = 3.0 * trng.choice([-1.0, 1.0], size=(8, D))
            yh, _, _ = forward(blk0, x)
            y = -yh
            hb = block_from_flat(run_training(R0 + R, lr=0.1, **cfg)["final"], D, w)
            rms_h = np.sqrt(trigger_loss(hb, x, y))
            adv, hook = backdoor_attack(R0, b, x, y)
            fb = block_from_flat(
                run_training(R0 + R, lr=0.1, adversary=adv, hook=hook, **cfg)["final"], D, w)
            impl.append(1 - np.sqrt(trigger_loss(fb, x, y)) / rms_h)
        ax.semilogx(widths, impl, "o-", color=col, label=f"budget {b}")
    ax.set_xlabel("student width (d_ff), low-rank teacher d_ff=8")
    ax.set_ylabel("backdoor implanted")
    ax.set_title("Over-parameterization widens the stealth window")
    ax.legend()
    _save(fig, "backdoor_capacity.png")


def main():
    for f in (fig_fp_crux, fig_adaptive, fig_compressed, fig_multiround,
              fig_curvature, fig_backdoor, fig_backdoor_capacity):
        f()


if __name__ == "__main__":
    main()
