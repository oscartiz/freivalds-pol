"""Multi-round DeMo training, with an optional budget-constrained adversary.

This drives the hardest question in the design: a cheater who stays *below* the per-step
detection threshold every round is never caught -- but does its effect accumulate? DeMo's
error feedback is built to preserve gradient information across rounds, so it also faithfully
preserves an injected bias. This module runs the real transformer block under DeMo
compression for many rounds so the accumulation can be measured (`experiments/multiround.py`).
"""

from __future__ import annotations

import numpy as np

from .compressor import compress
from .transformer import TransformerBlock, block_step, forward

ORDER = ("Wq", "Wk", "Wv", "Wo", "W1", "W2", "g1", "g2")


def flatten_params(blk) -> np.ndarray:
    return np.concatenate([getattr(blk, k).ravel() for k in ORDER])


def _apply(blk, vec, lr):
    i = 0
    for k in ORDER:
        p = getattr(blk, k)
        p -= lr * vec[i:i + p.size].reshape(p.shape)
        i += p.size


def set_params(blk, vec):
    """Overwrite a block's parameters in place from a flattened vector (in ORDER)."""
    i = 0
    for k in ORDER:
        p = getattr(blk, k)
        p[...] = vec[i:i + p.size].reshape(p.shape)
        i += p.size
    return blk


def block_from_flat(vec, d, d_ff):
    blk = TransformerBlock.init(d, d_ff, np.random.default_rng(0))
    return set_params(blk, vec)


def eval_loss(blk, *, task_seed=100, d=64, d_ff=128, T=32, teacher_d_ff=None,
              batches=8, seed=999) -> float:
    """Held-out loss against the same teacher on fresh (test) batches."""
    teacher = TransformerBlock.init(d, teacher_d_ff or d_ff, np.random.default_rng(task_seed))
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(batches):
        X = rng.normal(size=(T, d))
        Y, _, _ = forward(teacher, X)
        Y = Y + 0.01 * rng.normal(size=Y.shape)
        loss, _, _ = block_step(blk, X, Y)
        tot += loss
    return tot / batches


def run_training(R, *, adversary=None, hook=None, seed=0, task_seed=100, lr=0.05, decay=0.9,
                 tile=64, k=8, d=64, d_ff=128, T=32, teacher_d_ff=None, trajectory=False):
    """Train one block for R rounds with DeMo compression. ``adversary(g, t) -> g'`` may
    perturb the gradient each round; ``hook(t, blk)`` is called at the start of each round
    (e.g. to estimate a curvature direction). ``teacher_d_ff`` (default = ``d_ff``) sets the
    teacher's width independently, so a wide student can fit a low-rank task (the
    over-parameterized regime). Returns losses, final params, grad norms, and optionally the
    per-round parameter trajectory."""
    blk = TransformerBlock.init(d, d_ff, np.random.default_rng(seed))
    teacher = TransformerBlock.init(d, teacher_d_ff or d_ff, np.random.default_rng(task_seed))
    batch_rng = np.random.default_rng(task_seed + 1)
    n = flatten_params(blk).size
    m_prev = np.zeros(n)

    losses, gnorms, traj = [], [], []
    for t in range(R):
        if hook is not None:
            hook(t, blk)
        X = batch_rng.normal(size=(T, d))
        Y, _, _ = forward(teacher, X)
        Y = Y + 0.01 * batch_rng.normal(size=Y.shape)

        loss, grads, _ = block_step(blk, X, Y)
        g = np.concatenate([grads[kk].ravel() for kk in ORDER])
        gnorms.append(float(np.linalg.norm(g)))           # honest grad norm, pre-cheat
        if adversary is not None:
            g = adversary(g, t)

        _, m_next, applied = compress(g, m_prev, decay=decay, tile=tile, k=k)
        _apply(blk, applied, lr)
        m_prev = m_next[:n]

        losses.append(loss)
        if trajectory:
            traj.append(flatten_params(blk).astype(np.float32))

    return dict(losses=np.array(losses), final=flatten_params(blk),
                traj=(np.array(traj) if trajectory else None),
                gnorms=np.array(gnorms), n_params=n)


def aligned_adversary(budget_frac, direction):
    """Inject a fixed-direction bias of norm ``budget_frac * ||g||`` every round (worst case)."""
    direction = direction / np.linalg.norm(direction)

    def adv(g, t):
        return g + (budget_frac * np.linalg.norm(g)) * direction

    return adv


def random_adversary(budget_frac, seed=0):
    """Inject a *random*-direction perturbation of the same per-round budget (washes out)."""
    rng = np.random.default_rng(seed)

    def adv(g, t):
        d = rng.normal(size=g.size)
        d /= np.linalg.norm(d)
        return g + (budget_frac * np.linalg.norm(g)) * d

    return adv


def trigger_loss(blk, x_trig, y_trig) -> float:
    """MSE of the block's output on a trigger input against the adversary's target."""
    yhat, _, _ = forward(blk, x_trig)
    return float(np.mean((yhat - y_trig) ** 2))


def backdoor_attack(start, budget_frac, x_trig, y_trig):
    """Sub-threshold backdoor: each round inject the (normalized) gradient that drives the
    trigger output toward ``y_trig``. Returns (adversary, hook). The hook recomputes the
    backdoor gradient at the current parameters; the adversary adds a budget-limited dose so
    the per-round cheat stays under the detection threshold."""
    state = {"g": None}

    def hook(t, blk):
        if t >= start:
            _, grads, _ = block_step(blk, x_trig, y_trig)
            v = np.concatenate([grads[kk].ravel() for kk in ORDER])
            nv = np.linalg.norm(v)
            state["g"] = v / nv if nv > 0 else None

    def adv(g, t):
        if t >= start and state["g"] is not None:
            return g + (budget_frac * np.linalg.norm(g)) * state["g"]
        return g

    return adv, hook
