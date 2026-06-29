"""Multi-layer, multi-head transformer with an AdamW path, for scaling the §8/§9 security
analysis beyond the single-block toy.

The single-block ``transformer.py`` stays the reference used by the verifier transcript; this
module generalizes the same Llama-style block (RMSNorm + multi-head causal attention + GELU MLP)
to ``n_layers`` layers and ``n_heads`` heads, and adds a real optimizer (AdamW) alongside plain
SGD -- the §8/§9 "restoring force" argument depends on the optimizer, so it must be tested under
a real one. Backprop is validated against finite differences by ``grad_check``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transformer import gelu, gelu_grad, rmsnorm, rmsnorm_backward

PNAMES = ("Wq", "Wk", "Wv", "Wo", "W1", "W2", "g1", "g2")


@dataclass
class Transformer:
    layers: list[dict]      # each a dict of the 8 param arrays
    n_heads: int
    d: int
    d_ff: int

    @staticmethod
    def init(d, d_ff, n_layers, n_heads, rng):
        assert d % n_heads == 0, "d must be divisible by n_heads"
        s = 1.0 / np.sqrt(d)
        layers = [{
            "Wq": rng.normal(size=(d, d)) * s, "Wk": rng.normal(size=(d, d)) * s,
            "Wv": rng.normal(size=(d, d)) * s, "Wo": rng.normal(size=(d, d)) * s,
            "W1": rng.normal(size=(d, d_ff)) * s,
            "W2": rng.normal(size=(d_ff, d)) / np.sqrt(d_ff),
            "g1": np.ones(d), "g2": np.ones(d),
        } for _ in range(n_layers)]
        return Transformer(layers, n_heads, d, d_ff)

    def flat(self) -> np.ndarray:
        return np.concatenate([layer[p].ravel() for layer in self.layers for p in PNAMES])

    def set_flat(self, vec):
        i = 0
        for layer in self.layers:
            for p in PNAMES:
                a = layer[p]
                a[...] = vec[i:i + a.size].reshape(a.shape)
                i += a.size
        return self

    @property
    def n_params(self) -> int:
        return sum(layer[p].size for layer in self.layers for p in PNAMES)


def _layer_forward(p, X, H):
    T, d = X.shape
    dh = d // H
    scale = 1.0 / np.sqrt(dh)
    Xn1, c1 = rmsnorm(X, p["g1"])
    Q, K, V = Xn1 @ p["Wq"], Xn1 @ p["Wk"], Xn1 @ p["Wv"]
    Qh = Q.reshape(T, H, dh).transpose(1, 0, 2)
    Kh = K.reshape(T, H, dh).transpose(1, 0, 2)
    Vh = V.reshape(T, H, dh).transpose(1, 0, 2)
    Sraw = np.einsum("htd,hsd->hts", Qh, Kh)
    S = np.where(np.triu(np.ones((T, T), bool), 1), -1e30, Sraw * scale)
    e = np.exp(S - S.max(axis=-1, keepdims=True))
    P = e / e.sum(axis=-1, keepdims=True)
    Ctxh = np.einsum("hts,hsd->htd", P, Vh)
    Ctx = Ctxh.transpose(1, 0, 2).reshape(T, d)
    Aout = Ctx @ p["Wo"]
    X1 = X + Aout
    Xn2, c2 = rmsnorm(X1, p["g2"])
    Hpre = Xn2 @ p["W1"]
    Hact = gelu(Hpre)
    Yl = X1 + Hact @ p["W2"]
    cache = (c1, Xn1, Qh, Kh, Vh, P, Ctx, X1, c2, Xn2, Hpre, Hact, scale, H, dh)
    return Yl, cache


def _layer_backward(p, dY, cache):
    c1, Xn1, Qh, Kh, Vh, P, Ctx, X1, c2, Xn2, Hpre, Hact, scale, H, dh = cache
    T, d = dY.shape
    dX1 = dY.copy()
    dW2 = Hact.T @ dY
    dHpre = (dY @ p["W2"].T) * gelu_grad(Hpre)
    dW1 = Xn2.T @ dHpre
    dX1_n2, dg2 = rmsnorm_backward(dHpre @ p["W1"].T, c2)
    dX1 += dX1_n2

    dWo = Ctx.T @ dX1
    dCtxh = (dX1 @ p["Wo"].T).reshape(T, H, dh).transpose(1, 0, 2)
    dP = np.einsum("htd,hsd->hts", dCtxh, Vh)
    dVh = np.einsum("hts,htd->hsd", P, dCtxh)
    dSraw = P * (dP - (dP * P).sum(axis=-1, keepdims=True)) * scale
    dQh = np.einsum("hts,hsd->htd", dSraw, Kh)
    dKh = np.einsum("hts,htd->hsd", dSraw, Qh)
    dQ = dQh.transpose(1, 0, 2).reshape(T, d)
    dK = dKh.transpose(1, 0, 2).reshape(T, d)
    dV = dVh.transpose(1, 0, 2).reshape(T, d)
    dWq, dWk, dWv = Xn1.T @ dQ, Xn1.T @ dK, Xn1.T @ dV
    dXn1 = dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T
    dX_n1, dg1 = rmsnorm_backward(dXn1, c1)
    dX = dX1 + dX_n1
    grads = {"Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo,
             "W1": dW1, "W2": dW2, "g1": dg1, "g2": dg2}
    return dX, grads


def forward(model, X):
    h = X
    for layer in model.layers:
        h, _ = _layer_forward(layer, h, model.n_heads)
    return h


def loss_and_grads(model, X, Y):
    caches, h = [], X
    for layer in model.layers:
        h, c = _layer_forward(layer, h, model.n_heads)
        caches.append(c)
    resid = h - Y
    T, d = Y.shape
    loss = float(np.mean(resid ** 2))
    dh = (2.0 / (T * d)) * resid
    grads = []
    for layer, c in zip(reversed(model.layers), reversed(caches), strict=True):
        dh, g = _layer_backward(layer, dh, c)
        grads.append(g)
    grads.reverse()
    flat = np.concatenate([g[p].ravel() for g in grads for p in PNAMES])
    return loss, flat


def grad_check(model, X, Y, *, eps=1e-6, samples=8) -> float:
    _, g = loss_and_grads(model, X, Y)
    base = model.flat()
    worst = 0.0
    for idx in np.unique(np.linspace(0, base.size - 1, samples).astype(int)):
        v = base.copy()
        v[idx] = base[idx] + eps
        lp = float(np.mean((forward(model.set_flat(v), X) - Y) ** 2))
        v[idx] = base[idx] - eps
        lm = float(np.mean((forward(model.set_flat(v), X) - Y) ** 2))
        model.set_flat(base)
        fd = (lp - lm) / (2 * eps)
        worst = max(worst, abs(fd - g[idx]) / max(1e-8, abs(fd) + abs(g[idx])))
    return worst


class AdamW:
    """Decoupled-weight-decay Adam, operating on the flat parameter vector."""

    def __init__(self, n, lr=1e-2, wd=1e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.wd, self.b1, self.b2, self.eps = lr, wd, b1, b2, eps
        self.m, self.v, self.t = np.zeros(n), np.zeros(n), 0

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        return params - self.lr * (mhat / (np.sqrt(vhat) + self.eps) + self.wd * params)


def make_task(d, d_ff, n_layers, n_heads, T, rng):
    teacher = Transformer.init(d, d_ff, n_layers, n_heads, rng)
    X = rng.normal(size=(T, d))
    return X, forward(teacher, X) + 0.01 * rng.normal(size=(T, d))


def train(R, *, d=64, d_ff=128, n_layers=2, n_heads=4, T=32, lr=1e-2,
          optimizer="adamw", adversary=None, hook=None, seed=0, task_seed=100,
          teacher_layers=None, teacher_d_ff=None, trajectory=False):
    """Train the deep model for R rounds with AdamW (or plain SGD). ``adversary(g, t) -> g'``
    perturbs the gradient each round; ``hook(t, model)`` runs at the start of each round (e.g.
    to compute a backdoor gradient at the current params). Returns losses, final flat params,
    grad norms, traj."""
    model = Transformer.init(d, d_ff, n_layers, n_heads, np.random.default_rng(seed))
    teacher = Transformer.init(d, teacher_d_ff or d_ff, teacher_layers or n_layers, n_heads,
                               np.random.default_rng(task_seed))
    batch_rng = np.random.default_rng(task_seed + 1)
    opt = AdamW(model.n_params, lr=lr) if optimizer == "adamw" else None

    losses, gnorms, traj = [], [], []
    for _ in range(R):
        if hook is not None:
            hook(len(losses), model)
        X = batch_rng.normal(size=(T, d))
        Y = forward(teacher, X) + 0.01 * batch_rng.normal(size=(T, d))
        loss, g = loss_and_grads(model, X, Y)
        gnorms.append(float(np.linalg.norm(g)))
        if adversary is not None:
            g = adversary(g, len(losses))
        p = model.flat()
        p = opt.step(p, g) if opt else (p - lr * g)
        model.set_flat(p)
        losses.append(loss)
        if trajectory:
            traj.append(model.flat().astype(np.float32))
    return dict(losses=np.array(losses), final=model.flat(),
                gnorms=np.array(gnorms), n_params=model.n_params,
                traj=(np.array(traj) if trajectory else None))
