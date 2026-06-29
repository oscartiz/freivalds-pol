"""A real transformer-block training step (numpy), so the verifier runs on genuine attention
and projection GEMMs -- the matmuls a Psyche/DisTrO node actually computes.

A Llama-flavoured pre-norm block on one sequence of ``T`` tokens, width ``d``:

    Xn1 = RMSNorm(X, g1)
    Q,K,V = Xn1 @ Wq, Xn1 @ Wk, Xn1 @ Wv
    P   = softmax( (Q Kᵀ) / sqrt(d) + causal_mask )
    X1  = X + (P V) @ Wo                          # residual
    Xn2 = RMSNorm(X1, g2)
    Y   = X1 + GELU(Xn2 @ W1) @ W2                 # residual

Eight GEMMs are recorded per step -- the six weight projections plus the two data-dependent
attention products ``Q Kᵀ`` and ``P V`` -- and packaged as a ``StepTranscript``. Backprop is
hand-derived and validated against finite differences by ``grad_check`` (single head keeps
every product a clean 2-D matmul; multi-head is the same GEMMs repeated per head).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .commitments import hash_array
from .transcript import MatMulRecord, StepTranscript

_C = np.sqrt(2.0 / np.pi)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(_C * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    u = _C * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    du = _C * (1.0 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du


def rmsnorm(x, g, eps=1e-6):
    r = np.sqrt((x ** 2).mean(axis=1, keepdims=True) + eps)   # (T,1)
    y = (x / r) * g
    return y, (x, r, g)


def rmsnorm_backward(dy, cache):
    x, r, g = cache
    d = x.shape[1]
    dn = dy * g                                               # grad wrt normalized x
    dx = dn / r - x * ((dn * x).sum(axis=1, keepdims=True) / (d * r ** 3))
    dg = (dy * (x / r)).sum(axis=0)
    return dx, dg


def _softmax_causal(S):
    T = S.shape[0]
    masked = np.where(np.triu(np.ones((T, T), bool), k=1), -1e30, S)
    e = np.exp(masked - masked.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class TransformerBlock:
    Wq: np.ndarray
    Wk: np.ndarray
    Wv: np.ndarray
    Wo: np.ndarray
    W1: np.ndarray
    W2: np.ndarray
    g1: np.ndarray
    g2: np.ndarray

    @staticmethod
    def init(d, d_ff, rng):
        s = 1.0 / np.sqrt(d)
        return TransformerBlock(
            Wq=rng.normal(size=(d, d)) * s, Wk=rng.normal(size=(d, d)) * s,
            Wv=rng.normal(size=(d, d)) * s, Wo=rng.normal(size=(d, d)) * s,
            W1=rng.normal(size=(d, d_ff)) * s, W2=rng.normal(size=(d_ff, d)) / np.sqrt(d_ff),
            g1=np.ones(d), g2=np.ones(d),
        )

    def params_root(self) -> bytes:
        return hash_array(np.concatenate([p.ravel() for p in
                          (self.Wq, self.Wk, self.Wv, self.Wo, self.W1, self.W2,
                           self.g1, self.g2)]))


def make_task(d, d_ff, T, rng):
    """Inputs X and targets Y from a fixed random teacher block (a genuine sequence task)."""
    teacher = TransformerBlock.init(d, d_ff, rng)
    X = rng.normal(size=(T, d))
    Y, _, _ = forward(teacher, X)
    return X, Y + 0.01 * rng.normal(size=Y.shape)


def forward(blk, X):
    d = X.shape[1]
    scale = 1.0 / np.sqrt(d)
    Xn1, c1 = rmsnorm(X, blk.g1)
    Q, K, V = Xn1 @ blk.Wq, Xn1 @ blk.Wk, Xn1 @ blk.Wv
    Sraw = Q @ K.T
    P = _softmax_causal(Sraw * scale)
    Ctx = P @ V
    Aout = Ctx @ blk.Wo
    X1 = X + Aout
    Xn2, c2 = rmsnorm(X1, blk.g2)
    Hpre = Xn2 @ blk.W1
    Hact = gelu(Hpre)
    Mout = Hact @ blk.W2
    Y = X1 + Mout
    cache = dict(X=X, Xn1=Xn1, c1=c1, Q=Q, K=K, V=V, Sraw=Sraw, P=P, Ctx=Ctx,
                 X1=X1, Xn2=Xn2, c2=c2, Hpre=Hpre, Hact=Hact, scale=scale)
    records = [
        MatMulRecord("attn.Q", Xn1, blk.Wq, Q),
        MatMulRecord("attn.K", Xn1, blk.Wk, K),
        MatMulRecord("attn.V", Xn1, blk.Wv, V),
        MatMulRecord("attn.scores", Q, K.T, Sraw),
        MatMulRecord("attn.ctx", P, V, Ctx),
        MatMulRecord("attn.out", Ctx, blk.Wo, Aout),
        MatMulRecord("mlp.h", Xn2, blk.W1, Hpre),
        MatMulRecord("mlp.y", Hact, blk.W2, Mout),
    ]
    return Y, cache, records


def _loss_only(blk, X, Yt) -> float:
    Y, _, _ = forward(blk, X)
    return float(np.mean((Y - Yt) ** 2))


def block_step(blk, X, Yt, *, dtype="fp32"):
    """Forward + backward for MSE against Yt. Returns (loss, grads, [MatMulRecord])."""
    Y, c, recs = forward(blk, X)
    for r in recs:
        r.dtype = dtype
    T, d = Y.shape
    loss = float(np.mean((Y - Yt) ** 2))
    dY = (2.0 / (T * d)) * (Y - Yt)

    # Y = X1 + Mout
    dX1 = dY.copy()
    dW2 = c["Hact"].T @ dY
    dHact = dY @ blk.W2.T
    dHpre = dHact * gelu_grad(c["Hpre"])
    dW1 = c["Xn2"].T @ dHpre
    dXn2 = dHpre @ blk.W1.T
    dX1_n2, dg2 = rmsnorm_backward(dXn2, c["c2"])
    dX1 += dX1_n2

    # X1 = X + Aout
    dAout = dX1
    dWo = c["Ctx"].T @ dAout
    dCtx = dAout @ blk.Wo.T
    dP = dCtx @ c["V"].T
    dV = c["P"].T @ dCtx
    dS = c["P"] * (dP - (dP * c["P"]).sum(axis=1, keepdims=True))   # softmax backward
    dSraw = dS * c["scale"]
    dQ = dSraw @ c["K"]
    dK = dSraw.T @ c["Q"]
    dWq = c["Xn1"].T @ dQ
    dWk = c["Xn1"].T @ dK
    dWv = c["Xn1"].T @ dV
    dXn1 = dQ @ blk.Wq.T + dK @ blk.Wk.T + dV @ blk.Wv.T
    _, dg1 = rmsnorm_backward(dXn1, c["c1"])

    grads = dict(Wq=dWq, Wk=dWk, Wv=dWv, Wo=dWo, W1=dW1, W2=dW2, g1=dg1, g2=dg2)
    return loss, grads, recs


def step_transcript(blk, X, Yt, *, node_id="node-1", seed=1, dtype="fp32"):
    loss, grads, records = block_step(blk, X, Yt, dtype=dtype)
    update = np.concatenate([grads[k].ravel() for k in
                             ("Wq", "Wk", "Wv", "Wo", "W1", "W2", "g1", "g2")])
    transcript = StepTranscript(
        node_id=node_id, seed=seed,
        theta_root=blk.params_root(),
        shard_root=hash_array(np.concatenate([X.ravel(), Yt.ravel()])),
        update=update, matmuls=records,
    )
    return transcript, loss, grads


def grad_check(blk, X, Yt, *, eps=1e-6, samples=6) -> float:
    """Max relative error between analytic and finite-difference gradients (sampled entries)."""
    _, grads, _ = block_step(blk, X, Yt)
    worst = 0.0
    for name in ("Wq", "Wk", "Wv", "Wo", "W1", "W2", "g1", "g2"):
        flat = getattr(blk, name).ravel()       # view: writes go back into the param
        gflat = grads[name].ravel()
        for idx in np.unique(np.linspace(0, flat.size - 1, samples).astype(int)):
            orig = flat[idx]
            flat[idx] = orig + eps
            lp = _loss_only(blk, X, Yt)
            flat[idx] = orig - eps
            lm = _loss_only(blk, X, Yt)
            flat[idx] = orig
            fd = (lp - lm) / (2 * eps)
            denom = max(1e-8, abs(fd) + abs(gflat[idx]))
            worst = max(worst, abs(fd - gflat[idx]) / denom)
    return worst
