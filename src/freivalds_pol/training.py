"""A real training step (numpy), so the verifier runs on genuine gradients, not random matmuls.

A two-layer MLP  ``X -> H = relu(X W1) -> Yhat = H W2``  is trained with MSE against a fixed
random *teacher* network. One step's forward and backward passes are recorded as the five
GEMMs a node actually computes, packaged as a ``StepTranscript`` whose ``update`` is the
concatenated gradient. ``grad_check`` confirms the backprop (and therefore the recorded
matmuls) is correct via finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .commitments import hash_array
from .transcript import MatMulRecord, StepTranscript


def _relu(x):
    return np.maximum(x, 0.0)


@dataclass
class MLP:
    W1: np.ndarray
    W2: np.ndarray

    @staticmethod
    def init(d_in, d_hidden, d_out, rng):
        return MLP(
            W1=rng.normal(size=(d_in, d_hidden)) / np.sqrt(d_in),
            W2=rng.normal(size=(d_hidden, d_out)) / np.sqrt(d_hidden),
        )

    def params_root(self) -> bytes:
        return hash_array(np.concatenate([self.W1.ravel(), self.W2.ravel()]))


def make_task(d_in, d_hidden, d_out, batch, rng):
    """Inputs X and targets Y from a fixed random teacher MLP (a genuine regression task)."""
    teacher = MLP.init(d_in, d_hidden, d_out, rng)
    X = rng.normal(size=(batch, d_in))
    Y = _relu(X @ teacher.W1) @ teacher.W2 + 0.01 * rng.normal(size=(batch, d_out))
    return X, Y


def _loss_only(mlp, X, Y) -> float:
    Yhat = _relu(X @ mlp.W1) @ mlp.W2
    return float(np.mean((Yhat - Y) ** 2))


def training_step(mlp, X, Y, *, dtype="fp32"):
    """Forward + backward for MSE loss. Returns (loss, grads, [MatMulRecord] for every GEMM)."""
    Z1 = X @ mlp.W1                                    # GEMM 1 (forward)
    H = _relu(Z1)
    Yhat = H @ mlp.W2                                  # GEMM 2 (forward)
    resid = Yhat - Y
    n, d_out = Y.shape
    loss = float(np.mean(resid ** 2))

    dYhat = (2.0 / (n * d_out)) * resid               # d loss / d Yhat
    dW2 = H.T @ dYhat                                  # GEMM 3 (backward)
    dH = dYhat @ mlp.W2.T                              # GEMM 4 (backward)
    dZ1 = dH * (Z1 > 0)
    dW1 = X.T @ dZ1                                    # GEMM 5 (backward)

    records = [
        MatMulRecord("fwd.Z1", X, mlp.W1, Z1, dtype),
        MatMulRecord("fwd.Yhat", H, mlp.W2, Yhat, dtype),
        MatMulRecord("bwd.dW2", H.T, dYhat, dW2, dtype),
        MatMulRecord("bwd.dH", dYhat, mlp.W2.T, dH, dtype),
        MatMulRecord("bwd.dW1", X.T, dZ1, dW1, dtype),
    ]
    return loss, {"W1": dW1, "W2": dW2}, records


def step_transcript(mlp, X, Y, *, node_id="node-1", seed=1, dtype="fp32"):
    """Package one real training step as a StepTranscript (the node's claimed work)."""
    loss, grads, records = training_step(mlp, X, Y, dtype=dtype)
    update = np.concatenate([grads["W1"].ravel(), grads["W2"].ravel()])
    transcript = StepTranscript(
        node_id=node_id,
        seed=seed,
        theta_root=mlp.params_root(),
        shard_root=hash_array(np.concatenate([X.ravel(), Y.ravel()])),
        update=update,
        matmuls=records,
    )
    return transcript, loss, grads


def grad_check(mlp, X, Y, *, eps=1e-5, samples=25) -> float:
    """Max relative error between analytic and finite-difference gradients (sampled entries)."""
    _, grads, _ = training_step(mlp, X, Y)
    worst = 0.0
    for W, g in [(mlp.W1, grads["W1"]), (mlp.W2, grads["W2"])]:
        flat = W.ravel()           # view: writes go back into W
        gflat = g.ravel()
        for idx in np.unique(np.linspace(0, flat.size - 1, samples).astype(int)):
            orig = flat[idx]
            flat[idx] = orig + eps
            lp = _loss_only(mlp, X, Y)
            flat[idx] = orig - eps
            lm = _loss_only(mlp, X, Y)
            flat[idx] = orig
            fd = (lp - lm) / (2 * eps)
            denom = max(1e-8, abs(fd) + abs(gflat[idx]))
            worst = max(worst, abs(fd - gflat[idx]) / denom)
    return worst
