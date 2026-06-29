"""ZK matmul spot-check: a sumcheck argument for C = A·B, with honest cost accounting.

Run:  python -m experiments.zk_matmul   (after `pip install -e .`)

Proves one GEMM C = A·B over a prime field with the non-interactive sumcheck of `zk.py`, then
reports the cost honestly against two baselines:
  - recompute        (O(m·k·n))  -- the naive verifier
  - Freivalds        (O(n²))     -- this repo's main check (probabilistic, NOT private)

The point of this experiment is to be honest about what ZK buys and costs. The sumcheck *proof*
is succinct (3 field elements per contraction-bit round), but confirming the two final MLE
evaluations needs a polynomial commitment; the reference `RevealCommitment` recomputes them in
O(m·k)+O(k·n), so as prototyped the verifier is NOT faster than recompute and NOT hiding. A real
PCS (KZG/FRI) is required for the succinct, zero-knowledge win — and even then, ZK's value is
privacy + public verifiability, not beating Freivalds on raw speed.
"""

from __future__ import annotations

import time

import numpy as np

from freivalds_pol.zk import P, RevealCommitment, prove, verify_full, verify_sumcheck

M = K = N = 64        # power-of-two GEMM


def _rand(r, c, rng):
    return [[int(rng.integers(0, 1000)) for _ in range(c)] for _ in range(r)]


def _matmul(A, B):
    return [[sum(A[i][t] * B[t][j] for t in range(len(B)))  % P for j in range(len(B[0]))]
            for i in range(len(A))]


def _time(fn, reps=1):
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    return out, (time.perf_counter() - t0) / reps


def main():
    rng = np.random.default_rng(0)
    A, B = _rand(M, K, rng), _rand(K, N, rng)
    C, t_recompute = _time(lambda: _matmul(A, B))

    proof, t_prove = _time(lambda: prove(A, B, C))
    ok, _ = _time(lambda: verify_full(proof, A, B, RevealCommitment(), m=M, k=K, n=N))
    (sc_ok, _, _), t_sumcheck = _time(lambda: verify_sumcheck(proof, m=M, k=K, n=N))

    # Freivalds baseline (float), for scale reference
    Af, Bf, Cf = np.array(A, float), np.array(B, float), np.array(C, float)

    def freivalds():
        r = rng.choice([-1.0, 1.0], size=(N, 1))
        return np.abs(Af @ (Bf @ r) - Cf @ r).max()

    _, t_freivalds = _time(freivalds, reps=20)

    print(f"GEMM {M}x{K}x{N} over F_p (p = 2^61-1)\n")
    print(f"proof verifies: {ok}   (sumcheck chain alone: {sc_ok})")
    print(f"proof size: {3 * len(proof.rounds) + 2} field elements "
          f"({3 * len(proof.rounds)} round + 2 final), {len(proof.rounds)} rounds\n")
    print("cost (seconds):")
    print(f"  recompute  (O(mkn))          : {t_recompute:.4e}")
    print(f"  Freivalds  (O(n^2), float)   : {t_freivalds:.4e}")
    print(f"  zk prove                     : {t_prove:.4e}")
    print(f"  zk verify sumcheck (O(log k)): {t_sumcheck:.4e}")
    print("  zk verify full (+ reveal PCS): includes O(mk) MLE recompute (not succinct)\n")
    print("Honest reading:")
    print("  - The sumcheck proof is succinct in rounds and SOUND (tampered C is rejected).")
    print("  - With the reveal-PCS the verifier still recomputes the MLE openings, so it does")
    print("    NOT beat recompute and is NOT hiding. A real PCS (KZG/FRI) is required for the")
    print("    succinct + zero-knowledge win; that is future work (interface: zk.PolyCommitment).")
    print("  - Even with a real PCS, ZK costs far more than Freivalds; its value is PRIVACY and")
    print("    public verifiability (no interaction, no trusted verifier), which Freivalds lacks.")
    print("  - Floats need fixed-point encoding into F_p; ZK then proves the *quantized* matmul")
    print("    exactly, so the FP-threshold question (§5) moves into the quantization step.")


if __name__ == "__main__":
    main()
