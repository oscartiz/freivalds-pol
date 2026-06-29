"""Measure detection probability vs. number of challenges, and verifier cost vs recompute.

Run:  python -m experiments.run_detection   (after `pip install -e .`)

Reproduces the core table for the writeup: for each cheat in the threat model, the fraction
of cheating steps caught as the audit budget k (number of challenged matmuls) grows, plus
the verifier's asymptotic cost relative to full recompute-and-compare.
"""

from __future__ import annotations

import numpy as np

from freivalds_pol.adversary import lazy, poison_shard, wrong_compute
from freivalds_pol.challenge import sample_challenges
from freivalds_pol.transcript import MatMulRecord, StepTranscript
from freivalds_pol.verifier import verify_step

SHARD_ROOT = bytes([2]) * 32


def honest_transcript(n=128, layers=12, rng=None) -> StepTranscript:
    rng = np.random.default_rng() if rng is None else rng
    mms = []
    for i in range(layers):
        A = rng.normal(size=(n, n)).astype(np.float32)
        B = rng.normal(size=(n, n)).astype(np.float32)
        mms.append(MatMulRecord(f"layer{i}.gemm", A, B, A @ B))
    return StepTranscript(
        node_id="node-1",
        seed=int(rng.integers(0, 2**62)),
        theta_root=bytes([1]) * 32,
        shard_root=SHARD_ROOT,
        update=rng.normal(size=(n,)).astype(np.float32),
        matmuls=mms,
    )


def detection_rate(make_cheater, *, k, trials=300, n=128, layers=12, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    caught = 0
    for _ in range(trials):
        honest = honest_transcript(n=n, layers=layers, rng=rng)
        cheater = make_cheater(honest, rng=rng) if make_cheater else honest
        commitment = cheater.commitment()  # node commits to its own (possibly cheating) work
        challenges = sample_challenges(len(cheater.matmuls), k, rng=rng)
        res = verify_step(cheater, challenges, commitment,
                          expected_shard_root=SHARD_ROOT)
        caught += int(not res.accepted)
    return caught / trials


def main():
    layers, n = 12, 128
    cheats = {
        "honest (control)": None,
        "(a) lazy": lazy,
        "(b) wrong_compute 50%": lambda t, rng: wrong_compute(t, rng=rng, frac=0.5),
        "(d) poison_shard": poison_shard,
    }
    ks = [1, 2, 4, 8]

    print(f"detection rate over {layers} matmuls, n={n}\n")
    header = "cheat".ljust(24) + "".join(f"k={k}".rjust(9) for k in ks)
    print(header)
    print("-" * len(header))
    for name, fn in cheats.items():
        row = name.ljust(24)
        for k in ks:
            row += f"{detection_rate(fn, k=k, n=n, layers=layers):9.3f}"
        print(row)

    # Cost: Freivalds is O(k * n^2); recompute-and-compare is O(layers * n^3).
    print("\nverifier cost vs full recompute (lower is better):")
    for k in ks:
        ratio = (k * n**2) / (layers * n**3)
        print(f"  k={k}: {ratio:.4%} of recompute work")
    print("\nNote: float comparison uses a tolerance (see freivalds.freivalds_check); the "
          "honest control rate measures false positives under that tolerance.")


if __name__ == "__main__":
    main()
