"""End-to-end verification on a REAL transformer-block training step (not synthetic matmuls).

Run:  python -m experiments.real_step   (after `pip install -e .`)

Trains one step of a Llama-style pre-norm transformer block (RMSNorm, single-head causal
attention, GELU MLP) against a teacher block, records the eight GEMMs of the forward pass --
including the data-dependent attention products Q Kᵀ and P V -- and runs the full protocol
verifier (commitment + shard + two-sided calibrated Freivalds) on the honest step and on each
cheat. The gradient is first validated against finite differences, so the matmuls being
checked are genuinely a correct step.
"""

from __future__ import annotations

import numpy as np

from freivalds_pol.adversary import lazy, poison_shard, wrong_compute
from freivalds_pol.challenge import sample_challenges
from freivalds_pol.transformer import TransformerBlock, grad_check, make_task, step_transcript
from freivalds_pol.verifier import verify_step

D, D_FF, T = 128, 256, 64


def main():
    rng = np.random.default_rng(0)
    blk = TransformerBlock.init(D, D_FF, rng)
    X, Y = make_task(D, D_FF, T, rng)

    print(f"backprop finite-difference grad-check (max rel err): {grad_check(blk, X, Y):.2e}")
    t, loss, _ = step_transcript(blk, X, Y)
    print(f"real block step: loss={loss:.4f}, GEMMs={len(t.matmuls)} "
          f"({', '.join(r.name for r in t.matmuls)}), update dim={t.update.size}\n")

    commit, shard = t.commitment(), t.shard_root

    def all_ch(tr):
        return sample_challenges(len(tr.matmuls), len(tr.matmuls), rng=rng)

    print("verifier on the honest step and on each cheat (all 8 GEMMs challenged):")
    res = verify_step(t, all_ch(t), commit, expected_shard_root=shard)
    print(f"  {'honest':18s} -> accepted={res.accepted}   ({res.reason})")
    cheats = [
        ("lazy", lazy),
        ("wrong_compute 40%", lambda x, rng: wrong_compute(x, rng=rng, frac=0.4)),
        ("poison_shard", poison_shard),
    ]
    for name, fn in cheats:
        tc = fn(t, rng=rng)
        r = verify_step(tc, all_ch(tc), tc.commitment(), expected_shard_root=shard)
        print(f"  {name:18s} -> accepted={r.accepted}   ({r.reason})")


if __name__ == "__main__":
    main()
