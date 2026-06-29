"""Verifying the COMPRESSED update -- DisTrO's actual wire format, on a real transformer step.

Run:  python -m experiments.compressed   (after `pip install -e .`)

Takes the real gradient from one transformer-block step, compresses it DeMo-style (momentum
+ per-tile DCT + top-k + error feedback), then verifies the compressed payload per tile.
Shows that the honest payload verifies, that each compression cheat is caught with a
detection rate that climbs with the number of challenged tiles (1-(1-f)^c), and that the
verifier touches only a tiny fraction of the data.
"""

from __future__ import annotations

import numpy as np

from freivalds_pol.compressor import (
    cheat_bad_residual,
    cheat_fake_values,
    cheat_lazy,
    cheat_wrong_topk,
    compress,
    verify_compressed,
)
from freivalds_pol.transformer import TransformerBlock, make_task, step_transcript

D, D_FF, T = 128, 256, 64
TILE, K, DECAY = 64, 8, 0.9


def real_gradient(seed=0):
    rng = np.random.default_rng(seed)
    blk = TransformerBlock.init(D, D_FF, rng)
    X, Y = make_task(D, D_FF, T, rng)
    t, _, _ = step_transcript(blk, X, Y)
    return t.update                       # the dense gradient a node would compress


def main():
    grad = real_gradient()
    m_prev = np.random.default_rng(1).normal(size=grad.size) * 0.1
    cu, m_next, applied = compress(grad, m_prev, decay=DECAY, tile=TILE, k=K)
    n_tiles = cu.indices.shape[0]
    print(f"dense gradient dim = {grad.size}; compressed to {cu.indices.size} coeffs "
          f"({n_tiles} tiles x k={K}) = {cu.indices.size / grad.size:.1%} of the data\n")

    res = verify_compressed(grad, m_prev, m_next, cu, range(n_tiles), decay=DECAY)
    print(f"honest compressed update -> accepted={res.accepted} ({res.reason})\n")

    cheats = {
        "lazy (zeros)": cheat_lazy,
        "fake values": cheat_fake_values,
        "wrong top-k": cheat_wrong_topk,
        "bad residual": cheat_bad_residual,
    }
    challenges = [1, 2, 4, 8]
    frac = 0.25
    print(f"detection rate vs. #challenged tiles  (cheat corrupts {frac:.0%} of {n_tiles} tiles)\n")
    header = "cheat".ljust(18) + "".join(f"c={c}".rjust(9) for c in challenges)
    print(header + "\n" + "-" * len(header))
    for name, cheat in cheats.items():
        row = name.ljust(18)
        for c in challenges:
            caught = 0
            trials = 200
            for s in range(trials):
                rng = np.random.default_rng(s)
                cu_c, mn_c = cheat(cu, m_next, frac=frac, rng=rng)
                tiles = rng.choice(n_tiles, c, replace=False)
                r = verify_compressed(grad, m_prev, mn_c, cu_c, tiles, decay=DECAY)
                caught += int(not r.accepted)
            row += f"{caught / trials:9.3f}"
        print(row)

    print(f"\nverifier work per challenge ~ tile^2 = {TILE**2} ops; at c=8 it inspects "
          f"{8 / n_tiles:.1%} of the {n_tiles} tiles. The expensive gradient behind the")
    print("update is checked separately by Freivalds; here the DCT/top-k recompute is cheap.")


if __name__ == "__main__":
    main()
