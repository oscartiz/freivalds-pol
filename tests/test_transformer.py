import numpy as np

from freivalds_pol.adversary import lazy, poison_shard, wrong_compute
from freivalds_pol.challenge import sample_challenges
from freivalds_pol.transformer import (
    TransformerBlock,
    grad_check,
    make_task,
    step_transcript,
)
from freivalds_pol.verifier import verify_step


def _real_step(seed=0, d=16, d_ff=32, T=8):
    rng = np.random.default_rng(seed)
    blk = TransformerBlock.init(d, d_ff, rng)
    X, Y = make_task(d, d_ff, T, rng)
    return rng, blk, X, Y


def test_backprop_matches_finite_difference():
    _, blk, X, Y = _real_step()
    assert grad_check(blk, X, Y) < 1e-4


def test_records_are_the_eight_block_gemms():
    _, blk, X, Y = _real_step()
    t, _, _ = step_transcript(blk, X, Y)
    names = [r.name for r in t.matmuls]
    assert names == ["attn.Q", "attn.K", "attn.V", "attn.scores",
                     "attn.ctx", "attn.out", "mlp.h", "mlp.y"]
    for rec in t.matmuls:
        assert np.allclose(rec.C, rec.A @ rec.B)


def test_honest_block_step_verifies():
    rng, blk, X, Y = _real_step()
    t, _, _ = step_transcript(blk, X, Y)
    ch = sample_challenges(len(t.matmuls), len(t.matmuls), rng=rng)
    res = verify_step(t, ch, t.commitment(), expected_shard_root=t.shard_root)
    assert res.accepted, res.reason


def test_block_step_cheats_rejected():
    rng, blk, X, Y = _real_step()
    t, _, _ = step_transcript(blk, X, Y)
    shard = t.shard_root
    for fn in (lazy, lambda x, rng: wrong_compute(x, rng=rng, frac=0.4), poison_shard):
        tc = fn(t, rng=rng)
        ch = sample_challenges(len(tc.matmuls), len(tc.matmuls), rng=rng)
        res = verify_step(tc, ch, tc.commitment(), expected_shard_root=shard)
        assert not res.accepted
