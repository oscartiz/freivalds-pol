import numpy as np

from freivalds_pol.adversary import lazy, poison_shard, wrong_compute
from freivalds_pol.challenge import sample_challenges
from freivalds_pol.training import MLP, grad_check, make_task, step_transcript
from freivalds_pol.verifier import verify_step


def _real_step(seed=0, batch=24, d=(16, 24, 8)):
    rng = np.random.default_rng(seed)
    mlp = MLP.init(*d, rng)
    X, Y = make_task(*d, batch, rng)
    return rng, mlp, X, Y


def test_backprop_matches_finite_difference():
    rng, mlp, X, Y = _real_step()
    assert grad_check(mlp, X, Y) < 1e-3


def test_recorded_matmuls_are_consistent():
    # Every recorded GEMM must actually equal A @ B (the honest claim).
    _, mlp, X, Y = _real_step()
    t, _, _ = step_transcript(mlp, X, Y)
    assert len(t.matmuls) == 5
    for rec in t.matmuls:
        assert np.allclose(rec.C, rec.A @ rec.B)


def test_honest_real_step_verifies():
    rng, mlp, X, Y = _real_step()
    t, _, _ = step_transcript(mlp, X, Y)
    ch = sample_challenges(len(t.matmuls), len(t.matmuls), rng=rng)
    res = verify_step(t, ch, t.commitment(), expected_shard_root=t.shard_root)
    assert res.accepted, res.reason


def test_real_step_cheats_rejected():
    rng, mlp, X, Y = _real_step()
    t, _, _ = step_transcript(mlp, X, Y)
    shard = t.shard_root
    for fn in (lazy, lambda x, rng: wrong_compute(x, rng=rng, frac=0.4), poison_shard):
        tc = fn(t, rng=rng)
        ch = sample_challenges(len(tc.matmuls), len(tc.matmuls), rng=rng)
        res = verify_step(tc, ch, tc.commitment(), expected_shard_root=shard)
        assert not res.accepted
