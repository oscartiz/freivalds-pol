import numpy as np

from freivalds_pol.adversary import lazy, poison_shard, wrong_compute
from freivalds_pol.challenge import sample_challenges
from freivalds_pol.transcript import MatMulRecord, StepTranscript
from freivalds_pol.verifier import verify_step

SHARD = bytes([2]) * 32


def make_honest(rng, layers=8, n=48, dtype="fp32"):
    mms = []
    for i in range(layers):
        A = rng.normal(size=(n, n))
        B = rng.normal(size=(n, n))
        mms.append(MatMulRecord(f"l{i}", A, B, A @ B, dtype))
    return StepTranscript("node-1", 7, bytes([1]) * 32, SHARD,
                          rng.normal(size=(n,)), mms)


def test_honest_step_accepted():
    rng = np.random.default_rng(0)
    t = make_honest(rng)
    ch = sample_challenges(len(t.matmuls), 4, rng=rng)
    res = verify_step(t, ch, t.commitment(), expected_shard_root=SHARD)
    assert res.accepted, res.reason


def test_lazy_rejected_with_single_challenge():
    rng = np.random.default_rng(1)
    t = lazy(make_honest(rng), rng=rng)
    ch = sample_challenges(len(t.matmuls), 1, rng=rng)
    res = verify_step(t, ch, t.commitment(), expected_shard_root=SHARD)
    assert not res.accepted


def test_wrong_compute_rejected():
    rng = np.random.default_rng(2)
    t = wrong_compute(make_honest(rng), rng=rng, frac=1.0)
    ch = sample_challenges(len(t.matmuls), 4, rng=rng)
    res = verify_step(t, ch, t.commitment(), expected_shard_root=SHARD)
    assert not res.accepted


def test_poison_shard_rejected():
    rng = np.random.default_rng(3)
    t = poison_shard(make_honest(rng), rng=rng)
    ch = sample_challenges(len(t.matmuls), 1, rng=rng)
    res = verify_step(t, ch, t.commitment(), expected_shard_root=SHARD)
    assert not res.accepted
    assert "shard" in res.reason


def test_post_commitment_edit_rejected():
    rng = np.random.default_rng(4)
    t = make_honest(rng)
    good_commitment = t.commitment()
    t.matmuls[0].C = t.matmuls[0].C + 1.0
    ch = sample_challenges(len(t.matmuls), 1, rng=rng)
    res = verify_step(t, ch, good_commitment, expected_shard_root=SHARD)
    assert not res.accepted
    assert "commitment" in res.reason


def test_below_min_precision_rejected():
    rng = np.random.default_rng(5)
    t = make_honest(rng, dtype="bf16")  # node admits it computed the layer in bf16
    ch = sample_challenges(len(t.matmuls), 1, rng=rng)
    res = verify_step(t, ch, t.commitment(), expected_shard_root=SHARD, min_dtype="fp32")
    assert not res.accepted
    assert "precision" in res.reason
