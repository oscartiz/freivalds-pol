import numpy as np

from freivalds_pol.collusion import detect_free_riders, identity_bound
from freivalds_pol.transcript import MatMulRecord, StepTranscript

SHARD = bytes(32)


def _transcript(node_id, seed, update, rng):
    A = rng.normal(size=(8, 8))
    B = rng.normal(size=(8, 8))
    return StepTranscript(node_id, seed, bytes(32), SHARD, update,
                          [MatMulRecord("l0", A, B, A @ B)])


def test_distinct_honest_updates_not_flagged():
    rng = np.random.default_rng(0)
    ts = [_transcript(f"n{i}", i, rng.normal(size=20), rng) for i in range(5)]
    assert detect_free_riders(ts) == []


def test_copied_update_is_flagged():
    rng = np.random.default_rng(1)
    u = rng.normal(size=20)
    ts = [_transcript("honest", 1, u, rng),
          _transcript("copier", 2, u.copy(), rng),                 # exact copy
          _transcript("other", 3, rng.normal(size=20), rng)]
    groups = detect_free_riders(ts)
    assert len(groups) == 1 and set(groups[0]) == {"honest", "copier"}


def test_colluding_group_flagged_together():
    rng = np.random.default_rng(2)
    u = rng.normal(size=20)
    ts = [_transcript(f"c{i}", i, u.copy(), rng) for i in range(4)]
    groups = detect_free_riders(ts)
    assert len(groups) == 1 and len(groups[0]) == 4


def test_commitment_binds_identity_and_seed():
    rng = np.random.default_rng(3)
    t = _transcript("n", 7, rng.normal(size=20), rng)
    assert identity_bound(t)
