import numpy as np

from freivalds_pol.trainer import aligned_adversary, random_adversary, run_training


def test_honest_training_reduces_loss():
    h = run_training(80, lr=0.1)
    assert h["losses"][-1] < 0.8 * h["losses"][0]


def test_runs_are_deterministic():
    a = run_training(20, lr=0.1)
    b = run_training(20, lr=0.1)
    assert np.allclose(a["final"], b["final"])
    assert np.allclose(a["losses"], b["losses"])


def test_subthreshold_cheating_barely_moves_loss():
    R, lr, budget = 120, 0.1, 2e-3
    honest = run_training(R, lr=lr)
    direction = np.random.default_rng(0).normal(size=honest["n_params"])
    aligned = run_training(R, lr=lr, adversary=aligned_adversary(budget, direction))
    # a never-detected sub-threshold cheat does not meaningfully change the loss
    assert abs(aligned["losses"][-1] - honest["losses"][-1]) < 0.02


def test_subthreshold_drift_is_sublinear():
    # Drift grows much slower than the naive linear (free-accumulation) bound.
    R, lr, budget = 200, 0.1, 2e-3
    honest = run_training(R, lr=lr, trajectory=True)
    direction = np.random.default_rng(0).normal(size=honest["n_params"])
    aligned = run_training(R, lr=lr, trajectory=True,
                           adversary=aligned_adversary(budget, direction))
    drift = np.linalg.norm(aligned["traj"] - honest["traj"], axis=1)
    t = np.arange(1, R + 1)
    lo = R // 2
    p = np.polyfit(np.log(t[lo:]), np.log(drift[lo:]), 1)[0]
    assert p < 0.8   # clearly sublinear vs the naive bound's p = 1.0


def test_directed_bias_accumulates_faster_than_random():
    R, lr, budget = 200, 0.1, 2e-3
    honest = run_training(R, lr=lr, trajectory=True)
    direction = np.random.default_rng(0).normal(size=honest["n_params"])
    aligned = run_training(R, lr=lr, trajectory=True,
                           adversary=aligned_adversary(budget, direction))
    randomr = run_training(R, lr=lr, trajectory=True,
                           adversary=random_adversary(budget, seed=1))
    d_aligned = np.linalg.norm(aligned["traj"] - honest["traj"], axis=1)[-1]
    d_random = np.linalg.norm(randomr["traj"] - honest["traj"], axis=1)[-1]
    assert d_aligned > d_random   # error feedback gives a directed cheat a real edge
