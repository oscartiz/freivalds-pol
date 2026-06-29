import numpy as np

from freivalds_pol.model import Transformer, grad_check, make_task, train


def _setup(seed=0, n_layers=2, n_heads=4, d=16, d_ff=32, T=8):
    rng = np.random.default_rng(seed)
    m = Transformer.init(d, d_ff, n_layers, n_heads, rng)
    X, Y = make_task(d, d_ff, n_layers, n_heads, T, rng)
    return m, X, Y


def test_multilayer_multihead_backprop_matches_finite_difference():
    m, X, Y = _setup()
    assert grad_check(m, X, Y) < 1e-4


def test_flat_roundtrip():
    m, _, _ = _setup()
    v = m.flat().copy()
    m.set_flat(np.zeros_like(v))
    m.set_flat(v)
    assert np.array_equal(m.flat(), v)


def test_adamw_reduces_loss():
    h = train(60, d=16, d_ff=32, n_layers=2, n_heads=4, T=8, lr=1e-2)
    assert h["losses"][-1] < 0.6 * h["losses"][0]


def test_adamw_beats_sgd():
    cfg = dict(R=60, d=16, d_ff=32, n_layers=2, n_heads=4, T=8, lr=1e-2)
    adamw = train(optimizer="adamw", **cfg)
    sgd = train(optimizer="sgd", **cfg)
    assert adamw["losses"][-1] < sgd["losses"][-1]


def test_n_heads_must_divide_d():
    import pytest
    with pytest.raises(AssertionError):
        Transformer.init(d=16, d_ff=32, n_layers=1, n_heads=5, rng=np.random.default_rng(0))
