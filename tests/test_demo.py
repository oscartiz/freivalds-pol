import numpy as np

from freivalds_pol.compressor import dct_matrix
from freivalds_pol.demo import (
    _dct2,
    _from_blocks,
    _idct2,
    _to_blocks,
    decode,
    encode,
    verify,
)

C8 = 8


def test_2d_dct_roundtrips():
    C = dct_matrix(C8)
    rng = np.random.default_rng(0)
    blocks = rng.normal(size=(5, C8, C8))
    assert np.allclose(_idct2(_dct2(blocks, C), C), blocks, atol=1e-10)


def test_blocking_roundtrips():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(24, 16))
    assert np.allclose(_from_blocks(_to_blocks(W, 8), (24, 16), 8), W, atol=1e-12)


def test_constant_block_is_pure_dc():
    # 2D-DCT of a constant block puts ALL energy in the (0,0) DC coefficient = v*c.
    C = dct_matrix(C8)
    v = 0.37
    block = np.full((1, C8, C8), v)
    coeff = _dct2(block, C)[0]
    assert np.isclose(coeff[0, 0], v * C8, atol=1e-10)
    assert np.allclose(coeff.ravel()[1:], 0.0, atol=1e-10)


def test_known_single_coefficient_topk():
    # A block that is the inverse-DCT of a one-hot coefficient must report that exact top-1.
    C = dct_matrix(C8)
    coeff = np.zeros((1, C8, C8))
    coeff[0, 3, 5] = 2.5
    block = _idct2(coeff, C)              # exactly one coefficient present
    W = _from_blocks(block, (C8, C8), C8)
    du, _, _ = encode(W, np.zeros_like(W), decay=0.0, lr=1.0, chunk=C8, k=1)
    assert du.idx[0, 0] == 3 * C8 + 5
    assert np.isclose(du.val[0, 0], 2.5, atol=1e-9)


def test_error_feedback_invariant():
    rng = np.random.default_rng(2)
    grad = rng.normal(size=(16, 16))
    delta_prev = rng.normal(size=(16, 16)) * 0.1
    du, delta_next, applied = encode(grad, delta_prev, decay=0.9, lr=1.0, chunk=8, k=4)
    delta = 0.9 * delta_prev + grad
    assert np.allclose(applied + delta_next, delta, atol=1e-10)
    assert np.allclose(decode(du, dct_matrix(8)), applied, atol=1e-10)


def test_honest_payload_verifies_and_cheats_rejected():
    rng = np.random.default_rng(3)
    grad = rng.normal(size=(16, 16))
    delta_prev = rng.normal(size=(16, 16)) * 0.1
    du, delta_next, _ = encode(grad, delta_prev, decay=0.9, lr=1.0, chunk=8, k=4)
    blocks = range(du.idx.shape[0])
    assert verify(grad, delta_prev, delta_next, du, blocks, decay=0.9, lr=1.0).accepted

    # tamper a value
    du2 = encode(grad, delta_prev, decay=0.9, lr=1.0, chunk=8, k=4)[0]
    du2.val[0, 0] += 1.0
    assert not verify(grad, delta_prev, delta_next, du2, blocks, decay=0.9, lr=1.0).accepted

    # wrong gradient
    assert not verify(grad + 0.5, delta_prev, delta_next, du, blocks, decay=0.9, lr=1.0).accepted
