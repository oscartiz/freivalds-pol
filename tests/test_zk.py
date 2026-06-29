import numpy as np

from freivalds_pol.zk import (
    P,
    RevealCommitment,
    chi,
    mle_eval,
    prove,
    verify_full,
    verify_sumcheck,
)


def _rand(r, c, rng, hi=1000):
    return [[int(rng.integers(0, hi)) for _ in range(c)] for _ in range(r)]


def _matmul(A, B):
    k = len(B)
    return [[sum(A[i][t] * B[t][j] for t in range(k)) % P for j in range(len(B[0]))]
            for i in range(len(A))]


def test_chi_is_partition_of_unity():
    rng = np.random.default_rng(0)
    pt = [int(rng.integers(0, P)) for _ in range(5)]
    assert sum(chi(pt)) % P == 1


def test_mle_interpolates_matrix_at_boolean_points():
    rng = np.random.default_rng(1)
    M = _rand(4, 4, rng)
    for x in range(4):
        for z in range(4):
            px = [(x >> i) & 1 for i in range(2)]
            pz = [(z >> i) & 1 for i in range(2)]
            assert mle_eval(M, px, pz) == M[x][z] % P


def test_honest_proof_verifies():
    rng = np.random.default_rng(2)
    A, B = _rand(8, 16, rng), _rand(16, 8, rng)
    C = _matmul(A, B)
    assert verify_full(prove(A, B, C), A, B, RevealCommitment(), m=8, k=16, n=8)


def test_tampered_product_rejected():
    rng = np.random.default_rng(3)
    A, B = _rand(8, 8, rng), _rand(8, 8, rng)
    C = _matmul(A, B)
    for _ in range(10):
        Cb = [row[:] for row in C]
        i, j = int(rng.integers(0, 8)), int(rng.integers(0, 8))
        Cb[i][j] = (Cb[i][j] + 1 + int(rng.integers(0, P))) % P
        assert not verify_full(prove(A, B, Cb), A, B, RevealCommitment(), m=8, k=8, n=8)


def test_lying_final_evaluation_rejected():
    rng = np.random.default_rng(4)
    A, B = _rand(8, 8, rng), _rand(8, 8, rng)
    C = _matmul(A, B)
    pf = prove(A, B, C)
    pf.final_a = (pf.final_a + 1) % P
    assert not verify_full(pf, A, B, RevealCommitment(), m=8, k=8, n=8)


def test_sumcheck_chain_alone_accepts_honest():
    rng = np.random.default_rng(5)
    A, B = _rand(4, 8, rng), _rand(8, 4, rng)
    C = _matmul(A, B)
    ok, ry, _ = verify_sumcheck(prove(A, B, C), m=4, k=8, n=4)
    assert ok and len(ry) == 3
