"""A non-interactive sumcheck argument for matrix multiplication — the core of a ZK matmul check.

Goal (Milestone 4): prove `C = A·B` for a challenged GEMM succinctly and, ultimately, without
revealing `A`, `B`. This module implements the classical multilinear **sumcheck** reduction
(Lund-Fortnow-Karloff-Nisan / Thaler) over a prime field, made non-interactive with Fiat-Shamir:

  C̃(rx, rz) = Σ_{y∈{0,1}^kb} Ã(rx, y) · B̃(y, rz)

The prover sends one degree-2 polynomial per contraction-bit round (3 field elements each) plus
two final evaluations Ã(rx,ry), B̃(ry,rz). The verifier checks the round chain in O(log k) field
ops, then must confirm those two evaluations.

**What is real here:** the sumcheck protocol and its soundness (a wrong `C` is rejected with
probability ≥ 1 − 2·kb/|F| by Schwartz-Zippel) — see `tests/test_zk.py`.

**What is stubbed (honest scope, see docs/DESIGN.md §11b):** confirming the two final evaluations
*without the matrices* needs a **polynomial commitment** (KZG/FRI). `PolyCommitment` is the
interface; `RevealCommitment` is an INSECURE reference that opens by recomputing the MLE from the
data — so the prototype is sound but neither hiding nor succinct on the opening. A real PCS is
marked future work. Costs are measured honestly in `experiments/zk_matmul.py`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

P = 2**61 - 1                       # Mersenne prime field


def _inv(a):
    return pow(a % P, P - 2, P)


class Transcript:
    """Fiat-Shamir transcript: absorb messages, squeeze field challenges."""

    def __init__(self, label=b"zk-matmul"):
        self.h = hashlib.sha256(label)

    def absorb(self, *xs):
        for x in xs:
            self.h.update(int(x % P).to_bytes(8, "big"))
        return self

    def challenge(self):
        d = self.h.digest()
        self.h.update(d)
        return int.from_bytes(d[:8], "big") % P


def chi(point):
    """Equality table: chi(point)[x] = Π_i eq(bit_i(x), point_i), length 2^len(point)."""
    v = [1]
    for r in point:
        r %= P
        one_minus = (1 - r) % P
        v = [(x * one_minus) % P for x in v] + [(x * r) % P for x in v]
    return v


def _matvec(M, v):                  # M (rows x len(v)) @ v  -> length rows
    return [sum(M[i][j] * v[j] for j in range(len(v))) % P for i in range(len(M))]


def _vecmat(v, M):                  # v @ M (len(v) x cols) -> length cols
    cols = len(M[0])
    return [sum(v[i] * M[i][j] for i in range(len(M))) % P for j in range(cols)]


def mle_eval(M, px, py):
    """Multilinear extension of matrix M at (px, py): chi(px)^T M chi(py)."""
    cx, cy = chi(px), chi(py)
    return sum(cx[i] * M[i][j] % P * cy[j] for i in range(len(M)) for j in range(len(M[0]))) % P


def _log2(n):
    b = n.bit_length() - 1
    assert 1 << b == n, "dimensions must be powers of two"
    return b


@dataclass
class MatMulProof:
    rx: list
    rz: list
    rounds: list          # list of (s0, s1, s2) per contraction-bit round
    final_a: int
    final_b: int
    claim: int            # C̃(rx, rz)


def prove(A, B, C, transcript=None):
    """Produce a non-interactive sumcheck proof that C = A·B over the field."""
    t = transcript or Transcript()
    m, k, n = len(A), len(A[0]), len(B[0])
    mb, nb = _log2(m), _log2(n)
    _log2(k)  # assert k is a power of two
    # bind the statement (use commitment hashes in practice; here, dimensions)
    t.absorb(m, k, n)
    rx = [t.challenge() for _ in range(mb)]
    rz = [t.challenge() for _ in range(nb)]
    cx, cz = chi(rx), chi(rz)
    a = _vecmat(cx, A)                       # a[y] = Ã(rx, y),  length k
    b = _matvec(B, cz)                       # b[y] = B̃(y, rz),  length k
    claim = sum(cx[i] * C[i][j] % P * cz[j] for i in range(m) for j in range(n)) % P

    rounds = []
    a, b = a[:], b[:]
    while len(a) > 1:
        half = len(a) // 2
        s0 = sum(a[j] * b[j] for j in range(half)) % P
        s1 = sum(a[half + j] * b[half + j] for j in range(half)) % P
        s2 = sum(((2 * a[half + j] - a[j]) % P) * ((2 * b[half + j] - b[j]) % P)
                 for j in range(half)) % P
        rounds.append((s0, s1, s2))
        t.absorb(s0, s1, s2)
        r = t.challenge()
        a = [(a[j] + r * (a[half + j] - a[j])) % P for j in range(half)]
        b = [(b[j] + r * (b[half + j] - b[j])) % P for j in range(half)]
    return MatMulProof(rx, rz, rounds, a[0], b[0], claim)


def _interp_deg2(s0, s1, s2, r):
    # Lagrange through (0,s0),(1,s1),(2,s2) evaluated at r.
    r %= P
    l0 = (r - 1) * (r - 2) % P * _inv(2) % P
    l1 = r * (r - 2) % P * _inv(P - 1) % P
    l2 = r * (r - 1) % P * _inv(2) % P
    return (s0 * l0 + s1 * l1 + s2 * l2) % P


def verify_sumcheck(proof: MatMulProof, transcript=None, *, m, k, n):
    """Check the sumcheck chain. Returns (ok, ry, challenges) — caller still confirms the two
    final evaluations against commitments (see PolyCommitment)."""
    t = transcript or Transcript()
    t.absorb(m, k, n)
    rx = [t.challenge() for _ in range(_log2(m))]
    rz = [t.challenge() for _ in range(_log2(n))]
    if rx != proof.rx or rz != proof.rz:
        return False, None, None
    claim = proof.claim
    ry = []
    for (s0, s1, s2) in proof.rounds:
        if (s0 + s1) % P != claim % P:
            return False, None, None
        t.absorb(s0, s1, s2)
        r = t.challenge()
        ry.append(r)
        claim = _interp_deg2(s0, s1, s2, r)
    if proof.final_a * proof.final_b % P != claim % P:
        return False, None, None
    return True, ry, ry


class PolyCommitment:
    """Interface a real ZK matmul needs: commit to a matrix's MLE, open at a point with a proof
    the verifier checks WITHOUT the matrix. A real instance is KZG (pairings) or FRI."""

    def commit(self, M):                       # -> commitment
        raise NotImplementedError

    def open(self, M, px, py):                  # -> (value, opening_proof)
        raise NotImplementedError

    def verify_open(self, commitment, px, py, value, opening_proof) -> bool:
        raise NotImplementedError


class RevealCommitment(PolyCommitment):
    """INSECURE reference: 'commitment' is a hash; opening recomputes the MLE from the data.
    Makes the end-to-end flow runnable and sound, but is neither hiding nor succinct — exactly
    the piece a real PCS replaces. Do not use for anything real."""

    def commit(self, M):
        h = hashlib.sha256()
        for row in M:
            for x in row:
                h.update(int(x % P).to_bytes(8, "big"))
        return h.hexdigest()

    def open(self, M, px, py):
        return mle_eval(M, px, py), M           # "proof" reveals the matrix (insecure)

    def verify_open(self, commitment, px, py, value, opening_proof) -> bool:
        return self.commit(opening_proof) == commitment and mle_eval(opening_proof, px, py) == value


def verify_full(proof, A, B, pcs, *, m, k, n, transcript=None):
    """End-to-end check: sumcheck chain + confirm the two final MLE evaluations via the PCS.

    The verifier never uses A, B directly — it gets commitments and opening proofs from the
    prover. (With ``RevealCommitment`` the opening proof carries the matrix, so this is sound but
    not hiding/succinct; a real PCS makes the opening O(log) and zero-knowledge.)
    """
    ok, ry, _ = verify_sumcheck(proof, transcript, m=m, k=k, n=n)
    if not ok:
        return False
    # Sumcheck binds the high contraction-bit first, so the bound point is ry reversed
    # relative to chi()'s bit order.
    ry_pt = list(reversed(ry))
    a_commit, b_commit = pcs.commit(A), pcs.commit(B)
    a_val, a_proof = pcs.open(A, proof.rx, ry_pt)
    b_val, b_proof = pcs.open(B, ry_pt, proof.rz)
    if not pcs.verify_open(a_commit, proof.rx, ry_pt, a_val, a_proof):
        return False
    if not pcs.verify_open(b_commit, ry_pt, proof.rz, b_val, b_proof):
        return False
    return a_val == proof.final_a and b_val == proof.final_b
