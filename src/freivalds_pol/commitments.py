"""Merkle commitments over training-step data.

A node commits to its step transcript (model-state root, data-shard root, RNG seed,
matmul outputs, and the submitted update) by hashing each into a Merkle tree. The root is
cheap to publish on the coordinator / data-availability layer; later the node can open any
leaf with an O(log n) proof, so a challenger learns a specific value without the node
shipping the whole transcript. This is what lets the challenge be cheap *and* binding.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def _h(*chunks: bytes) -> bytes:
    d = hashlib.sha256()
    for c in chunks:
        d.update(c)
    return d.digest()


def hash_array(arr) -> bytes:
    """Deterministic hash of an ndarray, binding dtype and shape."""
    arr = np.ascontiguousarray(arr)
    header = f"{arr.dtype.str}|{arr.shape}".encode()
    return _h(b"arr", header, arr.tobytes())


def hash_leaf(data: bytes) -> bytes:
    return _h(b"leaf", data)


def merkle_root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return _h(b"empty")
    level = [hash_leaf(x) for x in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])  # duplicate last to pad to even width
        level = [_h(b"node", level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


@dataclass
class MerkleProof:
    index: int
    siblings: list[tuple[bytes, str]]  # (sibling_hash, "L" | "R")


def merkle_proof(leaves: list[bytes], index: int) -> MerkleProof:
    level = [hash_leaf(x) for x in leaves]
    siblings: list[tuple[bytes, str]] = []
    idx = index
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        if idx % 2:
            siblings.append((level[idx - 1], "L"))
        else:
            siblings.append((level[idx + 1], "R"))
        level = [_h(b"node", level[i], level[i + 1]) for i in range(0, len(level), 2)]
        idx //= 2
    return MerkleProof(index=index, siblings=siblings)


def verify_merkle_proof(leaf: bytes, proof: MerkleProof, root: bytes) -> bool:
    node = hash_leaf(leaf)
    for sib, side in proof.siblings:
        node = _h(b"node", sib, node) if side == "L" else _h(b"node", node, sib)
    return node == root
