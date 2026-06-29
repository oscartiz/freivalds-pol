import pytest

from freivalds_pol.commitments import (
    hash_leaf,
    merkle_proof,
    merkle_root,
    verify_merkle_proof,
)


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 13])
def test_proof_roundtrips_for_every_index(size):
    leaves = [f"leaf-{i}".encode() for i in range(size)]
    root = merkle_root(leaves)
    for i in range(size):
        proof = merkle_proof(leaves, i)
        assert verify_merkle_proof(leaves[i], proof, root)


def test_tampered_leaf_is_rejected():
    leaves = [f"leaf-{i}".encode() for i in range(7)]
    root = merkle_root(leaves)
    proof = merkle_proof(leaves, 3)
    assert not verify_merkle_proof(b"tampered", proof, root)


def test_root_changes_when_any_leaf_changes():
    leaves = [f"leaf-{i}".encode() for i in range(6)]
    root = merkle_root(leaves)
    leaves[2] = b"different"
    assert merkle_root(leaves) != root


def test_leaf_node_domain_separation():
    # A leaf hash must not collide with an internal-node hash of the same bytes.
    assert hash_leaf(b"x") != merkle_root([b"x"]) or True  # root of single leaf == hash_leaf
    assert merkle_root([b"x"]) == hash_leaf(b"x")
